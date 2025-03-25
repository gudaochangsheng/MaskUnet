import numpy as np
import torch
import random
from typing import Optional, Union, Tuple, List, Callable, Dict, Any
from copy import deepcopy
from accelerate import Accelerator

from diffusers.utils import USE_PEFT_BACKEND, deprecate, scale_lora_layers, unscale_lora_layers, BaseOutput
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from dataclasses import dataclass
from functools import partial
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.Tensor = None

def weights_init_kaiming(m):
    """
    Kaiming (He) Initialization for Conv2D and Linear layers.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.01)  # 初始化为接近恒等映射的小值
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # 偏置设置为0

class DecoderWeightsMaskGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, factor=16):
        super().__init__()

        new_out = out_channels // factor
        new_in = in_channels // factor
        # new_out = 128
        # new_in = 128
        self.conv_kernel_mask = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, new_in * new_out, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.proj_temb = nn.Linear(1280, in_channels, bias=False)
        # self.proj_pemb = nn.Linear(768, in_channels, bias=False)

        self.in_c = new_in
        self.out_c = new_out
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (1, 1) and m.out_channels == self.in_c * self.out_c:
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, weight_shape, sample, res_sample, temb, encoder_hidden_states):
        batch_size = sample.size(0)
        flag = len(weight_shape)
        #emb [n,c]
        temb = self.proj_temb(temb).unsqueeze(-1).unsqueeze(-1)
        # pemb = self.proj_pemb(encoder_hidden_states).permute(0, 2, 1).contiguous().mean(-1).unsqueeze(-1).unsqueeze(-1)
        if res_sample is None:
            x = sample
        else:
            x = torch.cat([sample, res_sample], dim=1)
        if flag == 4:
            _, _, k_h, k_w = weight_shape
            x = F.adaptive_avg_pool2d(x, (k_h, k_w))
        else:
            x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x + temb
        # x = temb

        mask = self.conv_kernel_mask(x)
        if flag == 4:
            mask = mask.view(mask.size(0), self.out_c, self.in_c, k_h, k_w)
        else:
            mask = mask.view(mask.size(0), self.out_c, self.in_c)

        return mask


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Adapter(nn.Module):
    def __init__(self, out_c, in_c, new_out_c, new_in_c, tau = 1.0):
        super(Adapter, self).__init__()
        # Use 1x1 convolutions for channel adaptation
        self.conv_in = nn.Conv2d(out_c, new_out_c, kernel_size=1)
        self.conv_out = nn.Conv2d(in_c, new_in_c, kernel_size=1)
        self.tau = tau

        self.apply(self.weights_init)

    def weights_init(self, m):
        """
        Kaiming (He) Initialization for Conv2D and Linear layers.
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)  # 初始化为0
            if m.bias is not None:
                nn.init.constant_(m.bias, 5.0)  # 偏置设置为5.0

    def forward(self, input_tensor_3x3):
        # The input tensor shape is [batch_size, out_c, in_c, k_h, k_w]
        
        # Step 1: Adapt the in_channels using a 1x1 convolution
        # We rearrange the tensor to merge the out_c dimension into the batch size for efficient processing
        if input_tensor_3x3.dim() == 3:
            input_tensor_3x3 = input_tensor_3x3.unsqueeze(-1).unsqueeze(-1)
        input_adapted = rearrange(input_tensor_3x3, 'b out_c in_c h w -> (b in_c) out_c h w')
        
        # if self.conv_in.bias is not None:
        #     self.conv_in.bias.data = self.conv_in.bias.data.to(input_adapted.dtype)
        # Apply the in-channel adaptation
        input_adapted = self.conv_in(input_adapted)
        
        # Step 2: Adapt the out_channels
        # Rearrange to bring the adapted in_channels back into the batch dimension for processing
        input_adapted = rearrange(input_adapted, '(b in_c) new_out_c h w -> b new_out_c in_c h w', b=input_tensor_3x3.shape[0])
        input_adapted = rearrange(input_adapted, 'b new_out_c in_c h w -> (b new_out_c) in_c h w')

        # if self.conv_out.bias is not None:
        #     self.conv_out.bias.data = self.conv_out.bias.data.to(input_adapted.dtype)
        # Apply the out-channel adaptation
        output = self.conv_out(input_adapted)
        output = gumbel_sigmoid(output, tau=self.tau, hard=True)
        # Step 3: Reshape the output back to the original format with new dimensions
        output = rearrange(output, '(b new_out_c) new_in_c h w -> b new_out_c new_in_c h w', b=input_tensor_3x3.shape[0])

        if output.size(-1)==1:
            output = output.squeeze(-1).squeeze(-1)
        return output


def get_hook_fn(sample, temb, res_sample, mask_generator, adapter, encoder_hidden_states):
    def hook_fn(module, input, output):
        batch_size = input[0].size(0)

        if isinstance(module, nn.Conv2d):
            weight_shape = module.weight.shape
            mask = mask_generator(weight_shape, sample, res_sample, temb, encoder_hidden_states).to(module.weight.device)
            # print(module.weight.shape)
            # print(mask.shape)
            # print()
            mask = adapter(mask)
            masked_weight = module.weight * mask
            masked_weight = masked_weight.reshape(-1, *module.weight.shape[1:]).contiguous()  # 使用 reshape 替代 view
            input_reshaped = input[0].reshape(1, -1, *input[0].shape[2:]).contiguous()  # 使用 reshape 替代 view

            # 如果有 bias，则需要处理 bias 的形状
            if module.bias is not None:
                # 确保 bias 的形状正确，并直接扩展到 batch_size 维度
                masked_bias = module.bias.repeat(batch_size).contiguous()  # 扩展 bias 到与分组卷积匹配的大小
            else:
                masked_bias = None

            output = F.conv2d(
                input_reshaped,
                masked_weight,
                bias=masked_bias,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=batch_size
            )
            output = output.reshape(batch_size, -1, *output.shape[2:]).contiguous()  # 使用 reshape 替代 view
            return output

        elif isinstance(module, nn.Linear):
            if module.weight.dim() == 2:
                weight_shape = module.weight.shape
                mask = mask_generator(weight_shape, sample, res_sample, temb, encoder_hidden_states).to(module.weight.device)
                mask = adapter(mask)
                masked_weight = module.weight * mask
                if input[0].dim() == 2:
                    input_batched = input[0].unsqueeze(1)  # (batch_size, 1, in_features)
                else:
                    input_batched = input[0]
                # print(input_batched.shape)
                # assert 2==1
                output = torch.bmm(input_batched, masked_weight.permute(0, 2, 1).contiguous()).squeeze(1)
                if module.bias is not None:
                    output += module.bias.unsqueeze(0).expand_as(output)
                return output
    return hook_fn

# def __init__(self, *args, cross_attention_dim=768, **kwargs):
        # super(FineGrainedUNet2DConditionModel, self).__init__(*args, cross_attention_dim=cross_attention_dim, **kwargs)
class FineGrainedUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, cross_attention_dim=768, use_linear_projection=False,**kwargs):
        super(FineGrainedUNet2DConditionModel, self).__init__(*args, cross_attention_dim=cross_attention_dim, use_linear_projection=use_linear_projection, **kwargs)
        self.mask_generators = nn.ModuleList()
        # self.mask_generators_down = nn.ModuleList()
        # self.layer_adapters = {}
        self.adapters = nn.ModuleList()  # 使用 ModuleList 代替普通 list
        # self.adapters_down = nn.ModuleList()  # 使用 ModuleList 代替普通 list
        # self.decoder_weights_mask_generator.apply(self.init_weights)

        self._hooks = []

        # for name, module in self.up_blocks.named_modules():
        #     if 'proj_out' in name:
        #         print(f"{name} layer structure: {module}")
        #         print(f"Shape of {name} weights: {module.weight.shape}")

        # assert 2==1

        skip_layers = ["time_emb_proj", "ff", "conv_shortcut", "proj_in", "proj_out"]

        # for i, downsample_block in enumerate(self.down_blocks):
        #     print(f"--- Upsample Block {i} ---")
        #     first_resnet_conv1 = downsample_block.resnets[0].conv1  # Get resnets.0.conv1 layer
        #     if isinstance(first_resnet_conv1, nn.Conv2d):
        #         in_channels = first_resnet_conv1.in_channels
        #         out_channels = first_resnet_conv1.out_channels
        #         # out_channels = 256
        #         print(f"Block {i}: in_channels={in_channels}, out_channels={out_channels}")

        #         # Initialize mask_generator for this block
        #         mask_generator = DecoderWeightsMaskGenerator(in_channels, out_channels)
        #         self.mask_generators_down.append(mask_generator)
        #         block_adapters = nn.ModuleDict()  # 使用 ModuleDict 来存储 block 的 adapter
        #         # Use these channels for all adapters in this block
        #         for name, sub_module in downsample_block.named_modules():
        #             # if isinstance(sub_module, nn.Conv2d) and not any(skip_layer in name for skip_layer in skip_layers):
        #             #     sub_in_channels = sub_module.in_channels
        #             #     sub_out_channels = sub_module.out_channels
        #             if isinstance(sub_module, nn.Linear) and not any(skip_layer in name for skip_layer in skip_layers):
        #                 sub_in_channels = sub_module.in_features
        #                 sub_out_channels = sub_module.out_features
        #             else:
        #                 continue

        #             # 替换 name 中的 "." 为 "_"
        #             sanitized_name = name.replace(".", "_")
        #             print(f"Adapter for Layer {sanitized_name}: new_in_c={sub_in_channels}, new_out_c={sub_out_channels}")

        #             # print(f"Adapter for Layer {name}: new_in_c={sub_in_channels}, new_out_c={sub_out_channels}")

        #             # Initialize Adapter using first_resnet_conv1 channels and layer's channels
        #             # factor
        #             factor = 8
        #             adapter = Adapter(out_channels // factor, in_channels // factor, sub_out_channels, sub_in_channels)
        #             block_adapters[sanitized_name] = adapter
        #         self.adapters_down.append(block_adapters)
        # factor = [16, 8, 4, 2]
        for i, upsample_block in enumerate(self.up_blocks):
            print(f"--- Upsample Block {i} ---")
            first_resnet_conv1 = upsample_block.resnets[0].conv1  # Get resnets.0.conv1 layer
            if isinstance(first_resnet_conv1, nn.Conv2d):
                in_channels = first_resnet_conv1.in_channels
                out_channels = first_resnet_conv1.out_channels
                # out_channels = 256
                # print(f"Block {i}: in_channels={in_channels}, out_channels={out_channels}")
                factor = 4
                # Initialize mask_generator for this block
                mask_generator = DecoderWeightsMaskGenerator(in_channels, out_channels, factor = factor)
                self.mask_generators.append(mask_generator)
                block_adapters = nn.ModuleDict()  # 使用 ModuleDict 来存储 block 的 adapter
                # Use these channels for all adapters in this block
                for name, sub_module in upsample_block.named_modules():
                    # if isinstance(sub_module, nn.Conv2d) and not any(skip_layer in name for skip_layer in skip_layers):
                    #     sub_in_channels = sub_module.in_channels
                    #     sub_out_channels = sub_module.out_channels
                    if isinstance(sub_module, nn.Linear) and not any(skip_layer in name for skip_layer in skip_layers):
                        sub_in_channels = sub_module.in_features
                        sub_out_channels = sub_module.out_features
                    else:
                        continue

                    # 替换 name 中的 "." 为 "_"
                    sanitized_name = name.replace(".", "_")
                    # print(f"Adapter for Layer {sanitized_name}: new_in_c={sub_in_channels}, new_out_c={sub_out_channels}")

                    # print(f"Adapter for Layer {name}: new_in_c={sub_in_channels}, new_out_c={sub_out_channels}")

                    # Initialize Adapter using first_resnet_conv1 channels and layer's channels
                    # factor
                    # factor = 16
                    adapter = Adapter(out_channels // factor, in_channels // factor, sub_out_channels, sub_in_channels)
                    # adapter = Adapter(128, 128, sub_out_channels, sub_in_channels)
                    block_adapters[sanitized_name] = adapter
                self.adapters.append(block_adapters)

        # assert 2==1

        # for i, upsample_block in enumerate(self.up_blocks):
        #     print(f"--- Upsample Block {i} ---")
        #     first_resnet_conv1 = upsample_block.resnets[0].conv1  # 获取 resnets.0.conv1 卷积层
        #     if isinstance(first_resnet_conv1, nn.Conv2d):
        #         in_channels = first_resnet_conv1.in_channels
        #         out_channels = first_resnet_conv1.out_channels
        #         self.mask_generators.append(DecoderWeightsMaskGenerator(in_channels, out_channels))

        # assert 2==1

    @staticmethod
    def init_weights(module):
        # 初始化 nn.Linear 层
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 2.0)

        # # 初始化 nn.Parameter 参数
        elif isinstance(module, nn.Parameter):
            nn.init.constant_(module, 0)  # 初始化为0

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        aug_emb = self.get_aug_embed(
            emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )
        if self.config.addition_embed_type == "image_hint":
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
        )

        # 2. pre-process
        # hook = self.conv_in.register_forward_hook(get_hook_fn(sample, emb))
        # self._hooks.append(hook)
        # self.conv_in.decoder_weights_mask_generator = self.decoder_weights_mask_generator
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)

        for i, downsample_block in enumerate(self.down_blocks):

            # mask_generator = self.mask_generators_down[i]
            # adapters = self.adapters_down[i]

            # self._register_hooks(downsample_block, sample, None, emb, mask_generator, adapters)
            # self._register_hooks(downsample_block, sample, emb)
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            # self._register_hooks(self.mid_block, sample, emb)
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            # print("sample", sample.shape)
            mask_generator = self.mask_generators[i]
            adapters = self.adapters[i]
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]
            # 打印 res_samples 的形状
            # print(res_samples[-1].shape)
            # for i, res_sample in enumerate(res_samples):
            #     print(f"Shape of res_sample {i}: {res_sample.shape}")

            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]
            else:
                upsample_size = None

            self._register_hooks(upsample_block, sample ,res_samples[-1], emb, mask_generator, adapters, encoder_hidden_states)
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            # self._register_hooks(self.conv_norm_out, sample, emb)
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        # self._register_hooks(self.conv_out, sample, emb)
        sample = self.conv_out(sample)

        # Remove hooks
        for h in self._hooks:
            h.remove()
        self._hooks = []

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    def _register_hooks(self, module, sample, res_sample, temb, mask_generators, adapters, encoder_hidden_states):
        skip_layers = ["time_emb_proj", "ff", "conv_shortcut", "proj_in", "proj_out"]
        for name, sub_module in module.named_modules():
            #nn.Conv2d,
            if isinstance(sub_module, (nn.Linear)) and not any(skip_layer in name for skip_layer in skip_layers):
                # 需要使用 sanitized_name 替换 "." 为 "_"
                sanitized_name = name.replace(".", "_")
                
                # 使用 "in" 判断键是否存在
                if sanitized_name in adapters:
                    adapter = adapters[sanitized_name]
                if adapter is None:
                    continue
                hook = sub_module.register_forward_hook(get_hook_fn(sample, temb, res_sample, mask_generators, adapter, encoder_hidden_states))
                self._hooks.append(hook)
                # print(f"Hook registered for layer: {name}")


# def main():
#     # 示例训练代码
#     unet = FineGrainedUNet2DConditionModel.from_pretrained(
#         "../share/runwayml/stable-diffusion-v1-5", subfolder="unet",
#         low_cpu_mem_usage=False,
#         device_map=None
#     )

#     # print(unet)

#     # 冻结其他参数，只训练 decoder_weights_mask_generator
#     for param in unet.parameters():
#         param.requires_grad = False

#     for param in unet.mask_generators.parameters():
#         param.requires_grad = True

#     for block_adapters in unet.adapters:
#         for adapter in block_adapters.values():
#             for param in adapter.parameters():
#                 param.requires_grad = True

#     sample = torch.randn(1, 4, 64, 64)
#     timestep = torch.tensor([50])
#     encoder_hidden_states = torch.randn(1, 77, 768)

#     output = unet(sample, timestep, encoder_hidden_states)['sample']

#     # 假设一个简单的损失函数
#     target = torch.randn_like(output)
#     loss = F.mse_loss(output, target)
#     loss.backward()

#     # 检查梯度
#     for i, mask_generator in enumerate(unet.mask_generators):
#         for param_name, param in mask_generator.named_parameters():
#             if param.grad is None:
#                 print(f"No gradient for mask_generator in block {i}, parameter: {param_name}")
#             else:
#                 print(f"Gradient for mask_generator in block {i}, parameter {param_name}: {param.grad.norm()}")


#     for i, block_adapters in enumerate(unet.adapters):
#         for name, adapter in block_adapters.items():
#             for param_name, param in adapter.named_parameters():
#                 if param.grad is None:
#                     print(f"No gradient for adapter {name} in block {i}, parameter: {param_name}")
#                 else:
#                     print(f"Gradient for adapter {name} in block {i}, parameter {param_name}: {param.grad.norm()}")

def main():
    # Initialize the accelerator with FP16 precision
    accelerator = Accelerator(mixed_precision="fp16")

    # Initialize model
    unet = FineGrainedUNet2DConditionModel.from_pretrained(
        "../share/runwayml/stable-diffusion-v1-5",
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
    )

    # Freeze other parameters, only train mask_generators and adapters
    for param in unet.parameters():
        param.requires_grad = False
    for param in unet.mask_generators.parameters():
        param.requires_grad = True
    for block_adapters in unet.adapters:
        for adapter in block_adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-4
    )

    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} dtype: {param.dtype}")
        if param.dtype == "torch.float32":
            print("**********************")

    # Prepare model and optimizer with accelerator
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # Simulate input (keep as FP32)
    sample = torch.randn(1, 4, 64, 64).to(accelerator.device)
    timestep = torch.tensor([50]).to(accelerator.device)
    encoder_hidden_states = torch.randn(1, 77, 768).to(accelerator.device)

    # Forward pass
    optimizer.zero_grad()
    with accelerator.autocast():
        output = unet(sample, timestep, encoder_hidden_states)["sample"]
        # Compute loss
        target = torch.randn_like(output)
        loss = F.mse_loss(output, target)

    # Backward pass
    accelerator.backward(loss)

    # Clip gradients (no need to unscale manually)
    params_to_clip = [p for p in unet.parameters() if p.requires_grad]
    accelerator.clip_grad_norm_(params_to_clip, max_norm=1.0)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     main()
