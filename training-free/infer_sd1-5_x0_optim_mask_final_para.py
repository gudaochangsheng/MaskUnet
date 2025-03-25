import torch
import torch.nn as nn
import os
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from sd_utils_x0_dpm import register_sd_forward, register_sdschedule_step
import ImageReward as RM
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from torchvision.transforms import ToPILImage
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from accelerate import Accelerator  # 导入 Accelerator
import gc
from hpsv2_loss import HPSV2Loss
import xformers
import time  # 导入 time 模块以记录时间
import json

def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than threshold to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: [..., num_features] unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if True, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as logits from the Gumbel-Sigmoid distribution.
      If hard=True, the returned samples are descretized according to threshold, otherwise they will
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

class MaskApplier:
    def __init__(self, unet):
        self.unet = unet
        self.masks = nn.ParameterDict()
        self.hooks = []
        self.mask_probs = {}
        self.mask_samples = {}
        self.initial_masks = {}  # 保存初始mask状态
        self.module_dict = {}  # 添加此字典来存储 safe_name 到 module 的映射

        # Initialize mask logits
        skip_layers = ["time_emb_proj", "ff", "conv_shortcut", "proj_in", "proj_out"]
        for name, module in self.unet.up_blocks.named_modules():
            if isinstance(module, (torch.nn.Linear)) and not any(skip_layer in name for skip_layer in skip_layers):
                safe_name = name.replace('.', '_')
                mask_shape = module.weight.shape
                # Initialize logits to a high value to start with masks mostly turned on
                mask_logit = torch.ones(mask_shape, device=module.weight.device, dtype=torch.float32)
                self.masks[safe_name] = nn.Parameter(mask_logit)
                self.initial_masks[safe_name] = mask_logit.clone()  # 保存初始状态
                self.module_dict[safe_name] = module  # 保存 safe_name 到 module 的映射
                hook = module.register_forward_hook(self.hook_fn(safe_name))
                self.hooks.append(hook)

    def hook_fn(self, layer_name):
        def apply_mask(module, input, output):
            # Use gumbel_sigmoid to sample mask
            logits = self.masks[layer_name].to(module.weight.device)
            mask = gumbel_sigmoid(logits, hard=True)
            batch_size = input[0].size(0)
            # Store the probabilities and samples for policy gradient update
            # with torch.no_grad():
            #     probs = torch.sigmoid(logits)
            #     self.mask_probs[layer_name] = probs.detach()
            #     self.mask_samples[layer_name] = mask.detach()
            # Apply the binary mask to the weights
            masked_weight = module.weight * mask
            # Recompute the output using the masked weights without modifying module.weight
            if isinstance(module, nn.Conv2d):
                bias = module.bias
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                groups = module.groups
                # Recompute the output
                return F.conv2d(input[0], masked_weight, bias, stride, padding, dilation, groups)
            elif isinstance(module, nn.Linear):
                weight_shape = module.weight.shape
                # print("input", input[0].shape)

                output = F.linear(input[0], masked_weight)
                if module.bias is not None:
                    output += module.bias
                return output
            else:
                return output  # For layers that are not Conv2d or Linear
        return apply_mask

    def reset_masks(self):
        # 重置所有 mask 到初始状态
        for name in self.masks.keys():
            self.masks[name].data = self.initial_masks[name].clone()


class MaskOptimizer:
    def __init__(self, unet, mask_applier, reward_model, hpsv2_model, accelerator, num_iters=20):
        self.unet = unet
        self.reward_model = reward_model
        self.hpsv2_model = hpsv2_model
        self.mask_applier = mask_applier
        self.num_iters = num_iters
        self.latent_cache = None
        self.accelerator = accelerator  # 保存 accelerator 实例

    def optimize_masks(self, prompt, pipe, num_iterations=20, seed=42, num_inference_steps=10):
        rm_input_ids = self.reward_model.module.blip.tokenizer(
                    prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt"
                ).input_ids.to(self.accelerator.device)
        rm_attention_mask = self.reward_model.module.blip.tokenizer(
            prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt"
        ).attention_mask.to(self.accelerator.device)
        gen_image = []
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.mask_applier.masks.parameters()), lr=1e-2)
        optimizer = self.accelerator.prepare(optimizer)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.mask_applier.masks.parameters()), lr=1e-2)
        # 对每个时间步执行独立的 mask 优化
        for stop_step in range(0, num_inference_steps):
            pipe.stop_step = stop_step
            if stop_step <= 6:
                num_iterations = 1
            else:
                num_iterations = 10
            for iteration in range(num_iterations):
                # generator = torch.Generator(device="cuda").manual_seed(seed)
                generator = None

                # Generate image
                with self.accelerator.autocast():
                    image, cur_latents = pipe.forward(prompt=prompt, num_inference_steps=num_inference_steps, 
                    guidance_scale=7.5, generator=generator, 
                    latents = self.latent_cache, return_dict=False,
                    widh=512,height=512)
                x_0_per_step = image

                # 保存生成的图像
                if stop_step == num_inference_steps-1 and iteration == num_iterations-1:
                    save_imge = x_0_per_step.squeeze(0)
                    to_pil = ToPILImage()
                    image_save = to_pil(save_imge.cpu())
                    gen_image.append(image_save)
                    # image_save.save(f"{save_path}/{prompt}/seed{seed}_step{stop_step+1}_iter{iteration+1}_x0.png")
                # gc.collect()
                # torch.cuda.empty_cache()  # 清理
                # 预处理图像
                rm_preprocess = Compose([
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
                x_0_per_step = rm_preprocess(x_0_per_step)
                x_0_per_step = x_0_per_step.to(self.accelerator.device)

                

                # 准备输入数据
                
                # torch.autograd.set_detect_anomaly(True)
                # Compute reward and loss
                with self.accelerator.autocast():
                    hpsv2_loss = self.hpsv2_model.score_grad(x_0_per_step, prompt)
                    # print("hpsv2", hpsv2_loss)
                    reward = self.reward_model.module.score_gard(rm_input_ids, rm_attention_mask, x_0_per_step)
                    loss = F.relu(-reward + 2).mean()
                    loss = loss + 5.0*hpsv2_loss
                    # print(loss)
                    


                
                self.accelerator.backward(loss)  # 使用 accelerator 进行反向传播

                # for name, param in self.mask_applier.masks.items():
                #     if param.grad is not None:
                #         print(f"Mask logits for {name}, grad mean: {param.grad.mean().item()}, grad std: {param.grad.std().item()}")

                # 可选：梯度裁剪
                # self.accelerator.clip_grad_norm_(self.unet.parameters(), max_norm=1.0)
                # self.accelerator.clip_grad_norm_(self.mask_applier.masks.parameters(), max_norm=1.0)

                optimizer.step()  # 更新参数
                # for name, param in self.unet.named_parameters():
                #     if param.requires_grad:
                #         mask = gumbel_sigmoid(param.data, tau=1.0, hard=True)
                #         zeros = (mask == 0).sum().item()
                #         ones = (mask == 1).sum().item()
                #         print(f"After update - {name}: max {param.data.max()}, min {param.data.min()}")
                #         print(f"Gumbel Sigmoid applied on {name}: Zeros={zeros}, Ones={ones}")
                optimizer.zero_grad()  # 重置优化器梯度

                # 打印参数更新信息
                # for name, param in self.unet.named_parameters():
                #     if param.requires_grad:
                #         print(f"After update - {name}: max {param.data.max()}, min {param.data.min()}")
                # for name, param in self.unet.named_parameters():
                #     if param.requires_grad:
                #         mask = gumbel_sigmoid(param.data, tau=1.0, hard=True)
                #         zeros = (mask == 0).sum().item()
                #         ones = (mask == 1).sum().item()
                #         print(f"After update - {name}: max {param.data.max()}, min {param.data.min()}")
                #         print(f"Gumbel Sigmoid applied on {name}: Zeros={zeros}, Ones={ones}")

                # 清理内存
                if stop_step == num_inference_steps-1 and iteration == num_iterations-1:
                    print(f"Step {stop_step+1}/{num_inference_steps}, Iteration {iteration + 1}/{num_iterations}, Reward: {reward.item()}")
                del image, x_0_per_step, reward, loss, hpsv2_loss
                gc.collect()
                torch.cuda.empty_cache()  # 添加这行代码释放显存
                
            self.latent_cache = cur_latents.detach().clone()
            # 每个时间步结束后重置mask
            # gc.collect()
            self.mask_applier.reset_masks()
            torch.cuda.empty_cache()
            # print(f"Mask reset after step {stop_step+1}")
        return gen_image

def main():
    # Set a fixed seed for reproducibility
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # 初始化 Accelerator
    accelerator = Accelerator(device_placement=True, mixed_precision='fp16')  # 使用混合精度

    # Load the UNet model
    # unet = UNet2DConditionModel.from_pretrained(
    #     "../share/runwayml/stable-diffusion-v1-5", subfolder="unet",
    #     torch_dtype=torch.float32
    # )

    # 创建 MaskApplier 实例
    

    # scheduler = DDIMScheduler.from_pretrained("../share/runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("../share/runwayml/stable-diffusion-v1-5", subfolder="scheduler", algorithm_type="dpmsolver++")
    # Load the Stable Diffusion pipeline with the modified UNet model
    pipe = StableDiffusionPipeline.from_pretrained(
        "../share/runwayml/stable-diffusion-v1-5",  torch_dtype=torch.float16,
        scheduler=scheduler
    )

    mask_applier = MaskApplier(pipe.unet)

    # 使用 accelerator.prepare() 一起包装模型和 MaskApplier
    unet, mask_applier = accelerator.prepare(pipe.unet, mask_applier)

    pipe.to(accelerator.device)

    # pipe.unet.enable_gradient_checkpointing()
    pipe.unet.enable_xformers_memory_efficient_attention()

    register_sd_forward(pipe)
    register_sdschedule_step(pipe.scheduler)

    def dummy_checker(images, **kwargs):
        # Return the images as is, and set NSFW detection results to False (not NSFW)
        return images, [False] * len(images)

    pipe.safety_checker = dummy_checker

    save_dir = "training-free_mask_fix_stop7_iter10"

    # Create MaskApplier and ImageReward instances
    reward_model = RM.load("ImageReward-v1.0").to(device = accelerator.device, dtype = torch.float16)
    reward_model = accelerator.prepare(reward_model)
    #HPSV2 model
    hpsv2_model = HPSV2Loss(
        dtype = torch.float16,
        device = accelerator.device,
        cache_dir = "./HPSV2_checkpoint"
        # memsave = True
    )
    hpsv2_model = accelerator.prepare(hpsv2_model)


    prompt_list_file = "./geneval/prompts/evaluation_metadata.jsonl"
    with open(prompt_list_file) as fp:
        metadatas = [json.loads(line) for line in fp]


    total_prompts = len(metadatas)
    num_processes = accelerator.num_processes
    prompts_per_process = total_prompts // num_processes
    start_index = accelerator.process_index * prompts_per_process
    end_index = start_index + prompts_per_process if accelerator.process_index != num_processes - 1 else total_prompts

    # Process prompts assigned to this process
    for idx in range(start_index, end_index):
        prompt = metadatas[idx]["prompt"]
        
        # Create output directory for each prompt
        outdir = f"{save_dir}/{idx:0>5}"
        os.makedirs(f"{outdir}/samples", exist_ok=True)

        # Save metadata for each prompt
        with open(f"{outdir}/metadata.jsonl", "w") as fp:
            json.dump(metadatas[idx], fp)

        # Create Mask Optimizer
        mask_optimizer = MaskOptimizer(unet, mask_applier, reward_model, hpsv2_model, accelerator)

        # Run the mask optimization process
        image = mask_optimizer.optimize_masks([prompt], pipe, seed=None, num_inference_steps=15)

        # Save the generated image
        for i, img in enumerate(image):
            img.save(f"{outdir}/samples/{i:05}.png")

        del image
        torch.cuda.empty_cache()  # 清空显存缓存

    accelerator.wait_for_everyone()
    print(f"Process {accelerator.process_index}: Finished generating images.")
    # sample_out = 0
    # for index, metadata in enumerate(metadatas):
    #     prompt = metadata["prompt"]
    #     outdir = f"{save_dir}"
    #     outpath = f"{outdir}/{index:0>5}"
    #     os.makedirs(f"{outpath}/samples", exist_ok=True)
    #     with open(f"{outpath}/metadata.jsonl", "w") as fp:
    #         json.dump(metadata, fp)
    # # prompt = "a cat wearing a hat and a dog wearing a glasses"
    # # Create Mask Optimizer
    #     mask_optimizer = MaskOptimizer(unet, mask_applier, reward_model, hpsv2_model, accelerator)

    #     # Optimize masks
    #     # start_time = time.time() 
    #     batch_prompts = [prompt for _ in range(1)]  # 假设每次处理1个 prompt
    #     image = mask_optimizer.optimize_masks(batch_prompts, pipe, seed = seed, num_inference_steps=15)
    #     # end_time = time.time()  # 记录图像生成结束时间
    #     # duration = end_time - start_time  # 计算生成时间
    #     for i, img in enumerate(image):
    #         img.save(f"{outpath}/samples/{sample_out + i:05}.png")
    #     sample_out += len(image)
    #     # print(f"Time taken for mask gen: {duration:.2f} seconds")
    # print("Image generation with optimized mask completed and saved.")

if __name__ == "__main__":
    main()