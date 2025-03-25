# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import torch.nn.functional as F
import torch.optim as optim
from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor
)
from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


import itertools
from typing import Any, Callable, Dict, Optional, Union, List

import spacy
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor
)
import numpy as np
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only


logger = logging.get_logger(__name__)


class SynGenMaskDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 image_encoder: CLIPVisionModelWithProjection = None,
                 requires_safety_checker: bool = True,
                 include_entities: bool = False,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,image_encoder,
                         requires_safety_checker)

        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        self.include_entities = include_entities

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(
            attention_maps=attention_maps
        )
        return attention_maps_list

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    # @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        attn_res=None,
        syngen_step_size: float = 20.0,
        parsed_prompt: str = None,
        num_intervention_steps: int = 25,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        if parsed_prompt:
            self.doc = parsed_prompt
        else:
            self.doc = self.parser(prompt)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # print(prompt_embeds.size())
        # assert 2==1

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attn_res = attn_res
        self.attention_store = AttentionStore(self.attn_res)
        self.register_attention_control()

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt:] if self.do_classifier_free_guidance else prompt_embeds
        )
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                if self.interrupt:
                    continue

                if i < self.stop_step:
                    continue
                
                if self.stop_step < num_intervention_steps:
                    latents = self._syngen_step(
                        latents,
                        text_embeddings,
                        t,
                        self.stop_step,
                        syngen_step_size,
                        cross_attention_kwargs,
                        prompt,
                        num_intervention_steps=num_intervention_steps,
                    )
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # print(self.unet)
                # assert 2==1
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents, pred_original_sample = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
                

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if i >= self.stop_step:
                    #     alpha_t = self.alphas[int(t.item())] ** 0.5
                    #     sigma_t = (1 - self.alphas[int(t.item())]) ** 0.5
                    #     latents = (latents - sigma_t * noise_pred) / alpha_t
                    latents_cur = latents
                    latents = pred_original_sample
                    break
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            has_nsfw_concept = None

        
        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        # image = (image / 2 + 0.5).clamp(0, 1)
        image = (image / 2 + 0.5).clamp(0, 1)
        # optimizer = optim.Adam(self.unet.parameters(), lr=1e-1)
        # input_tensor = image  # 模拟一个输入图像
        # target = torch.randn_like(input_tensor).to(noise_pred.device).half()  # 目标与输入形状相同
        # loss = F.mse_loss(image, target)
        # optimizer.zero_grad()
        # loss.backward()
        # for name, param in self.unet.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad}")
        #     else:
        #         print(f"No gradient for {name}")
        
        # assert 2==1

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            # return (image, has_nsfw_concept)
            return (image, latents_cur)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept), latents_cur

    def _syngen_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            cross_attention_kwargs,
            prompt,
            num_intervention_steps,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0)

                self.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                self.unet.zero_grad()
                # Get attention maps
                attention_maps = self._aggregate_and_get_attention_maps_per_token()
                loss = self._compute_loss(attention_maps=attention_maps, prompt=prompt)
                # Perform gradient update
                if i < num_intervention_steps:
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size
                        )
                    logger.info(f"Iteration {i} | Loss: {loss:0.4f}")

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    def _compute_loss(
            self, attention_maps: List[torch.Tensor], prompt: Union[str, List[str]]
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(self.tokenizer, prompt)
        loss = self._attribution_loss(attention_maps, prompt, attn_map_idx_to_wp)

        return loss

    def _attribution_loss(
            self,
            attention_maps: List[torch.Tensor],
            prompt: Union[str, List[str]],
            attn_map_idx_to_wp,
    ) -> torch.Tensor:
        if not self.subtrees_indices:
            self.subtrees_indices = self._extract_attribution_indices(prompt)
        subtrees_indices = self.subtrees_indices

        loss = 0

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            if noun and not modifier:
                if isinstance(noun, list) and len(noun) == 1:
                    processed_noun = noun[0]
                else:
                    processed_noun = noun
                loss += calculate_negative_loss(
                        attention_maps, modifier, processed_noun, subtree_indices, attn_map_idx_to_wp
                    )
            else:
                positive_loss, negative_loss = self._calculate_losses(
                    attention_maps,
                    all_subtree_pairs,
                    subtree_indices,
                    attn_map_idx_to_wp,
                )

                loss += positive_loss
                loss += negative_loss

        return loss

    def _calculate_losses(
            self,
            attention_maps,
            all_subtree_pairs,
            subtree_indices,
            attn_map_idx_to_wp,
    ):
        positive_loss = []
        negative_loss = []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            positive_loss.append(
                calculate_positive_loss(attention_maps, modifier, noun)
            )
            negative_loss.append(
                calculate_negative_loss(
                    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
                )
            )

        positive_loss = sum(positive_loss)
        negative_loss = sum(negative_loss)

        return positive_loss, negative_loss

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text.lower() == wp.lower():
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.lower().startswith(wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                                [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            if curr_collected_wp_indices:
                paired_indices.append(curr_collected_wp_indices)
            else:
                print(f"No wordpieces were aligned for {pair} in _align_indices")

        return paired_indices

    def _extract_attribution_indices(self, prompt):
        modifier_indices = []
        # extract standard attribution indices
        modifier_sets_1 = extract_attribution_indices(self.doc)
        modifier_indices_1 = self._align_indices(prompt, modifier_sets_1)
        if modifier_indices_1:
            modifier_indices.append(modifier_indices_1)

        # extract attribution indices with verbs in between
        modifier_sets_2 = extract_attribution_indices_with_verb_root(self.doc)
        modifier_indices_2 = self._align_indices(prompt, modifier_sets_2)
        if modifier_indices_2:
            modifier_indices.append(modifier_indices_2)

        modifier_sets_3 = extract_attribution_indices_with_verbs(self.doc)
        modifier_indices_3 = self._align_indices(prompt, modifier_sets_3)
        if modifier_indices_3:
            modifier_indices.append(modifier_indices_3)

        # entities only
        if self.include_entities:
            modifier_sets_4 = extract_entities_only(self.doc)
            modifier_indices_4 = self._align_indices(prompt, modifier_sets_4)
            modifier_indices.append(modifier_indices_4)

        # make sure there are no duplicates
        modifier_indices = unify_lists(modifier_indices)
        print(f"Final modifier indices collected:{modifier_indices}")

        return modifier_indices


def _get_attention_maps_list(
        attention_maps: torch.Tensor
) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list


def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst

from typing import Tuple
from dataclasses import dataclass
from diffusers.schedulers.scheduling_lcm import LCMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

def register_sdschedule_step(model):
    def sd_schedule_step(self):
        def step(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            generator=None,
            variance_noise: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ) -> Union[SchedulerOutput, Tuple]:
            """
            Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
            the multistep DPMSolver.

            Args:
                model_output (`torch.Tensor`):
                    The direct output from learned diffusion model.
                timestep (`int`):
                    The current discrete timestep in the diffusion chain.
                sample (`torch.Tensor`):
                    A current instance of a sample created by the diffusion process.
                generator (`torch.Generator`, *optional*):
                    A random number generator.
                variance_noise (`torch.Tensor`):
                    Alternative to generating noise with `generator` by directly providing the noise for the variance
                    itself. Useful for methods such as [`LEdits++`].
                return_dict (`bool`):
                    Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

            Returns:
                [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                    If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                    tuple is returned where the first element is the sample tensor.

            """
            if self.num_inference_steps is None:
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )

            if self.step_index is None:
                self._init_step_index(timestep)

            # Improve numerical stability for small number of steps
            lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
                self.config.euler_at_final
                or (self.config.lower_order_final and len(self.timesteps) < 15)
                or self.config.final_sigmas_type == "zero"
            )
            lower_order_second = (
                (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
            )

            model_output = self.convert_model_output(model_output, sample=sample)
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            self.model_outputs[-1] = model_output

            # Upcast to avoid precision issues when computing prev_sample
            sample = sample.to(torch.float32)
            if self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"] and variance_noise is None:
                noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=torch.float32
                )
            elif self.config.algorithm_type in ["sde-dpmsolver", "sde-dpmsolver++"]:
                noise = variance_noise.to(device=model_output.device, dtype=torch.float32)
            else:
                noise = None

            if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
                prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
            elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
                prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
            else:
                prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)

            if self.lower_order_nums < self.config.solver_order:
                self.lower_order_nums += 1

            # Cast sample back to expected dtype
            prev_sample = prev_sample.to(model_output.dtype)

            # upon completion increase step index by one
            self._step_index += 1

            if not return_dict:
                return (prev_sample, model_output)

            return SchedulerOutput(prev_sample=prev_sample)

        return step
    if model.__class__.__name__ == 'DPMSolverMultistepScheduler':
         model.step = sd_schedule_step(model)