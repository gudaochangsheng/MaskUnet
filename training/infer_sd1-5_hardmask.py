import torch
import os
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from hyper_unet6 import FineGrainedUNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from diffusers import PixArtAlphaPipeline
from safetensors.torch import load_file
import os

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], idx  # Return prompt and its index

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
# Set a fixed seed for reproducibility


def main_worker(rank, args):
    setup(rank, args.world_size)

    seed  =42

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with open(args.captions_file, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    model_path = "path_to_saved_model"
    unet = FineGrainedUNet2DConditionModel.from_pretrained(
        "../share/runwayml/stable-diffusion-v1-5", subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float16
    )

    input_dir = "./sd-naruto-model_hard_improve1_factor4_trans_decoder_sd1.5_factorablation_epoch15/checkpoint-9000"

    load_mask_path = os.path.join(input_dir, "mask_generators.pth")
    loaded_mask_state_dict = torch.load(load_mask_path)

    # Load weights for mask_generators
    unet.mask_generators.load_state_dict(loaded_mask_state_dict)

    # Load weights for adapters
    load_path_adapters = os.path.join(input_dir, "adapters.pth")
    adapters_state_dict = torch.load(load_path_adapters)
    unet.adapters.load_state_dict(adapters_state_dict)

    print("load success")
    scheduler = DDIMScheduler.from_pretrained("../share/runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    # Load the Stable Diffusion pipeline with the modified UNet model
    pipe = StableDiffusionPipeline.from_pretrained(
        "../share/runwayml/stable-diffusion-v1-5",  torch_dtype=torch.float16,
        scheduler=scheduler,
        unet = unet
    )
    pipe.to(f"cuda:{rank}")

    output_dir = "fake_images_sd1.5_mask_vis_9000_final"

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device=f"cuda:{rank}").manual_seed(args.seed) if args.seed else None
    def dummy_checker(images, **kwargs):
        # Return the images as is, and set NSFW detection results to False (not NSFW)
        return images, [False] * len(images)

    pipe.safety_checker = dummy_checker

    dataset = PromptDataset(prompts)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)

    for i, (batch_prompts, indices) in enumerate(dataloader):
        # Ensure batch_prompts is a list of strings
        batch_prompts = [prompt for prompt in batch_prompts]
        images = pipe(prompt=batch_prompts, generator=generator, num_inference_steps=50, guidance_scale=7.5).images

        # Save images immediately after generation
        for j, image in enumerate(images):
            output_filename = f"{output_dir}/{indices[j]:05}.png"
            image.save(output_filename)

    # Set the seed in the pipeline for deterministic image generation
    # generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate the image with a fixed seed #cute dragon creature
    # image = pipe(prompt="A naruto with green eyes and red legs.", generator=generator).images[0]
    # image = pipe(prompt="digital art of a little cat traveling around forest, wearing a scarf around the neck, carrying a tiny backpack on his back", generator=generator, num_inference_steps=10, guidance_scale=7.5).images[0]
    # image = pipe(prompt="cute dragon creature", generator=generator).images[0]
    # image.save("yoda-naruto.png")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of an inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="thuanz123/swiftbrush",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="captions.txt",
        required=True,
        help="Path to the captions file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="Random seed used for inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        required=False,
        help="Batch size for inference.",
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use.",
    )

    args = parser.parse_args()

    # Get the rank from the environment variable set by torchrun
    rank = int(os.environ["LOCAL_RANK"])
    
    main_worker(rank, args)