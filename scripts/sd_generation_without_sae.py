import os
import sys
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline

import fire

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(
    pipe_checkpoint,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="results/no_sae_generation/",
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Define your custom prompts
    prompts = [
        "An image of Architectures in Abstractionism style",
        "An image of Bears in Blossom Season style", 
        "An image of Flame in Color Fantasy style"
    ]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Generating {len(prompts)} images...")

    # Split prompts between processes for distributed generation
    with accelerator.split_between_processes(prompts) as local_prompts:
        if accelerator.is_main_process:
            progress_bar = tqdm(local_prompts, desc="Generating images")
        else:
            progress_bar = local_prompts

        local_images = []
        local_prompt_list = []
        
        for prompt in progress_bar:
            with torch.no_grad():
                # Generate image without any hooks (no SAE)
                image = model(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                ).images[0]
                
                local_images.append(image)
                local_prompt_list.append(prompt)

    # Gather results from all processes
    accelerator.wait_for_everyone()
    all_images = gather_object(local_images)
    all_prompts = gather_object(local_prompt_list)

    # Save images (only on main process)
    if accelerator.is_main_process:
        for i, (image, prompt) in enumerate(zip(all_images, all_prompts)):
            # Create a safe filename from the prompt
            safe_filename = prompt.replace("An image of ", "").replace(" in ", "_").replace(" style", "").replace(" ", "_")
            filename = f"{safe_filename}_seed{seed}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            image.save(filepath)
            print(f"Saved: {filename}")

    print("Generation complete!")


if __name__ == "__main__":
    fire.Fire(main)