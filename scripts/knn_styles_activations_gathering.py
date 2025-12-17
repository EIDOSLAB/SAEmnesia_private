"""
Gather feature activations from a SAE for a given hookpoint and save them to a file.
Save-optimized version to prevent OOM errors during file saving.
For themes/styles instead of object classes.
Collects UNCONDITIONAL activations (before text conditioning influence).
"""

import os
import sys

import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from UnlearnCanvas_resources.const import class_available, theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
import pickle
import gc
import time

import tqdm


def save_tensor_to_disk(tensor, filepath):
    """Save a tensor to disk using torch.save instead of pickle for better memory efficiency."""
    torch.save(tensor, filepath)
    

def load_tensor_from_disk(filepath):
    """Load a tensor from disk using torch.load."""
    return torch.load(filepath)


def main(checkpoint_path, hookpoint, pipe_path, save_dir, steps=100, seed=188):
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = os.path.join(save_dir, "temp_tensors")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Setup style prompts dictionary
    style_prompts_dict = {
        theme: [] for theme in theme_available if theme != "Seed_Images"
    }
    for class_avail in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_avail}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            prompt = prompts[0]
            prompt = prompt.strip()
            prompt = prompt if not prompt.endswith(".") else prompt[:-1]
            for theme in theme_available:
                if theme == "Seed_Images":
                    continue
                theme_prompt = f"{prompt} in {theme.replace('_', ' ')} style."
                style_prompts_dict[theme].append(theme_prompt)

    sae = Sae.load_from_disk(
        os.path.join(checkpoint_path, hookpoint), device="cuda"
    ).eval()

    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    # Instead of keeping everything in memory, we'll track file paths
    style_latents_paths = {}

    progress_bar = tqdm.tqdm(
        list(style_prompts_dict.keys()), total=len(style_prompts_dict)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    for theme in progress_bar:
        progress_bar.set_description(f"Processing theme: {theme}")
        prompts = style_prompts_dict[theme]
        
        # Clear memory before processing each theme
        torch.cuda.empty_cache()
        gc.collect()
        
        # CRITICAL: Set unconditional=True to extract unconditional activations
        _, acts_cache = pipe.run_with_cache(
            prompt=prompts,
            generator=generator,
            num_inference_steps=steps,
            save_input=True,  # Collect input activations
            save_output=False,  # Not collecting output
            positions_to_cache=[hookpoint],
            guidance_scale=9.0,  # Keep guidance for proper generation
            output_type="latent",  # prevent decoding to pixel space
            unconditional=True,  # KEY: Extract only unconditional activations
        )
        
        # Extract unconditional activations
        activations = acts_cache["input"][hookpoint].cpu()
        assert activations.shape[0] == len(prompts)
        assert activations.shape[1] == steps
        
        # Process SAE latents
        sae_latents = []
        with torch.no_grad():
            for i in range(len(prompts)):
                sae_in = activations[i].reshape(steps, -1, sae.d_in)
                top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                sae_out = torch.zeros(
                    (top_acts.shape[0], sae.num_latents),
                    device=sae.device,
                    dtype=top_acts.dtype,
                ).scatter(-1, top_indices, top_acts)
                sae_out = sae_out.reshape(steps, -1, sae.num_latents).cpu()
                sae_latents.append(sae_out.mean(1).to(dtype=torch.float16))
        
        # Save latents to temporary file
        latent_tensor = torch.stack(sae_latents)
        latent_path = os.path.join(temp_dir, f"{theme}_latents.pt")
        save_tensor_to_disk(latent_tensor, latent_path)
        style_latents_paths[theme] = latent_path
        
        # Free up memory
        del activations, sae_latents, latent_tensor, acts_cache
        torch.cuda.empty_cache()
        gc.collect()

    # First, save style_latents_dict
    print("Building and saving style_latents_dict...")
    style_latents_dict = {}
    
    for theme, path in style_latents_paths.items():
        style_latents_dict[theme] = load_tensor_from_disk(path)
    
    latents_file = os.path.join(save_dir, f"style_latents_dict_{hookpoint}.pkl")
    with open(latents_file, "wb") as f:
        pickle.dump(style_latents_dict, f)
    print(f"Saved to {latents_file}")
    
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)