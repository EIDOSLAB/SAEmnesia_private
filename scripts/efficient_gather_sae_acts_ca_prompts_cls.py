"""
Gather feature activations from a SAE for a given hookpoint and save them to a file.
Save-optimized version to prevent OOM errors during file saving.
For object classes instead of styles.
"""

import os
import sys
import json

import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae
from UnlearnCanvas_resources.const import class_available

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
    
    # Setup object prompts dictionary
    cls_prompts_dict = {class_avail: [] for class_avail in class_available}
    
    for class_avail in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_avail}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            prompt = [p.strip() for p in prompts]
            cls_prompts_dict[class_avail].extend(prompt)

    sae = Sae.load_from_disk(
        os.path.join(checkpoint_path, hookpoint), device="cuda"
    ).eval()

    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    # Detect model type from model_index.json
    model_index_path = os.path.join(pipe_path, "model_index.json")
    is_sdxl = False
    
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        # SDXL has text_encoder_2, SD1.5 doesn't
        is_sdxl = "text_encoder_2" in model_index
        
    if is_sdxl:
        print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
        PipelineClass = HookedStableDiffusionXLPipeline
    else:
        print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
        PipelineClass = HookedStableDiffusionPipeline

    pipe = PipelineClass.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    # Instead of keeping everything in memory, we'll track file paths
    cls_latents_paths = {}

    progress_bar = tqdm.tqdm(
        list(cls_prompts_dict.keys()), total=len(cls_prompts_dict)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    for class_avail in progress_bar:
        progress_bar.set_description(f"Processing class: {class_avail}")
        prompts = cls_prompts_dict[class_avail]
        
        # Clear memory before processing each class
        torch.cuda.empty_cache()
        gc.collect()
        
        _, acts_cache = pipe.run_with_cache(
            prompt=prompts,
            generator=generator,
            num_inference_steps=steps,
            save_input=False,
            save_output=True,
            positions_to_cache=[hookpoint],
            guidance_scale=9.0,
            output_type="latent",  # prevent decoding to pixel space
        )
        
        activations = acts_cache["output"][hookpoint].cpu()
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
        latent_path = os.path.join(temp_dir, f"{class_avail}_latents.pt")
        save_tensor_to_disk(latent_tensor, latent_path)
        cls_latents_paths[class_avail] = latent_path
        
        # Free up memory
        del activations, sae_latents, latent_tensor, acts_cache
        torch.cuda.empty_cache()
        gc.collect()

    # First, save cls_latents_dict
    print("Building and saving cls_latents_dict...")
    cls_latents_dict = {}
    
    for class_avail, path in cls_latents_paths.items():
        cls_latents_dict[class_avail] = load_tensor_from_disk(path)
    
    latents_file = os.path.join(save_dir, f"cls_latents_dict_{hookpoint}.pkl")
    with open(latents_file, "wb") as f:
        pickle.dump(cls_latents_dict, f)
    print(f"Saved to {latents_file}")
    
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)