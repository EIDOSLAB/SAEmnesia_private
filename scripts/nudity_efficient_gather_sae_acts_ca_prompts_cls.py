"""
Process single category at a time to avoid memory issues.
Run this script twice: once for 'nudity' and once for 'non_nudity'
Maintains exact same structure as object classes script.
"""

import os
import sys
import fire
import torch
import random
from diffusers.utils.import_utils import is_xformers_available

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
import pickle
import gc
import tqdm


def subsample_prompts(prompts, n_samples, seed=42):
    """Randomly subsample prompts."""
    random.seed(seed)
    if len(prompts) <= n_samples:
        return prompts
    return random.sample(prompts, n_samples)


def main(checkpoint_path, hookpoint, pipe_path, save_dir, 
         category, prompts_file, steps='100', seed='188', 
         batch_size='2', save_every='20', max_prompts=None):
    """
    Process a single category (nudity or non_nudity).
    
    Args:
        checkpoint_path: Path to SAE checkpoint
        hookpoint: Which hookpoint to use
        pipe_path: Path to diffusion model
        save_dir: Directory to save results
        category: Either 'nudity' or 'non_nudity'
        prompts_file: Path to text file with prompts for this category
        steps: Number of diffusion steps (default: 50 for nudity)
        seed: Random seed
        batch_size: Number of prompts to process at once
        save_every: Save intermediate results every N prompts
        max_prompts: Limit number of prompts (for subsampling)
    """
    steps = int(steps)
    seed = int(seed)
    batch_size = int(batch_size)
    save_every = int(save_every)
    
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = os.path.join(save_dir, "temp_tensors", category)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load prompts
    with open(prompts_file, 'r') as f:
        all_prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(all_prompts)} total prompts for {category}")
    
    # Subsample if requested
    if max_prompts:
        prompts = subsample_prompts(all_prompts, int(max_prompts), seed)
        print(f"Subsampled to {len(prompts)} prompts")
    else:
        prompts = all_prompts
        print(f"Using all {len(prompts)} prompts (no subsampling)")
    
    print(f"Processing {len(prompts)} {category} prompts")
    print(f"Each prompt will have shape: [steps={steps}, num_latents]")
    print(f"Final tensor shape will be: [{len(prompts)}, {steps}, num_latents]")

    # Load SAE
    print("Loading SAE...")
    sae = Sae.load_from_disk(
        os.path.join(checkpoint_path, hookpoint), device="cuda"
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    # Load pipeline
    print("Loading pipeline...")
    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()
    
    # Enable memory optimizations
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Process in chunks
    chunk_files = []
    chunk_idx = 0
    chunk_latents = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    
    progress_bar = tqdm.tqdm(range(num_batches), desc=f"Processing {category}")
    
    for batch_idx in progress_bar:
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        
        progress_bar.set_postfix({
            'batch': f'{batch_idx+1}/{num_batches}',
            'prompts': f'{end_idx}/{len(prompts)}',
            'GPU_MB': f'{torch.cuda.memory_allocated()/1024**2:.0f}'
        })
        
        try:
            _, acts_cache = pipe.run_with_cache(
                prompt=batch_prompts,
                generator=generator,
                num_inference_steps=steps,
                save_input=False,
                save_output=True,
                positions_to_cache=[hookpoint],
                guidance_scale=9.0,
                output_type="latent",
            )
            
            activations = acts_cache["output"][hookpoint].cpu()
            
            with torch.no_grad():
                for i in range(len(batch_prompts)):
                    sae_in = activations[i].reshape(steps, -1, sae.d_in)
                    top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                    sae_out = torch.zeros(
                        (top_acts.shape[0], sae.num_latents),
                        device=sae.device,
                        dtype=top_acts.dtype,
                    ).scatter(-1, top_indices, top_acts)
                    sae_out = sae_out.reshape(steps, -1, sae.num_latents).cpu()
                    # Average spatial dimension but KEEP timesteps
                    # Result shape: [steps, num_latents]
                    chunk_latents.append(sae_out.mean(1).to(dtype=torch.float16))
            
            del activations, acts_cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save chunk periodically
            if len(chunk_latents) >= save_every:
                chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.pt")
                torch.save(torch.stack(chunk_latents), chunk_file)
                chunk_files.append(chunk_file)
                print(f"\n  Saved chunk {chunk_idx} ({len(chunk_latents)} samples)")
                chunk_latents = []
                chunk_idx += 1
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nOOM at batch {batch_idx}. Reduce batch_size further.")
                torch.cuda.empty_cache()
                gc.collect()
            raise e
    
    # Save final chunk
    if chunk_latents:
        chunk_file = os.path.join(temp_dir, f"chunk_{chunk_idx:04d}.pt")
        torch.save(torch.stack(chunk_latents), chunk_file)
        chunk_files.append(chunk_file)
        print(f"\nSaved final chunk {chunk_idx}")
    
    # Merge chunks
    print(f"\nMerging {len(chunk_files)} chunks...")
    all_latents = []
    for chunk_file in tqdm.tqdm(chunk_files, desc="Loading chunks"):
        chunk_data = torch.load(chunk_file, map_location='cpu')
        all_latents.append(chunk_data)
        del chunk_data
        gc.collect()
    
    # Stack to create final tensor: [num_prompts, steps, num_latents]
    final_tensor = torch.cat(all_latents, dim=0)
    print(f"\nFinal tensor shape: {final_tensor.shape}")
    print(f"  [num_prompts={final_tensor.shape[0]}, timesteps={final_tensor.shape[1]}, num_latents={final_tensor.shape[2]}]")
    
    # Save individual category result
    output_file = os.path.join(save_dir, f"{category}_latents_{hookpoint}.pt")
    torch.save(final_tensor, output_file)
    print(f"\nSaved to {output_file}")
    
    # Clean up temp files
    for chunk_file in chunk_files:
        os.remove(chunk_file)
    
    print("Done!")


if __name__ == "__main__":
    fire.Fire(main)