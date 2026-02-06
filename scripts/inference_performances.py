import os
import pickle
import sys
import json
import time
from collections import defaultdict

import numpy as np
import torch
from accelerate import Accelerator
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae

sys.path.append("..")

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


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


class SAEReconstructionHook:
    """Hook that only performs SAE reconstruction without any unlearning
    
    This hook is designed to work with the hooked pipeline's custom hook interface.
    It maintains internal state to track timesteps across the denoising process.
    """
    
    def __init__(self, sae, steps, start_timestep=0):
        self.sae = sae
        self.steps = steps
        self.start_timestep = start_timestep
        self.reconstruction_times = []
        self.current_timestep = 0
        self.call_count = 0
        
    def __call__(self, *args, **kwargs):
        """
        Flexible hook interface that handles multiple calling conventions:
        - Standard PyTorch: hook(module, args, output)
        - Custom 4-arg: hook(module, args, kwargs, output)
        """
        # Handle different calling conventions
        if len(args) == 3:
            # Standard PyTorch hook: (module, input_args, output)
            module, input_args, output = args
            # Estimate timestep based on call count
            # Each timestep typically calls the hook once per hookpoint
            current_step = self.call_count % self.steps
        elif len(args) == 4:
            # Custom hook: (module, input_args, kwargs_dict, output)
            module, input_args, kwargs_dict, output = args
            current_step = kwargs_dict.get("current_step", self.call_count % self.steps)
        else:
            raise ValueError(f"Unexpected number of arguments: {len(args)}")
        
        self.call_count += 1
        
        # Handle tuple outputs (some modules return multiple values)
        is_tuple_output = isinstance(output, tuple)
        if is_tuple_output:
            # Assume the first element is the main tensor
            actual_output = output[0]
            rest_of_output = output[1:]
        else:
            actual_output = output
            rest_of_output = None
        
        # Only apply reconstruction in the specified timestep range
        if current_step < self.start_timestep or current_step >= self.steps:
            return output
        
        # Debug: Print shape on first call
        if self.call_count == 1:
            print(f"[DEBUG] Hook called with output shape: {actual_output.shape}")
            print(f"[DEBUG] Output dtype: {actual_output.dtype}")
        
        # Time the SAE reconstruction
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                # The SAE expects (batch_size, sequence_length, embedding_dim)
                # Store original shape for potential reshaping
                original_shape = actual_output.shape
                
                # Handle different tensor shapes
                if len(original_shape) == 4:
                    # (batch, channels, height, width) -> (batch, height*width, channels)
                    batch, channels, height, width = original_shape
                    reshaped = actual_output.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
                elif len(original_shape) == 3:
                    # Already in correct format (batch, seq_len, dim)
                    reshaped = actual_output
                elif len(original_shape) == 2:
                    # (batch, dim) -> (batch, 1, dim)
                    reshaped = actual_output.unsqueeze(1)
                else:
                    raise ValueError(f"Unexpected tensor shape: {original_shape}")
                
                # Encode to SAE latents
                sae_latents = self.sae.encode(reshaped)
                # Decode back to reconstruct
                reconstructed = self.sae.decode(sae_latents)
                
                # Reshape back to original format
                if len(original_shape) == 4:
                    batch, channels, height, width = original_shape
                    reconstructed = reconstructed.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
                elif len(original_shape) == 2:
                    reconstructed = reconstructed.squeeze(1)
                # For 3D, already in correct format
            
            end_time = time.perf_counter()
            self.reconstruction_times.append(end_time - start_time)
            
            # Return in the same format as input
            if is_tuple_output:
                return (reconstructed,) + rest_of_output
            else:
                return reconstructed
                
        except Exception as e:
            # If reconstruction fails, return original output and log error
            print(f"[ERROR] SAE reconstruction failed: {e}")
            print(f"  Output shape: {actual_output.shape}")
            import traceback
            traceback.print_exc()
            return output
    
    def reset(self):
        """Reset counters for a new generation"""
        self.call_count = 0
        self.current_timestep = 0


def benchmark_run(
    model,
    prompts,
    generator,
    steps,
    guidance_scale,
    steering_hooks,
    warmup=False,
):
    """Run inference and return timing information"""
    
    # Synchronize CUDA before starting
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        images = model.run_with_hooks(
            prompt=prompts,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            position_hook_dict=steering_hooks,
        )
    
    # Synchronize CUDA after completion
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return total_time, images


def main(
    pipe_checkpoint,
    hookpoint,
    sae_checkpoint,
    seed=188,
    steps=50,
    guidance_scale=7.5,
    num_warmup_runs=2,
    num_benchmark_runs=5,
    batch_size=1,
    start_timestep=0,
    test_prompts=None,
):
    """
    Benchmark SAE reconstruction overhead by comparing inference times.
    
    Args:
        pipe_checkpoint: Path to the pretrained diffusion model checkpoint
        hookpoint: Position in the model where SAE hooks are applied
        sae_checkpoint: Path to SAE checkpoint
        seed: Random seed for generation
        steps: Number of inference steps
        guidance_scale: Classifier-free guidance scale
        num_warmup_runs: Number of warmup runs before benchmarking
        num_benchmark_runs: Number of runs to average for benchmarking
        batch_size: Number of images to generate per run
        start_timestep: Timestep from which to start applying SAE reconstruction
        test_prompts: List of prompts to use (default: simple test prompts)
    """
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("=" * 80)
        print("SAE RECONSTRUCTION OVERHEAD BENCHMARK")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Steps: {steps}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Batch size: {batch_size}")
        print(f"SAE active timesteps: {start_timestep} to {steps}")
        print(f"Warmup runs: {num_warmup_runs}")
        print(f"Benchmark runs: {num_benchmark_runs}")
        print("=" * 80)

    # Detect model type
    model_index_path = os.path.join(pipe_checkpoint, "model_index.json")
    is_sdxl = False
    
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        is_sdxl = "text_encoder_2" in model_index
    
    # Load appropriate pipeline
    if is_sdxl:
        if accelerator.is_main_process:
            print("🎯 Detected SDXL model")
        PipelineClass = HookedStableDiffusionXLPipeline
        try:
            model = PipelineClass.from_pretrained(
                pipe_checkpoint,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        except:
            model = PipelineClass.from_pretrained(
                pipe_checkpoint,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
    else:
        if accelerator.is_main_process:
            print("🎯 Detected SD1.5 model")
        PipelineClass = HookedStableDiffusionPipeline
        model = PipelineClass.from_pretrained(
            pipe_checkpoint,
            torch_dtype=torch.float16,
        )
    
    # Disable safety checker
    if hasattr(model.pipe, 'safety_checker'):
        model.pipe.safety_checker = None
    
    model = model.to(device)
    model.pipe.vae = model.pipe.vae.to(dtype=torch.float32)

    # Enable optimizations
    if is_sdxl and hasattr(model.pipe, 'enable_vae_slicing'):
        model.pipe.enable_vae_slicing()
    
    if hasattr(model.pipe, 'disable_vae_tiling'):
        model.pipe.disable_vae_tiling()

    if is_xformers_available():
        import xformers
        if accelerator.is_main_process:
            print("✓ Enabling xFormers memory efficient attention")
        model.enable_xformers_memory_efficient_attention()

    # Load SAE
    if accelerator.is_main_process:
        print(f"📦 Loading SAE from {sae_checkpoint}")
    sae = load_sae(sae_checkpoint, hookpoint, device)

    # Prepare test prompts
    if test_prompts is None:
        test_prompts = [
            "A photograph of a cat",
            "A painting of a sunset over mountains",
            "A modern office building",
            "A bowl of fresh fruit",
        ]
    
    # Use only batch_size prompts
    test_prompts = test_prompts[:batch_size]
    
    if accelerator.is_main_process:
        print(f"📝 Using {len(test_prompts)} prompt(s):")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"  {i}. {prompt}")
        print()

    seed_everything(seed)
    
    # Store results
    results = {
        'baseline': [],
        'with_sae': [],
        'sae_reconstruction_times': [],
    }

    # ========================================
    # WARMUP PHASE
    # ========================================
    if accelerator.is_main_process:
        print("🔥 Warming up...")
    
    for i in range(num_warmup_runs):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        
        # Warmup without SAE
        _, _ = benchmark_run(
            model, test_prompts, generator, steps, guidance_scale, {}, warmup=True
        )
        
        # Warmup with SAE
        sae_hook = SAEReconstructionHook(sae, steps, start_timestep)
        steering_hooks = {hookpoint: sae_hook}
        _, _ = benchmark_run(
            model, test_prompts, generator, steps, guidance_scale, steering_hooks, warmup=True
        )
        
        if accelerator.is_main_process:
            print(f"  Warmup run {i+1}/{num_warmup_runs} completed")
    
    if accelerator.is_main_process:
        print("✓ Warmup completed\n")

    # ========================================
    # BASELINE BENCHMARK (No SAE)
    # ========================================
    if accelerator.is_main_process:
        print("📊 Benchmarking BASELINE (no SAE)...")
    
    for i in tqdm(range(num_benchmark_runs), disable=not accelerator.is_main_process, desc="Baseline"):
        generator = torch.Generator(device=device).manual_seed(seed + num_warmup_runs + i)
        
        total_time, _ = benchmark_run(
            model, test_prompts, generator, steps, guidance_scale, {}
        )
        
        results['baseline'].append(total_time)
        
        if accelerator.is_main_process:
            print(f"  Run {i+1}: {total_time:.3f}s")
    
    # ========================================
    # SAE BENCHMARK (With Reconstruction)
    # ========================================
    if accelerator.is_main_process:
        print(f"\n📊 Benchmarking WITH SAE RECONSTRUCTION...")
    
    for i in tqdm(range(num_benchmark_runs), disable=not accelerator.is_main_process, desc="With SAE"):
        generator = torch.Generator(device=device).manual_seed(seed + num_warmup_runs + num_benchmark_runs + i)
        
        # Create a fresh hook for each run to reset state
        sae_hook = SAEReconstructionHook(sae, steps, start_timestep)
        steering_hooks = {hookpoint: sae_hook}
        
        total_time, _ = benchmark_run(
            model, test_prompts, generator, steps, guidance_scale, steering_hooks
        )
        
        results['with_sae'].append(total_time)
        results['sae_reconstruction_times'].append(sae_hook.reconstruction_times)
        
        if accelerator.is_main_process:
            avg_recon_time = np.mean(sae_hook.reconstruction_times) if sae_hook.reconstruction_times else 0
            print(f"  Run {i+1}: {total_time:.3f}s (avg SAE recon: {avg_recon_time*1000:.2f}ms, {len(sae_hook.reconstruction_times)} reconstructions)")


    # ========================================
    # RESULTS SUMMARY
    # ========================================
    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        
        baseline_times = np.array(results['baseline'])
        sae_times = np.array(results['with_sae'])
        
        print(f"\n{'Metric':<40} {'Baseline':<15} {'With SAE':<15} {'Overhead':<15}")
        print("-" * 80)
        
        # Mean times
        baseline_mean = baseline_times.mean()
        sae_mean = sae_times.mean()
        overhead_abs = sae_mean - baseline_mean
        overhead_pct = (overhead_abs / baseline_mean) * 100
        
        print(f"{'Mean time (s)':<40} {baseline_mean:<15.3f} {sae_mean:<15.3f} {overhead_abs:.3f}s ({overhead_pct:.1f}%)")
        
        # Std deviation
        baseline_std = baseline_times.std()
        sae_std = sae_times.std()
        
        print(f"{'Std deviation (s)':<40} {baseline_std:<15.3f} {sae_std:<15.3f}")
        
        # Min/Max
        baseline_min = baseline_times.min()
        sae_min = sae_times.min()
        
        print(f"{'Min time (s)':<40} {baseline_min:<15.3f} {sae_min:<15.3f}")
        
        baseline_max = baseline_times.max()
        sae_max = sae_times.max()
        
        print(f"{'Max time (s)':<40} {baseline_max:<15.3f} {sae_max:<15.3f}")
        
        # Per-image times
        baseline_per_img = baseline_mean / batch_size
        sae_per_img = sae_mean / batch_size
        
        print(f"{'Time per image (s)':<40} {baseline_per_img:<15.3f} {sae_per_img:<15.3f}")
        
        # SAE reconstruction statistics
        all_recon_times = []
        for run_recon_times in results['sae_reconstruction_times']:
            all_recon_times.extend(run_recon_times)
        
        all_recon_times = np.array(all_recon_times) * 1000  # Convert to ms
        
        print("\n" + "-" * 80)
        print("SAE RECONSTRUCTION STATISTICS (per timestep)")
        print("-" * 80)
        print(f"Mean reconstruction time:     {all_recon_times.mean():.2f}ms")
        print(f"Std deviation:                {all_recon_times.std():.2f}ms")
        print(f"Min reconstruction time:      {all_recon_times.min():.2f}ms")
        print(f"Max reconstruction time:      {all_recon_times.max():.2f}ms")
        print(f"Total reconstructions:        {len(all_recon_times)}")
        
        # Active timesteps
        active_steps = steps - start_timestep
        expected_reconstructions = active_steps * num_benchmark_runs * batch_size
        print(f"Expected reconstructions:     {expected_reconstructions}")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: SAE adds {overhead_abs:.3f}s ({overhead_pct:.1f}%) overhead per run")
        print(f"         Average {all_recon_times.mean():.2f}ms per reconstruction step")
        print("=" * 80)
        
        # Save detailed results
        results_dict = {
            'config': {
                'pipe_checkpoint': pipe_checkpoint,
                'hookpoint': hookpoint,
                'sae_checkpoint': sae_checkpoint,
                'steps': steps,
                'guidance_scale': guidance_scale,
                'batch_size': batch_size,
                'start_timestep': start_timestep,
                'num_warmup_runs': num_warmup_runs,
                'num_benchmark_runs': num_benchmark_runs,
            },
            'baseline': {
                'times': baseline_times.tolist(),
                'mean': float(baseline_mean),
                'std': float(baseline_std),
                'min': float(baseline_min),
                'max': float(baseline_max),
            },
            'with_sae': {
                'times': sae_times.tolist(),
                'mean': float(sae_mean),
                'std': float(sae_std),
                'min': float(sae_min),
                'max': float(sae_max),
            },
            'overhead': {
                'absolute_seconds': float(overhead_abs),
                'percentage': float(overhead_pct),
                'per_image_baseline': float(baseline_per_img),
                'per_image_sae': float(sae_per_img),
            },
            'sae_reconstruction': {
                'mean_ms': float(all_recon_times.mean()),
                'std_ms': float(all_recon_times.std()),
                'min_ms': float(all_recon_times.min()),
                'max_ms': float(all_recon_times.max()),
                'total_reconstructions': len(all_recon_times),
            }
        }
        
        output_file = 'sae_benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    fire.Fire(main)