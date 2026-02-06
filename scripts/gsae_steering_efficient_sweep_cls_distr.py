"""
Memory-optimized script for hyperparameter sweep for object unlearning using decoder-based steering.
"""
import os
import pickle
import sys
import gc
import json
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline, HookedStableDiffusionXLPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import class_available, theme_available
# from UnlearnCanvas_resources.const import class_available_subsample as class_available
# from UnlearnCanvas_resources.const import theme_available_subsample as theme_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


class SAEDecoderSteeringHook:
    """
    Hook that applies steering using SAE decoder columns for selected features.
    Similar to SAEMaskedUnlearningHook but steers using decoder columns instead of masking.
    """
    def __init__(
        self,
        concept_to_unlearn,
        percentile,
        alpha,
        feature_importance_fn,
        concept_latents_dict,
        sae,
        steps,
        seed,
        gamma=1.0,
    ):
        """
        Args:
            concept_to_unlearn: List of concepts to steer away from
            percentile: Percentile threshold for selecting important features
            alpha: Steering factor (negative to suppress, positive to enhance)
            feature_importance_fn: Function to compute feature importance
            concept_latents_dict: Dictionary mapping concepts to their latent representations
            sae: Trained SAE model with decoder D
            steps: Total number of denoising steps
            seed: Random seed
            gamma: Balancing parameter for feature weighting
        """
        self.concept_to_unlearn = concept_to_unlearn
        self.percentile = percentile
        self.alpha = alpha
        self.feature_importance_fn = feature_importance_fn
        self.concept_latents_dict = concept_latents_dict
        self.sae = sae
        self.steps = steps
        self.seed = seed
        self.gamma = gamma
        self.current_step = 0
        
        # Precompute important features and their decoder columns
        self.important_features = self._get_important_features()
        
    def _get_important_features(self):
        """Get important feature indices using the feature importance function."""
        important_features = {}
        
        for concept in self.concept_to_unlearn:
            if concept not in self.concept_latents_dict:
                continue
            
            concept_latents = self.concept_latents_dict[concept]
            
            # Determine number of timesteps
            if len(concept_latents.shape) == 3:
                n_timesteps = concept_latents.shape[1]
            elif len(concept_latents.shape) == 2:
                n_timesteps = concept_latents.shape[0]
            else:
                n_timesteps = 1
            
            # Compute feature importance scores across all timesteps
            all_scores = []
            for timestep in range(n_timesteps):
                scores = self.feature_importance_fn(
                    style_latents_dict=self.concept_latents_dict,
                    target_style=concept,
                    timestep=timestep,
                    seed=self.seed,
                )
                all_scores.append(scores)
            
            # Average importance scores across timesteps
            if len(all_scores) > 0:
                feature_scores = torch.stack(all_scores).mean(dim=0)
            else:
                # Fallback
                if len(concept_latents.shape) == 3:
                    feature_scores = concept_latents.mean(dim=(0, 1))
                elif len(concept_latents.shape) == 2:
                    feature_scores = concept_latents.mean(dim=0)
                else:
                    feature_scores = concept_latents
            
            # Ensure float for quantile
            if feature_scores.dtype not in [torch.float32, torch.float64]:
                feature_scores = feature_scores.float()
            
            important_features[concept] = feature_scores
        
        return important_features
    
    def __call__(self, module, input, output):
        """Apply steering during forward pass."""
        # Extract residual stream x from output
        if isinstance(output, tuple):
            x = output[0]
            return_tuple = True
        else:
            x = output
            return_tuple = False
        
        # Handle classifier-free guidance
        if x.shape[0] % 2 == 0:
            x_uncond, x_cond = x.chunk(2)
            x_cond_steered = self._apply_steering(x_cond)
            x_steered = torch.cat([x_uncond, x_cond_steered], dim=0)  # ✓ Fixed!
        else:
            x_steered = self._apply_steering(x)
        
        self.current_step += 1
        
        if return_tuple:
            return (x_steered,) + output[1:]
        return x_steered
    
    def _apply_steering(self, x):
        """
        Apply steering using decoder columns for selected features.
        """
        device = x.device
        dtype = x.dtype
        original_shape = x.shape
        
        # Handle different tensor shapes
        if len(x.shape) == 4:
            # Conv layer: [batch, channels, height, width]
            batch_size, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            d_model = channels
        elif len(x.shape) == 3:
            # Transformer layer: [batch, seq_len, d_model]
            batch_size, seq_len, d_model = x.shape
        else:
            return x
        
        # Compute residual stream magnitude for normalization
        x_norm = torch.norm(x, dim=-1, keepdim=True)  # [batch, seq_len, 1]
        
        # Initialize steering vector accumulator
        total_steering = torch.zeros_like(x)
        
        # Process each concept
        for concept in self.concept_to_unlearn:
            if concept not in self.important_features:
                continue
            
            feature_scores = self.important_features[concept]
            
            # Ensure float for quantile
            if feature_scores.dtype not in [torch.float32, torch.float64]:
                feature_scores = feature_scores.float()
            
            # Get top features based on percentile threshold
            threshold = torch.quantile(feature_scores, self.percentile / 100.0)
            important_indices = torch.where(feature_scores > threshold)[0]
            
            if len(important_indices) == 0:
                continue
            
            # Get decoder columns
            if hasattr(self.sae.W_dec, 'weight'):
                decoder = self.sae.W_dec.weight.data
            elif hasattr(self.sae.W_dec, 'data'):
                decoder = self.sae.W_dec.data
            else:
                decoder = self.sae.W_dec
            
            decoder = decoder.to(device).to(dtype)
            
            # Check dimension match
            if decoder.shape[1] != d_model:
                continue
            
            # Process each important feature
            for feat_idx in important_indices:
                feat_idx = feat_idx.item()
                
                # Get decoder column D·,i
                D_i = decoder[feat_idx]  # [d_model]
                
                # Compute balancing parameter
                gamma_i = 1
                
                # Compute normalization factor βi = ||x||2 / ||D·,i||2
                D_i_norm = torch.norm(D_i)
                beta_i = x_norm / (D_i_norm + 1e-8)  # [batch, seq_len, 1]
                
                # Add steering vector contribution
                steering_contribution = beta_i * gamma_i * D_i.unsqueeze(0).unsqueeze(0)
                total_steering += steering_contribution
        
        # Apply steering: x̂ = x + α × total_steering
        x_steered = x + self.alpha * total_steering
        
        # Reshape back if needed
        if len(original_shape) == 4:
            batch_size, channels, height, width = original_shape
            x_steered = x_steered.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        return x_steered


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


def empty_cache():
    """Helper function to clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_class_prompts(limit_themes=50):
    """Load and prepare class prompts (moved to separate function to improve memory management)"""
    class_prompt_dict = {class_: [] for class_ in class_available}
    for class_to_unlearn in class_available:
        with open(
            os.path.join(
                "/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning",
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_to_unlearn}.txt",
            ),
            "r",
        ) as prompt_file:
            prompts = prompt_file.readlines()
            for i, theme in enumerate(theme_available):
                if i >= limit_themes:
                    break
                if theme != "Seed_Images":
                    theme_prompt = prompts[i].strip()
                    theme_prompt = (
                        theme_prompt
                        if not theme_prompt.endswith(".")
                        else theme_prompt[:-1]
                    )
                    theme_prompt = f"{theme_prompt} in {theme.replace('_', ' ')} style."
                    class_prompt_dict[class_to_unlearn].append(theme_prompt)
    return class_prompt_dict


def process_batch(model, batch_prompts, batch_classes, class_to_unlearn, percentile, alpha, 
                 hookpoint, sae, class_latents_dict, steps, seed, guidance_scale, generator, gamma=1.0):
    """Process a single batch of prompts using decoder-based steering"""
    steering_hooks = {}
    steering_hooks[hookpoint] = SAEDecoderSteeringHook(
        concept_to_unlearn=[class_to_unlearn],
        percentile=percentile,
        alpha=alpha,
        feature_importance_fn=compute_feature_importance,
        concept_latents_dict=class_latents_dict,
        sae=sae,
        steps=steps,
        seed=seed,
        gamma=gamma,
    )
    
    with torch.no_grad():
        images = model.run_with_hooks(
            prompt=batch_prompts,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            position_hook_dict=steering_hooks,
        )
    
    return images, batch_classes


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    seed=188,
    steps=100,
    percentiles=[99.99, 99.995, 99.999],
    alphas=[-0.1, -0.2, -0.3, -0.4, -0.5, -1.0, -5.0, -10.0],
    gamma=1.0,
    guidance_scale=9.0,
    output_dir="sweep_results/decoder_steering/class20/",
    limit_themes=50,
    batch_size=64,
):
    """
    Hyperparameter sweep for decoder-based steering unlearning.
    
    Args:
        pipe_checkpoint: Path to pretrained diffusion model
        hookpoint: Model layer to apply steering
        class_latents_path: Path to class latents dictionary
        sae_checkpoint: Path to trained SAE
        seed: Random seed
        steps: Number of inference steps
        percentiles: List of percentile thresholds (high values = fewer features)
        alphas: List of steering factors (negative = suppress, positive = enhance)
        gamma: Balancing parameter for feature weighting
        guidance_scale: CFG scale
        output_dir: Output directory for results
        limit_themes: Number of themes to use
        batch_size: Batch size for processing
    """
    accelerator = Accelerator()
    device = accelerator.device

    # Detect model type from model_index.json
    model_index_path = os.path.join(pipe_checkpoint, "model_index.json")
    is_sdxl = False
    
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
        is_sdxl = "text_encoder_2" in model_index
        
    if is_sdxl:
        if accelerator.is_main_process:
            print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
        PipelineClass = HookedStableDiffusionXLPipeline
    else:
        if accelerator.is_main_process:
            print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
        PipelineClass = HookedStableDiffusionPipeline

    # Load model with memory optimization options
    model = PipelineClass.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True if os.path.exists(pipe_checkpoint) else False,
    )
    
    # Enable memory efficient attention
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
    
    model = model.to(device)

    # Convert VAE to float32 to prevent black images with SDXL
    if is_sdxl:
        model.pipe.vae = model.pipe.vae.to(dtype=torch.float32)
    if accelerator.is_main_process:
        print(f"✓ VAE converted to float32 (dtype: {model.pipe.vae.dtype})")
    
    # For SDXL, enable VAE optimizations
    if is_sdxl and hasattr(model.pipe, 'enable_vae_slicing'):
        model.pipe.enable_vae_slicing()
    
    # Disable VAE tiling if it causes issues
    if hasattr(model.pipe, 'disable_vae_tiling'):
        model.pipe.disable_vae_tiling()
    
    # Set up generator
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    if accelerator.is_main_process:
        print('Loading SAE...')
    
    # Load SAE
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    if accelerator.is_main_process:
        print('Loading class latents...')
    
    # Load class latents
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)
    
    empty_cache()
    
    if accelerator.is_main_process:
        print('Preparing class prompts...')
    
    # Get class prompts
    class_prompt_dict = get_class_prompts(limit_themes)
    
    # Set up progress bar
    total_iterations = len(alphas) * len(class_available) * len(percentiles)
    if accelerator.is_main_process:
        progress_bar = tqdm(total=total_iterations)
        print(f"Starting decoder-based steering sweep:")
        print(f"  - {len(alphas)} alpha values: {alphas}")
        print(f"  - {len(percentiles)} percentiles: {percentiles}")
        print(f"  - {len(class_available)} classes")
        print(f"  - gamma: {gamma}")
    
    # Main processing loop
    for alpha in alphas:
        for percentile in percentiles:
            for class_to_unlearn in class_available:
                if accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Alpha: {alpha} Percentile: {percentile} Class: {class_to_unlearn}"
                    )
                
                # Create output directory
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_alpha_{alpha}/{class_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                
                # Prepare all prompts
                all_prompts = [
                    (class_name, prompt)
                    for class_name, prompts in class_prompt_dict.items()
                    for prompt in prompts
                ]
                
                # Process in distributed manner across GPUs/processes
                all_images = []
                all_classes = []
                
                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, prompt in local_tuples]
                    local_classes = [class_name for class_name, _ in local_tuples]
                    
                    # Process in smaller batches to save memory
                    for i in range(0, len(local_prompts), batch_size):
                        batch_prompts = local_prompts[i:i+batch_size]
                        batch_classes = local_classes[i:i+batch_size]
                        
                        batch_images, batch_classes = process_batch(
                            model, batch_prompts, batch_classes, class_to_unlearn, percentile, 
                            alpha, hookpoint, sae, class_latents_dict, steps, seed, 
                            guidance_scale, generator, gamma
                        )
                        
                        all_images.extend(batch_images)
                        all_classes.extend(batch_classes)
                        
                        # Clear memory after each batch
                        empty_cache()
                
                # Gather results across processes
                accelerator.wait_for_everyone()
                gathered_images = gather_object(all_images)
                gathered_classes = gather_object(all_classes)
                
                # Save results
                if accelerator.is_main_process:
                    for i, (img, object_class) in enumerate(zip(gathered_images, gathered_classes)):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}_{i}.jpg",
                            )
                        )
                    progress_bar.update(1)
                
                # Final memory cleanup for this iteration
                empty_cache()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("Processing complete!")


if __name__ == "__main__":
    fire.Fire(main)