import os
import pickle
import sys
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
from SAE.unlearning_utils import compute_feature_importance, get_percentile_threshold

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import (
    class_available,
    theme_available,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available

print("=" * 80)
print("LOADING conditional_gsae_steering.py - WITH CONCEPT DETECTION")
print("=" * 80)


class SAEDecoderSteeringHook:
    """
    Hook that applies steering using SAE decoder columns for selected features.
    Uses compute_feature_importance to identify discriminative features.
    
    ENHANCEMENT: Only applies steering if the concept is detected by checking
    if the important features' activations exceed the average across all concepts.
    If concept is NOT detected, returns original activation unchanged.
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
        start_timestep=0,
        detection_threshold_multiplier=1.5,  # Concept must activate X times above average
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
            start_timestep: Timestep from which to start applying steering
            detection_threshold_multiplier: Multiplier for detection threshold (higher = stricter)
        """
        print(f'[INIT] Starting hook initialization...', flush=True)
        
        self.concept_to_unlearn = concept_to_unlearn
        self.percentile = percentile
        self.alpha = alpha
        self.feature_importance_fn = feature_importance_fn
        self.concept_latents_dict = concept_latents_dict
        self.sae = sae
        self.steps = steps
        self.seed = seed
        self.gamma = gamma
        self.start_timestep = start_timestep
        self.current_step = 0
        self.detection_threshold_multiplier = detection_threshold_multiplier
        
        # Debug counters
        self.debug_concept_detected_count = 0
        self.debug_concept_not_detected_count = 0
        
        print(f'[INIT] SAE seed: {self.seed}', flush=True)
        print(f'[INIT] Alpha: {self.alpha}, Percentile: {self.percentile}', flush=True)
        print(f'[INIT] Detection threshold multiplier: {self.detection_threshold_multiplier}', flush=True)
        print(f'[INIT] Concepts to unlearn: {self.concept_to_unlearn}', flush=True)
        print(f'[INIT] Steering will start at timestep: {self.start_timestep}', flush=True)
        
        # Precompute important features (same as original)
        self.important_features = self._get_important_features()
        
        # Precompute per-timestep data for concept detection
        self._precompute_detection_thresholds()
        
        print(f'[INIT] Hook initialization complete!', flush=True)
    
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
    
    def _precompute_detection_thresholds(self):
        """
        Precompute per-timestep feature indices and average activations for concept detection.
        Same logic as SAEMaskedUnlearningHook.
        """
        self.top_feature_idxs = []
        self.all_concept_avg_acts = []
        
        for timestep in range(self.steps):
            timestep_feature_idxs = []
            timestep_all_concept_avg_acts = []
            
            for concept in self.concept_to_unlearn:
                # Get feature importance scores for this concept at this timestep
                feature_scores = self.feature_importance_fn(
                    self.concept_latents_dict, concept, timestep, seed=self.seed
                )
                feature_scores = feature_scores.float()
                
                # Get threshold and find top features
                percentile_threshold = get_percentile_threshold(feature_scores, self.percentile)
                top_feature_idxs = torch.where(feature_scores > percentile_threshold)[0]
                timestep_feature_idxs.append(top_feature_idxs)
                
                # Compute average activation across ALL concepts for these features
                all_concept_avg_acts = torch.zeros((len(top_feature_idxs)))
                for c in self.concept_latents_dict:
                    all_concept_avg_acts += self.concept_latents_dict[c][
                        :, timestep, top_feature_idxs
                    ].mean(dim=0)
                all_concept_avg_acts /= len(self.concept_latents_dict)
                timestep_all_concept_avg_acts.append(all_concept_avg_acts)
            
            self.top_feature_idxs.append(torch.cat(timestep_feature_idxs))
            self.all_concept_avg_acts.append(torch.cat(timestep_all_concept_avg_acts))
        
        print(f'[INIT] Precomputed {len(self.top_feature_idxs)} timesteps for detection', flush=True)
        print(f'[INIT] Top features per timestep: {[len(idx) for idx in self.top_feature_idxs[:5]]}...', flush=True)
    
    def _detect_concept(self, x, device):
        """
        Use SAE encoder to detect if concept is present.
        Returns True if the important feature activates significantly above average threshold.
        
        Uses detection_threshold_multiplier to require activation to be X times the average
        before considering the concept as "present".
        """
        top_features = self.top_feature_idxs[self.current_step].to(device)
        avg_acts = self.all_concept_avg_acts[self.current_step].to(device)
        
        # Encode with SAE to get latent activations
        sae_input, _, _ = self.sae.preprocess_input(x)
        pre_acts = self.sae.pre_acts(sae_input)
        top_acts, top_indices = self.sae.select_topk(pre_acts)
        
        # Create full latent tensor
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.sae.W_dec.mT.shape[-1],))
        latents = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        latents = latents.reshape(x.shape[0], -1, self.sae.num_latents)
        
        # Get activations for the important features
        latent_acts = latents[:, :, top_features]  # [batch, seq_len, num_features]
        
        # Compute mean activation of the important feature(s) across all positions
        mean_activation = latent_acts.mean().item()
        mean_threshold = avg_acts.mean().item()
        
        # Apply threshold multiplier - concept must activate X times above average to be "present"
        # Higher multiplier = stricter detection = fewer false positives
        adjusted_threshold = mean_threshold * self.detection_threshold_multiplier
        
        has_concept = mean_activation > adjusted_threshold
        
        # Debug info for first few timesteps
        if self.current_step < 3:
            print(f'[DETECT DEBUG] Timestep {self.current_step}: '
                  f'mean_act={mean_activation:.4f}, threshold={adjusted_threshold:.4f} '
                  f'(base={mean_threshold:.4f} x {self.detection_threshold_multiplier}), '
                  f'has_concept={has_concept}', flush=True)
        
        return has_concept
    
    def __call__(self, module, input, output):
        """Apply steering during forward pass, but only if concept is detected."""
        # Only apply steering after start_timestep
        if self.current_step < self.start_timestep:
            self.current_step += 1
            return output
        
        # Extract residual stream x from output
        if isinstance(output, tuple):
            x = output[0]
            return_tuple = True
        else:
            x = output
            return_tuple = False
        
        device = x.device
        
        # Handle classifier-free guidance
        if x.shape[0] % 2 == 0:
            x_uncond, x_cond = x.chunk(2)
            
            # Prepare x_cond for SAE (flatten if needed)
            if len(x_cond.shape) == 4:
                batch_size, channels, height, width = x_cond.shape
                x_cond_flat = x_cond.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            else:
                x_cond_flat = x_cond
            
            # Check if concept is detected using SAE encoder
            has_concept = self._detect_concept(x_cond_flat, device)
            
            # Debug logging
            if self.current_step % 10 == 0:
                print(f'[DEBUG] Timestep {self.current_step}: has_concept={has_concept}', flush=True)
            
            if has_concept:
                self.debug_concept_detected_count += 1
                # Apply steering (original formulation)
                x_cond_steered = self._apply_steering(x_cond)
                x_steered = torch.cat([x_uncond, x_cond_steered], dim=0)
                
                if self.current_step % 10 == 0:
                    diff = (x_cond_steered - x_cond).abs().mean().item()
                    print(f'[DEBUG] Timestep {self.current_step}: Applied steering, diff={diff:.6f}', flush=True)
            else:
                self.debug_concept_not_detected_count += 1
                # No steering - return original
                x_steered = x
        else:
            # No CFG case
            if len(x.shape) == 4:
                batch_size, channels, height, width = x.shape
                x_flat = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            else:
                x_flat = x
            
            has_concept = self._detect_concept(x_flat, device)
            
            if has_concept:
                self.debug_concept_detected_count += 1
                x_steered = self._apply_steering(x)
            else:
                self.debug_concept_not_detected_count += 1
                x_steered = x
        
        self.current_step += 1
        
        # Print summary at end
        if self.current_step == self.steps:
            print(f'[SUMMARY] Concept detected: {self.debug_concept_detected_count}, '
                  f'Not detected: {self.debug_concept_not_detected_count}', flush=True)
        
        if return_tuple:
            return (x_steered,) + output[1:]
        return x_steered
    
    def _apply_steering(self, x):
        """
        Apply steering using decoder columns for selected features.
        EXACT SAME FORMULATION as original:
        x̂ = x + α × Σ(βi × γi × D·,i)
        where βi = ||x||2 / ||D·,i||2
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
                gamma_i = self.gamma
                
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


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    class_params_path,
    gamma=1.0,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="eval_results/steering_results/",
    start_from=0,
    start_timestep=0,
    detection_threshold_multiplier=1.1,  # Higher = stricter concept detection
):
    """
    Generate images with steering vector-based unlearning using per-class alpha values.
    Only applies steering when concept is detected by SAE encoder.
    
    Args:
        detection_threshold_multiplier: Multiplier for detection threshold. 
            - 1.0 = concept detected if activation > average
            - 1.5 = concept detected if activation > 1.5x average (default, stricter)
            - 2.0 = concept detected if activation > 2x average (even stricter)
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
    
    # Load appropriate pipeline
    if is_sdxl:
        if accelerator.is_main_process:
            print("🎯 Detected SDXL model - using HookedStableDiffusionXLPipeline")
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
            print("🎯 Detected SD1.5 model - using HookedStableDiffusionPipeline")
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
    
    if accelerator.is_main_process:
        print(f"✓ VAE converted to float32 (dtype: {model.pipe.vae.dtype})")

    # Enable optimizations
    if is_sdxl and hasattr(model.pipe, 'enable_vae_slicing'):
        model.pipe.enable_vae_slicing()
    
    if hasattr(model.pipe, 'disable_vae_tiling'):
        model.pipe.disable_vae_tiling()

    if is_xformers_available():
        import xformers
        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems, please update xFormers to at least 0.0.17."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    if accelerator.is_main_process:
        print(f"Generator device: {device}")
        print(f"Model dtype: {model.pipe.unet.dtype}")
        print(f"Steering parameters: gamma={gamma}")
        print(f"Steering will be applied from timestep {start_timestep} to {steps}")
    
    # Load SAE and class latents
    sae = load_sae(sae_checkpoint, hookpoint, device)
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)
    
    # Load class parameters (contains percentile and alpha for each class)
    class_params = torch.load(class_params_path, weights_only=False)
    
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("LOADED CLASS PARAMETERS:")
        print("="*80)
        for class_name in class_available:
            if class_name in class_params:
                # Support both 'alpha' and 'multiplier' keys
                alpha = class_params[class_name].get("alpha", class_params[class_name].get("multiplier"))
                percentile = class_params[class_name]["percentile"]
                print(f"{class_name:20s} → alpha={alpha:7.2f}, percentile={percentile:.4f}")
        print("="*80 + "\n")

    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    classes_to_process = class_available[:]
    
    if accelerator.is_main_process:
        pipeline_type = "SDXL" if is_sdxl else "SD1.5"
        print(f"Using {pipeline_type} pipeline with conditional steering (only when concept detected)")
        print(f"Starting from index {start_from}, processing {len(classes_to_process)} classes")
    
    progress_bar = tqdm(
        classes_to_process,
        total=len(classes_to_process),
        disable=not accelerator.is_main_process,
        initial=0,
    )
    
    for class_to_unlearn in progress_bar:
        # Get class-specific parameters
        if class_to_unlearn not in class_params:
            if accelerator.is_main_process:
                print(f"Warning: No parameters found for {class_to_unlearn}, skipping...")
            continue
        
        # Support both 'alpha' and 'multiplier' keys
        class_alpha = class_params[class_to_unlearn].get("alpha", class_params[class_to_unlearn].get("multiplier"))
        class_percentile = class_params[class_to_unlearn]["percentile"]
        
        if accelerator.is_main_process:
            progress_bar.set_description(
                f"Steering away from {class_to_unlearn} (α={class_alpha:.2f}, p={class_percentile:.2f})"
            )
                
        output_path = os.path.join(output_dir, f"{class_to_unlearn}")
        os.makedirs(output_path, exist_ok=True)
        
        # Check if already completed
        if start_from > 0:
            expected_files = len(theme_avail) * len(class_available) + len(class_available)
            existing_files = len([f for f in os.listdir(output_path) if f.endswith(f'_seed{seed}.jpg')])
            
            if existing_files >= expected_files and accelerator.is_main_process:
                print(f"Skipping {class_to_unlearn} - already completed ({existing_files} files found)")
                continue
        
        for test_theme in theme_avail:
            input_classes = []
            input_themes = []
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            
            with accelerator.split_between_processes(class_theme_pairs) as local_classes_themes:
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                
                # Setup SAE decoder steering hook with class-specific parameters
                steering_hooks = {}
                steering_hooks[hookpoint] = SAEDecoderSteeringHook(
                    concept_to_unlearn=[class_to_unlearn],
                    percentile=class_percentile,
                    alpha=class_alpha,
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=class_latents_dict,
                    sae=sae,
                    steps=steps,
                    seed=seed,
                    gamma=gamma,
                    start_timestep=start_timestep,
                    detection_threshold_multiplier=detection_threshold_multiplier,
                )
                
                with torch.no_grad():
                    images = model.run_with_hooks(
                        prompt=local_prompts,
                        generator=generator,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        position_hook_dict=steering_hooks,
                    )
                
                for object_class, theme in local_classes_themes:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            
            if accelerator.is_main_process:
                for img, object_class, theme in zip(images, input_classes, input_themes):
                    if theme == "":
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}.jpg",
                            )
                        )
                    else:
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}.jpg",
                            )
                        )
        
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)