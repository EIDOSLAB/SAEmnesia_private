"""
SAE Feature Visualizer for Diffusion Models - Simplified Version

Supports two visualization modes:
1. top_latent_ablation: Shows which patches are ablated (color-coded by latent activity)
2. smooth_heatmap: Shows traditional activation heatmaps

Multi-latent support: Specify n_latents to visualize multiple top-scoring features simultaneously.
Color coding for ablation mode with multiple latents:
- Red: All top-n latents active
- Yellow: Only second-highest latent active
- Green: Only highest latent active
- White: No latents active

Usage:
    python visualize_sae_features_clean.py \
        --sae_path path/to/sae/checkpoint \
        --pipe_path path/to/stable-diffusion \
        --hookpoint unet.up_blocks.1.attentions.1 \
        --scores_json path/to/scores.json \
        --concept_name Dogs \
        --prompt "a photo of a dog" \
        --mode top_latent_ablation \
        --n_latents 2 \
        --output_dir visualizations
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import cv2
import fire

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.sae import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline


class SAEFeatureVisualizer:
    """Simplified SAE feature visualizer supporting two modes with multi-latent capability."""
    
    def __init__(
        self,
        sae_path: str,
        pipe_path: str,
        hookpoint: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        class_latents_path: Optional[str] = None,
        class_params_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.hookpoint = hookpoint
        
        # Load class data if provided
        self.class_latents_dict = None
        self.class_params = None
        
        if class_latents_path is not None:
            import pickle
            print(f"Loading class latents from {class_latents_path}")
            with open(class_latents_path, 'rb') as f:
                self.class_latents_dict = pickle.load(f)

        if class_params_path is not None:
            print(f"Loading class params from {class_params_path}")
            self.class_params = torch.load(class_params_path)

        # Load SAE
        print(f"Loading SAE from {sae_path}")
        self.sae = Sae.load_from_disk(sae_path, device=self.device)
        self.sae = self.sae.to(dtype=self.dtype).eval()
        self.sae.cfg.batch_topk = False
        self.sae.cfg.sample_topk = False
        
        # Load diffusion pipeline
        print(f"Loading Stable Diffusion pipeline from {pipe_path}")
        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            pipe_path,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)
        
        print(f"Visualizer initialized for hookpoint: {hookpoint}")
    
    @torch.no_grad()
    def generate_with_activations(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, List[Image.Image], torch.Tensor]:
        """Generate an image and extract SAE activations at each timestep."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        print(f"Generating image with prompt: '{prompt}'")
        
        # Run pipeline
        final_image, intermediate_images, cache_dict = self.pipe.run_with_cache_intermediate(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            positions_to_cache=[self.hookpoint],
            output_type="pil",
            save_output=True,
        )
        
        # Extract and process activations
        activations = cache_dict["output"][self.hookpoint]
        batch_size, timesteps, spatial_tokens, channels = activations.shape
        
        # Process through SAE
        print(f"Processing activations through SAE...")
        activations_flat = activations.reshape(batch_size * timesteps * spatial_tokens, channels)
        
        sae_latents = []
        chunk_size = 1000
        
        for i in range(0, activations_flat.shape[0], chunk_size):
            chunk = activations_flat[i:i+chunk_size].to(self.device)
            pre_acts = self.sae.pre_acts(chunk)
            top_acts, top_indices = self.sae.select_topk(pre_acts)
            
            sae_out = torch.zeros(
                (top_acts.shape[0], self.sae.num_latents),
                device=self.device,
                dtype=top_acts.dtype,
            ).scatter(-1, top_indices, top_acts)
            
            sae_latents.append(sae_out.cpu())
        
        sae_latents = torch.cat(sae_latents, dim=0)
        sae_latents = sae_latents.reshape(batch_size, timesteps, spatial_tokens, self.sae.num_latents)
        sae_latents = sae_latents[0]  # Take first batch element
        
        return final_image[0], intermediate_images, sae_latents
    
    def create_activation_heatmap(
        self,
        sae_activations: torch.Tensor,
        feature_idx: int,
        timestep_idx: int,
        target_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """Create a smooth heatmap for a feature at a timestep."""
        feature_acts = sae_activations[timestep_idx, :, feature_idx]
        spatial_size = int(np.sqrt(feature_acts.shape[0]))
        
        heatmap = feature_acts.reshape(spatial_size, spatial_size).cpu().numpy()
        
        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.ones_like(heatmap) * 0.5
        
        # Resize
        heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
        return heatmap
    
    def overlay_heatmap(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay a heatmap on an image."""
        img_array = np.array(image)
        
        # Ensure RGB
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def create_multi_latent_ablation_mask(
        self,
        sae_activations: torch.Tensor,
        feature_indices: List[int],
        timestep_idx: int,
        concept_name: str,
        target_size: Tuple[int, int] = (512, 512),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create a color-coded ablation mask for multiple latents.
        
        Returns:
            mask: Array with values:
                - 0: No latents active
                - 1: Only highest latent active (green)
                - 2: Only second latent active (yellow)
                - 3: Both/all latents active (red)
            stats: Dictionary with patch statistics (before resizing)
        """
        from SAE.unlearning_utils import compute_feature_importance, get_percentile_threshold
        
        percentile = self.class_params[concept_name]["percentile"]
        
        # Get activations for this timestep
        timestep_acts = sae_activations[timestep_idx]  # [spatial_tokens, num_latents]
        spatial_size = int(np.sqrt(timestep_acts.shape[0]))
        
        # Compute feature importance
        feature_scores = compute_feature_importance(
            self.class_latents_dict, concept_name, timestep_idx
        ).float()
        percentile_threshold = get_percentile_threshold(feature_scores, percentile)
        
        # Create mask for each latent
        latent_masks = []
        
        for i, feat_idx in enumerate(feature_indices):
            # Check if feature is important at this timestep (for informational purposes)
            is_important = feature_scores[feat_idx] > percentile_threshold
            
            # Note: We don't use the importance check to skip visualization
            # because we've already selected it as a top-N feature globally.
            # During actual unlearning, the per-timestep importance check matters,
            # but for visualization we want to see all selected features.
            
            # Compute threshold
            all_class_avg_act = torch.zeros(1)
            for class_name in self.class_latents_dict:
                all_class_avg_act += self.class_latents_dict[class_name][
                    :, timestep_idx, feat_idx
                ].mean()
            all_class_avg_act /= len(self.class_latents_dict)
            threshold = all_class_avg_act.item()
            
            # Get activations for this feature
            feature_acts = timestep_acts[:, feat_idx].cpu().numpy()
            feature_acts_grid = feature_acts.reshape(spatial_size, spatial_size)
            
            print(f"\nLatent {i+1} (feature {feat_idx}) at timestep {timestep_idx}:")
            print(f"  Importance score: {feature_scores[feat_idx]:.4f} (threshold: {percentile_threshold:.4f})")
            print(f"  *** {'IMPORTANT' if is_important else 'NOT IMPORTANT'} at this timestep ***")
            print(f"  Activation threshold: {threshold:.4f}")
            print(f"  Max activation in image: {feature_acts_grid.max():.4f}")
            print(f"  Mean activation in image: {feature_acts_grid.mean():.4f}")
            print(f"  Patches above threshold: {(feature_acts_grid > threshold).sum()}/{feature_acts_grid.size}")
            
            # Create binary mask
            mask = (feature_acts_grid > threshold).astype(np.float32)
            
            latent_masks.append(mask)
        
        # Combine masks with color coding
        combined_mask = np.zeros((spatial_size, spatial_size))
        
        if len(latent_masks) == 1:
            # Single latent: just use binary (0 or 1)
            combined_mask = latent_masks[0]
        elif len(latent_masks) == 2:
            # Two latents: 0=none, 1=first only, 2=second only, 3=both
            first_only = (latent_masks[0] == 1) & (latent_masks[1] == 0)
            second_only = (latent_masks[0] == 0) & (latent_masks[1] == 1)
            both = (latent_masks[0] == 1) & (latent_masks[1] == 1)
            
            combined_mask[first_only] = 1
            combined_mask[second_only] = 2
            combined_mask[both] = 3
        else:
            # Multiple latents: 0=none, 1=first only, 2=others, 3=all
            all_active = np.all(np.array(latent_masks) == 1, axis=0)
            first_only = (latent_masks[0] == 1) & ~all_active
            others_active = (np.any(np.array(latent_masks[1:]) == 1, axis=0)) & (latent_masks[0] == 0)
            
            combined_mask[first_only] = 1
            combined_mask[others_active] = 2
            combined_mask[all_active] = 3
        
        # Compute statistics on the original spatial grid (before resizing)
        stats = {
            'n_green': int(np.sum(combined_mask == 1)),
            'n_yellow': int(np.sum(combined_mask == 2)),
            'n_red': int(np.sum(combined_mask == 3)),
            'n_total': int(combined_mask.size),
        }
        
        # Resize
        combined_mask = cv2.resize(
            combined_mask,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return combined_mask, stats
    
    def overlay_ablation_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay color-coded ablation mask on image.
        
        Mask values:
        - 0: No color (transparent)
        - 1: Green (highest latent only)
        - 2: Yellow (second latent only)
        - 3: Red (all latents)
        """
        img_array = np.array(image)
        
        # Ensure RGB
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # Create colored overlay
        overlay_colored = np.zeros_like(img_array)
        
        # Color mapping
        colors = {
            1: (0, 255, 0),    # Green - highest latent only
            2: (255, 255, 0),  # Yellow - second latent only
            3: (255, 0, 0),    # Red - all latents
        }
        
        for value, color in colors.items():
            overlay_colored[mask == value] = color
        
        # Create alpha mask
        alpha_mask = np.where(mask > 0, alpha, 0)
        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
        # Blend
        overlay = img_array.astype(np.float32) * (1 - alpha_mask) + overlay_colored.astype(np.float32) * alpha_mask
        overlay = overlay.astype(np.uint8)
        
        return overlay
    
    def _ensure_pil_image(self, img) -> Image.Image:
        """Convert various image formats to PIL Image."""
        # Handle list/tuple
        if isinstance(img, (list, tuple)):
            img = img[0]
        
        # If already PIL Image, return it
        if isinstance(img, Image.Image):
            return img
        
        # Convert tensor to numpy
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Process numpy array
        if isinstance(img, np.ndarray):
            # Remove batch dimension
            while img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            
            # Transpose if channels first
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            
            # Convert to uint8
            if img.dtype in (np.float32, np.float64):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            return Image.fromarray(img)
        
        # If we get here, try one more conversion attempt
        raise TypeError(f"Cannot convert type {type(img)} to PIL Image")
    
    def visualize_top_latent_ablation(
        self,
        prompt: str,
        scores_json: str,
        concept_name: str,
        n_latents: int = 1,
        timesteps_to_show: List[int] = [47, 30, 10, 1],
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.5,
    ):
        """
        Visualize top n latents with ablation highlighting across timesteps.
        
        Color coding:
        - Red: All top-n latents active
        - Yellow: Only second-highest latent active
        - Green: Only highest latent active
        - White: No latents active
        """
        if self.class_latents_dict is None or self.class_params is None:
            raise ValueError("Must provide class_latents_path and class_params_path")
        
        # Load scores
        with open(scores_json, 'r') as f:
            scores_data = json.load(f)
        
        if concept_name not in scores_data.get('scores', {}):
            concept_name = concept_name.replace('_', ' ')
        
        concept_scores = scores_data['scores'][concept_name]
        if isinstance(concept_scores[0], list):
            concept_scores = np.mean(concept_scores, axis=0)
        
        # Get top n feature indices
        top_feature_indices = np.argsort(concept_scores)[::-1][:n_latents]
        top_scores = [concept_scores[i] for i in top_feature_indices]
        
        print(f"\nTop {n_latents} feature(s) for '{concept_name}':")
        for idx, score in zip(top_feature_indices, top_scores):
            print(f"  Feature {idx}: {score:.4f}")
        
        # Generate image
        final_image, intermediate_images, sae_activations = self.generate_with_activations(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        # Create figure
        n_timesteps = len(timesteps_to_show)
        fig, axes = plt.subplots(1, n_timesteps, figsize=(5 * n_timesteps, 5))
        
        if n_timesteps == 1:
            axes = [axes]
        
        for idx, t in enumerate(timesteps_to_show):
            img = self._ensure_pil_image(intermediate_images[t])
            
            # Create multi-latent ablation mask
            mask, stats = self.create_multi_latent_ablation_mask(
                sae_activations,
                top_feature_indices.tolist(),
                t,
                concept_name,
            )
            
            # Create overlay
            overlay = self.overlay_ablation_mask(img, mask, alpha=alpha)
            
            # Plot
            axes[idx].imshow(overlay)
            axes[idx].axis('off')
            axes[idx].set_title(f'Timestep {t}', fontsize=12, fontweight='bold')
            
            # Add statistics (using pre-computed stats from original grid)
            stats_text = []
            
            if n_latents == 1:
                pct = (stats['n_green'] / stats['n_total'] * 100) if stats['n_total'] > 0 else 0
                stats_text.append(f'Active: {pct:.1f}%')
                stats_text.append(f'({stats["n_green"]}/{stats["n_total"]} patches)')
            elif n_latents == 2:
                stats_text.append(f'Green (1st only): {stats["n_green"]}')
                stats_text.append(f'Yellow (2nd only): {stats["n_yellow"]}')
                stats_text.append(f'Red (both): {stats["n_red"]}')
            else:
                stats_text.append(f'Green (1st): {stats["n_green"]}')
                stats_text.append(f'Yellow (others): {stats["n_yellow"]}')
                stats_text.append(f'Red (all): {stats["n_red"]}')
            
            axes[idx].text(
                0.5, -0.05,
                '\n'.join(stats_text),
                transform=axes[idx].transAxes,
                ha='center',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Title
        feature_str = f"Feature {top_feature_indices[0]}" if n_latents == 1 else f"Top {n_latents} Features"
        fig.suptitle(
            f'{feature_str} - Ablation Masks for: "{concept_name}"\n'
            f'Prompt: "{prompt}"',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_smooth_heatmap(
        self,
        prompt: str,
        scores_json: str,
        concept_name: str,
        n_latents: int = 1,
        timesteps_to_show: List[int] = [47, 30, 10, 1],
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.6,
    ):
        """Visualize top n latents with smooth heatmaps across timesteps."""
        # Load scores
        with open(scores_json, 'r') as f:
            scores_data = json.load(f)
        
        if concept_name not in scores_data.get('scores', {}):
            concept_name = concept_name.replace('_', ' ')
        
        concept_scores = scores_data['scores'][concept_name]
        if isinstance(concept_scores[0], list):
            concept_scores = np.mean(concept_scores, axis=0)
        
        # Get top n feature indices
        top_feature_indices = np.argsort(concept_scores)[::-1][:n_latents]
        top_scores = [concept_scores[i] for i in top_feature_indices]
        
        print(f"\nTop {n_latents} feature(s) for '{concept_name}':")
        for idx, score in zip(top_feature_indices, top_scores):
            print(f"  Feature {idx}: {score:.4f}")
        
        # Generate image
        final_image, intermediate_images, sae_activations = self.generate_with_activations(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        # Create figure
        n_timesteps = len(timesteps_to_show)
        fig = plt.figure(figsize=(5 * n_timesteps, 5 * n_latents))
        gs = GridSpec(n_latents, n_timesteps, figure=fig, hspace=0.25, wspace=0.15)
        
        for feat_row, (feat_idx, score) in enumerate(zip(top_feature_indices, top_scores)):
            for col_idx, t in enumerate(timesteps_to_show):
                ax = fig.add_subplot(gs[feat_row, col_idx])
                
                img = self._ensure_pil_image(intermediate_images[t])
                
                # Create heatmap
                heatmap = self.create_activation_heatmap(sae_activations, feat_idx, t)
                overlay = self.overlay_heatmap(img, heatmap, alpha=alpha)
                
                # Plot
                ax.imshow(overlay)
                ax.axis('off')
                
                # Title
                if feat_row == 0:
                    ax.set_title(f'Timestep {t}', fontsize=12, fontweight='bold')
                
                if col_idx == 0:
                    ax.set_ylabel(
                        f'Feature {feat_idx}\n(Score: {score:.3f})',
                        fontsize=11,
                        fontweight='bold',
                        rotation=0,
                        ha='right',
                        va='center',
                        labelpad=40
                    )
                
                # Stats
                max_act = heatmap.max()
                ax.text(
                    0.5, -0.02,
                    f'Max: {max_act:.3f}',
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )
        
        # Title
        feature_str = f"Feature {top_feature_indices[0]}" if n_latents == 1 else f"Top {n_latents} Features"
        fig.suptitle(
            f'{feature_str} - Heatmaps for: "{concept_name}"\nPrompt: "{prompt}"',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main(
    sae_path: str,
    pipe_path: str,
    hookpoint: str = "unet.up_blocks.1.attentions.1",
    prompt: str = "a photo of a dog",
    mode: str = "top_latent_ablation",  # or "smooth_heatmap"
    scores_json: str = None,
    concept_name: str = None,
    n_latents: int = 1,
    timesteps_to_show: str = "47,30,10,1",
    num_inference_steps: int = 50,
    seed: Optional[int] = 188,
    output_dir: str = "visualizations",
    device: str = "cuda",
    class_latents_path: Optional[str] = None,
    class_params_path: Optional[str] = None,
    alpha: float = 0.5,
):
    """
    Main function to run SAE feature visualizations.
    
    Args:
        mode: 'top_latent_ablation' or 'smooth_heatmap'
        n_latents: Number of top-scoring latents to visualize (1-5 recommended)
        alpha: Transparency of overlay (0.0-1.0)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = SAEFeatureVisualizer(
        sae_path=sae_path,
        pipe_path=pipe_path,
        hookpoint=hookpoint,
        device=device,
        class_latents_path=class_latents_path,
        class_params_path=class_params_path,
    )
    
    # Parse timesteps
    if isinstance(timesteps_to_show, str):
        timesteps = [int(t.strip()) for t in timesteps_to_show.split(',')]
    elif isinstance(timesteps_to_show, (list, tuple)):
        timesteps = [int(t) for t in timesteps_to_show]
    else:
        timesteps = [int(timesteps_to_show)]
    
    # Create output path
    save_path = os.path.join(
        output_dir,
        f"{mode}_{concept_name}_n{n_latents}.png"
    )
    
    if mode == "top_latent_ablation":
        if class_latents_path is None or class_params_path is None:
            raise ValueError("Must provide class_latents_path and class_params_path for ablation mode")
        
        visualizer.visualize_top_latent_ablation(
            prompt=prompt,
            scores_json=scores_json,
            concept_name=concept_name,
            n_latents=n_latents,
            timesteps_to_show=timesteps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path,
            alpha=alpha,
        )
    
    elif mode == "smooth_heatmap":
        visualizer.visualize_smooth_heatmap(
            prompt=prompt,
            scores_json=scores_json,
            concept_name=concept_name,
            n_latents=n_latents,
            timesteps_to_show=timesteps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path,
            alpha=alpha,
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'top_latent_ablation' or 'smooth_heatmap'")
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    fire.Fire(main)