"""
SAE Feature Visualizer for Diffusion Models - Enhanced Version

This script visualizes SAE feature activations on generated images to understand
what patterns each feature responds to. This enhanced version also saves images
with the top 5 latent activation values displayed on each patch with color coding.

Usage:
    python visualize_sae_features.py \
        --sae_path path/to/sae/checkpoint \
        --pipe_path path/to/stable-diffusion \
        --hookpoint unet.up_blocks.1.attentions.1 \
        --scores_json path/to/scores.json \
        --prompt "a photo of a dog" \
        --output_dir visualizations
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import fire

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.sae import Sae
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline


class SAEFeatureVisualizer:
    """
    Visualizer for SAE features in diffusion models.
    
    This class handles:
    1. Extracting SAE activations during generation
    2. Predicting x0 at each timestep
    3. Creating activation heatmaps
    4. Overlaying heatmaps on images
    5. Generating publication-quality visualizations
    6. [NEW] Creating value-annotated patches showing top 5 latent activations
    """
    
    def __init__(
        self,
        sae_path: str,
        pipe_path: str,
        hookpoint: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        class_latents_path: Optional[str] = None,  # NEW
        class_params_path: Optional[str] = None,   # NEW
    ):
        """
        Initialize the visualizer.
        
        Args:
            sae_path: Path to SAE checkpoint
            pipe_path: Path to Stable Diffusion model
            hookpoint: Layer to extract activations from (e.g., 'unet.up_blocks.1.attentions.1')
            device: Device to run on
            dtype: Data type for computations
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.hookpoint = hookpoint
        
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
        print(f"SAE dimensions: d_in={self.sae.d_in}, num_latents={self.sae.num_latents}")
    
    @torch.no_grad()
    def generate_with_activations(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, List[Image.Image], torch.Tensor]:
        """
        Generate an image and extract SAE activations at each timestep.
        
        Args:
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            
        Returns:
            final_image: The final generated image
            intermediate_images: List of x0 predictions at each timestep
            sae_activations: SAE feature activations [timesteps, spatial_tokens, num_latents]
        """
        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
        
        print(f"Generating image with prompt: '{prompt}'")
        
        # Run pipeline with intermediate caching
        final_image, intermediate_images, cache_dict = self.pipe.run_with_cache_intermediate(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            positions_to_cache=[self.hookpoint],
            output_type="pil",
            save_output=True,
        )
        
        # Extract activations from cache
        activations = cache_dict["output"][self.hookpoint]  # [batch, timesteps, spatial, channels]
        
        # Process through SAE
        print(f"Processing activations through SAE...")
        print(f"Raw activation shape: {activations.shape}")
        
        batch_size, timesteps, spatial_tokens, channels = activations.shape
        
        # Reshape for SAE processing
        activations_flat = activations.reshape(batch_size * timesteps * spatial_tokens, channels)
        
        # Get SAE latent activations
        sae_latents = []
        chunk_size = 1000  # Process in chunks to avoid OOM
        
        for i in range(0, activations_flat.shape[0], chunk_size):
            chunk = activations_flat[i:i+chunk_size].to(self.device)
            
            # Get pre-activations (before top-k selection)
            pre_acts = self.sae.pre_acts(chunk)
            
            # Apply top-k selection
            top_acts, top_indices = self.sae.select_topk(pre_acts)
            
            # Create dense representation
            sae_out = torch.zeros(
                (top_acts.shape[0], self.sae.num_latents),
                device=self.device,
                dtype=top_acts.dtype,
            ).scatter(-1, top_indices, top_acts)
            
            sae_latents.append(sae_out.cpu())
        
        sae_latents = torch.cat(sae_latents, dim=0)
        
        # Reshape back to [batch, timesteps, spatial, num_latents]
        sae_latents = sae_latents.reshape(batch_size, timesteps, spatial_tokens, self.sae.num_latents)
        
        # Take the first batch element (single image generation)
        sae_latents = sae_latents[0]  # [timesteps, spatial, num_latents]
        
        print(f"SAE activations shape: {sae_latents.shape}")
        
        return final_image[0], intermediate_images, sae_latents
    
    def create_activation_heatmap(
        self,
        sae_activations: torch.Tensor,
        feature_idx: int,
        timestep_idx: int,
        target_size: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        """
        Create a heatmap for a specific feature at a specific timestep.

        Args:
            sae_activations: Tensor of shape [timesteps, spatial_tokens, num_latents]
            feature_idx: Index of the feature to visualize
            timestep_idx: Index of the timestep
            target_size: Target image size (H, W)

        Returns:
            heatmap: numpy array of shape [H, W] with values in [0, 1]
        """
        # Extract activations for this feature and timestep
        feature_acts = sae_activations[timestep_idx, :, feature_idx]  # [spatial_tokens]

        # Determine spatial dimensions
        spatial_size = int(np.sqrt(feature_acts.shape[0]))

        # Reshape to 2D grid
        heatmap = feature_acts.reshape(spatial_size, spatial_size).cpu().numpy()

        # Ensure float type
        heatmap = heatmap.astype(np.float32)

        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            # If all values are the same, create a uniform heatmap
            heatmap = np.ones_like(heatmap) * 0.5

        # cv2.resize expects (width, height), so swap target_size
        # target_size is (H, W), but cv2 wants (W, H)
        target_size_cv2 = (target_size[1], target_size[0])

        # Resize to target size using bicubic interpolation
        heatmap = cv2.resize(heatmap, target_size_cv2, interpolation=cv2.INTER_CUBIC)

        return heatmap
    
    def overlay_heatmap(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        colormap: str = 'jet',
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay a heatmap on an image.

        Args:
            image: PIL Image
            heatmap: numpy array [H, W] with values in [0, 1]
            colormap: matplotlib colormap name
            alpha: transparency of the heatmap overlay

        Returns:
            overlay: RGB image with heatmap overlay
        """
        # Convert image to numpy array
        img_array = np.array(image)

        # Remove any batch dimensions
        while img_array.ndim == 4 and img_array.shape[0] == 1:
            img_array = img_array[0]

        # Remove any batch dimensions from heatmap
        while heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]

        # Ensure image is RGB (3 channels)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        elif img_array.ndim != 3 or img_array.shape[2] != 3:
            raise ValueError(f"Unexpected image shape after processing: {img_array.shape}")

        # Ensure heatmap is 2D
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap should be 2D, got shape: {heatmap.shape}")

        # Ensure heatmap matches image dimensions
        if heatmap.shape[:2] != img_array.shape[:2]:
            heatmap = cv2.resize(
                heatmap, 
                (img_array.shape[1], img_array.shape[0]),  # (width, height)
                interpolation=cv2.INTER_CUBIC
            )

        # Apply colormap
        cmap = plt.get_cmap(colormap)
        heatmap_colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)

        # Ensure both arrays have the same shape
        assert img_array.shape == heatmap_colored.shape, \
            f"Shape mismatch: img_array {img_array.shape} vs heatmap_colored {heatmap_colored.shape}"

        # Blend images
        overlay = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay
    
    def create_single_feature_value_image(
        self,
        image: Image.Image,
        sae_activations: torch.Tensor,
        feature_idx: int,
        timestep_idx: int,
        target_size: Tuple[int, int] = (512, 512),
        class_latents_dict: Optional[Dict] = None,  # NEW: dictionary of class activations
        class_to_check: Optional[str] = None,  # NEW: which class to check ablation for
        percentile: Optional[float] = None,  # NEW: percentile threshold used in unlearning
    ) -> Image.Image:
        """
        Create an image showing the activation value and ablation status for the top activating patch.
        Shows whether the patch would be ablated during unlearning of a specific class.

        Args:
            image: PIL Image to use as base
            sae_activations: Tensor of shape [timesteps, spatial_tokens, num_latents]
            feature_idx: Index of the feature to visualize
            timestep_idx: Index of the timestep
            target_size: Target image size (H, W)
            class_latents_dict: Dictionary mapping class names to their activation tensors
                               Shape per class: [num_samples, timesteps, num_latents]
            class_to_check: Name of the class being unlearned (to compute ablation threshold)
            percentile: Percentile threshold used for selecting important features

        Returns:
            annotated_image: PIL Image with top activation value and ablation status displayed
        """
        # Get activations for this feature and timestep
        feature_acts = sae_activations[timestep_idx, :, feature_idx]  # [spatial_tokens]

        # Determine spatial dimensions
        spatial_size = int(np.sqrt(feature_acts.shape[0]))

        # Reshape to 2D grid
        acts_grid = feature_acts.reshape(spatial_size, spatial_size).cpu().numpy()

        # Find the top activating patch
        top_idx = np.argmax(acts_grid)
        top_i, top_j = np.unravel_index(top_idx, acts_grid.shape)
        top_value = acts_grid[top_i, top_j]

        # Compute ablation threshold if we have the necessary info
        is_ablated = None
        avg_threshold = None
        if class_latents_dict is not None and class_to_check is not None and percentile is not None:
            # Check if this feature would be selected for ablation
            from SAE.unlearning_utils import compute_feature_importance, get_percentile_threshold

            # Compute feature importance scores for this class at this timestep
            feature_scores = compute_feature_importance(
                class_latents_dict, class_to_check, timestep_idx
            ).float()

            # Get percentile threshold
            percentile_threshold = get_percentile_threshold(feature_scores, percentile)

            # Check if this feature is in the top features to ablate
            is_important_feature = feature_scores[feature_idx] > percentile_threshold

            if is_important_feature:
                # Compute average activation threshold across all classes
                # This matches the logic in SAEMaskedUnlearningHook.__init__
                all_class_avg_act = torch.zeros(1)
                for class_name in class_latents_dict:
                    # class_latents_dict[class_name] shape: [num_samples, timesteps, num_latents]
                    all_class_avg_act += class_latents_dict[class_name][
                        :, timestep_idx, feature_idx
                    ].mean()
                all_class_avg_act /= len(class_latents_dict)
                avg_threshold = all_class_avg_act.item()

                # Determine if this patch would be ablated
                # Matches the condition in SAEMaskedUnlearningHook.__call__ line 131-133
                is_ablated = top_value > avg_threshold
            else:
                # Feature is not selected for ablation
                is_ablated = False

        # Create a new image with white background
        width, height = target_size
        result_img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(result_img)

        # Try to load fonts
        try:
            # Adjust font size based on spatial dimensions
            font_size = max(8, min(24, width // (spatial_size * 3)))
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            small_font_size = max(6, font_size - 4)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", small_font_size)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Calculate patch dimensions
        patch_h = height / spatial_size
        patch_w = width / spatial_size

        # Create colormap
        cmap = cm.get_cmap('RdYlGn')  # Red (low) -> Yellow (mid) -> Green (high)
        norm = Normalize(vmin=acts_grid.min(), vmax=acts_grid.max())

        # Draw grid lines for all patches
        for i in range(spatial_size + 1):
            y = int(i * patch_h)
            draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=1)
        for j in range(spatial_size + 1):
            x = int(j * patch_w)
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=1)

        # Draw only the top activating patch
        i, j = top_i, top_j

        # Get patch bounds
        y1 = int(i * patch_h)
        y2 = int((i + 1) * patch_h)
        x1 = int(j * patch_w)
        x2 = int((j + 1) * patch_w)

        # Get color based on value
        color_rgba = cmap(norm(top_value))
        color_rgb = tuple(int(c * 255) for c in color_rgba[:3])

        # Draw colored background
        draw.rectangle([x1, y1, x2, y2], fill=color_rgb, outline=(0, 0, 0), width=2)

        # Format activation value text
        value_text = f"{top_value:.3f}"

        # Get text bounding boxes
        value_bbox = draw.textbbox((0, 0), value_text, font=font)
        value_text_width = value_bbox[2] - value_bbox[0]
        value_text_height = value_bbox[3] - value_bbox[1]

        # Calculate text position
        if is_ablated is not None:
            # If showing ablation status, place activation value in upper part
            value_text_x = x1 + (patch_w - value_text_width) / 2
            value_text_y = y1 + (patch_h * 0.3 - value_text_height) / 2

            # Format ablation status text
            ablation_text = "ABLATED" if is_ablated else "NOT ABLATED"
            ablation_color = (255, 0, 0) if is_ablated else (0, 128, 0)  # Red if ablated, green if not

            ablation_bbox = draw.textbbox((0, 0), ablation_text, font=small_font)
            ablation_text_width = ablation_bbox[2] - ablation_bbox[0]
            ablation_text_height = ablation_bbox[3] - ablation_bbox[1]

            # Place ablation status in middle part of patch
            ablation_text_x = x1 + (patch_w - ablation_text_width) / 2
            ablation_text_y = y1 + patch_h * 0.5 - ablation_text_height / 2

            # Show threshold in bottom part if available
            if avg_threshold is not None:
                threshold_text = f"Thr: {avg_threshold:.3f}"
                threshold_bbox = draw.textbbox((0, 0), threshold_text, font=small_font)
                threshold_text_width = threshold_bbox[2] - threshold_bbox[0]
                threshold_text_height = threshold_bbox[3] - threshold_bbox[1]
                threshold_text_x = x1 + (patch_w - threshold_text_width) / 2
                threshold_text_y = y1 + patch_h * 0.8 - threshold_text_height / 2
                draw.text((threshold_text_x, threshold_text_y), threshold_text, fill=(100, 100, 100), font=small_font)

            # Draw texts
            draw.text((value_text_x, value_text_y), value_text, fill=(0, 0, 0), font=font)
            draw.text((ablation_text_x, ablation_text_y), ablation_text, fill=ablation_color, font=small_font)
        else:
            # Just center the activation value
            value_text_x = x1 + (patch_w - value_text_width) / 2
            value_text_y = y1 + (patch_h - value_text_height) / 2
            draw.text((value_text_x, value_text_y), value_text, fill=(0, 0, 0), font=font)

        return result_img
    
    def create_top5_activation_image(
        self,
        image: Image.Image,
        sae_activations: torch.Tensor,
        timestep_idx: int,
        top_feature_indices: List[int],
        patch_size: int = 64,
        class_latents_dict: Optional[Dict] = None,
        class_to_check: Optional[str] = None,
        percentile: Optional[float] = None,
    ) -> Image.Image:
        """
        Create an image showing the activation value of the TOP SCORING feature for each patch
        with ablation status indicator (A/NA).
        
        The top scoring feature is the SAME across all patches (the most important feature
        for this object at this timestep).
        
        Args:
            image: PIL Image to annotate
            sae_activations: Tensor of shape [timesteps, spatial_tokens, num_latents]
            timestep_idx: Index of the timestep
            top_feature_indices: List of feature indices ranked by importance score
            patch_size: Size of each patch in pixels for annotation
            class_latents_dict: Dictionary mapping class names to their activation tensors
            class_to_check: Name of the class being unlearned
            percentile: Percentile threshold used for selecting important features
            
        Returns:
            annotated_image: PIL Image with activation values and ablation status overlaid
        """
        # Get activations for this timestep
        timestep_acts = sae_activations[timestep_idx]  # [spatial_tokens, num_latents]
        
        # Determine spatial dimensions
        spatial_size = int(np.sqrt(timestep_acts.shape[0]))
        
        # THE TOP SCORING FEATURE IS THE FIRST ONE IN THE LIST
        top_feature_id = top_feature_indices[0]
        
        print(f"\n=== Showing activations for TOP SCORING FEATURE: {top_feature_id} ===")
        
        # Compute ablation threshold for this feature
        threshold = None
        if class_latents_dict is not None and class_to_check is not None and percentile is not None:
            from SAE.unlearning_utils import compute_feature_importance, get_percentile_threshold
            
            # Compute feature importance scores for this class at this timestep
            feature_scores = compute_feature_importance(
                class_latents_dict, class_to_check, timestep_idx
            ).float()
            
            # Get percentile threshold
            percentile_threshold = get_percentile_threshold(feature_scores, percentile)
            
            is_important = feature_scores[top_feature_id] > percentile_threshold
            
            if is_important:
                # Compute average activation threshold across all classes
                all_class_avg_act = torch.zeros(1)
                for class_name in class_latents_dict:
                    all_class_avg_act += class_latents_dict[class_name][
                        :, timestep_idx, top_feature_id
                    ].mean()
                all_class_avg_act /= len(class_latents_dict)
                threshold = all_class_avg_act.item()
                print(f"Feature {top_feature_id}: IMPORTANT, threshold={threshold:.4f}, score={feature_scores[top_feature_id]:.4f}")
            else:
                print(f"Feature {top_feature_id}: NOT important (won't be ablated)")
        
        # Create a copy of the image
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Create overlay image
        overlay = Image.new('RGBA', (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8)
        except:
            font = ImageFont.load_default()
        
        # Calculate patch dimensions
        patch_h = height / spatial_size
        patch_w = width / spatial_size
        
        # Get activations for the TOP SCORING feature across all patches
        feature_acts = timestep_acts[:, top_feature_id].cpu().numpy()  # [spatial_tokens]
        feature_acts_grid = feature_acts.reshape(spatial_size, spatial_size)
        
        # Find global min/max for consistent color coding
        global_min = feature_acts_grid.min()
        global_max = feature_acts_grid.max()
        
        # Create colormap
        cmap = cm.get_cmap('RdYlGn')  # Red (low) -> Yellow (mid) -> Green (high)
        norm = Normalize(vmin=global_min, vmax=global_max)
        
        print(f"Activation range: [{global_min:.4f}, {global_max:.4f}]")
        
        # Iterate through patches
        for i in range(spatial_size):
            for j in range(spatial_size):
                # Get patch bounds
                y1 = int(i * patch_h)
                y2 = int((i + 1) * patch_h)
                x1 = int(j * patch_w)
                x2 = int((j + 1) * patch_w)
                
                # Get activation value for this patch (for the top scoring feature)
                activation_value = feature_acts_grid[i, j]
                
                # Get color based on value
                color_rgba = cmap(norm(activation_value))
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
                
                # Determine ablation status
                ablation_status = ""
                if threshold is not None:
                    is_ablated = activation_value > threshold
                    ablation_status = " A" if is_ablated else " NA"
                
                # Format text with ablation status
                text = f"{activation_value:.2f}{ablation_status}"
                
                # Get text bounding box for background
                text_bbox = draw.textbbox((x1 + 2, y1 + 2), text, font=font)
                
                # Draw colored background
                draw.rectangle(text_bbox, fill=color_rgb + (200,))
                
                # Draw text
                draw.text((x1 + 2, y1 + 2), text, fill=(0, 0, 0, 255), font=font)
        
        # Composite overlay onto original image
        base_img = image.convert('RGBA')
        result = Image.alpha_composite(base_img, overlay)
        result = result.convert('RGB')
        
        return result
    
    def visualize_feature_across_timesteps(
        self,
        prompt: str,
        feature_idx: int,
        timesteps_to_show: List[int] = [0, 10, 20, 30, 40, 49],
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Visualize a single feature's evolution across timesteps (Figure 5 style).
        
        Args:
            prompt: Generation prompt
            feature_idx: Index of the feature to visualize
            timesteps_to_show: List of timestep indices to display
            num_inference_steps: Total inference steps
            seed: Random seed
            save_path: Path to save the visualization
        """
        # Generate image with activations
        final_image, intermediate_images, sae_activations = self.generate_with_activations(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        # Create figure
        n_timesteps = len(timesteps_to_show)
        fig, axes = plt.subplots(2, n_timesteps, figsize=(4 * n_timesteps, 8))
        
        if n_timesteps == 1:
            axes = axes.reshape(2, 1)
        
        for idx, t in enumerate(timesteps_to_show):
            # Get intermediate image
            img = intermediate_images[t]
            
            # Create heatmap
            heatmap = self.create_activation_heatmap(
                sae_activations,
                feature_idx,
                t,
            )
            
            # Create overlay
            overlay = self.overlay_heatmap(img, heatmap, alpha=0.6)
            
            # Plot intermediate image
            axes[0, idx].imshow(img)
            axes[0, idx].axis('off')
            axes[0, idx].set_title(f'Timestep {t}', fontsize=12)
            
            # Plot heatmap overlay
            axes[1, idx].imshow(overlay)
            axes[1, idx].axis('off')
            
            # Add activation statistics
            max_act = heatmap.max()
            mean_act = heatmap.mean()
            axes[1, idx].text(
                0.5, -0.05,
                f'Max: {max_act:.3f}\nMean: {mean_act:.3f}',
                transform=axes[1, idx].transAxes,
                ha='center',
                va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        # Add overall title
        fig.suptitle(
            f'Feature {feature_idx} Evolution\nPrompt: "{prompt}"',
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
    
    def visualize_top_features_for_concept(
        self,
        prompt: str,
        scores_json: str,
        concept_name: str,
        top_k: int = 10,
        timestep_idx: int = 25,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        save_values_image: bool = True,
    ):
        """
        Visualize top-k features for a specific concept at a given timestep.
        
        Args:
            prompt: Generation prompt (should contain the concept)
            scores_json: Path to JSON file with pre-computed scores
            concept_name: Name of the concept (e.g., 'Dogs', 'Cats')
            top_k: Number of top features to show
            timestep_idx: Timestep index to visualize
            num_inference_steps: Total inference steps
            seed: Random seed
            save_path: Path to save the visualization
            save_values_image: Whether to save the values-annotated image
        """
        # Load scores
        print(f"Loading scores from {scores_json}")
        with open(scores_json, 'r') as f:
            scores_data = json.load(f)
        
        # Find concept scores
        if concept_name not in scores_data.get('scores', {}):
            concept_name_normalized = concept_name.replace('_', ' ')
            if concept_name_normalized not in scores_data.get('scores', {}):
                raise ValueError(f"Concept '{concept_name}' not found in scores JSON")
            concept_name = concept_name_normalized
        
        concept_scores = scores_data['scores'][concept_name]
        
        # Handle 2D scores (timestep x latent)
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            # Average across timesteps
            concept_scores = np.mean(concept_scores, axis=0)
        
        # Get top-k feature indices
        top_k = 1
        top_feature_indices = np.argsort(concept_scores)[::-1][:top_k]
        top_scores = [concept_scores[i] for i in top_feature_indices]
        
        print(f"Top {top_k} features for '{concept_name}':")
        for idx, score in zip(top_feature_indices, top_scores):
            print(f"  Feature {idx}: {score:.4f}")
        
        # Generate image with activations
        final_image, intermediate_images, sae_activations = self.generate_with_activations(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        # Get image at specified timestep
        img = intermediate_images[timestep_idx]
        
        # Handle different image types and convert to PIL Image
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy
            img = img.cpu().numpy()
            
            # Remove batch dimension if present
            while img.ndim == 4 and img.shape[0] == 1:
                img = img[0]
            
            # If channels first, transpose to channels last
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                img = np.transpose(img, (1, 2, 0))
            
            # Ensure it's in [0, 255] uint8 format
            if img.dtype in (np.float32, np.float64):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            img = Image.fromarray(img)
        
        elif isinstance(img, np.ndarray):
            # Remove batch dimension if present
            while img.ndim == 4 and img.shape[0] == 1:
                img = img[0]

            # Ensure it's in [0, 255] uint8 format
            if img.dtype in (np.float32, np.float64):
                # Assuming values are in [0, 1] range
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Convert to PIL Image
            img = Image.fromarray(img)

        # Final check: img should now be a PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image after conversion, got {type(img)}")
        
        # Create figure with grid
        n_cols = 5
        n_rows = (top_k + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows + 2))
        gs = GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.3, wspace=0.2)
        
        # Show original image in top row
        ax_orig = fig.add_subplot(gs[0, :])
        ax_orig.imshow(img)
        ax_orig.axis('off')
        ax_orig.set_title(
            f'Generated Image at Timestep {timestep_idx}\nPrompt: "{prompt}"',
            fontsize=14,
            fontweight='bold',
            pad=10
        )
        
        # Show each feature
        for idx, (feat_idx, score) in enumerate(zip(top_feature_indices, top_scores)):
            row = (idx // n_cols) + 1
            col = idx % n_cols
            
            # Create heatmap
            heatmap = self.create_activation_heatmap(
                sae_activations,
                feat_idx,
                timestep_idx,
            )
            
            # Create overlay
            overlay = self.overlay_heatmap(img, heatmap, alpha=0.6)
            
            # Plot
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(overlay)
            ax.axis('off')
            ax.set_title(
                f'Feature {feat_idx}\nScore: {score:.4f}',
                fontsize=10,
                fontweight='bold'
            )
            
            # Add activation info
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
            
            # Save individual numbered activation image if save_path provided
            if save_path:
                value_img = self.create_single_feature_value_image(
                    img,
                    sae_activations,
                    feat_idx,
                    timestep_idx,
                    class_latents_dict=self.class_latents_dict,  # Pass class latents
                    class_to_check=concept_name,  # The class being analyzed
                    percentile=self.class_params[concept_name]["percentile"] if self.class_params else None,
                )
                value_path = save_path.replace('.png', f'_feat{feat_idx}_t{timestep_idx}_values.png')
                value_img.save(value_path)
                print(f"Saved numbered activation image for feature {feat_idx} to {value_path}")
        
        # Add overall title
        fig.suptitle(
            f'Top {top_k} Features for Unlearning: "{concept_name}"',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            
            # Save values-annotated image
            if save_values_image:
                values_img = self.create_top5_activation_image(
                    img,
                    sae_activations,
                    timestep_idx,
                    top_feature_indices.tolist(),
                    class_latents_dict=self.class_latents_dict,  # Add this
                    class_to_check=concept_name,  # Add this
                    percentile=self.class_params[concept_name]["percentile"] if self.class_params else None,  # Add this
                )
                values_path = save_path.replace('.png', '_values.png')
                values_img.save(values_path)
                print(f"Saved values-annotated image to {values_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_top_features_grid(
        self,
        prompt: str,
        scores_json: str,
        concept_name: str,
        top_k: int = 5,
        timesteps_to_show: List[int] = [47, 30, 10, 1],
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
        save_individual_images: bool = True,
        save_values_images: bool = True,
    ):
        """
        Visualize top-k features across multiple timesteps in a grid layout (Figure 6 style).
        Now also saves images with activation values for each timestep.
        
        Args:
            prompt: Generation prompt (should contain the concept)
            scores_json: Path to JSON file with pre-computed scores
            concept_name: Name of the concept (e.g., 'Dogs', 'Cats')
            top_k: Number of top features to show (default: 5)
            timesteps_to_show: List of timestep indices to display (default: [47, 30, 10, 1])
            num_inference_steps: Total inference steps
            seed: Random seed
            save_path: Path to save the visualization
            save_individual_images: Whether to save individual grid cells as separate images
            save_values_images: Whether to save value-annotated images for each timestep
        """
        # Load scores
        print(f"Loading scores from {scores_json}")
        with open(scores_json, 'r') as f:
            scores_data = json.load(f)
        
        # Find concept scores
        if concept_name not in scores_data.get('scores', {}):
            concept_name_normalized = concept_name.replace('_', ' ')
            if concept_name_normalized not in scores_data.get('scores', {}):
                raise ValueError(f"Concept '{concept_name}' not found in scores JSON")
            concept_name = concept_name_normalized
        
        concept_scores = scores_data['scores'][concept_name]
        
        # Handle 2D scores (timestep x latent)
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            # Average across timesteps
            concept_scores = np.mean(concept_scores, axis=0)
        
        # Get top-k feature indices
        top_feature_indices = np.argsort(concept_scores)[::-1][:top_k]
        top_scores = [concept_scores[i] for i in top_feature_indices]
        
        print(f"Top {top_k} features for '{concept_name}':")
        for idx, score in zip(top_feature_indices, top_scores):
            print(f"  Feature {idx}: {score:.4f}")
        
        # Generate image with activations
        final_image, intermediate_images, sae_activations = self.generate_with_activations(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        
        # Create main grid figure
        n_timesteps = len(timesteps_to_show)
        fig = plt.figure(figsize=(5 * n_timesteps, 6 * top_k))
        gs = GridSpec(top_k + 1, n_timesteps, figure=fig, hspace=0.25, wspace=0.15)
        
        # Row 0: Show intermediate images at each timestep
        for col_idx, t in enumerate(timesteps_to_show):
            ax = fig.add_subplot(gs[0, col_idx])
            img = intermediate_images[t]
            
            # Handle different image types and convert to PIL Image
            if isinstance(img, Image.Image):
                # Already a PIL Image, just use it
                pass
            elif isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
                # Remove batch dimension if present
                while img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                # If channels first, transpose to channels last
                if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                    img = np.transpose(img, (1, 2, 0))
                # Convert to uint8
                if img.dtype in (np.float32, np.float64):
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                img = Image.fromarray(img)
            elif isinstance(img, np.ndarray):
                # Remove batch dimension if present
                while img.ndim == 4 and img.shape[0] == 1:
                    img = img[0]
                # Convert to uint8
                if img.dtype in (np.float32, np.float64):
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                img = Image.fromarray(img)
            else:
                # Fallback: try to handle as numpy array
                if hasattr(img, 'shape'):
                    img = np.array(img)
                    while img.ndim == 4 and img.shape[0] == 1:
                        img = img[0]
                    if img.dtype in (np.float32, np.float64):
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    img = Image.fromarray(img)
            
            # Verify img is properly formatted before imshow
            if isinstance(img, Image.Image):
                # PIL Image is fine for imshow
                pass
            elif isinstance(img, np.ndarray):
                # Ensure no batch dimension
                if img.ndim == 4:
                    raise ValueError(f"Image still has batch dimension: {img.shape}. Expected 3D array.")
            
            if isinstance(img, list):
                img = img[0]
            
            # Remove batch dimension if present
            if hasattr(img, 'squeeze'):
                while len(img.shape) == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
            elif hasattr(img, 'shape') and len(img.shape) == 4:
                img = img[0]
                img = img.squeeze(0)  # Remove the first dimension
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'Timestep {t}', fontsize=14, fontweight='bold')
            
            # Save values-annotated image for this timestep
            if save_values_images and save_path:
                values_img = self.create_top5_activation_image(
                    img,
                    sae_activations,
                    t,
                    top_feature_indices.tolist(),
                    class_latents_dict=self.class_latents_dict,  # Add this
                    class_to_check=concept_name,  # Add this
                    percentile=self.class_params[concept_name]["percentile"] if self.class_params else None,  # Add this
                )
                values_path = save_path.replace('.png', f'_timestep{t}_values.png')
                values_img.save(values_path)
                print(f"Saved values image for timestep {t} to {values_path}")
        
        # Remaining rows: Show feature heatmaps
        for feat_row, (feat_idx, score) in enumerate(zip(top_feature_indices, top_scores)):
            for col_idx, t in enumerate(timesteps_to_show):
                ax = fig.add_subplot(gs[feat_row + 1, col_idx])
                
                # Get image and convert to PIL Image
                img = intermediate_images[t]
                if isinstance(img, Image.Image):
                    # Already a PIL Image, just use it
                    pass
                elif isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                    # Remove batch dimension if present
                    while img.ndim == 4 and img.shape[0] == 1:
                        img = img[0]
                    # If channels first, transpose to channels last
                    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
                        img = np.transpose(img, (1, 2, 0))
                    # Convert to uint8
                    if img.dtype in (np.float32, np.float64):
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                elif isinstance(img, np.ndarray):
                    # Remove batch dimension if present
                    while img.ndim == 4 and img.shape[0] == 1:
                        img = img[0]
                    # Convert to uint8
                    if img.dtype in (np.float32, np.float64):
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                else:
                    # Fallback: try to handle as numpy array
                    if hasattr(img, 'shape'):
                        img = np.array(img)
                        while img.ndim == 4 and img.shape[0] == 1:
                            img = img[0]
                        if img.dtype in (np.float32, np.float64):
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        img = Image.fromarray(img)
                
                # Create heatmap
                heatmap = self.create_activation_heatmap(
                    sae_activations,
                    feat_idx,
                    t,
                )
                
                # Create overlay
                overlay = self.overlay_heatmap(img, heatmap, alpha=0.6)
                
                # Plot
                ax.imshow(overlay)
                ax.axis('off')
                
                # Add title only in first column
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
                
                # Add activation stats
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
                
                # Save individual heatmap overlay image if requested
                if save_individual_images and save_path:
                    individual_path = save_path.replace(
                        '.png',
                        f'_feat{feat_idx}_t{t}.png'
                    )
                    overlay_pil = Image.fromarray(overlay)
                    overlay_pil.save(individual_path)
                    
                    # Also save numbered activation values image
                    value_img = self.create_single_feature_value_image(
                        img,
                        sae_activations,
                        feat_idx,
                        t,
                        class_latents_dict=self.class_latents_dict,  # Pass class latents
                        class_to_check=concept_name,  # The class being analyzed
                        percentile=self.class_params[concept_name]["percentile"] if self.class_params else None,
                    )
                    value_path = save_path.replace(
                        '.png',
                        f'_feat{feat_idx}_t{t}_values.png'
                    )
                    value_img.save(value_path)
        
        # Add overall title
        fig.suptitle(
            f'Top {top_k} Features for Concept: "{concept_name}"\nPrompt: "{prompt}"',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved grid visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main(
    sae_path: str,
    pipe_path: str,
    hookpoint: str = "unet.up_blocks.1.attentions.1",
    prompt: str = "a photo of a dog",
    mode: str = "feature_timeline",  # or "top_features" or "top_features_grid"
    feature_idx: Optional[int] = None,
    scores_json: Optional[str] = None,
    concept_name: Optional[str] = None,
    top_k: int = 10,
    timestep: int = 25,
    timesteps_to_show: str = "0,10,20,30,40,49",
    num_inference_steps: int = 50,
    seed: Optional[int] = 188,
    output_dir: str = "visualizations",
    device: str = "cuda",
    save_individual_images: bool = True,
    save_values_images: bool = True,
    class_latents_path: Optional[str] = None,  # NEW
    class_params_path: Optional[str] = None,   # NEW
):
    """
    Main function to run SAE feature visualizations.
    
    Args:
        sae_path: Path to SAE checkpoint directory
        pipe_path: Path to Stable Diffusion model
        hookpoint: Layer to visualize (e.g., 'unet.up_blocks.1.attentions.1')
        prompt: Text prompt for generation
        mode: Visualization mode ('feature_timeline', 'top_features', or 'top_features_grid')
        feature_idx: Feature index for timeline mode
        scores_json: Path to scores JSON for top_features mode
        concept_name: Concept name for top_features mode
        top_k: Number of top features to show
        timestep: Timestep index for top_features mode
        timesteps_to_show: Comma-separated timesteps for timeline/grid mode
        num_inference_steps: Number of denoising steps
        seed: Random seed
        output_dir: Output directory for visualizations
        device: Device to run on
        save_individual_images: Whether to save individual grid cells as separate images
        save_values_images: Whether to save value-annotated images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = SAEFeatureVisualizer(
        sae_path=sae_path,
        pipe_path=pipe_path,
        hookpoint=hookpoint,
        device=device,
        class_latents_path=class_latents_path,  # NEW
        class_params_path=class_params_path,    # NEW
    )
    
    if mode == "feature_timeline":
        # Visualize single feature across timesteps (Figure 5)
        if feature_idx is None:
            raise ValueError("Must specify feature_idx for feature_timeline mode")
        
        # Handle both string and tuple/list inputs
        if isinstance(timesteps_to_show, str):
            timesteps = [int(t.strip()) for t in timesteps_to_show.split(',')]
        elif isinstance(timesteps_to_show, (list, tuple)):
            timesteps = [int(t) for t in timesteps_to_show]
        else:
            timesteps = [timesteps_to_show]
        
        save_path = os.path.join(
            output_dir,
            f"feature_{feature_idx}_timeline.png"
        )
        
        visualizer.visualize_feature_across_timesteps(
            prompt=prompt,
            feature_idx=feature_idx,
            timesteps_to_show=timesteps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path,
        )
    
    elif mode == "top_features":
        # Visualize top-k features for a concept (Figure 6)
        if scores_json is None or concept_name is None:
            raise ValueError("Must specify scores_json and concept_name for top_features mode")
        
        save_path = os.path.join(
            output_dir,
            f"top_{top_k}_features_{concept_name}.pdf"
        )
        
        visualizer.visualize_top_features_for_concept(
            prompt=prompt,
            scores_json=scores_json,
            concept_name=concept_name,
            top_k=top_k,
            timestep_idx=timestep,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path,
            save_values_image=save_values_images,
        )

    elif mode == "top_features_grid":
        # Visualize top-k features across timesteps in grid (Figure 6)
        if scores_json is None or concept_name is None:
            raise ValueError("Must specify scores_json and concept_name for top_features_grid mode")
        
        # Handle both string and tuple/list inputs
        if isinstance(timesteps_to_show, str):
            timesteps = [int(t.strip()) for t in timesteps_to_show.split(',')]
        elif isinstance(timesteps_to_show, (list, tuple)):
            timesteps = [int(t) for t in timesteps_to_show]
        else:
            timesteps = [timesteps_to_show]
        
        save_path = os.path.join(
            output_dir,
            f"top_{top_k}_features_grid_{concept_name}.pdf"
        )
        
        visualizer.visualize_top_features_grid(
            prompt=prompt,
            scores_json=scores_json,
            concept_name=concept_name,
            top_k=top_k,
            timesteps_to_show=timesteps,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path,
            save_individual_images=save_individual_images,
            save_values_images=save_values_images,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    fire.Fire(main)