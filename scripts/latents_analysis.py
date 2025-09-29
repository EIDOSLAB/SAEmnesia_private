#!/usr/bin/env python
"""
Plot latent activations for a specific object using SAE model.

This script loads a single shard of data for a specified object, computes
latent activations using the SAE model, and creates bar plots showing
the activation values with a horizontal line at zero.
"""

import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datasets import Dataset as HFDataset

# Add parent directory to path for SAE imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from SAE.sae import Sae
except ImportError:
    print("Error: Could not import SAE module. Make sure the SAE directory is in the parent directory.")
    sys.exit(1)


def load_single_shard_for_object(data_path, hookpoint, object_name, shard_idx=0, dtype=torch.float32):
    """
    Load a single shard of data for a specific object.
    
    Args:
        data_path: Base path to the data directory
        hookpoint: Name of the hookpoint (e.g., 'unet.up_blocks.1.attentions.1')
        object_name: Name of the object to load (e.g., 'Architectures')
        shard_idx: Index of the shard to load (default: 0 for first shard)
        dtype: PyTorch data type for the tensors
    
    Returns:
        HFDataset containing the loaded data
    """
    base_path = Path(data_path)
    object_dir = base_path / hookpoint / object_name
    
    if not object_dir.exists():
        raise ValueError(f"Object directory does not exist: {object_dir}")
    
    print(f"Loading data from: {object_dir}")
    
    # Load the full dataset first
    dataset = HFDataset.load_from_disk(str(object_dir), keep_in_memory=False)
    
    print(f"Full dataset size: {len(dataset)}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # Get the number of shards by looking at arrow files
    arrow_files = list(object_dir.glob("data-*.arrow"))
    num_shards = len(arrow_files)
    print(f"Found {num_shards} shards")
    
    if shard_idx >= num_shards:
        raise ValueError(f"Shard index {shard_idx} is out of range. Available shards: 0-{num_shards-1}")
    
    # Calculate shard boundaries
    shard_size = len(dataset) // num_shards
    start_idx = shard_idx * shard_size
    
    if shard_idx == num_shards - 1:
        # Last shard gets any remaining samples
        end_idx = len(dataset)
    else:
        end_idx = start_idx + shard_size
    
    print(f"Loading shard {shard_idx}: samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
    
    # Select the shard
    shard_dataset = dataset.select(range(start_idx, end_idx))
    
    # Set format for torch tensors
    shard_dataset.set_format(
        type="torch",
        columns=["activations", "timestep"] + (["object_label"] if "object_label" in shard_dataset.column_names else []),
        dtype=dtype,
    )
    
    return shard_dataset


def get_latent_activations(sae_model, activations):
    """
    Compute latent activations using the SAE model.
    
    Args:
        sae_model: Loaded SAE model
        activations: Input activations tensor
    
    Returns:
        Latent activations tensor
    """
    sae_model.eval()
    
    with torch.no_grad():
        print(f"    Input activations shape: {activations.shape}")
        print(f"    Input activations stats - Max: {activations.max().item():.4f}, Min: {activations.min().item():.4f}")
        
        # Handle different tensor shapes
        if len(activations.shape) == 3:
            # If 3D [batch, seq, features], reshape to 2D
            original_shape = activations.shape
            activations = activations.reshape(-1, activations.shape[-1])
            print(f"    Reshaped from 3D to 2D: {activations.shape}")
            
            # Get pre-activations (latent activations)
            latent_acts = sae_model.pre_acts(activations)
            print(f"    Pre-acts shape after model: {latent_acts.shape}")
            
            # Reshape back if needed
            latent_acts = latent_acts.reshape(original_shape[0], original_shape[1], -1)
            # Take mean over sequence dimension
            latent_acts = latent_acts.mean(dim=1)
            print(f"    After mean over sequence: {latent_acts.shape}")
            
        elif len(activations.shape) == 2:
            # If 2D [batch, features], directly compute
            print(f"    Computing pre_acts for 2D tensor: {activations.shape}")
            latent_acts = sae_model.pre_acts(activations)
            print(f"    Pre-acts shape: {latent_acts.shape}")
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
        
        print(f"    Final latent_acts shape: {latent_acts.shape}")
        print(f"    Final latent_acts stats - Max: {latent_acts.max().item():.4f}, Min: {latent_acts.min().item():.4f}")
        
        # Check if we're getting the right activations by looking at the distribution
        print(f"    Number of positive latents: {(latent_acts > 0).sum().item()}")
        print(f"    Number of latents > 0.1: {(latent_acts > 0.1).sum().item()}")
        print(f"    Number of latents > 0.5: {(latent_acts > 0.5).sum().item()}")
    
    return latent_acts


def plot_latent_activations(latent_activations, sample_idx, object_name, save_dir=None):
    """
    Create a bar plot of latent activations with a horizontal line at zero.
    
    Args:
        latent_activations: Tensor of latent activations for one sample
        sample_idx: Index of the sample being plotted
        object_name: Name of the object for the title
        save_dir: Optional directory to save the plot
    """
    # Convert to numpy for plotting
    acts = latent_activations.cpu().numpy()
    
    print(f"    Plotting - acts shape: {acts.shape}")
    print(f"    Plotting - acts stats: Max={np.max(acts):.4f}, Min={np.min(acts):.4f}, Mean={np.mean(acts):.4f}")
    print(f"    Plotting - Non-zero activations: {np.count_nonzero(acts)}")
    
    # Find the actual max and its index for verification
    max_idx = np.argmax(acts)
    max_val = acts[max_idx]
    print(f"    Max activation at index {max_idx}: {max_val:.4f}")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create bar plot with wider bars
    latent_indices = np.arange(len(acts))
    colors = ['red' if x < 0 else 'blue' for x in acts]
    bars = ax.bar(latent_indices, acts, color=colors, alpha=0.7, width=0.9)
    
    # Highlight the maximum activation bar in green for easy identification
    max_color = 'green'
    bars[max_idx].set_color(max_color)
    bars[max_idx].set_alpha(0.9)
    bars[max_idx].set_linewidth(2)  # Make it thicker
    bars[max_idx].set_edgecolor('darkgreen')  # Add border
    
    # Also highlight top 3 activations for better visibility
    top_3_indices = np.argsort(acts)[-3:][::-1]
    for i, idx in enumerate(top_3_indices):
        if i == 0:  # Already colored green above
            continue
        elif i == 1:  # Second highest in orange
            bars[idx].set_color('orange')
            bars[idx].set_alpha(0.9)
            bars[idx].set_linewidth(1)
        elif i == 2:  # Third highest in yellow
            bars[idx].set_color('gold')
            bars[idx].set_alpha(0.9)
            bars[idx].set_linewidth(1)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Latent Index')
    ax.set_ylabel('Activation Value')
    ax.set_title(f'{object_name} - Sample {sample_idx} - Latent Activations')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    max_act = np.max(acts)
    min_act = np.min(acts)
    mean_act = np.mean(acts)
    num_positive = np.sum(acts > 0)
    num_negative = np.sum(acts < 0)
    
    # Find top 3 activating latents
    top_3_indices = np.argsort(acts)[-3:][::-1]  # Get indices of top 3, in descending order
    top_3_values = acts[top_3_indices]
    
    top_3_text = "Top 3 Latents:\n" + "\n".join([f"#{idx}: {val:.3f}" for idx, val in zip(top_3_indices, top_3_values)])
    
    # Verification text to check if max matches
    verify_text = f"\nColors:\nGreen = Max (#{max_idx})\nOrange = 2nd\nYellow = 3rd"
    
    stats_text = f'Max: {max_act:.3f}\nMin: {min_act:.3f}\nMean: {mean_act:.3f}\nPos: {num_positive}\nNeg: {num_negative}\n\n{top_3_text}{verify_text}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        save_path = Path(save_dir) / f'{object_name}_sample_{sample_idx}_latents.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    
def apply_multipliers_to_activations(latent_acts, object_name, multipliers_dict):
    """
    Apply multipliers to latent activations based on percentile thresholding.
    
    Args:
        latent_acts: Tensor of latent activations
        object_name: Name of the object to get multiplier for
        multipliers_dict: Dictionary containing class parameters
    
    Returns:
        Modified latent activations tensor
    """
    if multipliers_dict is None or object_name not in multipliers_dict:
        print(f"    No multipliers found for object: {object_name}")
        return latent_acts
    
    # Get parameters for this object from the .pth file structure
    params = multipliers_dict[object_name]
    percentile = params["percentile"]  # Directly from the saved structure
    multiplier = params["multiplier"]  # Directly from the saved structure
    
    print(f"    Applying multiplier {multiplier} at percentile {percentile} for {object_name}")
    
    # Convert to numpy for percentile calculation
    acts_np = latent_acts.cpu().numpy()
    
    # Calculate the threshold value at the specified percentile
    threshold = np.percentile(acts_np, percentile)
    print(f"    Threshold at {percentile}th percentile: {threshold:.4f}")
    
    # Create a mask for values above the threshold
    mask = acts_np > threshold
    num_affected = np.sum(mask)
    print(f"    Number of latents above threshold: {num_affected}")
    
    # Apply multiplier to values above threshold
    modified_acts = acts_np.copy()
    modified_acts[mask] = modified_acts[mask] * multiplier
    
    # Convert back to tensor
    return torch.from_numpy(modified_acts).to(latent_acts.device)


def plot_comparison_latent_activations(original_acts, modified_acts, sample_idx, object_name, multiplier, save_dir=None):
    """
    Create a comparison plot showing original vs modified latent activations.
    
    Args:
        original_acts: Original latent activations tensor
        modified_acts: Modified latent activations tensor  
        sample_idx: Index of the sample being plotted
        object_name: Name of the object for the title
        multiplier: Multiplier value used
        save_dir: Optional directory to save the plot
    """
    # Convert to numpy for plotting
    orig_acts = original_acts.cpu().numpy()
    mod_acts = modified_acts.cpu().numpy()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Find important latents from BOTH original and modified plots
    top_3_indices_orig = np.argsort(orig_acts)[-3:][::-1]
    top_3_indices_mod = np.argsort(mod_acts)[-3:][::-1]
    
    # Combine all important indices (top 3 from both plots)
    important_indices = set(top_3_indices_orig) | set(top_3_indices_mod)
    important_indices = sorted(list(important_indices), key=lambda x: max(orig_acts[x], mod_acts[x]), reverse=True)
    
    print(f"    Important latents (top 3 from either plot): {important_indices}")
    
    # Plot original activations WITH ENHANCED STYLING
    latent_indices = np.arange(len(orig_acts))
    colors_orig = ['red' if x < 0 else 'blue' for x in orig_acts]
    bars_orig = ax1.bar(latent_indices, orig_acts, color=colors_orig, alpha=0.7, width=0.9)
    
    # Find max for original activations
    max_idx_orig = np.argmax(orig_acts)
    
    # Highlight important latents in original plot
    color_map = ['green', 'orange', 'gold', 'cyan', 'magenta', 'lime']  # Extended color palette
    for i, idx in enumerate(important_indices):
        if i < len(color_map):
            color = color_map[i]
            bars_orig[idx].set_color(color)
            bars_orig[idx].set_alpha(0.9)
            if idx == max_idx_orig:
                bars_orig[idx].set_linewidth(2)
                bars_orig[idx].set_edgecolor('darkgreen')
            else:
                bars_orig[idx].set_linewidth(1)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title(f'{object_name} - Sample {sample_idx} - Original Latent Activations')
    ax1.set_ylabel('Original Activation Value')
    ax1.grid(True, alpha=0.3)
    
    # Plot modified activations WITH ENHANCED STYLING
    colors_mod = ['red' if x < 0 else 'blue' for x in mod_acts]
    bars_mod = ax2.bar(latent_indices, mod_acts, color=colors_mod, alpha=0.7, width=0.9)
    
    # Find max for modified activations
    max_idx_mod = np.argmax(mod_acts)
    
    # Highlight important latents in modified plot (same colors as original for consistency)
    for i, idx in enumerate(important_indices):
        if i < len(color_map):
            color = color_map[i]
            bars_mod[idx].set_color(color)
            bars_mod[idx].set_alpha(0.9)
            if idx == max_idx_mod:
                bars_mod[idx].set_linewidth(2)
                bars_mod[idx].set_edgecolor('darkgreen')
            else:
                bars_mod[idx].set_linewidth(1)
    
    # Additionally highlight bars that were significantly changed by multiplier in purple
    # But only if they're not already highlighted as important latents
    changed_mask = np.abs(orig_acts - mod_acts) > 1e-6
    for i, changed in enumerate(changed_mask):
        if changed and i not in important_indices:
            bars_mod[i].set_color('purple')
            bars_mod[i].set_alpha(0.8)
            bars_mod[i].set_linewidth(1)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title(f'{object_name} - Sample {sample_idx} - Modified Latent Activations (Multiplier: {multiplier})')
    ax2.set_xlabel('Latent Index')
    ax2.set_ylabel('Modified Activation Value')
    ax2.grid(True, alpha=0.3)
    
    # Enhanced statistics for both plots
    max_act_orig = np.max(orig_acts)
    min_act_orig = np.min(orig_acts)
    mean_act_orig = np.mean(orig_acts)
    num_positive_orig = np.sum(orig_acts > 0)
    num_negative_orig = np.sum(orig_acts < 0)
    
    max_act_mod = np.max(mod_acts)
    min_act_mod = np.min(mod_acts)
    mean_act_mod = np.mean(mod_acts)
    num_positive_mod = np.sum(mod_acts > 0)
    num_negative_mod = np.sum(mod_acts < 0)
    changed_count = np.sum(changed_mask)
    
    # Create legend for important latents (top from either plot)
    important_latents_text = "Important Latents:\n"
    for i, idx in enumerate(important_indices[:6]):  # Show up to 6
        orig_val = orig_acts[idx]
        mod_val = mod_acts[idx]
        color_name = ['Green', 'Orange', 'Gold', 'Cyan', 'Magenta', 'Lime'][i]
        important_latents_text += f"{color_name} #{idx}: {orig_val:.3f}→{mod_val:.3f}\n"
    
    # Original plot stats
    stats_text_orig = f'Max: {max_act_orig:.3f}\nMin: {min_act_orig:.3f}\nMean: {mean_act_orig:.3f}\nPos: {num_positive_orig}\nNeg: {num_negative_orig}\n\n{important_latents_text}'
    ax1.text(0.02, 0.98, stats_text_orig, transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Modified plot stats  
    stats_text_mod = f'Max: {max_act_mod:.3f}\nMin: {min_act_mod:.3f}\nMean: {mean_act_mod:.3f}\nPos: {num_positive_mod}\nNeg: {num_negative_mod}\nChanged: {changed_count}\n\nPurple = Other changed latents'
    ax2.text(0.02, 0.98, stats_text_mod, transform=ax2.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        save_path = Path(save_dir) / f'{object_name}_sample_{sample_idx}_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_average_latent_activations(sae_model, dataset, device, max_samples=None):
    """
    Compute average latent activations across all samples in the dataset.
    
    Args:
        sae_model: Loaded SAE model
        dataset: Dataset containing activation samples
        device: Device to use for computation
        max_samples: Maximum number of samples to process (None for all)
    
    Returns:
        Average latent activations tensor
    """
    sae_model.eval()
    
    all_latent_acts = []
    total_samples = len(dataset)
    
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    print(f"Computing average latent activations across {total_samples} samples...")
    
    with torch.no_grad():
        for i in range(total_samples):
            if i % 100 == 0:
                print(f"  Processing sample {i+1}/{total_samples}")
            
            try:
                sample = dataset[i]
                activations = sample['activations'].unsqueeze(0).to(device)
                
                # Get latent activations
                latent_acts = get_latent_activations(sae_model, activations)
                latent_acts = latent_acts.squeeze(0)  # Remove batch dimension
                
                all_latent_acts.append(latent_acts.cpu())
                
            except Exception as e:
                print(f"  Error processing sample {i}: {e}")
                continue
    
    if not all_latent_acts:
        raise ValueError("No valid samples were processed")
    
    # Stack all activations and compute mean
    all_latent_acts = torch.stack(all_latent_acts)
    average_latent_acts = all_latent_acts.mean(dim=0)
    
    print(f"Computed average across {len(all_latent_acts)} valid samples")
    
    return average_latent_acts


def plot_average_latent_activations(average_latent_acts, object_name, shard_idx, total_samples, save_dir=None):
    """
    Create a bar plot of average latent activations across all samples.
    
    Args:
        average_latent_acts: Tensor of average latent activations
        object_name: Name of the object for the title
        shard_idx: Index of the shard
        total_samples: Total number of samples used for averaging
        save_dir: Optional directory to save the plot
    """
    # Convert to numpy for plotting
    acts = average_latent_acts.cpu().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create bar plot with wider bars
    latent_indices = np.arange(len(acts))
    colors = ['red' if x < 0 else 'blue' for x in acts]
    bars = ax.bar(latent_indices, acts, color=colors, alpha=0.7, width=0.9)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Customize the plot
    ax.set_xlabel('Latent Index')
    ax.set_ylabel('Average Activation Value')
    ax.set_title(f'{object_name} - Shard {shard_idx} - Average Latent Activations ({total_samples} samples)')
    ax.grid(True, alpha=0.3)
    
    # Add some statistics to the plot
    max_act = np.max(acts)
    min_act = np.min(acts)
    mean_act = np.mean(acts)
    num_positive = np.sum(acts > 0)
    num_negative = np.sum(acts < 0)
    std_act = np.std(acts)
    
    # Find top 3 activating latents
    top_3_indices = np.argsort(acts)[-3:][::-1]  # Get indices of top 3, in descending order
    top_3_values = acts[top_3_indices]
    
    top_3_text = "Top 3 Latents:\n" + "\n".join([f"#{idx}: {val:.4f}" for idx, val in zip(top_3_indices, top_3_values)])
    
    stats_text = f'Max: {max_act:.4f}\nMin: {min_act:.4f}\nMean: {mean_act:.4f}\nStd: {std_act:.4f}\nPos: {num_positive}\nNeg: {num_negative}\n\n{top_3_text}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        save_path = Path(save_dir) / f'{object_name}_shard_{shard_idx}_average_latents.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved average plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function to load data, compute latent activations, and create plots.
    """
    parser = argparse.ArgumentParser(description="Plot latent activations for a specific object using SAE model.")
    
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the SAE model directory"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to the base data directory"
    )
    parser.add_argument(
        "--object_name", 
        type=str, 
        required=True,
        help="Name of the object to analyze (e.g., 'Architectures')"
    )
    parser.add_argument(
        "--hookpoint", 
        type=str,
        default="unet.up_blocks.1.attentions.1",
        help="Hookpoint name (default: unet.up_blocks.1.attentions.1)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="Number of individual samples to plot (default: 5)"
    )
    parser.add_argument(
        "--shard_idx", 
        type=int, 
        default=0,
        help="Index of the shard to load (default: 0)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--save_dir", 
        type=str,
        help="Directory to save plots (if not specified, plots will be displayed)"
    )
    parser.add_argument(
        "--use_float16", 
        action="store_true",
        help="Use float16 precision"
    )
    parser.add_argument(
        "--max_average_samples", 
        type=int,
        help="Maximum number of samples to use for average computation (default: all samples in shard)"
    )
    parser.add_argument(
        "--multipliers_path", 
        type=str,
        help="Path to .pth file containing class multipliers (optional)"
    )

    
    args = parser.parse_args()
    
    # Set up device and dtype
    device = torch.device(args.device)
    dtype = torch.float16 if args.use_float16 else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Create save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Load SAE model
        print(f"Loading SAE model from: {args.model_path}")
        sae_model = Sae.load_from_disk(args.model_path, device=device)
        sae_model = sae_model.to(dtype=dtype)
        print(f"Model loaded successfully. Number of latents: {sae_model.num_latents}")
        
        # Load multipliers if provided
        multipliers_dict = None
        if args.multipliers_path:
            print(f"Loading multipliers from: {args.multipliers_path}")
            multipliers_dict = torch.load(args.multipliers_path)
            print(f"Loaded multipliers for {len(multipliers_dict)} classes")
            if args.object_name in multipliers_dict:
                params = multipliers_dict[args.object_name]
                print(f"Found parameters for {args.object_name}: percentile={params['percentile']}, multiplier={params['multiplier']}")
            else:
                print(f"WARNING: No parameters found for {args.object_name}")
        
        # Load data for the specified object
        print(f"Loading data for object: {args.object_name}")
        dataset = load_single_shard_for_object(
            args.data_path, 
            args.hookpoint, 
            args.object_name, 
            args.shard_idx,
            dtype
        )
        
        # Check if we have enough samples
        available_samples = len(dataset)
        num_samples = min(args.num_samples, available_samples)
        
        if available_samples < args.num_samples:
            print(f"Warning: Only {available_samples} samples available, plotting {num_samples}")
        
        print(f"Processing {num_samples} individual samples...")
        
        # Process and plot each individual sample
        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}")
            
            # Get activations for this sample
            sample = dataset[i]
            activations = sample['activations'].unsqueeze(0).to(device)  # Add batch dimension
            
            print(f"  Sample {i} activations shape: {activations.shape}")
            
            # Compute original latent activations
            latent_acts = get_latent_activations(sae_model, activations)
            latent_acts = latent_acts.squeeze(0)  # Remove batch dimension
            
            print(f"  Latent activations shape: {latent_acts.shape}")
            print(f"  Latent stats - Max: {latent_acts.max().item():.3f}, Min: {latent_acts.min().item():.3f}, Mean: {latent_acts.mean().item():.3f}")
            
            # Create original plot
            plot_latent_activations(latent_acts, i, args.object_name, args.save_dir)
            
            # If multipliers are provided, also create comparison plots
            if multipliers_dict and args.object_name in multipliers_dict:
                print(f"  Creating comparison plot with multipliers...")
                
                # Apply multipliers
                modified_latent_acts = apply_multipliers_to_activations(
                    latent_acts, 
                    args.object_name, 
                    multipliers_dict
                )
                
                # Get multiplier value for the title
                multiplier = multipliers_dict[args.object_name]["multiplier"]
                
                # Create comparison plot
                plot_comparison_latent_activations(
                    latent_acts, 
                    modified_latent_acts, 
                    i, 
                    args.object_name, 
                    multiplier,
                    args.save_dir
                )
        
        # Compute and plot average latent activations across the whole shard
        print(f"\nComputing average latent activations for the whole shard...")
        average_latent_acts = compute_average_latent_activations(
            sae_model, 
            dataset, 
            device, 
            max_samples=args.max_average_samples
        )
        
        # Determine how many samples were actually used
        samples_used = len(dataset) if args.max_average_samples is None else min(args.max_average_samples, len(dataset))
        
        print(f"Average latent stats - Max: {average_latent_acts.max().item():.4f}, Min: {average_latent_acts.min().item():.4f}, Mean: {average_latent_acts.mean().item():.4f}")
        
        # Plot average activations
        plot_average_latent_activations(
            average_latent_acts, 
            args.object_name, 
            args.shard_idx, 
            samples_used,
            args.save_dir
        )
        
        print("All plots created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)