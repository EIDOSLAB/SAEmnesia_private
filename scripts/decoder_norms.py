#!/usr/bin/env python
"""
Extract decoder column norms for assigned concepts from a trained SAE.

This script:
1. Loads the SAE model
2. Loads the concept-to-latent assignments from score JSON files
3. Extracts the decoder column norms for each assigned latent
4. Computes cosine similarities between object decoder columns
5. Analyzes correlation between avg cosine similarity and multipliers
6. Saves the results to JSON files
7. Creates histograms and correlation plots
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine

# Add parent directory to path for imports (adjust as needed)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from SAE.sae import Sae
except ImportError:
    print("Warning: Could not import SAE from SAE.sae")
    print("Attempting alternative import paths...")
    try:
        from sae import Sae
    except ImportError:
        print("Please ensure the SAE module is in your Python path")
        sys.exit(1)


def normalize_concept_name(name):
    """Convert between underscore and space formats for concept names."""
    return name.replace('_', ' ')


def find_concept_in_scores(concept_name, scores):
    """Find concept in scores dict, trying both original and normalized names."""
    if concept_name in scores:
        return concept_name, scores[concept_name]
    
    normalized_name = normalize_concept_name(concept_name)
    if normalized_name in scores:
        return normalized_name, scores[normalized_name]
    
    underscore_name = concept_name.replace(' ', '_')
    if underscore_name in scores:
        return underscore_name, scores[underscore_name]
    
    return None, None


def get_best_latent_from_scores(concept_scores, num_latents, assigned_latents):
    """Get the best available latent for a concept based on scores."""
    # Handle both 2D (timestep x latent) and 1D (latent) score arrays
    if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
        avg_scores = np.mean(concept_scores, axis=0)
    else:
        avg_scores = concept_scores
    
    # Find the highest scoring latent that's not already assigned
    sorted_scores = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)
    
    for latent_idx, score in sorted_scores:
        if latent_idx < num_latents and latent_idx not in assigned_latents:
            return latent_idx, score
    
    return None, None


def load_concept_assignments(object_scores_path, style_scores_path, num_latents):
    """
    Load and compute concept-to-latent assignments from score files.
    
    Returns:
        object_to_latent: dict mapping object names to latent indices
        style_to_latent: dict mapping style names to latent indices
    """
    object_to_latent = {}
    style_to_latent = {}
    assigned_latents = set()
    
    # Load object scores
    object_scores = {}
    if object_scores_path and Path(object_scores_path).exists():
        with open(object_scores_path, 'r') as f:
            object_data = json.load(f)
            object_scores = object_data.get('scores', {})
        print(f"Loaded {len(object_scores)} object scores")
    
    # Load style scores
    style_scores = {}
    if style_scores_path and Path(style_scores_path).exists():
        with open(style_scores_path, 'r') as f:
            style_data = json.load(f)
            style_scores = style_data.get('scores', {})
        print(f"Loaded {len(style_scores)} style scores")
    
    # Collect all concepts with their best scores
    concept_priorities = []
    
    # Add objects (with priority boost)
    for concept_name, concept_scores in object_scores.items():
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            avg_scores = np.mean(concept_scores, axis=0)
        else:
            avg_scores = concept_scores
        best_score = max(avg_scores) if len(avg_scores) > 0 else 0
        concept_priorities.append((concept_name, best_score + 1.0, 'object', avg_scores))
    
    # Add styles
    for concept_name, concept_scores in style_scores.items():
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            avg_scores = np.mean(concept_scores, axis=0)
        else:
            avg_scores = concept_scores
        best_score = max(avg_scores) if len(avg_scores) > 0 else 0
        concept_priorities.append((concept_name, best_score, 'style', avg_scores))
    
    # Sort by priority score (highest first)
    concept_priorities.sort(key=lambda x: x[1], reverse=True)
    
    # Assign in priority order
    for concept_name, priority_score, concept_type, avg_scores in concept_priorities:
        sorted_scores = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)
        
        for latent_idx, score in sorted_scores:
            if latent_idx < num_latents and latent_idx not in assigned_latents:
                if concept_type == 'object':
                    object_to_latent[concept_name] = latent_idx
                else:
                    style_to_latent[concept_name] = latent_idx
                assigned_latents.add(latent_idx)
                break
    
    return object_to_latent, style_to_latent


def compute_cosine_similarity_matrix(decoder_weight, latent_indices):
    """
    Compute pairwise cosine similarities between decoder columns.
    
    Args:
        decoder_weight: The decoder weight matrix
        latent_indices: List of latent indices to compute similarities for
        
    Returns:
        similarity_matrix: NxN matrix of cosine similarities
    """
    n = len(latent_indices)
    similarity_matrix = np.zeros((n, n))
    
    # Get decoder columns for the specified latents
    if decoder_weight.shape[0] == len(latent_indices) or decoder_weight.shape[0] > max(latent_indices):
        # Decoder is [num_latents, d_in] - each row is a latent
        vectors = decoder_weight[latent_indices].cpu().numpy()
    else:
        # Decoder is [d_in, num_latents] - each column is a latent
        vectors = decoder_weight[:, latent_indices].T.cpu().numpy()
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_vectors = vectors / norms
    
    # Compute cosine similarities (dot product of normalized vectors)
    similarity_matrix = normalized_vectors @ normalized_vectors.T
    
    return similarity_matrix


def analyze_object_similarities(decoder_weight, object_to_latent, multipliers=None, output_dir=None):
    """
    Analyze cosine similarities between object decoder columns.
    
    Args:
        decoder_weight: The decoder weight matrix
        object_to_latent: Dict mapping object names to latent indices
        multipliers: Optional dict mapping object names to multiplier values
        output_dir: Optional directory to save plots
        
    Returns:
        dict with similarity analysis results
    """
    object_names = sorted(object_to_latent.keys())
    latent_indices = [object_to_latent[name] for name in object_names]
    
    print(f"\nComputing cosine similarities for {len(object_names)} objects...")
    
    # Compute similarity matrix
    sim_matrix = compute_cosine_similarity_matrix(decoder_weight, latent_indices)
    
    # Create object name to index mapping
    name_to_idx = {name: i for i, name in enumerate(object_names)}
    
    # Compute average similarities for each object (excluding self-similarity)
    avg_similarities = {}
    avg_abs_similarities = {}
    avg_positive_similarities = {}
    
    for i, obj_name in enumerate(object_names):
        # Get all similarities for this object (excluding diagonal)
        sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        
        # Standard average
        avg_similarities[obj_name] = float(np.mean(sims))
        
        # Average of absolute values
        avg_abs_similarities[obj_name] = float(np.mean(np.abs(sims)))
        
        # Average of positive values only (ReLU-like)
        positive_sims = sims[sims > 0]
        if len(positive_sims) > 0:
            avg_positive_similarities[obj_name] = float(np.mean(positive_sims))
        else:
            avg_positive_similarities[obj_name] = 0.0
    
    results = {
        "similarity_matrix": {
            "object_names": object_names,
            "matrix": sim_matrix.tolist()
        },
        "average_similarities": avg_similarities,
        "average_absolute_similarities": avg_abs_similarities,
        "average_positive_similarities": avg_positive_similarities,
        "statistics": {
            "avg_sim": {
                "mean": float(np.mean(list(avg_similarities.values()))),
                "std": float(np.std(list(avg_similarities.values()))),
                "min": float(np.min(list(avg_similarities.values()))),
                "max": float(np.max(list(avg_similarities.values())))
            },
            "avg_abs_sim": {
                "mean": float(np.mean(list(avg_abs_similarities.values()))),
                "std": float(np.std(list(avg_abs_similarities.values()))),
                "min": float(np.min(list(avg_abs_similarities.values()))),
                "max": float(np.max(list(avg_abs_similarities.values())))
            },
            "avg_pos_sim": {
                "mean": float(np.mean(list(avg_positive_similarities.values()))),
                "std": float(np.std(list(avg_positive_similarities.values()))),
                "min": float(np.min(list(avg_positive_similarities.values()))),
                "max": float(np.max(list(avg_positive_similarities.values())))
            }
        }
    }
    
    # If multipliers provided, compute correlations
    if multipliers is not None:
        print("\nComputing correlations with multipliers...")
        
        # Filter to objects that have multipliers
        common_objects = [obj for obj in object_names if obj in multipliers]
        
        if len(common_objects) > 1:
            mult_values = [multipliers[obj] for obj in common_objects]
            avg_sim_values = [avg_similarities[obj] for obj in common_objects]
            avg_abs_sim_values = [avg_abs_similarities[obj] for obj in common_objects]
            avg_pos_sim_values = [avg_positive_similarities[obj] for obj in common_objects]
            
            # Compute correlations
            corr_standard, p_standard = stats.pearsonr(avg_sim_values, mult_values)
            corr_abs, p_abs = stats.pearsonr(avg_abs_sim_values, mult_values)
            corr_pos, p_pos = stats.pearsonr(avg_pos_sim_values, mult_values)
            
            results["correlations"] = {
                "standard": {
                    "correlation": float(corr_standard),
                    "p_value": float(p_standard),
                    "n_objects": len(common_objects)
                },
                "absolute": {
                    "correlation": float(corr_abs),
                    "p_value": float(p_abs),
                    "n_objects": len(common_objects)
                },
                "positive_only": {
                    "correlation": float(corr_pos),
                    "p_value": float(p_pos),
                    "n_objects": len(common_objects)
                }
            }
            
            print(f"\nCorrelation Results (n={len(common_objects)}):")
            print(f"  Standard avg similarity vs multiplier: r={corr_standard:.4f}, p={p_standard:.4f}")
            print(f"  Absolute avg similarity vs multiplier: r={corr_abs:.4f}, p={p_abs:.4f}")
            print(f"  Positive avg similarity vs multiplier: r={corr_pos:.4f}, p={p_pos:.4f}")
            
            # Create correlation plots
            if output_dir:
                plot_correlations(
                    common_objects, mult_values, 
                    avg_sim_values, avg_abs_sim_values, avg_pos_sim_values,
                    corr_standard, corr_abs, corr_pos,
                    output_dir
                )
        else:
            print(f"Warning: Only {len(common_objects)} objects have both similarities and multipliers")
    
    return results


def plot_correlations(object_names, multipliers, avg_sims, avg_abs_sims, avg_pos_sims,
                      corr_standard, corr_abs, corr_pos, output_dir):
    """Create correlation scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Standard average similarity
    axes[0].scatter(avg_sims, multipliers, alpha=0.6, s=50)
    axes[0].set_xlabel('Average Cosine Similarity', fontsize=11)
    axes[0].set_ylabel('Multiplier Value', fontsize=11)
    axes[0].set_title(f'Standard Avg Similarity vs Multiplier\nr = {corr_standard:.3f}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(avg_sims, multipliers, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(avg_sims), max(avg_sims), 100)
    axes[0].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)
    
    # Plot 2: Absolute average similarity
    axes[1].scatter(avg_abs_sims, multipliers, alpha=0.6, s=50, color='orange')
    axes[1].set_xlabel('Average |Cosine Similarity|', fontsize=11)
    axes[1].set_ylabel('Multiplier Value', fontsize=11)
    axes[1].set_title(f'Absolute Avg Similarity vs Multiplier\nr = {corr_abs:.3f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(avg_abs_sims, multipliers, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(avg_abs_sims), max(avg_abs_sims), 100)
    axes[1].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)
    
    # Plot 3: Positive average similarity
    axes[2].scatter(avg_pos_sims, multipliers, alpha=0.6, s=50, color='green')
    axes[2].set_xlabel('Average Positive Cosine Similarity', fontsize=11)
    axes[2].set_ylabel('Multiplier Value', fontsize=11)
    axes[2].set_title(f'Positive Avg Similarity vs Multiplier\nr = {corr_pos:.3f}', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(avg_pos_sims, multipliers, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(avg_pos_sims), max(avg_pos_sims), 100)
    axes[2].plot(x_line, p(x_line), "r--", alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "similarity_multiplier_correlations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation plots saved to {output_path}")


def create_histogram(norms, output_path, title="Decoder Column Norms Distribution"):
    """
    Create and save a histogram of decoder column norms.
    
    Args:
        norms: numpy array of decoder norms
        output_path: Path to save the histogram image
        title: Title for the histogram
    """
    plt.figure(figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = plt.hist(norms, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical lines for statistics
    mean_val = np.mean(norms)
    median_val = np.median(norms)
    std_val = np.std(norms)
    
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'Mean + 1 STD: {mean_val + std_val:.4f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'Mean - 1 STD: {mean_val - std_val:.4f}')
    
    # Labels and title
    plt.xlabel('Decoder Column Norm', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    
    # Add text box with statistics
    stats_text = f'N: {len(norms)}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {np.min(norms):.4f}\nMax: {np.max(norms):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to {output_path}")


def load_multipliers(multipliers_path):
    """Load multiplier values from JSON file."""
    if not multipliers_path or not Path(multipliers_path).exists():
        return None
    
    with open(multipliers_path, 'r') as f:
        data = json.load(f)
    
    # Handle different possible JSON structures
    if 'multipliers' in data:
        return data['multipliers']
    elif 'objects' in data:
        return {k: v.get('multiplier', v.get('value', 1.0)) 
                for k, v in data['objects'].items()}
    else:
        # Assume it's a flat dict of name: multiplier
        return data


def extract_decoder_norms(sae_path, object_scores_path=None, style_scores_path=None, 
                          object_to_latent=None, style_to_latent=None, device="cpu",
                          histogram_dir=None, multipliers_path=None):
    """
    Extract decoder column norms for assigned concepts and analyze similarities.
    
    Args:
        sae_path: Path to the SAE checkpoint
        object_scores_path: Path to object scores JSON (optional if mappings provided)
        style_scores_path: Path to style scores JSON (optional if mappings provided)
        object_to_latent: Pre-computed object-to-latent mapping (optional)
        style_to_latent: Pre-computed style-to-latent mapping (optional)
        device: Device to load model on
        histogram_dir: Directory to save histogram (optional)
        multipliers_path: Path to multipliers JSON file (optional)
    
    Returns:
        dict containing decoder norms for each concept
        numpy array of all norms
    """
    # Load SAE
    print(f"Loading SAE from {sae_path}")
    sae = Sae.load_from_disk(sae_path, device=device)
    
    # Get decoder weights
    decoder_weight = sae.W_dec
    
    if hasattr(decoder_weight, 'data'):
        decoder_weight = decoder_weight.data
    
    print(f"Decoder weight shape: {decoder_weight.shape}")
    print(f"Number of latents: {sae.num_latents}")
    
    # Determine which dimension corresponds to latents
    if decoder_weight.shape[0] == sae.num_latents:
        norms = torch.norm(decoder_weight, dim=1).cpu().numpy()
        print("Using row-wise norms (decoder shape: [num_latents, d_in])")
    else:
        norms = torch.norm(decoder_weight, dim=0).cpu().numpy()
        print("Using column-wise norms (decoder shape: [d_in, num_latents])")
    
    print(f"Computed {len(norms)} decoder norms")
    
    # Create histogram if directory specified
    if histogram_dir:
        histogram_path = Path(histogram_dir) / "decoder_norms_histogram.png"
        create_histogram(norms, histogram_path, title="SAE Decoder Column Norms Distribution")
    
    # Get concept-to-latent mappings
    if object_to_latent is None or style_to_latent is None:
        if object_scores_path is None and style_scores_path is None:
            raise ValueError("Must provide either scores paths or pre-computed mappings")
        object_to_latent, style_to_latent = load_concept_assignments(
            object_scores_path, style_scores_path, sae.num_latents
        )
    
    # Load multipliers if provided
    multipliers = load_multipliers(multipliers_path) if multipliers_path else None
    
    # Analyze object similarities
    similarity_results = None
    if len(object_to_latent) > 1:
        similarity_results = analyze_object_similarities(
            decoder_weight, object_to_latent, multipliers, histogram_dir
        )
    
    # Build results
    results = {
        "metadata": {
            "sae_path": str(sae_path),
            "num_latents": sae.num_latents,
            "decoder_shape": list(decoder_weight.shape),
            "total_objects": len(object_to_latent),
            "total_styles": len(style_to_latent),
        },
        "objects": {},
        "styles": {},
        "all_norms": {}
    }
    
    # Add similarity results if computed
    if similarity_results:
        results["similarity_analysis"] = similarity_results
    
    # Add all norms
    for i, norm in enumerate(norms):
        results["all_norms"][str(i)] = float(norm)
    
    # Add object norms
    print("\nObject decoder norms:")
    for concept_name, latent_idx in sorted(object_to_latent.items()):
        norm = float(norms[latent_idx])
        results["objects"][concept_name] = {
            "latent_index": latent_idx,
            "decoder_norm": norm
        }
        
        # Add similarity info if available
        if similarity_results and concept_name in similarity_results["average_similarities"]:
            results["objects"][concept_name].update({
                "avg_similarity": similarity_results["average_similarities"][concept_name],
                "avg_absolute_similarity": similarity_results["average_absolute_similarities"][concept_name],
                "avg_positive_similarity": similarity_results["average_positive_similarities"][concept_name]
            })
        
        print(f"  {concept_name}: latent {latent_idx}, norm = {norm:.6f}")
    
    # Add style norms
    print("\nStyle decoder norms:")
    for concept_name, latent_idx in sorted(style_to_latent.items()):
        norm = float(norms[latent_idx])
        results["styles"][concept_name] = {
            "latent_index": latent_idx,
            "decoder_norm": norm
        }
        print(f"  {concept_name}: latent {latent_idx}, norm = {norm:.6f}")
    
    # Summary statistics
    object_norms = [results["objects"][c]["decoder_norm"] for c in results["objects"]]
    style_norms = [results["styles"][c]["decoder_norm"] for c in results["styles"]]
    all_concept_norms = object_norms + style_norms
    
    results["statistics"] = {
        "object_norms": {
            "mean": float(np.mean(object_norms)) if object_norms else 0,
            "std": float(np.std(object_norms)) if object_norms else 0,
            "min": float(np.min(object_norms)) if object_norms else 0,
            "max": float(np.max(object_norms)) if object_norms else 0,
        },
        "style_norms": {
            "mean": float(np.mean(style_norms)) if style_norms else 0,
            "std": float(np.std(style_norms)) if style_norms else 0,
            "min": float(np.min(style_norms)) if style_norms else 0,
            "max": float(np.max(style_norms)) if style_norms else 0,
        },
        "all_concept_norms": {
            "mean": float(np.mean(all_concept_norms)) if all_concept_norms else 0,
            "std": float(np.std(all_concept_norms)) if all_concept_norms else 0,
            "min": float(np.min(all_concept_norms)) if all_concept_norms else 0,
            "max": float(np.max(all_concept_norms)) if all_concept_norms else 0,
        },
        "all_latent_norms": {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
        }
    }
    
    print(f"\nStatistics:")
    print(f"  Object norms: mean={results['statistics']['object_norms']['mean']:.4f}, "
          f"std={results['statistics']['object_norms']['std']:.4f}")
    print(f"  Style norms: mean={results['statistics']['style_norms']['mean']:.4f}, "
          f"std={results['statistics']['style_norms']['std']:.4f}")
    print(f"  All latent norms: mean={results['statistics']['all_latent_norms']['mean']:.4f}, "
          f"std={results['statistics']['all_latent_norms']['std']:.4f}")
    
    return results, norms


def main():
    parser = argparse.ArgumentParser(
        description="Extract decoder column norms and analyze similarities for assigned concepts from a trained SAE"
    )
    
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to the SAE checkpoint directory"
    )
    parser.add_argument(
        "--object_scores_path",
        type=str,
        default=None,
        help="Path to the JSON file containing object scores"
    )
    parser.add_argument(
        "--style_scores_path",
        type=str,
        default=None,
        help="Path to the JSON file containing style scores"
    )
    parser.add_argument(
        "--multipliers_path",
        type=str,
        default=None,
        help="Path to JSON file containing multiplier values for objects"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="decoder_norms.json",
        help="Path to save the output JSON file"
    )
    parser.add_argument(
        "--all_norms_output_path",
        type=str,
        default="all_decoder_norms.json",
        help="Path to save the JSON file with all decoder norms"
    )
    parser.add_argument(
        "--histogram_dir",
        type=str,
        default="norms_multipliers_analysis",
        help="Directory to save histograms and plots"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (cpu or cuda)"
    )
    
    args = parser.parse_args()
    
    # Create histogram directory if it doesn't exist
    histogram_dir = Path(args.histogram_dir)
    histogram_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract norms and analyze similarities
    results, all_norms = extract_decoder_norms(
        sae_path=args.sae_path,
        object_scores_path=args.object_scores_path,
        style_scores_path=args.style_scores_path,
        device=args.device,
        histogram_dir=args.histogram_dir,
        multipliers_path=args.multipliers_path
    )
    
    # Save main results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Save all norms to separate JSON
    all_norms_output_path = Path(args.all_norms_output_path)
    all_norms_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_norms_data = {
        "metadata": {
            "sae_path": str(args.sae_path),
            "num_latents": len(all_norms),
            "description": "All decoder column norms indexed by latent index"
        },
        "statistics": {
            "mean": float(np.mean(all_norms)),
            "std": float(np.std(all_norms)),
            "min": float(np.min(all_norms)),
            "max": float(np.max(all_norms)),
            "median": float(np.median(all_norms)),
            "percentile_25": float(np.percentile(all_norms, 25)),
            "percentile_75": float(np.percentile(all_norms, 75)),
            "percentile_90": float(np.percentile(all_norms, 90)),
            "percentile_95": float(np.percentile(all_norms, 95)),
            "percentile_99": float(np.percentile(all_norms, 99)),
        },
        "norms": {str(i): float(norm) for i, norm in enumerate(all_norms)}
    }
    
    with open(all_norms_output_path, 'w') as f:
        json.dump(all_norms_data, f, indent=2)
    
    print(f"All decoder norms saved to {all_norms_output_path}")
    print(f"Visualizations saved to {histogram_dir}/")


if __name__ == "__main__":
    main()