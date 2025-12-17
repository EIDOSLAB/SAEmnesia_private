#!/usr/bin/env python
"""
Compute and visualize correlations between object decoder cosine similarities and multipliers.

This script:
1. Loads decoder norms JSON (with similarity analysis)
2. Loads multipliers from .pth file
3. Computes correlations between avg similarities and multipliers
4. Creates scatter plots for all three similarity types
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import stats


def load_decoder_norms(json_path):
    """Load decoder norms and similarity analysis from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_multipliers(pth_path):
    """Load multipliers from .pth file."""
    params = torch.load(pth_path, map_location='cpu')
    return params


def normalize_name(name):
    """Normalize concept names for matching."""
    return name.lower().replace('_', ' ').strip()


def match_similarities_with_multipliers(norms_data, multipliers_data):
    """
    Match object similarities with their multipliers.
    
    Returns:
        dict with matched data for each object
    """
    objects = norms_data.get('objects', {})
    
    # Check if similarity analysis exists
    if 'similarity_analysis' not in norms_data:
        print("ERROR: No similarity analysis found in JSON file.")
        print("Please re-run extract_decoder_norms.py to generate similarity analysis.")
        return None
    
    sim_analysis = norms_data['similarity_analysis']
    avg_sims = sim_analysis.get('average_similarities', {})
    avg_abs_sims = sim_analysis.get('average_absolute_similarities', {})
    avg_pos_sims = sim_analysis.get('average_positive_similarities', {})
    
    # Match objects with multipliers
    matched = {}
    
    for obj_name in objects.keys():
        # Try to find multiplier for this object
        mult_value = None
        
        # Try exact match first
        if obj_name in multipliers_data:
            mult_value = multipliers_data[obj_name]['multiplier']
        else:
            # Try normalized matching
            norm_name = normalize_name(obj_name)
            for mult_key, mult_data in multipliers_data.items():
                if normalize_name(mult_key) == norm_name:
                    mult_value = mult_data['multiplier']
                    break
        
        if mult_value is not None and obj_name in avg_sims:
            matched[obj_name] = {
                'multiplier': mult_value,
                'avg_similarity': avg_sims[obj_name],
                'avg_absolute_similarity': avg_abs_sims[obj_name],
                'avg_positive_similarity': avg_pos_sims[obj_name],
                'decoder_norm': objects[obj_name]['decoder_norm'],
                'latent_index': objects[obj_name]['latent_index']
            }
    
    return matched


def compute_correlations(matched_data):
    """Compute Pearson correlations for all three similarity types."""
    object_names = list(matched_data.keys())
    
    multipliers = np.array([matched_data[obj]['multiplier'] for obj in object_names])
    avg_sims = np.array([matched_data[obj]['avg_similarity'] for obj in object_names])
    avg_abs_sims = np.array([matched_data[obj]['avg_absolute_similarity'] for obj in object_names])
    avg_pos_sims = np.array([matched_data[obj]['avg_positive_similarity'] for obj in object_names])
    
    # Compute correlations
    corr_standard, p_standard = stats.pearsonr(avg_sims, multipliers)
    corr_abs, p_abs = stats.pearsonr(avg_abs_sims, multipliers)
    corr_pos, p_pos = stats.pearsonr(avg_pos_sims, multipliers)
    
    # Also compute Spearman
    spearman_standard, sp_standard = stats.spearmanr(avg_sims, multipliers)
    spearman_abs, sp_abs = stats.spearmanr(avg_abs_sims, multipliers)
    spearman_pos, sp_pos = stats.spearmanr(avg_pos_sims, multipliers)
    
    return {
        'standard': {
            'pearson_r': corr_standard,
            'pearson_p': p_standard,
            'spearman_r': spearman_standard,
            'spearman_p': sp_standard,
            'data': (avg_sims, multipliers)
        },
        'absolute': {
            'pearson_r': corr_abs,
            'pearson_p': p_abs,
            'spearman_r': spearman_abs,
            'spearman_p': sp_abs,
            'data': (avg_abs_sims, multipliers)
        },
        'positive_only': {
            'pearson_r': corr_pos,
            'pearson_p': p_pos,
            'spearman_r': spearman_pos,
            'spearman_p': sp_pos,
            'data': (avg_pos_sims, multipliers)
        },
        'object_names': object_names
    }


def create_correlation_plots(correlations, matched_data, output_dir):
    """Create scatter plots for all three correlation types."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    plot_configs = [
        ('standard', 'Average Cosine Similarity', 'steelblue', 0),
        ('absolute', 'Average |Cosine Similarity|', 'coral', 1),
        ('positive_only', 'Average Positive Cosine Similarity', 'forestgreen', 2)
    ]
    
    for corr_type, xlabel, color, idx in plot_configs:
        ax = axes[idx]
        corr_data = correlations[corr_type]
        x_vals, y_vals = corr_data['data']
        
        # Scatter plot
        ax.scatter(x_vals, y_vals, s=100, alpha=0.7, c=color, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='red', linewidth=2, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Multiplier Value', fontsize=12, fontweight='bold')
        
        # Title with correlation info
        title = f'{corr_type.replace("_", " ").title()} vs Multiplier\n'
        title += f'Pearson r = {corr_data["pearson_r"]:.3f} (p = {corr_data["pearson_p"]:.4f})\n'
        title += f'Spearman ρ = {corr_data["spearman_r"]:.3f} (p = {corr_data["spearman_p"]:.4f})'
        ax.set_title(title, fontsize=11)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add R² value
        ss_res = np.sum((y_vals - p(x_vals)) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f'Cosine Similarity vs Multiplier Correlations (n={len(correlations["object_names"])})', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "similarity_multiplier_correlations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrelation plots saved to {output_path}")


def create_detailed_scatter(correlations, matched_data, output_dir):
    """Create detailed scatter plot with object labels."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    object_names = correlations['object_names']
    
    plot_configs = [
        ('standard', 'Average Cosine Similarity', 'steelblue', 0),
        ('absolute', 'Average |Cosine Similarity|', 'coral', 1),
        ('positive_only', 'Average Positive Cosine Similarity', 'forestgreen', 2)
    ]
    
    for corr_type, xlabel, color, idx in plot_configs:
        ax = axes[idx]
        corr_data = correlations[corr_type]
        x_vals, y_vals = corr_data['data']
        
        # Scatter plot
        scatter = ax.scatter(x_vals, y_vals, s=120, alpha=0.6, c=color, edgecolors='black', linewidth=1)
        
        # Add labels for each point
        for i, obj_name in enumerate(object_names):
            ax.annotate(obj_name, (x_vals[i], y_vals[i]), 
                       xytext=(3, 3), textcoords='offset points', 
                       fontsize=8, alpha=0.8)
        
        # Add trend line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='red', linewidth=2.5, alpha=0.7, 
                label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Multiplier Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{corr_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        
        # Grid and legend
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        
        # Add correlation text box
        corr_text = (f'Pearson: r={corr_data["pearson_r"]:.3f}, p={corr_data["pearson_p"]:.4f}\n'
                    f'Spearman: ρ={corr_data["spearman_r"]:.3f}, p={corr_data["spearman_p"]:.4f}')
        ax.text(0.05, 0.95, corr_text, 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Detailed Similarity-Multiplier Correlations with Object Labels (n={len(object_names)})', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "similarity_multiplier_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed scatter plots saved to {output_path}")


def print_summary(correlations, matched_data):
    """Print summary statistics and results."""
    print("\n" + "="*80)
    print("COSINE SIMILARITY vs MULTIPLIER CORRELATION ANALYSIS")
    print("="*80)
    print(f"\nNumber of objects analyzed: {len(correlations['object_names'])}")
    
    print("\n" + "-"*80)
    print("1. STANDARD AVERAGE SIMILARITY")
    print("-"*80)
    std_data = correlations['standard']
    print(f"   Pearson correlation:  r = {std_data['pearson_r']:7.4f}  (p = {std_data['pearson_p']:.6f})")
    print(f"   Spearman correlation: ρ = {std_data['spearman_r']:7.4f}  (p = {std_data['spearman_p']:.6f})")
    sig = '***' if std_data['pearson_p'] < 0.001 else '**' if std_data['pearson_p'] < 0.01 else '*' if std_data['pearson_p'] < 0.05 else 'ns'
    print(f"   Significance: {sig}")
    
    print("\n" + "-"*80)
    print("2. ABSOLUTE AVERAGE SIMILARITY (|cos|)")
    print("-"*80)
    abs_data = correlations['absolute']
    print(f"   Pearson correlation:  r = {abs_data['pearson_r']:7.4f}  (p = {abs_data['pearson_p']:.6f})")
    print(f"   Spearman correlation: ρ = {abs_data['spearman_r']:7.4f}  (p = {abs_data['spearman_p']:.6f})")
    sig = '***' if abs_data['pearson_p'] < 0.001 else '**' if abs_data['pearson_p'] < 0.01 else '*' if abs_data['pearson_p'] < 0.05 else 'ns'
    print(f"   Significance: {sig}")
    
    print("\n" + "-"*80)
    print("3. POSITIVE-ONLY AVERAGE SIMILARITY (ReLU-like)")
    print("-"*80)
    pos_data = correlations['positive_only']
    print(f"   Pearson correlation:  r = {pos_data['pearson_r']:7.4f}  (p = {pos_data['pearson_p']:.6f})")
    print(f"   Spearman correlation: ρ = {pos_data['spearman_r']:7.4f}  (p = {pos_data['spearman_p']:.6f})")
    sig = '***' if pos_data['pearson_p'] < 0.001 else '**' if pos_data['pearson_p'] < 0.01 else '*' if pos_data['pearson_p'] < 0.05 else 'ns'
    print(f"   Significance: {sig}")
    
    print("\n" + "="*80)
    print("DETAILED OBJECT DATA (sorted by multiplier)")
    print("="*80)
    
    # Sort by multiplier
    sorted_objects = sorted(matched_data.items(), key=lambda x: x[1]['multiplier'])
    
    print(f"\n{'Object':<20} {'Mult':>8} {'AvgSim':>10} {'Avg|Sim|':>10} {'AvgPosSim':>10} {'Norm':>10}")
    print("-"*80)
    for obj_name, data in sorted_objects:
        print(f"{obj_name:<20} {data['multiplier']:8.2f} {data['avg_similarity']:10.4f} "
              f"{data['avg_absolute_similarity']:10.4f} {data['avg_positive_similarity']:10.4f} "
              f"{data['decoder_norm']:10.4f}")
    
    print("\n" + "="*80)
    
    # Additional statistics
    mults = np.array([data['multiplier'] for data in matched_data.values()])
    avg_sims = np.array([data['avg_similarity'] for data in matched_data.values()])
    avg_abs_sims = np.array([data['avg_absolute_similarity'] for data in matched_data.values()])
    avg_pos_sims = np.array([data['avg_positive_similarity'] for data in matched_data.values()])
    
    print("\nDESCRIPTIVE STATISTICS")
    print("-"*80)
    print(f"\nMultipliers:")
    print(f"  Mean ± Std: {mults.mean():.3f} ± {mults.std():.3f}")
    print(f"  Range: [{mults.min():.3f}, {mults.max():.3f}]")
    
    print(f"\nAverage Similarities:")
    print(f"  Mean ± Std: {avg_sims.mean():.4f} ± {avg_sims.std():.4f}")
    print(f"  Range: [{avg_sims.min():.4f}, {avg_sims.max():.4f}]")
    
    print(f"\nAverage |Similarities|:")
    print(f"  Mean ± Std: {avg_abs_sims.mean():.4f} ± {avg_abs_sims.std():.4f}")
    print(f"  Range: [{avg_abs_sims.min():.4f}, {avg_abs_sims.max():.4f}]")
    
    print(f"\nAverage Positive Similarities:")
    print(f"  Mean ± Std: {avg_pos_sims.mean():.4f} ± {avg_pos_sims.std():.4f}")
    print(f"  Range: [{avg_pos_sims.min():.4f}, {avg_pos_sims.max():.4f}]")
    
    print("\n" + "="*80)


def save_results_json(correlations, matched_data, output_path):
    """Save correlation results to JSON."""
    results = {
        'metadata': {
            'n_objects': len(correlations['object_names']),
            'object_names': correlations['object_names']
        },
        'correlations': {
            'standard': {
                'pearson_r': float(correlations['standard']['pearson_r']),
                'pearson_p': float(correlations['standard']['pearson_p']),
                'spearman_r': float(correlations['standard']['spearman_r']),
                'spearman_p': float(correlations['standard']['spearman_p'])
            },
            'absolute': {
                'pearson_r': float(correlations['absolute']['pearson_r']),
                'pearson_p': float(correlations['absolute']['pearson_p']),
                'spearman_r': float(correlations['absolute']['spearman_r']),
                'spearman_p': float(correlations['absolute']['spearman_p'])
            },
            'positive_only': {
                'pearson_r': float(correlations['positive_only']['pearson_r']),
                'pearson_p': float(correlations['positive_only']['pearson_p']),
                'spearman_r': float(correlations['positive_only']['spearman_r']),
                'spearman_p': float(correlations['positive_only']['spearman_p'])
            }
        },
        'objects': matched_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute correlations between cosine similarities and multipliers"
    )
    
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to decoder_norms.json (with similarity analysis)"
    )
    parser.add_argument(
        "--multipliers_pth",
        type=str,
        required=True,
        help="Path to class_params.pth file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="similarity_multiplier_analysis",
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading decoder norms and similarity analysis...")
    norms_data = load_decoder_norms(args.json_path)
    
    print("Loading multipliers...")
    multipliers_data = load_multipliers(args.multipliers_pth)
    print(f"  Found {len(multipliers_data)} objects with multipliers")
    
    # Match data
    print("\nMatching similarities with multipliers...")
    matched_data = match_similarities_with_multipliers(norms_data, multipliers_data)
    
    if matched_data is None or len(matched_data) == 0:
        print("ERROR: No matching data found!")
        return
    
    print(f"  Successfully matched {len(matched_data)} objects")
    
    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(matched_data)
    
    # Print summary
    print_summary(correlations, matched_data)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_correlation_plots(correlations, matched_data, output_dir)
    create_detailed_scatter(correlations, matched_data, output_dir)
    
    # Save results
    results_path = output_dir / "correlation_results.json"
    save_results_json(correlations, matched_data, results_path)
    
    print(f"\n✓ Analysis complete! All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()