#!/usr/bin/env python
"""
Visualize the relationship between decoder column norms, optimal multipliers, and average latent activations.

This script:
1. Loads the decoder norms from the JSON file
2. Loads the optimal multipliers from the .pth file
3. Loads the average latent activations from the JSON file
4. Creates visualizations to understand their relationship
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_decoder_norms(json_path):
    """Load decoder norms from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_multipliers(pth_path):
    """Load multipliers from .pth file."""
    params = torch.load(pth_path, map_location='cpu')
    return params


def load_avg_activations(json_path):
    """Load average latent activations from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('results', {})


def normalize_name(name):
    """Normalize concept names for matching."""
    return name.lower().replace('_', ' ').strip()


def match_concepts(norms_data, multipliers_data, avg_activations_data=None):
    """Match concepts between norms, multipliers, and avg activations data."""
    matched = []
    
    # Create normalized name lookup for norms
    norms_lookup = {}
    for concept, data in norms_data.get('objects', {}).items():
        norms_lookup[normalize_name(concept)] = {
            'original_name': concept,
            'latent_index': data['latent_index'],
            'decoder_norm': data['decoder_norm']
        }
    
    # Create normalized name lookup for avg activations
    activations_lookup = {}
    if avg_activations_data:
        for concept, data in avg_activations_data.items():
            activations_lookup[normalize_name(concept)] = {
                'original_name': concept,
                'avg_activation': data['avg_activation'],
                'std_activation': data.get('std_activation', 0)
            }
    
    # Match with multipliers
    for concept, params in multipliers_data.items():
        norm_key = normalize_name(concept)
        if norm_key in norms_lookup:
            entry = {
                'concept': concept,
                'latent_index': norms_lookup[norm_key]['latent_index'],
                'decoder_norm': norms_lookup[norm_key]['decoder_norm'],
                'multiplier': params['multiplier'],
                'percentile': params['percentile']
            }
            
            # Add avg activation if available
            if norm_key in activations_lookup:
                entry['avg_activation'] = activations_lookup[norm_key]['avg_activation']
                entry['std_activation'] = activations_lookup[norm_key]['std_activation']
            else:
                entry['avg_activation'] = None
                entry['std_activation'] = None
                if avg_activations_data:
                    print(f"Warning: No avg activation found for concept '{concept}'")
            
            matched.append(entry)
        else:
            print(f"Warning: No norm found for concept '{concept}'")
    
    return matched


def create_visualizations(matched_data, output_dir, has_activations=False):
    """Create various visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    concepts = [d['concept'] for d in matched_data]
    norms = np.array([d['decoder_norm'] for d in matched_data])
    multipliers = np.array([d['multiplier'] for d in matched_data])
    latent_indices = [d['latent_index'] for d in matched_data]
    
    # Extract avg activations if available
    if has_activations:
        avg_activations = np.array([d['avg_activation'] if d['avg_activation'] is not None else 0 for d in matched_data])
        std_activations = np.array([d['std_activation'] if d['std_activation'] is not None else 0 for d in matched_data])
    
    # Set up style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Scatter plot: Norm vs Multiplier
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(norms, multipliers, s=100, alpha=0.7, c=range(len(concepts)), cmap='tab20')
    
    # Add labels for each point
    for i, concept in enumerate(concepts):
        ax.annotate(concept, (norms[i], multipliers[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Decoder Column Norm', fontsize=12)
    ax.set_ylabel('Optimal Multiplier', fontsize=12)
    ax.set_title('Decoder Norm vs Optimal Multiplier', fontsize=14)
    
    # Add correlation info
    correlation = np.corrcoef(norms, multipliers)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'norm_vs_multiplier_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Bar chart: Three bars for each concept with multiple y-axes
    if has_activations:
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        x = np.arange(len(concepts))
        width = 0.25
        
        # Left axis: Decoder Norms
        bars1 = ax1.bar(x - width, norms, width, label='Decoder Norm', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Concept', fontsize=12)
        ax1.set_ylabel('Decoder Norm', color='steelblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(concepts, rotation=45, ha='right', fontsize=9)
        
        # Right axis 1: Multipliers (absolute value)
        ax2 = ax1.twinx()
        abs_multipliers = np.abs(multipliers)
        bars2 = ax2.bar(x, abs_multipliers, width, label='|Multiplier|', color='coral', alpha=0.8)
        max_abs = np.max(abs_multipliers)
        ax2.set_ylim(0, max_abs * 1.1)
        ax2.set_ylabel('|Multiplier|', color='coral', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='coral')
        
        # Right axis 2: Avg Activations
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        bars3 = ax3.bar(x + width, avg_activations, width, label='Avg Activation', color='forestgreen', alpha=0.8)
        ax3.set_ylabel('Avg Activation', color='forestgreen', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='forestgreen')
        
        # Calculate correlations
        from scipy import stats
        pearson_norm_mult, p_nm = stats.pearsonr(norms, abs_multipliers)
        pearson_norm_act, p_na = stats.pearsonr(norms, avg_activations)
        pearson_mult_act, p_ma = stats.pearsonr(abs_multipliers, avg_activations)
        
        # Add correlation text box
        corr_text = (f'Correlations:\n'
                    f'Norm vs |Mult|: r={pearson_norm_mult:.3f}\n'
                    f'Norm vs AvgAct: r={pearson_norm_act:.3f}\n'
                    f'|Mult| vs AvgAct: r={pearson_mult_act:.3f}')
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend
        ax1.legend([bars1, bars2, bars3], ['Decoder Norm', '|Multiplier|', 'Avg Activation'], 
                  loc='upper right')
        
        plt.title('Decoder Norms vs Multipliers vs Avg Activations by Concept', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'norm_vs_multiplier_vs_activation_bars.png', dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Original two-bar version
        fig, ax1 = plt.subplots(figsize=(18, 8))
        
        x = np.arange(len(concepts))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, norms, width, label='Decoder Norm', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Concept', fontsize=12)
        ax1.set_ylabel('Decoder Norm', color='steelblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(concepts, rotation=45, ha='right', fontsize=9)
        
        ax2 = ax1.twinx()
        abs_multipliers = np.abs(multipliers)
        bars2 = ax2.bar(x + width/2, abs_multipliers, width, label='Multiplier', color='coral', alpha=0.8)
        max_abs = np.max(abs_multipliers)
        ax2.set_ylim(0, max_abs * 1.1)
        ax2.set_ylabel('Multiplier', color='coral', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='coral')
        yticks = ax2.get_yticks()
        ax2.set_yticklabels([f'{-y:.1f}' for y in yticks])
        
        from scipy import stats
        pearson_corr, pearson_p = stats.pearsonr(norms, abs_multipliers)
        spearman_corr, spearman_p = stats.spearmanr(norms, abs_multipliers)
        
        corr_text = f'Pearson r = {pearson_corr:.3f} (p = {pearson_p:.3f})\nSpearman ρ = {spearman_corr:.3f} (p = {spearman_p:.3f})'
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.title('Decoder Norms vs Multipliers by Concept', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'norm_vs_multiplier_bars.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Sorted by norm with all three metrics
    sorted_indices = np.argsort(norms)
    sorted_concepts = [concepts[i] for i in sorted_indices]
    sorted_norms = norms[sorted_indices]
    sorted_multipliers = multipliers[sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(sorted_concepts))
    
    ax1.set_xlabel('Concept (sorted by norm)', fontsize=12)
    ax1.set_ylabel('Decoder Norm', color='steelblue', fontsize=12)
    line1 = ax1.plot(x, sorted_norms, 'o-', color='steelblue', label='Decoder Norm', markersize=8)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_concepts, rotation=45, ha='right', fontsize=9)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Multiplier', color='coral', fontsize=12)
    line2 = ax2.plot(x, sorted_multipliers, 's-', color='coral', label='Multiplier', markersize=8)
    ax2.tick_params(axis='y', labelcolor='coral')
    
    if has_activations:
        sorted_activations = avg_activations[sorted_indices]
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        line3 = ax3.plot(x, sorted_activations, '^-', color='forestgreen', label='Avg Activation', markersize=8)
        ax3.set_ylabel('Avg Activation', color='forestgreen', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='forestgreen')
        
        lines = line1 + line2 + line3
        labels = ['Decoder Norm', 'Multiplier', 'Avg Activation']
    else:
        lines = line1 + line2
        labels = ['Decoder Norm', 'Multiplier']
    
    ax1.legend(lines, labels, loc='upper left')
    
    title_suffix = ' with Avg Activations' if has_activations else ''
    ax1.set_title(f'Metrics (Sorted by Norm){title_suffix}', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_sorted_by_norm.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap / Table view with avg activations
    fig, ax = plt.subplots(figsize=(16, 10))
    
    sorted_by_mult = sorted(matched_data, key=lambda x: x['multiplier'])
    
    table_data = []
    for d in sorted_by_mult:
        row = [
            d['concept'],
            d['latent_index'],
            f"{d['decoder_norm']:.4f}",
            f"{d['multiplier']:.1f}"
        ]
        if has_activations:
            avg_act = d['avg_activation'] if d['avg_activation'] is not None else 'N/A'
            row.append(f"{avg_act:.4f}" if isinstance(avg_act, float) else avg_act)
        table_data.append(row)
    
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ['Concept', 'Latent Index', 'Decoder Norm', 'Multiplier']
    col_widths = [0.25, 0.15, 0.2, 0.2]
    if has_activations:
        col_labels.append('Avg Activation')
        col_widths.append(0.2)
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on values
    for i, d in enumerate(sorted_by_mult):
        mult_color = plt.cm.RdYlGn((d['multiplier'] - min(multipliers)) / (max(multipliers) - min(multipliers) + 1e-6))
        table[(i+1, 3)].set_facecolor(mult_color)
        
        norm_color = plt.cm.Blues((d['decoder_norm'] - min(norms)) / (max(norms) - min(norms) + 1e-6))
        table[(i+1, 2)].set_facecolor(norm_color)
        
        if has_activations and d['avg_activation'] is not None:
            act_color = plt.cm.Greens((d['avg_activation'] - min(avg_activations)) / (max(avg_activations) - min(avg_activations) + 1e-6))
            table[(i+1, 4)].set_facecolor(act_color)
    
    ax.set_title('Concept Parameters Summary (Sorted by Multiplier)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Regression plots
    if multipliers.max() != multipliers.min():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.scatter(norms, multipliers, s=100, alpha=0.7, c='steelblue')
        
        z = np.polyfit(norms, multipliers, 1)
        p = np.poly1d(z)
        x_line = np.linspace(norms.min(), norms.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='coral', linewidth=2, label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        for i, concept in enumerate(concepts):
            ax.annotate(concept, (norms[i], multipliers[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Decoder Column Norm', fontsize=12)
        ax.set_ylabel('Optimal Multiplier', fontsize=12)
        ax.set_title('Decoder Norm vs Optimal Multiplier with Linear Fit', fontsize=14)
        ax.legend()
        
        ss_res = np.sum((multipliers - p(norms)) ** 2)
        ss_tot = np.sum((multipliers - np.mean(multipliers)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        ax.text(0.05, 0.90, f'R² = {r_squared:.3f}', 
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'norm_vs_multiplier_regression.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 6. NEW: Stacked bar plot (Avg Activation + |Multiplier|) vs Decoder Norm
    if has_activations:
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        x = np.arange(len(concepts))
        width = 0.35
        
        # Sort by decoder norm for better visualization
        sorted_idx = np.argsort(norms)
        sorted_concepts = [concepts[i] for i in sorted_idx]
        sorted_norms = norms[sorted_idx]
        sorted_multipliers = np.abs(multipliers[sorted_idx])
        sorted_activations = avg_activations[sorted_idx]
        
        # Left side: Decoder Norms as bars
        bars_norm = ax1.bar(x - width/2, sorted_norms, width, label='Decoder Norm', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Concept (sorted by Decoder Norm)', fontsize=12)
        ax1.set_ylabel('Decoder Norm', color='steelblue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_concepts, rotation=45, ha='right', fontsize=9)
        
        # Right side: Stacked bars (Avg Activation bottom, |Multiplier| top)
        ax2 = ax1.twinx()
        bars_act = ax2.bar(x + width/2, sorted_activations, width, label='Avg Activation', color='forestgreen', alpha=0.8)
        bars_mult = ax2.bar(x + width/2, sorted_multipliers, width, bottom=sorted_activations, label='|Multiplier|', color='coral', alpha=0.8)
        ax2.set_ylabel('Avg Activation + |Multiplier|', color='black', fontsize=12)
        
        # Compute the sum and correlation
        stacked_sum = sorted_activations + sorted_multipliers
        
        # Use original (unsorted) arrays for correlation
        sum_act_mult = avg_activations + np.abs(multipliers)
        corr_norm_sum = np.corrcoef(norms, sum_act_mult)[0, 1]
        
        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(norms, sum_act_mult)
        spearman_r, spearman_p = stats.spearmanr(norms, sum_act_mult)
        
        # Add correlation text box
        corr_text = (f'Norm vs (AvgAct + |Mult|):\n'
                    f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})\n'
                    f'Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3f})')
        ax1.text(0.02, 0.98, corr_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend
        ax1.legend([bars_norm, bars_act, bars_mult], ['Decoder Norm', 'Avg Activation', '|Multiplier|'], 
                  loc='upper right')
        
        plt.title('Decoder Norm vs Stacked (Avg Activation + |Multiplier|)', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'norm_vs_stacked_activation_multiplier.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also print the correlation
        print(f"\n*** Stacked Sum Correlation ***")
        print(f"Correlation (Norm vs AvgAct + |Mult|): Pearson r = {pearson_r:.4f}, Spearman ρ = {spearman_r:.4f}")
    
    # 7. Scatter matrix if we have activations
    if has_activations:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Norm vs Multiplier
        axes[0, 0].scatter(norms, multipliers, s=80, alpha=0.7, c='steelblue')
        axes[0, 0].set_xlabel('Decoder Norm')
        axes[0, 0].set_ylabel('Multiplier')
        axes[0, 0].set_title(f'Norm vs Mult (r={np.corrcoef(norms, multipliers)[0,1]:.3f})')
        
        # Norm vs Avg Activation
        axes[0, 1].scatter(norms, avg_activations, s=80, alpha=0.7, c='forestgreen')
        axes[0, 1].set_xlabel('Decoder Norm')
        axes[0, 1].set_ylabel('Avg Activation')
        axes[0, 1].set_title(f'Norm vs AvgAct (r={np.corrcoef(norms, avg_activations)[0,1]:.3f})')
        
        # Norm vs (AvgAct + |Mult|) - NEW
        sum_act_mult = avg_activations + np.abs(multipliers)
        axes[0, 2].scatter(norms, sum_act_mult, s=80, alpha=0.7, c='purple')
        axes[0, 2].set_xlabel('Decoder Norm')
        axes[0, 2].set_ylabel('AvgAct + |Mult|')
        axes[0, 2].set_title(f'Norm vs Sum (r={np.corrcoef(norms, sum_act_mult)[0,1]:.3f})')
        
        # Multiplier vs Avg Activation
        axes[1, 0].scatter(multipliers, avg_activations, s=80, alpha=0.7, c='coral')
        axes[1, 0].set_xlabel('Multiplier')
        axes[1, 0].set_ylabel('Avg Activation')
        axes[1, 0].set_title(f'Mult vs AvgAct (r={np.corrcoef(multipliers, avg_activations)[0,1]:.3f})')
        
        # Norm * Avg Activation vs Multiplier (test if product predicts multiplier)
        product = norms * avg_activations
        axes[1, 1].scatter(product, multipliers, s=80, alpha=0.7, c='teal')
        axes[1, 1].set_xlabel('Norm × Avg Activation')
        axes[1, 1].set_ylabel('Multiplier')
        axes[1, 1].set_title(f'Norm×AvgAct vs Mult (r={np.corrcoef(product, multipliers)[0,1]:.3f})')
        
        # |Multiplier| vs Avg Activation
        axes[1, 2].scatter(np.abs(multipliers), avg_activations, s=80, alpha=0.7, c='orange')
        axes[1, 2].set_xlabel('|Multiplier|')
        axes[1, 2].set_ylabel('Avg Activation')
        axes[1, 2].set_title(f'|Mult| vs AvgAct (r={np.corrcoef(np.abs(multipliers), avg_activations)[0,1]:.3f})')
        
        plt.suptitle('Pairwise Relationships', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nDecoder Norms:")
    print(f"  Mean: {norms.mean():.4f}")
    print(f"  Std:  {norms.std():.4f}")
    print(f"  Min:  {norms.min():.4f} ({concepts[np.argmin(norms)]})")
    print(f"  Max:  {norms.max():.4f} ({concepts[np.argmax(norms)]})")
    
    print(f"\nMultipliers:")
    print(f"  Mean: {multipliers.mean():.4f}")
    print(f"  Std:  {multipliers.std():.4f}")
    print(f"  Min:  {multipliers.min():.4f} ({concepts[np.argmin(multipliers)]})")
    print(f"  Max:  {multipliers.max():.4f} ({concepts[np.argmax(multipliers)]})")
    
    if has_activations:
        print(f"\nAvg Activations:")
        print(f"  Mean: {avg_activations.mean():.4f}")
        print(f"  Std:  {avg_activations.std():.4f}")
        print(f"  Min:  {avg_activations.min():.4f} ({concepts[np.argmin(avg_activations)]})")
        print(f"  Max:  {avg_activations.max():.4f} ({concepts[np.argmax(avg_activations)]})")
    
    print(f"\nCorrelations:")
    print(f"  Norm vs Multiplier: {np.corrcoef(norms, multipliers)[0, 1]:.4f}")
    if has_activations:
        print(f"  Norm vs Avg Activation: {np.corrcoef(norms, avg_activations)[0, 1]:.4f}")
        print(f"  Multiplier vs Avg Activation: {np.corrcoef(multipliers, avg_activations)[0, 1]:.4f}")
        sum_act_mult = avg_activations + np.abs(multipliers)
        print(f"  Norm vs (AvgAct + |Mult|): {np.corrcoef(norms, sum_act_mult)[0, 1]:.4f}")
    
    print("\n" + "="*60)
    print("DETAILED DATA (sorted by norm)")
    print("="*60)
    
    if has_activations:
        print(f"{'Concept':<15} {'Latent':<8} {'Norm':<12} {'Multiplier':<12} {'AvgAct':<12}")
        print("-"*59)
        for i in np.argsort(norms):
            avg_act = avg_activations[i] if has_activations else 'N/A'
            print(f"{concepts[i]:<15} {latent_indices[i]:<8} {norms[i]:<12.4f} {multipliers[i]:<12.1f} {avg_act:<12.4f}")
    else:
        print(f"{'Concept':<15} {'Latent':<8} {'Norm':<12} {'Multiplier':<12}")
        print("-"*47)
        for i in np.argsort(norms):
            print(f"{concepts[i]:<15} {latent_indices[i]:<8} {norms[i]:<12.4f} {multipliers[i]:<12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize relationship between decoder norms, multipliers, and avg activations"
    )
    
    parser.add_argument(
        "--norms_json",
        type=str,
        required=True,
        help="Path to decoder_norms.json"
    )
    parser.add_argument(
        "--multipliers_pth",
        type=str,
        required=True,
        help="Path to class_params.pth"
    )
    parser.add_argument(
        "--avg_activations_json",
        type=str,
        default=None,
        help="Path to avg_latent_activations.json (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./norm_multiplier_analysis",
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading decoder norms...")
    norms_data = load_decoder_norms(args.norms_json)
    
    print("Loading multipliers...")
    multipliers_data = load_multipliers(args.multipliers_pth)
    
    # Load avg activations if provided
    avg_activations_data = None
    if args.avg_activations_json:
        print("Loading average activations...")
        avg_activations_data = load_avg_activations(args.avg_activations_json)
        print(f"  Found {len(avg_activations_data)} concepts with avg activations")
    
    # Match concepts
    print("Matching concepts...")
    matched_data = match_concepts(norms_data, multipliers_data, avg_activations_data)
    
    if not matched_data:
        print("ERROR: No matching concepts found!")
        return
    
    print(f"Found {len(matched_data)} matching concepts")
    
    # Check if we have activations data
    has_activations = avg_activations_data is not None and any(d['avg_activation'] is not None for d in matched_data)
    
    # Create visualizations
    create_visualizations(matched_data, args.output_dir, has_activations=has_activations)


if __name__ == "__main__":
    main()