import argparse
import json
import torch
import numpy as np
import os 
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configure matplotlib for LaTeX-style plots without requiring LaTeX installation
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern', 'Times New Roman', 'DejaVu Serif'],
    'text.usetex': False,  # Use matplotlib's LaTeX-style fonts instead
    'mathtext.fontset': 'cm',  # Computer Modern math fonts
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 22,
    'lines.linewidth': 3.5,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'grid.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.4
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from SAE.unlearning_utils import compute_feature_importance

# Predefined y-limits for nudity concepts (you'll need to populate these with actual values)
CONCEPT_Y_LIMITS = {
    'nudity': {'max': 0.05, 'min': -0.01},  # Placeholder values - update with actual data
    'non_nudity': {'max': 0.05, 'min': -0.01}  # Placeholder values - update with actual data
}

def load_latents_dict(latents_path):
    if latents_path.endswith('.pt') or latents_path.endswith('.pth'):
        return torch.load(latents_path)
    elif latents_path.endswith('.npy'):
        arr = np.load(latents_path, allow_pickle=True).item()
        return {k: torch.tensor(v) for k, v in arr.items()}
    elif latents_path.endswith('.pkl'):
        with open(latents_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file type for latents dict.")

def get_concept_y_limits(concept_name, padding_factor=0.05):
    """
    Get y-limits for a specific concept with padding.
    
    Args:
        concept_name (str): Name of the concept ('nudity' or 'non_nudity')
        padding_factor (float): Fraction of range to use as padding
    
    Returns:
        tuple: (y_min, y_max) with padding applied
    """
    if concept_name in CONCEPT_Y_LIMITS:
        limits = CONCEPT_Y_LIMITS[concept_name]
        min_val = limits['min']
        max_val = limits['max']
        
        # Calculate range and padding
        range_val = max_val - min_val
        if range_val > 0:
            padding = max(range_val * padding_factor, 0.001)
        else:
            padding = max(abs(max_val) * 0.1, 0.001)
        
        y_min = min_val - padding
        y_max = max_val + padding
        
        return y_min, y_max
    else:
        # Fallback: calculate from data if concept not in predefined limits
        print(f"Warning: Concept '{concept_name}' not found in predefined limits. Using data-based limits.")
        return None, None

def plot_concept_scores(all_scores, output_dir=None, save_format='pdf', plot_style='line', 
                        marker_size=6, line_width=2.5, use_predefined_limits=False, 
                        padding_factor=0.05):
    """
    Plot average scores for each concept with neuron index on x-axis and score on y-axis.
    Enhanced for LaTeX paper publication with better visibility and typography.
    
    Args:
        all_scores (dict): Dictionary with concept names as keys and lists of score arrays as values
        output_dir (str, optional): Directory to save plots. If None, plots are displayed
        save_format (str): Format to save plots ('png', 'pdf', 'svg', etc.)
        plot_style (str): 'line', 'scatter', or 'both'
        marker_size (float): Size of markers for scatter plots
        line_width (float): Width of lines for line plots
        use_predefined_limits (bool): Whether to use predefined y-limits
        padding_factor (float): Padding factor for y-limits
    """
    concepts = list(all_scores.keys())
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme for nudity categories
    color_map = {
        'nudity': '#d62728',  # Red
        'non_nudity': '#2ca02c'  # Green
    }
    
    # Individual plots for each concept (averaged across timesteps)
    for concept in concepts:
        timestep_scores = all_scores[concept]
        
        # Convert to numpy arrays and compute average across timesteps
        score_arrays = [np.array(scores) for scores in timestep_scores]
        avg_scores = np.mean(score_arrays, axis=0)
        neuron_indices = np.arange(len(avg_scores))
        
        # Create figure with appropriate size for papers
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use concept-specific color
        color = color_map.get(concept, '#2ca02c')
        
        if plot_style == 'line':
            ax.plot(neuron_indices, avg_scores, linewidth=line_width, color=color)
        elif plot_style == 'scatter':
            ax.scatter(neuron_indices, avg_scores, s=marker_size, alpha=0.7, color=color)
        elif plot_style == 'both':
            ax.plot(neuron_indices, avg_scores, linewidth=line_width, alpha=0.8, color=color)
            ax.scatter(neuron_indices, avg_scores, s=marker_size, alpha=0.9, color=color)
        
        # Enhanced title and labels
        ax.set_xlabel('Neuron Index', fontweight='bold')
        # ax.set_ylabel('Score', fontweight='bold')
        
        # Set y-limits based on predefined values or calculate from data
        if use_predefined_limits:
            y_min, y_max = get_concept_y_limits(concept, padding_factor)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
                print(f"Concept '{concept}': Using predefined y_limits = [{y_min:.6f}, {y_max:.6f}]")
            else:
                # Fallback to data-based calculation
                min_score = np.min(avg_scores)
                max_score = np.max(avg_scores)
                score_range = max_score - min_score
                if score_range > 0:
                    padding = max(score_range * padding_factor, 0.001)
                else:
                    padding = max(abs(max_score) * 0.1, 0.001)
                y_min = min_score - padding
                y_max = max_score + padding
                ax.set_ylim(y_min, y_max)
                print(f"Concept '{concept}': Using data-based y_limits = [{y_min:.6f}, {y_max:.6f}], range = {score_range:.6f}")
        else:
            # Original data-based calculation
            min_score = np.min(avg_scores)
            max_score = np.max(avg_scores)
            score_range = max_score - min_score
            if score_range > 0:
                padding = max(score_range * padding_factor, 0.001)
            else:
                padding = max(abs(max_score) * 0.1, 0.001)
            y_min = min_score - padding
            y_max = max_score + padding
            ax.set_ylim(y_min, y_max)
            print(f"Concept '{concept}': Using data-based y_limits = [{y_min:.6f}, {y_max:.6f}], range = {score_range:.6f}")
        
        # Enhanced grid - set this BEFORE removing ticks to ensure it's preserved
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.0)
        ax.set_axisbelow(True)
        
        # Remove y-ticks AFTER setting up the grid
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        
        # Add statistics with improved formatting
        mean_score = np.mean(avg_scores)
        max_score = np.max(avg_scores)
        min_score = np.min(avg_scores)
        max_neuron = np.argmax(avg_scores)
        
        # Thicker mean line
        ax.axhline(mean_score, color='red', linestyle='--', alpha=0.8, linewidth=2.0,
                   label=f'Mean: {mean_score:.4f}')
        
        # Add zero line if it's in the visible range
        current_ylim = ax.get_ylim()
        if current_ylim[0] <= 0 <= current_ylim[1]:
            ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.0)
        
        # Improved statistics text box
        textstr = (f'Max: {max_score:.4f} (neuron {max_neuron})\n'
                   f'Min: {min_score:.4f}')
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, 
                    edgecolor='black', linewidth=1.2)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=20,
               verticalalignment='top', bbox=props, family='monospace')
        
        # Improve tick formatting for x-axis only
        ax.tick_params(axis='x', which='major', length=6, width=1.2)
        ax.tick_params(axis='x', which='minor', length=4, width=1.0)
        
        # Add minor ticks
        ax.minorticks_on()
        
        plt.tight_layout()
        
        if output_dir:
            filename = f'{concept}_avg_scores.{save_format}'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none',
                       format=save_format, pad_inches=0.1)
            print(f"Saved average score plot for {concept} to {filepath}")
        else:
            plt.show()
        
        plt.close()

def plot_timestep_histograms(all_scores, concept, output_dir=None, save_format='pdf', bins=50, alpha=0.8):
    """
    Plot histograms for a specific concept across different timesteps.
    Enhanced for LaTeX paper publication.
    
    Args:
        all_scores (dict): Dictionary with concept names as keys and lists of score arrays as values
        concept (str): Specific concept to plot ('nudity' or 'non_nudity')
        output_dir (str, optional): Directory to save plots
        save_format (str): Format to save plots
        bins (int): Number of bins for histograms
        alpha (float): Transparency of histogram bars
    """
    if concept not in all_scores:
        print(f"Concept '{concept}' not found in scores.")
        return
    
    timestep_scores = all_scores[concept]
    num_timesteps = len(timestep_scores)
    
    # Create subplot grid with better spacing
    n_cols = min(4, num_timesteps)
    n_rows = (num_timesteps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle(f'Score Distributions Across Timesteps for {concept.replace("_", " ").title()}', 
                fontweight='bold', fontsize=18, y=0.98)
    
    if num_timesteps == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for timestep in range(num_timesteps):
        row = timestep // n_cols
        col = timestep % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        scores = timestep_scores[timestep]
        
        # Enhanced histogram with better styling
        n, bins_edges, patches = ax.hist(scores, bins=bins, alpha=alpha, 
                                        edgecolor='black', linewidth=1.2,
                                        color='#1f77b4', density=False)
        
        ax.set_title(f'Timestep {timestep}', fontweight='bold', pad=10)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_axisbelow(True)
        
        # Enhanced mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {mean_score:.4f}')
        
        # Add legend for each subplot
        ax.legend(loc='upper right', fontsize=10)
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', length=4, width=1.0)
    
    # Hide empty subplots
    for i in range(num_timesteps, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{concept}_timesteps_histogram.{save_format}'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none',
                   format=save_format, pad_inches=0.1)
        print(f"Saved timestep histograms for {concept} to {filepath}")
    else:
        plt.show()
    
    plt.close()

def main(args):
    # Load the latents dictionary
    latents_dict = load_latents_dict(args.latents_path)
    concepts = list(latents_dict.keys())
    num_timesteps = args.num_timesteps
    
    print(f"Found concepts: {concepts}")
    print(f"Processing {num_timesteps} timesteps")
    
    all_scores = {}
    for concept in concepts:
        print(f"\nComputing scores for '{concept}'...")
        all_scores[concept] = []
        for timestep in range(num_timesteps):
            scores = compute_feature_importance(latents_dict, concept, timestep)
            all_scores[concept].append(scores.cpu().tolist())
        print(f"  Completed {num_timesteps} timesteps")
    
    # Save scores to JSON
    output = {
        "concept_type": "nudity",
        "num_timesteps": num_timesteps,
        "concepts": concepts,
        "scores": all_scores
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved scores to {args.output_json}")
    
    # Generate score plots if requested
    if args.plot_scores:
        plot_output_dir = args.plot_output_dir if args.plot_output_dir else None
        
        print("\nGenerating concept score plots...")
        plot_concept_scores(
            all_scores, 
            output_dir=plot_output_dir,
            save_format=args.save_format,
            plot_style=args.plot_style,
            marker_size=args.marker_size,
            line_width=args.line_width,
            use_predefined_limits=args.use_predefined_limits,
            padding_factor=args.padding_factor
        )
        
        # Generate timestep histograms for specific concept if requested
        if args.plot_timesteps_concept:
            print(f"\nGenerating timestep histograms for {args.plot_timesteps_concept}...")
            plot_timestep_histograms(
                all_scores,
                args.plot_timesteps_concept,
                output_dir=plot_output_dir,
                save_format=args.save_format,
                bins=args.bins,
                alpha=args.alpha
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save nudity concept feature importance scores with enhanced plotting for LaTeX papers.")
    parser.add_argument("--model_checkpoint", type=str, required=False, 
                        help="Path to the model checkpoint (not used for score computation, but kept for clarity).")
    parser.add_argument("--latents_path", type=str, required=True, 
                        help="Path to the nudity latents dictionary (.pt, .pkl, or .npy).")
    parser.add_argument("--num_timesteps", type=int, required=True,
                        help="Number of timesteps to process per concept.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to output JSON file.")
    
    # Score plotting arguments with enhanced defaults
    parser.add_argument("--plot_scores", action="store_true",
                        help="Generate score plots for each concept (neuron index vs score).")
    parser.add_argument("--plot_output_dir", type=str, default=None,
                        help="Directory to save plots. If not specified, plots will be displayed.")
    parser.add_argument("--save_format", type=str, default="pdf", choices=["png", "pdf", "svg", "jpg"],
                        help="Format to save plots.")
    parser.add_argument("--plot_style", type=str, default="line", choices=["line", "scatter", "both"],
                        help="Style of plot: line, scatter, or both.")
    parser.add_argument("--marker_size", type=float, default=10.0,
                        help="Size of markers for scatter plots.")
    parser.add_argument("--line_width", type=float, default=3.5,
                        help="Width of lines for line plots.")
    parser.add_argument("--plot_timesteps_concept", type=str, default=None,
                        help="Generate timestep-specific histograms for a particular concept (e.g., 'nudity' or 'non_nudity').")
    parser.add_argument("--bins", type=int, default=50,
                        help="Number of bins for histograms.")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="Transparency for histogram bars.")
    
    # New arguments for y-limit control
    parser.add_argument("--use_predefined_limits", type=bool, default=False,
                        help="Use predefined y-limits for consistent scaling across concepts.")
    parser.add_argument("--padding_factor", type=float, default=0.05,
                        help="Padding factor for y-limits (fraction of range).")
    
    args = parser.parse_args()
    main(args)