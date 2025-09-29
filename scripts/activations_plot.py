#!/usr/bin/env python
"""
Simple SAE latent histogram generator - ignores styles, just loads random activations per object.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import random
import time

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.sae import Sae

def log_with_timestamp(message):
    """Print message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def load_object_datasets_simple(base_dirs, hookpoint, samples_per_object=1000, seed=42, dtype=torch.float32):
    """
    Simple loading: just grab random samples from each object directory, ignore styles completely.
    """
    log_with_timestamp("Starting simple dataset loading (ignoring styles)...")
    random.seed(seed)
    np.random.seed(seed)
    
    sampled_data = {}
    
    for base_dir in base_dirs:
        log_with_timestamp(f"Processing base directory: {base_dir}")
        base_path = Path(base_dir)
        hookpoint_dir = base_path / hookpoint
        
        if not hookpoint_dir.exists():
            log_with_timestamp(f"SKIP: Hookpoint directory does not exist: {hookpoint_dir}")
            continue
        
        log_with_timestamp(f"Found hookpoint directory: {hookpoint_dir}")
        
        # Find all object directories (skip metadata)
        concept_subdirs = [d for d in hookpoint_dir.iterdir() if d.is_dir() and d.name != 'metadata']
        log_with_timestamp(f"Found {len(concept_subdirs)} object directories")
        
        for i, concept_dir in enumerate(concept_subdirs):
            object_name = concept_dir.name
            log_with_timestamp(f"Processing object {i+1}/{len(concept_subdirs)}: {object_name}")
            
            if not (concept_dir / "dataset_info.json").exists():
                log_with_timestamp(f"  SKIP: No dataset_info.json in {concept_dir}")
                continue
            
            try:
                # Load the dataset
                log_with_timestamp(f"  Loading dataset from: {concept_dir}")
                start_time = time.time()
                dataset = HFDataset.load_from_disk(str(concept_dir), keep_in_memory=False)
                load_time = time.time() - start_time
                log_with_timestamp(f"  Loaded {len(dataset)} samples in {load_time:.2f}s")
                
                # Sample random indices
                n_samples = min(len(dataset), samples_per_object)
                if len(dataset) > samples_per_object:
                    # Randomly sample indices
                    sampled_indices = random.sample(range(len(dataset)), n_samples)
                    log_with_timestamp(f"  Randomly sampling {n_samples} out of {len(dataset)} samples")
                else:
                    # Use all samples
                    sampled_indices = list(range(len(dataset)))
                    log_with_timestamp(f"  Using all {n_samples} samples")
                
                # Extract the sampled activations
                start_time = time.time()
                sampled_dataset = dataset.select(sampled_indices)
                
                # Set format to torch
                sampled_dataset.set_format(
                    type="torch",
                    columns=["activations"],
                    dtype=dtype,
                )
                
                # Extract activations as tensor
                activations = torch.stack([item['activations'] for item in sampled_dataset])
                extract_time = time.time() - start_time
                
                sampled_data[object_name] = activations
                log_with_timestamp(f"  Extracted {activations.shape} in {extract_time:.2f}s")
                
            except Exception as e:
                log_with_timestamp(f"  ERROR loading {object_name}: {e}")
                continue
    
    log_with_timestamp(f"Simple loading completed! Got {len(sampled_data)} objects")
    return sampled_data

def compute_object_latent_activations_fast(sae, sampled_data, batch_size=64, device="cuda", dtype=torch.float32):
    """Fast computation of average latent activations using pre-sampled data."""
    log_with_timestamp("Computing latent activations...")
    model = sae.module if hasattr(sae, 'module') else sae
    model.eval()
    
    object_activations = {}
    
    with torch.no_grad():
        for i, (object_name, activations) in enumerate(sampled_data.items()):
            log_with_timestamp(f"Processing object {i+1}/{len(sampled_data)}: {object_name} ({activations.shape[0]} samples)")
            
            all_latent_activations = []
            n_batches = (activations.shape[0] + batch_size - 1) // batch_size
            
            # Process in batches
            for batch_idx in range(0, activations.shape[0], batch_size):
                batch_num = batch_idx // batch_size + 1
                if batch_num % max(1, n_batches // 4) == 0:  # Show progress every 25%
                    log_with_timestamp(f"    Batch {batch_num}/{n_batches}")
                
                batch_activations = activations[batch_idx:batch_idx+batch_size].to(device, dtype=dtype)
                
                try:
                    # Handle reshaping
                    if len(batch_activations.shape) == 3:
                        original_shape = batch_activations.shape
                        batch_activations = batch_activations.reshape(-1, batch_activations.shape[-1])
                    
                    # Get pre-activations from SAE
                    pre_acts = model.pre_acts(batch_activations)
                    
                    # Reshape back if needed
                    if len(original_shape) == 3:
                        batch_size_actual = original_shape[0]
                        seq_len = original_shape[1]
                        pre_acts = pre_acts.reshape(batch_size_actual, seq_len, -1)
                        pre_acts = pre_acts.mean(dim=1)  # Average over sequence
                    
                    # Apply softmax to get probabilities
                    probs = F.softmax(pre_acts, dim=1)
                    
                    # Store activations
                    all_latent_activations.append(probs.cpu())
                    
                except Exception as e:
                    log_with_timestamp(f"    ERROR processing batch {batch_num}: {e}")
                    continue
            
            if all_latent_activations:
                # Compute average across all samples for this object
                combined_activations = torch.cat(all_latent_activations, dim=0)
                avg_activations = combined_activations.mean(dim=0)
                object_activations[object_name] = avg_activations.numpy()
                log_with_timestamp(f"    Computed average activations: {avg_activations.shape}")
            else:
                log_with_timestamp(f"    ERROR: No valid activations for {object_name}")
    
    return object_activations

def create_individual_histograms(object_activations, output_dir, title_prefix="Validation Set", top_k=20):
    """Create individual PDF histograms for each object in separate directories."""
    log_with_timestamp(f"Creating individual histograms for {len(object_activations)} objects...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_with_timestamp(f"Output directory: {output_path}")
    
    # Create histogram for each object
    for object_name, activations in object_activations.items():
        log_with_timestamp(f"Creating histogram for: {object_name}")
        
        # Create object directory
        object_dir = output_path / object_name
        object_dir.mkdir(parents=True, exist_ok=True)
        
        # PDF path
        pdf_path = object_dir / f"{object_name}_latent_histogram.pdf"
        
        # CSV path for all latent activations
        csv_path = object_dir / f"{object_name}_all_latent_activations.csv"
        
        with PdfPages(pdf_path) as pdf:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create histogram
            n_latents = len(activations)
            latent_indices = np.arange(n_latents)
            
            bars = ax.bar(latent_indices, activations, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_title(f'{title_prefix}: {object_name}\n({n_latents} latents)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Latent Index', fontsize=14)
            ax.set_ylabel('Average Activation', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            max_activation = np.max(activations)
            max_latent = np.argmax(activations)
            mean_activation = np.mean(activations)
            std_activation = np.std(activations)
            
            # Statistics text box
            stats_text = f'Max: {max_activation:.6f} (Latent {max_latent})\n'
            stats_text += f'Mean: {mean_activation:.6f}\n'
            stats_text += f'Std: {std_activation:.6f}\n'
            stats_text += f'Total Latents: {n_latents}'
            
            ax.text(0.02, 0.98, stats_text, 
                   transform=ax.transAxes, verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
            
            # Highlight maximum bar
            bars[max_latent].set_color('red')
            bars[max_latent].set_alpha(0.9)
            
            # Set x-axis ticks
            if n_latents > 100:
                tick_step = max(1, n_latents // 20)
                ax.set_xticks(range(0, n_latents, tick_step))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save histogram page
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create detailed statistics pages (split if needed)
            top_k_actual = min(top_k, n_latents)
            top_indices = np.argsort(activations)[-top_k_actual:][::-1]  # Top k indices, descending
            
            # Calculate pages needed
            rows_per_page = 25
            n_pages = (top_k_actual + rows_per_page - 1) // rows_per_page
            
            for page_num in range(n_pages):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.axis('off')
                
                # Calculate range for this page
                start_idx = page_num * rows_per_page
                end_idx = min(start_idx + rows_per_page, top_k_actual)
                page_indices = top_indices[start_idx:end_idx]
                
                # Prepare data for this page
                stats_data = []
                for i, latent_idx in enumerate(page_indices):
                    rank = start_idx + i + 1
                    stats_data.append([
                        f"{rank}",
                        f"{latent_idx}",
                        f"{activations[latent_idx]:.6f}"
                    ])
                
                # Create table
                table = ax.table(cellText=stats_data,
                                colLabels=['Rank', 'Latent Index', 'Activation'],
                                cellLoc='center',
                                loc='center',
                                colWidths=[0.2, 0.3, 0.3])
                
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                
                # Style the table
                for (i, j), cell in table.get_celld().items():
                    if i == 0:  # Header row
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(color='white')
                    else:
                        cell.set_facecolor('#f1f1f2')
                        # Highlight rank 1 (only on first page)
                        if page_num == 0 and j == 0 and int(stats_data[i-1][0]) == 1:
                            cell.set_facecolor('#ffcccc')
                
                # Title with page info
                # if n_pages > 1:
                #     ax.set_title(f'{object_name} - Top {top_k_actual} Most Active Latents (Page {page_num + 1}/{n_pages})', 
                #                 fontsize=16, fontweight='bold', pad=20)
                # else:
                #     ax.set_title(f'{object_name} - Top {top_k_actual} Most Active Latents', 
                #                 fontsize=16, fontweight='bold', pad=20)
                
                # Save statistics page
                pdf.savefig(fig, bbox_inches='tight', dpi=300)
                plt.close()
        
        log_with_timestamp(f"  Saved PDF: {pdf_path}")
        
        # Save CSV with all latent activations
        log_with_timestamp(f"  Saving CSV with all {n_latents} latent activations...")
        
        # Create CSV data with all latents (sorted by activation value, descending)
        all_indices = np.argsort(activations)[::-1]  # All indices, highest activation first
        csv_data = []
        
        for rank, latent_idx in enumerate(all_indices, 1):
            csv_data.append([
                rank,
                latent_idx,
                activations[latent_idx]
            ])
        
        # Write to CSV
        import csv
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Rank', 'Latent_Index', 'Average_Activation'])  # Header
            writer.writerows(csv_data)
        
        log_with_timestamp(f"  Saved CSV: {csv_path}")
    
    log_with_timestamp(f"All histograms and CSV files saved to: {output_path}")

def main():
    """Main function to generate individual SAE latent activation histograms."""
    log_with_timestamp("=== Simple SAE Histogram Generator Started ===")
    
    parser = argparse.ArgumentParser(description="Generate individual histograms of SAE latent activations for objects (ignoring styles).")
    
    parser.add_argument("--sae_path", type=str, required=True, 
                       help="Path to the trained SAE model directory")
    parser.add_argument("--activations_dir", type=str, required=True,
                       help="Path to the activations directory")
    parser.add_argument("--hookpoint", type=str, required=True,
                       help="Name of the hookpoint/layer to analyze")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory (will create subdirs for each object)")
    parser.add_argument("--samples_per_object", type=int, default=1000,
                       help="Number of samples to use per object")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for processing")
    parser.add_argument("--top_k_latents", type=int, default=60,
                       help="Number of top latents to show in statistics table")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--use_float16", action="store_true",
                       help="Use float16 precision")
    
    args = parser.parse_args()
    
    log_with_timestamp(f"Arguments: {vars(args)}")
    
    # Set device and dtype
    device = torch.device(args.device)
    dtype = torch.float16 if args.use_float16 else torch.float32
    
    log_with_timestamp(f"Using device: {device}, dtype: {dtype}")
    
    # Load SAE model
    log_with_timestamp(f"Loading SAE model from: {args.sae_path}")
    try:
        start_time = time.time()
        sae = Sae.load_from_disk(args.sae_path, device=device)
        sae = sae.to(dtype=dtype)
        sae.eval()
        load_time = time.time() - start_time
        log_with_timestamp(f"Loaded SAE model with {sae.num_latents} latents in {load_time:.2f}s")
    except Exception as e:
        log_with_timestamp(f"FATAL: Failed to load SAE model: {e}")
        return
    
    # Load and sample data (simple approach)
    try:
        sampled_data = load_object_datasets_simple(
            [args.activations_dir], 
            args.hookpoint,
            samples_per_object=args.samples_per_object,
            seed=args.seed,
            dtype=dtype
        )
    except Exception as e:
        log_with_timestamp(f"FATAL: Failed to load and sample data: {e}")
        return
    
    if not sampled_data:
        log_with_timestamp("FATAL: No object data loaded")
        return
    
    # Compute latent activations
    try:
        object_activations = compute_object_latent_activations_fast(
            sae, 
            sampled_data, 
            batch_size=args.batch_size,
            device=device,
            dtype=dtype
        )
    except Exception as e:
        log_with_timestamp(f"FATAL: Failed to compute activations: {e}")
        return
    
    if not object_activations:
        log_with_timestamp("FATAL: No object activations computed")
        return
    
    # Create individual histograms
    try:
        create_individual_histograms(
            object_activations, 
            args.output_dir, 
            title_prefix=f"Random Samples ({args.hookpoint})",
            top_k=args.top_k_latents
        )
        log_with_timestamp("SUCCESS: Generated all histograms!")
        log_with_timestamp(f"   Output directory: {args.output_dir}")
        log_with_timestamp(f"   Objects analyzed: {len(object_activations)}")
        log_with_timestamp(f"   Latents per object: {len(next(iter(object_activations.values())))}")
    except Exception as e:
        log_with_timestamp(f"FATAL: Failed to create histograms: {e}")
        return
    
    log_with_timestamp("=== Simple SAE Histogram Generator Completed ===")

if __name__ == "__main__":
    main()