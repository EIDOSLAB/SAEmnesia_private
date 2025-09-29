#!/usr/bin/env python
"""
Activation Ecosystem Multipliers Algorithm

This approach analyzes the full activation distribution structure:
- Peak latents: Core concept features
- Supporting latents: Secondary features that can compensate
- Background latents: Context and style features

Key insight: Classes with good "support structure" are robust to gentle suppression,
while peak-dependent classes need aggressive suppression.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset as HFDataset
import json
from tqdm import tqdm

# Add parent directory to path for SAE imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

try:
    from SAE.sae import Sae
except ImportError:
    print("Error: Could not import SAE module. Make sure the SAE directory is in the parent directory.")
    sys.exit(1)


def load_class_data(data_path, hookpoint, class_name, max_samples=None, dtype=torch.float32):
    """Load data for a specific class."""
    base_path = Path(data_path)
    class_dir = base_path / hookpoint / class_name
    
    if not class_dir.exists():
        raise ValueError(f"Class directory does not exist: {class_dir}")
    
    print(f"Loading data for {class_name} from: {class_dir}")
    
    # Load the full dataset
    dataset = HFDataset.load_from_disk(str(class_dir), keep_in_memory=False)
    
    print(f"  Full dataset size: {len(dataset)}")
    
    # Limit samples if requested
    if max_samples is not None and len(dataset) > max_samples:
        print(f"  Limiting to {max_samples} samples")
        dataset = dataset.select(range(max_samples))
    
    # Set format for torch tensors
    dataset.set_format(
        type="torch",
        columns=["activations", "timestep"] + (["object_label"] if "object_label" in dataset.column_names else []),
        dtype=dtype,
    )
    
    return dataset


def get_latent_activations_batch(sae_model, activations):
    """Compute latent activations for a batch."""
    sae_model.eval()
    
    with torch.no_grad():
        # Handle different tensor shapes
        if len(activations.shape) == 3:
            # If 3D [batch, seq, features], reshape to 2D
            original_shape = activations.shape
            activations = activations.reshape(-1, activations.shape[-1])
            
            # Get pre-activations (latent activations)
            latent_acts = sae_model.pre_acts(activations)
            
            # Reshape back and take mean over sequence dimension
            latent_acts = latent_acts.reshape(original_shape[0], original_shape[1], -1)
            latent_acts = latent_acts.mean(dim=1)
            
        elif len(activations.shape) == 2:
            # If 2D [batch, features], directly compute
            latent_acts = sae_model.pre_acts(activations)
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
    
    return latent_acts


def compute_latent_statistics(sae_model, dataset, device, batch_size=32):
    """Compute comprehensive statistics for latent activations."""
    sae_model.eval()
    
    all_latent_acts = []
    total_samples = len(dataset)
    
    print(f"    Computing latent statistics across {total_samples} samples...")
    
    # Process in batches
    for i in tqdm(range(0, total_samples, batch_size), desc="    Processing batches"):
        end_idx = min(i + batch_size, total_samples)
        batch_indices = list(range(i, end_idx))
        
        try:
            # Get batch of samples
            batch_samples = [dataset[idx] for idx in batch_indices]
            batch_activations = torch.stack([sample['activations'] for sample in batch_samples]).to(device)
            
            # Get latent activations for the batch
            latent_acts = get_latent_activations_batch(sae_model, batch_activations)
            
            all_latent_acts.append(latent_acts.cpu())
            
        except Exception as e:
            print(f"    Error processing batch {i//batch_size}: {e}")
            continue
    
    if not all_latent_acts:
        raise ValueError("No valid samples were processed")
    
    # Concatenate all activations
    all_latent_acts = torch.cat(all_latent_acts, dim=0)
    print(f"    Processed {all_latent_acts.shape[0]} samples successfully")
    
    # Compute comprehensive statistics
    stats = {}
    
    # Basic statistics
    stats['mean'] = all_latent_acts.mean(dim=0)
    stats['std'] = all_latent_acts.std(dim=0)
    stats['max'] = all_latent_acts.max(dim=0)[0]
    stats['min'] = all_latent_acts.min(dim=0)[0]
    
    # Sparsity measures
    stats['sparsity'] = (all_latent_acts > 0.01).float().mean(dim=0)
    stats['global_sparsity'] = (all_latent_acts > 0.01).float().mean().item()
    
    # Summary statistics
    stats['total_samples'] = all_latent_acts.shape[0]
    stats['num_latents'] = all_latent_acts.shape[1]
    
    return stats


def analyze_activation_ecosystem(max_acts):
    """
    Analyze the full activation distribution structure.
    
    Key insight: The distribution of supporting activations determines
    how robust a concept is to suppression of peak activations.
    """
    
    max_val = max_acts.max().item()
    total_latents = max_acts.shape[0]
    
    # Define activation tiers based on peak strength
    peak_threshold = max_val * 0.7        # 70%+ of max - core concept features
    strong_threshold = max_val * 0.3      # 30-70% of max - strong supporting features  
    medium_threshold = max_val * 0.1      # 10-30% of max - medium supporting features
    weak_threshold = max_val * 0.03       # 3-10% of max - weak background features
    very_weak_threshold = max_val * 0.01  # 1-3% of max - noise level
    
    # Count latents in each tier
    peak_latents = (max_acts > peak_threshold).sum().item()
    strong_latents = ((max_acts > strong_threshold) & (max_acts <= peak_threshold)).sum().item()
    medium_latents = ((max_acts > medium_threshold) & (max_acts <= strong_threshold)).sum().item()
    weak_latents = ((max_acts > weak_threshold) & (max_acts <= medium_threshold)).sum().item()
    very_weak_latents = ((max_acts > very_weak_threshold) & (max_acts <= weak_threshold)).sum().item()
    silent_latents = (max_acts <= very_weak_threshold).sum().item()
    
    # Calculate ecosystem health metrics
    
    # Support ratio: How many supporting features per peak feature?
    support_ratio = (strong_latents + medium_latents) / max(peak_latents, 1)
    
    # Background density: What fraction of latents contribute to concept?
    active_latents = peak_latents + strong_latents + medium_latents + weak_latents
    background_density = active_latents / total_latents
    
    # Feature hierarchy: How well-structured is the feature hierarchy?
    if peak_latents > 0:
        hierarchy_score = (strong_latents / max(peak_latents, 1)) + (medium_latents / max(strong_latents + peak_latents, 1))
    else:
        hierarchy_score = 0
    
    # Compensation potential: Can lower tiers compensate for peak loss?
    compensation_strength = (strong_latents * 0.5 + medium_latents * 0.3 + weak_latents * 0.1) / max(peak_latents, 1)
    
    # Fragility score: How dependent is the concept on peak features?
    if total_latents > 0:
        peak_dominance = peak_latents / total_latents
        fragility_score = peak_dominance / (background_density + 1e-8)
    else:
        fragility_score = 0
    
    return {
        'max_activation': max_val,
        'thresholds': {
            'peak': peak_threshold,
            'strong': strong_threshold,
            'medium': medium_threshold,
            'weak': weak_threshold,
            'very_weak': very_weak_threshold
        },
        'tier_counts': {
            'peak_latents': peak_latents,
            'strong_latents': strong_latents,
            'medium_latents': medium_latents,
            'weak_latents': weak_latents,
            'very_weak_latents': very_weak_latents,
            'silent_latents': silent_latents,
            'total_active': active_latents
        },
        'ecosystem_metrics': {
            'support_ratio': support_ratio,
            'background_density': background_density,
            'hierarchy_score': hierarchy_score,
            'compensation_strength': compensation_strength,
            'fragility_score': fragility_score
        }
    }


def compute_ecosystem_multipliers(stats, class_name):
    """
    Compute multipliers based on activation ecosystem analysis.
    
    Core theory:
    - Peak-dependent concepts (low support) need strong suppression
    - Well-supported concepts (high compensation) work with gentle suppression  
    - Competing-features concepts need targeted suppression
    - Weak/diffuse concepts need careful gentle suppression
    """
    
    max_acts = stats['max']
    global_sparsity = stats['global_sparsity']
    
    # Analyze the activation ecosystem
    ecosystem = analyze_activation_ecosystem(max_acts)
    
    # Extract key metrics
    max_activation = ecosystem['max_activation']
    tier_counts = ecosystem['tier_counts']
    metrics = ecosystem['ecosystem_metrics']
    
    peak_latents = tier_counts['peak_latents']
    strong_latents = tier_counts['strong_latents']
    medium_latents = tier_counts['medium_latents']
    weak_latents = tier_counts['weak_latents']
    
    support_ratio = metrics['support_ratio']
    background_density = metrics['background_density']
    compensation_strength = metrics['compensation_strength']
    fragility_score = metrics['fragility_score']
    hierarchy_score = metrics['hierarchy_score']
    
    print(f"    Ecosystem analysis for {class_name}:")
    print(f"      Max activation: {max_activation:.3f}")
    print(f"      Peak latents (>70% max): {peak_latents}")
    print(f"      Strong latents (30-70% max): {strong_latents}")
    print(f"      Medium latents (10-30% max): {medium_latents}")
    print(f"      Weak latents (3-10% max): {weak_latents}")
    print(f"      Support ratio: {support_ratio:.3f}")
    print(f"      Background density: {background_density:.3f}")
    print(f"      Compensation strength: {compensation_strength:.3f}")
    print(f"      Fragility score: {fragility_score:.3f}")
    print(f"      Hierarchy score: {hierarchy_score:.3f}")
    
    # PATTERN CLASSIFICATION BASED ON ECOSYSTEM STRUCTURE
    
    # Pattern 1: Peak-Dependent / Fragile (Horses-like)
    # - Very few peak latents, low support, low compensation
    if (peak_latents <= 2 and support_ratio < 3.0 and 
        compensation_strength < 2.0 and max_activation > 15.0):
        multiplier = -5.0
        strategy = "peak_dependent_fragile"
        reasoning = f"Very fragile: {peak_latents} peak latents, support_ratio={support_ratio:.1f}"
        confidence = "high"
        
    # Pattern 2: Moderately Peak-Dependent
    # - Few peak latents but some support
    elif (peak_latents <= 3 and support_ratio < 5.0 and 
          compensation_strength < 4.0 and max_activation > 12.0):
        multiplier = -3.0
        strategy = "moderately_peak_dependent"
        reasoning = f"Moderately fragile: {peak_latents} peak latents, support_ratio={support_ratio:.1f}"
        confidence = "high"
        
    # Pattern 3: Competing Features (Architectures-like)
    # - Multiple peak latents with high support but poor hierarchy
    elif (peak_latents >= 4 and support_ratio > 8.0 and 
          hierarchy_score < 1.5 and max_activation > 15.0):
        multiplier = -2.5
        strategy = "competing_features"
        reasoning = f"Competing systems: {peak_latents} peak latents, poor hierarchy={hierarchy_score:.1f}"
        confidence = "high"
        
    # Pattern 4: Well-Supported Robust (Bears-like)
    # - Good peak latents with strong support structure
    elif (2 <= peak_latents <= 4 and support_ratio > 5.0 and 
          compensation_strength > 3.0 and max_activation > 20.0):
        multiplier = -1.0
        strategy = "well_supported_robust"
        reasoning = f"Robust: good support_ratio={support_ratio:.1f}, compensation={compensation_strength:.1f}"
        confidence = "high"
        
    # Pattern 5: Over-Distributed
    # - Too many features, unclear concept encoding
    elif (support_ratio > 15.0 and background_density > 0.3):
        multiplier = -0.8
        strategy = "over_distributed"
        reasoning = f"Over-distributed: support_ratio={support_ratio:.1f}, density={background_density:.1f}"
        confidence = "medium"
        
    # Pattern 6: Weak Concept
    # - Low peak activation or very sparse
    elif (max_activation < 10.0 or background_density < 0.05):
        multiplier = -0.5
        strategy = "weak_concept"
        reasoning = f"Weak: max_act={max_activation:.1f}, density={background_density:.1f}"
        confidence = "medium"
        
    # Pattern 7: Standard Balanced
    # - Reasonable structure, standard approach
    elif (peak_latents > 0 and support_ratio > 2.0 and max_activation > 8.0):
        if compensation_strength > 2.0:
            multiplier = -1.0
        else:
            multiplier = -1.5
        strategy = "standard_balanced"
        reasoning = f"Standard: support_ratio={support_ratio:.1f}, compensation={compensation_strength:.1f}"
        confidence = "medium"
        
    else:
        # Fallback based on activation strength
        if max_activation > 25.0:
            multiplier = -1.5
        elif max_activation > 15.0:
            multiplier = -1.0
        else:
            multiplier = -0.8
        strategy = "fallback"
        reasoning = f"Fallback: max_act={max_activation:.1f}"
        confidence = "low"
    
    # ADAPTIVE PERCENTILE SELECTION
    # More selective percentiles for stronger multipliers or fragile concepts
    if multiplier <= -3.0 or fragility_score > 0.5:
        percentile = 99.999   # Standard selective
    elif multiplier <= -2.0:
        percentile = 99.999   # Standard selective
    elif global_sparsity < 0.15:
        percentile = 99.99    # Less selective for sparse concepts
    else:
        percentile = 99.999   # Standard
    
    # BOUNDS AND ROUNDING
    multiplier = max(-10.0, min(-0.3, multiplier))
    multiplier = round(multiplier, 1)
    
    print(f"    → Strategy: {strategy}")
    print(f"    → Multiplier: {multiplier}, Percentile: {percentile}")
    print(f"    → Reasoning: {reasoning}")
    print(f"    → Confidence: {confidence}")
    
    result = {
        'multiplier': multiplier,
        'percentile': percentile,
        'strategy': strategy,
        'reasoning': reasoning,
        'confidence': confidence,
        'ecosystem_analysis': ecosystem,
        'decision_factors': {
            'pattern_detected': strategy,
            'key_metrics': {
                'peak_latents': peak_latents,
                'support_ratio': support_ratio,
                'compensation_strength': compensation_strength,
                'fragility_score': fragility_score,
                'background_density': background_density
            },
            'thresholds_used': {
                'peak_dependent': peak_latents <= 2 and support_ratio < 3.0,
                'competing_features': peak_latents >= 4 and support_ratio > 8.0,
                'well_supported': 2 <= peak_latents <= 4 and support_ratio > 5.0,
                'weak_concept': max_activation < 10.0 or background_density < 0.05
            }
        }
    }
    
    return result


def save_results(all_results, output_path):
    """Save results in both .pth and .json formats."""
    
    # Prepare data for .pth format (compatible with existing code)
    pth_data = {}
    for class_name, result in all_results.items():
        pth_data[class_name] = {
            'multiplier': result['multiplier'],
            'percentile': result['percentile']
        }
    
    # Save .pth file
    pth_path = output_path.with_suffix('.pth')
    torch.save(pth_data, pth_path)
    print(f"Saved .pth format to: {pth_path}")
    
    # Prepare data for .json format (detailed analysis)
    json_data = {}
    for class_name, result in all_results.items():
        json_data[class_name] = result
    
    # Save .json file
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved .json format to: {json_path}")


def main():
    """Main function to compute ecosystem-based adaptive multipliers."""
    
    parser = argparse.ArgumentParser(description="Compute ecosystem-based adaptive multipliers.")
    
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
        "--hookpoint", 
        type=str,
        default="unet.up_blocks.1.attentions.1",
        help="Hookpoint name (default: unet.up_blocks.1.attentions.1)"
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Output path for saving results (without extension)"
    )
    parser.add_argument(
        "--max_samples_per_class", 
        type=int,
        default=5000,
        help="Maximum samples per class for analysis (default: 5000)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--use_float16", 
        action="store_true",
        help="Use float16 precision"
    )
    
    args = parser.parse_args()
    
    # Set up device and dtype
    device = torch.device(args.device)
    dtype = torch.float16 if args.use_float16 else torch.float32
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    try:
        # Load SAE model
        print(f"Loading SAE model from: {args.model_path}")
        sae_model = Sae.load_from_disk(args.model_path, device=device)
        sae_model = sae_model.to(dtype=dtype)
        print(f"Model loaded successfully. Number of latents: {sae_model.num_latents}")
        
        # Find all available classes
        hookpoint_dir = Path(args.data_path) / args.hookpoint
        if not hookpoint_dir.exists():
            raise ValueError(f"Hookpoint directory does not exist: {hookpoint_dir}")
        
        class_dirs = [d for d in hookpoint_dir.iterdir() if d.is_dir() and not d.name.startswith('tmp')]
        class_names = [d.name for d in class_dirs]
        
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Process each class
        all_results = {}
        
        for i, class_name in enumerate(class_names):
            print(f"\n[{i+1}/{len(class_names)}] Processing class: {class_name}")
            
            try:
                # Load class data
                dataset = load_class_data(
                    args.data_path, 
                    args.hookpoint, 
                    class_name, 
                    max_samples=args.max_samples_per_class,
                    dtype=dtype
                )
                
                # Compute statistics
                stats = compute_latent_statistics(
                    sae_model, 
                    dataset, 
                    device, 
                    batch_size=args.batch_size
                )
                
                # Compute ecosystem-based parameters
                result = compute_ecosystem_multipliers(stats, class_name)
                
                # Store results
                all_results[class_name] = result
                
                print(f"  Final parameters:")
                print(f"    Multiplier: {result['multiplier']:.1f}")
                print(f"    Percentile: {result['percentile']:.5f}")
                print(f"    Strategy: {result['strategy']}")
                print(f"    Confidence: {result['confidence']}")
                
            except Exception as e:
                print(f"  Error processing {class_name}: {e}")
                continue
        
        # Save results
        if all_results:
            output_path = Path(args.output_path)
            save_results(all_results, output_path)
            
            print(f"\nProcessing completed successfully!")
            print(f"Processed {len(all_results)} classes")
            
            # Print summary
            print("\nSummary of computed parameters:")
            print("-" * 100)
            print(f"{'Class':<15} {'Multiplier':<10} {'Percentile':<12} {'Strategy':<25} {'Confidence':<10}")
            print("-" * 100)
            
            multipliers = []
            strategies = {}
            
            for class_name, result in all_results.items():
                mult = result['multiplier']
                perc = result['percentile']
                strategy = result['strategy']
                confidence = result['confidence']
                
                multipliers.append(mult)
                strategies[strategy] = strategies.get(strategy, 0) + 1
                
                print(f"{class_name:<15} {mult:<10.1f} {perc:<12.5f} {strategy:<25} {confidence:<10}")
            
            print(f"\nMultiplier distribution:")
            print(f"  Range: {min(multipliers):.1f} to {max(multipliers):.1f}")
            print(f"  Mean: {np.mean(multipliers):.2f}")
            print(f"  Std: {np.std(multipliers):.2f}")
            
            # Count by ranges
            very_strong = sum(1 for m in multipliers if m <= -3.0)
            strong = sum(1 for m in multipliers if -3.0 < m <= -2.0)
            moderate = sum(1 for m in multipliers if -2.0 < m <= -1.0)
            weak = sum(1 for m in multipliers if m > -1.0)
            
            print(f"  Very strong (≤-3.0): {very_strong} classes")
            print(f"  Strong (-3.0 to -2.0): {strong} classes")
            print(f"  Moderate (-2.0 to -1.0): {moderate} classes")
            print(f"  Weak (>-1.0): {weak} classes")
            
            print(f"\nStrategy distribution:")
            for strategy, count in sorted(strategies.items()):
                print(f"  {strategy}: {count} classes")
                
        else:
            print("No classes were processed successfully.")
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)