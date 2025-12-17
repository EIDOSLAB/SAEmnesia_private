#!/usr/bin/env python
"""
Compute average activation of assigned latents for each object.

This script:
1. Loads object-to-latent assignments from scores JSON files
2. Loads a subsample of activations for each object
3. Computes the average activation of the assigned latent for samples containing that object
4. Saves results to a JSON file
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

import torch
from datasets import Dataset as HFDataset, concatenate_datasets


def load_object_latent_assignments(scores_json_path):
    """
    Load object scores and determine assigned latent for each object.
    
    The assigned latent is the one with the highest average score across timesteps.
    """
    with open(scores_json_path, 'r') as f:
        scores_data = json.load(f)
    
    scores = scores_data.get('scores', {})
    object_to_latent = {}
    
    for concept_name, concept_scores in scores.items():
        # Handle both 2D (timestep x latent) and 1D (latent) score arrays
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            # 2D: Average across timesteps first
            avg_scores = np.mean(concept_scores, axis=0)
        else:
            # 1D: Already averaged or single values
            avg_scores = np.array(concept_scores)
        
        # Find the latent with highest score
        assigned_latent = int(np.argmax(avg_scores))
        object_to_latent[concept_name] = {
            'assigned_latent': assigned_latent,
            'assignment_score': float(avg_scores[assigned_latent])
        }
    
    return object_to_latent


def normalize_concept_name(name):
    """Convert between underscore and space formats for concept names."""
    return name.replace('_', ' ')


def find_concept_assignment(concept_name, object_to_latent):
    """Find concept in assignments dict, trying both original and normalized names."""
    # Try original name first
    if concept_name in object_to_latent:
        return concept_name, object_to_latent[concept_name]
    
    # Try with underscores replaced by spaces
    normalized_name = normalize_concept_name(concept_name)
    if normalized_name in object_to_latent:
        return normalized_name, object_to_latent[normalized_name]
    
    # Try with spaces replaced by underscores
    underscore_name = concept_name.replace(' ', '_')
    if underscore_name in object_to_latent:
        return underscore_name, object_to_latent[underscore_name]
    
    return None, None


def load_activations_for_hookpoint(activations_dir, hookpoint, max_samples_per_object=100, dtype=torch.float32):
    """
    Load a subsample of activations for each object from the dataset.
    
    Returns a dict mapping object_name -> list of activation tensors
    """
    base_path = Path(activations_dir)
    hookpoint_dir = base_path / hookpoint
    
    if not hookpoint_dir.exists():
        raise FileNotFoundError(f"Hookpoint directory does not exist: {hookpoint_dir}")
    
    object_activations = {}
    
    # Find all concept subdirectories
    concept_subdirs = [d for d in hookpoint_dir.iterdir() if d.is_dir() and d.name != 'metadata']
    
    print(f"Found {len(concept_subdirs)} concept directories in {hookpoint_dir}")
    
    for concept_dir in concept_subdirs:
        concept_name = concept_dir.name
        
        # Check if this is a valid HuggingFace dataset
        if not (concept_dir / "dataset_info.json").exists():
            print(f"  Skipping {concept_name} - not a valid dataset")
            continue
        
        print(f"  Loading concept '{concept_name}'...")
        
        try:
            # Load the dataset
            dataset = HFDataset.load_from_disk(str(concept_dir), keep_in_memory=False)
            
            # Subsample if needed
            num_samples = min(len(dataset), max_samples_per_object)
            if num_samples < len(dataset):
                # Random subsample
                indices = np.random.choice(len(dataset), num_samples, replace=False)
                dataset = dataset.select(indices)
            
            # Set format for torch
            dataset.set_format(type="torch", columns=["activations"], dtype=dtype)
            
            # Collect activations
            activations_list = []
            for i in range(len(dataset)):
                act = dataset[i]['activations']
                activations_list.append(act)
            
            object_activations[concept_name] = activations_list
            print(f"    Loaded {len(activations_list)} samples")
            
        except Exception as e:
            print(f"    Error loading {concept_name}: {e}")
            continue
    
    return object_activations


def compute_average_latent_activation(activations_list, latent_idx, sae_checkpoint_path=None, device="cuda"):
    """
    Compute the average activation of a specific latent across all samples.
    
    If sae_checkpoint_path is provided, run activations through the SAE to get latent activations.
    Otherwise, assume activations are already latent activations (pre_acts).
    """
    if not activations_list:
        return None
    
    # Stack all activations
    all_activations = torch.stack(activations_list)
    
    # If we have an SAE, use it to get latent activations
    if sae_checkpoint_path is not None:
        # Add parent directory to path for imports
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.dirname(script_dir))
        
        from SAE.sae import Sae
        
        sae = Sae.load_from_disk(sae_checkpoint_path, device=device)
        sae.eval()
        
        with torch.no_grad():
            # Move activations to device
            all_activations = all_activations.to(device)
            
            # Handle 3D activations (batch, timesteps, features)
            original_shape = all_activations.shape
            if len(original_shape) == 3:
                batch_size, seq_len, features = original_shape
                all_activations = all_activations.reshape(-1, features)
            
            # Get pre-activations (latent activations before top-k selection)
            pre_acts = sae.pre_acts(all_activations)
            
            # Reshape back if needed
            if len(original_shape) == 3:
                pre_acts = pre_acts.reshape(batch_size, seq_len, -1)
                # Average over timesteps
                pre_acts = pre_acts.mean(dim=1)
            
            # Get the activation of the specific latent
            latent_activations = pre_acts[:, latent_idx]
            
            # Compute average
            avg_activation = latent_activations.mean().item()
            std_activation = latent_activations.std().item()
            
            return {
                'mean': avg_activation,
                'std': std_activation,
                'num_samples': len(activations_list)
            }
    else:
        # Assume activations are raw model activations
        # We need the SAE to compute latent activations
        raise ValueError("SAE checkpoint path is required to compute latent activations from raw model activations")


def main():
    parser = argparse.ArgumentParser(
        description="Compute average activation of assigned latents for each object."
    )
    
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Path to the concept activations directory"
    )
    parser.add_argument(
        "--object_scores_path",
        type=str,
        required=True,
        help="Path to the JSON file containing pre-computed object scores"
    )
    parser.add_argument(
        "--sae_path",
        type=str,
        required=True,
        help="Path to the SAE checkpoint directory"
    )
    parser.add_argument(
        "--hookpoint",
        type=str,
        required=True,
        help="Name of the hookpoint to process"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="avg_latent_activations.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--max_samples_per_object",
        type=int,
        default=100,
        help="Maximum number of samples to use per object"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("Computing Average Latent Activations for Objects")
    print("=" * 60)
    
    # Step 1: Load object-to-latent assignments from scores
    print(f"\n1. Loading object-to-latent assignments from {args.object_scores_path}")
    object_to_latent = load_object_latent_assignments(args.object_scores_path)
    print(f"   Found {len(object_to_latent)} objects with latent assignments")
    
    # Step 2: Load activations for each object
    print(f"\n2. Loading activations from {args.activations_dir}/{args.hookpoint}")
    object_activations = load_activations_for_hookpoint(
        args.activations_dir,
        args.hookpoint,
        max_samples_per_object=args.max_samples_per_object
    )
    print(f"   Loaded activations for {len(object_activations)} objects")
    
    # Step 3: Compute average latent activation for each object
    print(f"\n3. Computing average latent activations using SAE from {args.sae_path}")
    
    results = {}
    
    for object_name, activations_list in object_activations.items():
        # Find the assigned latent for this object
        found_name, assignment = find_concept_assignment(object_name, object_to_latent)
        
        if assignment is None:
            print(f"   ⚠️  No latent assignment found for '{object_name}', skipping")
            continue
        
        assigned_latent = assignment['assigned_latent']
        assignment_score = assignment['assignment_score']
        
        print(f"   Processing '{object_name}' (assigned latent: {assigned_latent})...")
        
        try:
            stats = compute_average_latent_activation(
                activations_list,
                assigned_latent,
                sae_checkpoint_path=args.sae_path,
                device=args.device
            )
            
            if stats is not None:
                results[object_name] = {
                    'assigned_latent': assigned_latent,
                    'assignment_score': assignment_score,
                    'avg_activation': stats['mean'],
                    'std_activation': stats['std'],
                    'num_samples': stats['num_samples']
                }
                print(f"      Avg activation: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['num_samples']})")
            
        except Exception as e:
            print(f"      Error: {e}")
            continue
    
    # Step 4: Save results
    print(f"\n4. Saving results to {args.output_json}")
    
    output_data = {
        'metadata': {
            'activations_dir': str(args.activations_dir),
            'object_scores_path': str(args.object_scores_path),
            'sae_path': str(args.sae_path),
            'hookpoint': args.hookpoint,
            'max_samples_per_object': args.max_samples_per_object,
            'seed': args.seed
        },
        'results': results
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"   ✅ Saved results for {len(results)} objects")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Object':<25} {'Latent':<8} {'Avg Act':<12} {'Std':<12} {'Samples':<8}")
    print("-" * 65)
    
    for obj_name, data in sorted(results.items()):
        print(f"{obj_name:<25} {data['assigned_latent']:<8} {data['avg_activation']:<12.6f} {data['std_activation']:<12.6f} {data['num_samples']:<8}")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()