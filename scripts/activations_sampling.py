#!/usr/bin/env python
"""
Script to subsample concept activation tensors to reduce memory usage.
Takes large pickle files containing tensors and creates smaller versions by sampling a subset.
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import random

def subsample_concept_files(
    input_dir,
    output_dir,
    samples_per_concept=100000,
    seed=42,
    sample_strategy="random",
    dtype=torch.float16
):
    """
    Subsample large concept activation tensors to create more manageable files.
    
    Args:
        input_dir: Directory containing original pickle files
        output_dir: Directory to save subsampled files
        samples_per_concept: Number of samples to keep per concept (default: 100,000)
        seed: Random seed for reproducibility
        sample_strategy: Strategy for sampling ("random", "first", or "stride")
        dtype: Data type for the output tensors (default: float16)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of pickle files
    input_path = Path(input_dir)
    pkl_files = list(input_path.glob("*.pkl"))
    print(f"Found {len(pkl_files)} concept files in {input_dir}")
    
    # Process each file
    for pkl_file in tqdm(pkl_files, desc="Processing concept files"):
        concept_name = pkl_file.stem  # Get filename without extension
        
        try:
            # Load the original pickle file
            with open(pkl_file, "rb") as f:
                data_dict = pickle.load(f)
            
            # Extract the tensor
            if concept_name in data_dict:
                original_tensor = data_dict[concept_name]
            else:
                # Try the first key in the dictionary
                first_key = next(iter(data_dict.keys()), None)
                if first_key is None:
                    print(f"Empty dictionary for '{concept_name}', skipping")
                    continue
                
                original_tensor = data_dict[first_key]
                print(f"Using key '{first_key}' instead of '{concept_name}'")
            
            # Verify it's a tensor
            if not isinstance(original_tensor, torch.Tensor):
                if isinstance(original_tensor, np.ndarray):
                    original_tensor = torch.tensor(original_tensor)
                else:
                    print(f"Data for '{concept_name}' is not a tensor or numpy array: {type(original_tensor)}, skipping")
                    continue
            
            # Get original dimensions
            original_shape = original_tensor.shape
            print(f"Original tensor shape: {original_shape}")
            
            # Calculate how many samples to take
            n_samples = min(samples_per_concept, original_shape[0])
            
            # Sample the tensor according to the chosen strategy
            if sample_strategy == "random":
                # Random sampling (without replacement)
                indices = torch.randperm(original_shape[0])[:n_samples]
                sampled_tensor = original_tensor[indices]
            
            elif sample_strategy == "first":
                # Take the first n samples
                sampled_tensor = original_tensor[:n_samples]
            
            elif sample_strategy == "stride":
                # Take samples with a stride to cover the whole tensor
                stride = max(1, original_shape[0] // n_samples)
                indices = torch.arange(0, original_shape[0], stride)[:n_samples]
                sampled_tensor = original_tensor[indices]
            
            else:
                raise ValueError(f"Unknown sampling strategy: {sample_strategy}")
            
            # Convert to the desired dtype to save memory
            sampled_tensor = sampled_tensor.to(dtype)
            
            # Create the new dictionary with the same structure
            sampled_dict = {concept_name: sampled_tensor}
            
            # Save the subsampled tensor
            output_file = Path(output_dir) / pkl_file.name
            with open(output_file, "wb") as f:
                pickle.dump(sampled_dict, f)
            
            # Report stats
            memory_reduction = original_tensor.element_size() * original_tensor.numel() / (sampled_tensor.element_size() * sampled_tensor.numel())
            print(f"Processed '{concept_name}': {original_shape} → {sampled_tensor.shape}, {memory_reduction:.1f}x memory reduction")
            
            # Free memory
            del original_tensor
            del sampled_tensor
            del data_dict
            del sampled_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error processing '{concept_name}': {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Subsample concept activation tensors to reduce memory usage.')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Directory containing original pickle files')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save subsampled files')
    parser.add_argument('--samples', type=int, default=100000, 
                        help='Number of samples to keep per concept')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--strategy', type=str, choices=['random', 'first', 'stride'], default='random',
                        help='Strategy for sampling: random, first, or stride')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'], default='float16',
                        help='Data type for output tensors')
    
    args = parser.parse_args()
    
    # Map string dtype to torch dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    
    print(f"Subsampling concept files:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Samples per concept: {args.samples}")
    print(f"  Sampling strategy: {args.strategy}")
    print(f"  Output dtype: {args.dtype}")
    print(f"  Random seed: {args.seed}")
    
    # Run the subsampling
    subsample_concept_files(
        args.input_dir,
        args.output_dir,
        samples_per_concept=args.samples,
        seed=args.seed,
        sample_strategy=args.strategy,
        dtype=dtype_map[args.dtype]
    )
    
    print(f"\nSubsampling complete! Subsampled files saved to {args.output_dir}")

if __name__ == "__main__":
    main()