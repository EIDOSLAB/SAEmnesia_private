#!/usr/bin/env python
"""
Inspect the structure of pickle files in a directory to help 
understand their format for proper handling.
"""

import os
import sys
import pickle
from pathlib import Path
import torch
import numpy as np

def inspect_pickle_file(filepath):
    """Inspect the structure of a single pickle file."""
    print(f"\nInspecting file: {filepath}")
    
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print("Dictionary structure:")
            for key, value in data.items():
                value_type = type(value)
                value_shape = getattr(value, 'shape', 'N/A')
                
                if hasattr(value, 'dtype'):
                    value_dtype = value.dtype
                else:
                    value_dtype = 'N/A'
                
                print(f"  '{key}': type={value_type}, shape={value_shape}, dtype={value_dtype}")
                
                # If value is also a dict, peek inside
                if isinstance(value, dict) and len(value) < 10:
                    print(f"    Nested dictionary keys: {list(value.keys())}")
        
        elif isinstance(data, (torch.Tensor, np.ndarray)):
            print(f"Shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Mean value: {data.mean()}")
            print(f"Standard deviation: {data.std()}")
        
        else:
            print(f"Unrecognized data format: {type(data)}")
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_pickle.py <directory_or_file>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        inspect_pickle_file(path)
    elif path.is_dir():
        pkl_files = list(path.glob("*.pkl"))
        print(f"Found {len(pkl_files)} pickle files")
        
        # Inspect first 3 files (or fewer if there aren't that many)
        for i, pkl_file in enumerate(pkl_files[:3]):
            inspect_pickle_file(pkl_file)
    else:
        print(f"Path not found: {path}")

if __name__ == "__main__":
    main()