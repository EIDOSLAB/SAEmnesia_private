#!/usr/bin/env python3
import pickle
import os
import numpy as np
import torch
import argparse
import sys

def load_pickle(file_path):
    """Load a pickle file and return its contents."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_shape_info(obj, prefix=""):
    """Get shape information from tensor objects."""
    if isinstance(obj, torch.Tensor):
        return f"{prefix}Shape: {obj.shape}, Type: {obj.dtype}"
    elif isinstance(obj, np.ndarray):
        return f"{prefix}Shape: {obj.shape}, Type: {obj.dtype}"
    elif isinstance(obj, dict):
        results = [f"{prefix}Dict with {len(obj)} keys: {list(obj.keys())}"]
        for key, value in obj.items():
            results.append(get_shape_info(value, prefix=f"{prefix}  '{key}': "))
        return "\n".join(results)
    elif isinstance(obj, list):
        results = [f"{prefix}List with {len(obj)} elements"]
        if len(obj) > 0:
            first_elem = obj[0]
            results.append(get_shape_info(first_elem, prefix=f"{prefix}  First element: "))
        return "\n".join(results)
    else:
        return f"{prefix}Not a tensor: {type(obj)}"

def analyze_pickle(file_path, category_name):
    """Analyze the contents of a pickle file and print shape information."""
    print(f"Processing: {file_path}")
    data = load_pickle(file_path)
    
    if data is None:
        return
    
    print(f"\n=== {category_name} Shape Analysis ===")
    shape_info = get_shape_info(data)
    print(shape_info)

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Analyze activation shapes in pickle files.')
    parser.add_argument('--index-file', type=str, help='Path to index file mapping categories to file paths')
    parser.add_argument('--base-path', type=str, 
                        default="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.1/activations",
                        help='Base path for activation files')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=["Cats", "Dogs", "Human", "Trees", "Architectures"],
                        help='Categories to analyze (space separated)')
    parser.add_argument('--all', action='store_true', help='Analyze all available categories')
    
    args = parser.parse_args()
    
    # Dictionary to store category -> file path mappings
    category_paths = {}
    
    # Option 1: Parse the index file if specified
    if args.index_file and os.path.exists(args.index_file):
        print(f"Reading index from {args.index_file}")
        with open(args.index_file, 'r') as f:
            for line in f:
                if ':' in line:
                    category, path = line.strip().split(':', 1)
                    category_paths[category] = path
    # Option 2: Use base path and default categories
    else:
        print(f"Using base path: {args.base_path}")
        all_categories = [
            "Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", 
            "Fishes", "Flame", "Flowers", "Frogs", "Horses", "Human", 
            "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", 
            "Towers", "Trees", "Waterfalls"
        ]
        
        categories_to_map = all_categories if args.all else args.categories
        for category in categories_to_map:
            category_paths[category] = f"{args.base_path}/{category}.pkl"
    
    # Print info about what we're going to analyze
    print(f"Found {len(category_paths)} categories")
    categories_to_analyze = list(category_paths.keys()) if args.all else args.categories
    print(f"Will analyze {len(categories_to_analyze)} categories: {', '.join(categories_to_analyze)}")
    
    # Analyze the specified categories
    for category in categories_to_analyze:
        if category in category_paths:
            print(f"\nProcessing category: {category}")
            analyze_pickle(category_paths[category], category)
        else:
            print(f"Category '{category}' not found in the available paths")

if __name__ == "__main__":
    # Print script information
    print(f"Running activation shape analysis script")
    print(f"Start time: {np.datetime_as_string(np.datetime64('now'))}")
    print(f"Arguments: {' '.join(sys.argv[1:])}")
    
    main()
    
    print(f"\nEnd time: {np.datetime_as_string(np.datetime64('now'))}")