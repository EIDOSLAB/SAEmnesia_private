"""
Extremely memory-efficient recovery script that processes files in chunks
and handles each class completely separately to avoid OOM errors.
Saves activations in an activations directory with one pickle per class.
"""

import os
import sys
import torch
import pickle
import gc
import argparse
import time
import psutil
from tqdm import tqdm

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")

def load_tensor_from_disk_with_retry(filepath, max_retries=3, chunk_size=None):
    """Load a tensor from disk using torch.load with retry logic and optional chunking"""
    for attempt in range(max_retries):
        try:
            if chunk_size is not None:
                # If chunk_size is provided, use memory-mapped loading
                return torch.load(filepath, map_location=torch.device('cpu'))
            else:
                return torch.load(filepath)
        except Exception as e:
            print(f"Error loading {filepath} (attempt {attempt+1}/{max_retries}): {e}")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)  # Wait before retrying
    
    raise RuntimeError(f"Failed to load tensor after {max_retries} attempts")

def process_individual_class(class_name, temp_file_path, output_dir, hookpoint, file_type):
    """Process a single class and save it directly to its own pickle file"""
    print(f"\nProcessing {class_name}...")
    
    try:
        # Create appropriate output directory based on file_type
        if file_type == "activations":
            indiv_dir = os.path.join(output_dir, "activations")
        else:
            indiv_dir = os.path.join(output_dir, f"individual_{file_type}")
        
        os.makedirs(indiv_dir, exist_ok=True)
        
        # For activations, use pickle instead of .pt
        if file_type == "activations":
            output_path = os.path.join(indiv_dir, f"{class_name}.pkl")
        else:
            output_path = os.path.join(indiv_dir, f"{class_name}.pt")
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"Skipping {class_name} - already processed")
            return
        
        print_memory_usage()
        print(f"Loading tensor from {temp_file_path}")
        
        # Load tensor with chunking for activations (which are likely larger)
        if file_type == "activations":
            tensor = load_tensor_from_disk_with_retry(temp_file_path, chunk_size=1000)
        else:
            tensor = load_tensor_from_disk_with_retry(temp_file_path)
            
        print(f"Loaded tensor with shape {tensor.shape}")
        print_memory_usage()
        
        # Save to individual file - for activations use pickle directly
        print(f"Saving to {output_path}")
        if file_type == "activations":
            with open(output_path, "wb") as f:
                pickle.dump({class_name: tensor}, f)
        else:
            torch.save(tensor, output_path)
        
        # Add reference to main reference file
        dict_path = os.path.join(output_dir, f"cls_{file_type}_dict_{hookpoint}.pkl.refs")
        with open(dict_path, "a") as f:
            f.write(f"{class_name}:{output_path}\n")
            
        print(f"Successfully processed {class_name}")
        
        # Important: clear memory
        del tensor
        gc.collect()
        torch.cuda.empty_cache()
        print_memory_usage()
        
    except Exception as e:
        print(f"Error processing {class_name}: {e}")
        return False
    
    return True

def combine_individual_files_to_dict(output_dir, hookpoint, file_type):
    """Combine individual files into a dictionary - skip for activations"""
    # For activations, we don't need to combine since each class has its own pickle
    if file_type == "activations":
        print("Skipping combination for activations as each class already has its own pickle file")
        return True
    
    print(f"\nCombining individual {file_type} files into dictionary...")
    
    refs_file = os.path.join(output_dir, f"cls_{file_type}_dict_{hookpoint}.pkl.refs")
    if not os.path.exists(refs_file):
        print(f"No reference file found at {refs_file}")
        return False
    
    # Read references
    class_paths = {}
    with open(refs_file, "r") as f:
        for line in f:
            if ":" in line:
                class_name, path = line.strip().split(":", 1)
                class_paths[class_name] = path
    
    print(f"Found {len(class_paths)} classes to combine")
    
    # Combine into dictionary one by one
    output_path = os.path.join(output_dir, f"cls_{file_type}_dict_{hookpoint}.pkl")
    
    # Initialize with empty dict
    with open(output_path, "wb") as f:
        pickle.dump({}, f)
    
    # Add each class
    for class_name, path in tqdm(class_paths.items()):
        try:
            # Load current dict
            with open(output_path, "rb") as f:
                class_dict = pickle.load(f)
            
            # Add new class
            if os.path.exists(path):
                class_dict[class_name] = torch.load(path)
                
                # Save updated dict
                with open(output_path, "wb") as f:
                    pickle.dump(class_dict, f)
                    
                print(f"Added {class_name} to dictionary")
            else:
                print(f"Warning: File {path} not found for class {class_name}")
            
            # Clear memory
            del class_dict
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error adding {class_name} to dictionary: {e}")
    
    print(f"Combined dictionary saved to {output_path}")
    return True

def create_master_activations_index(output_dir, hookpoint):
    """Create a master index file for activations directory"""
    activations_dir = os.path.join(output_dir, "activations")
    if not os.path.exists(activations_dir):
        print(f"Activations directory {activations_dir} does not exist")
        return False
    
    # List all pickle files in the activations directory
    pickle_files = [f for f in os.listdir(activations_dir) if f.endswith('.pkl')]
    
    if not pickle_files:
        print("No activation pickle files found")
        return False
    
    # Create index file
    index_path = os.path.join(output_dir, f"activations_index_{hookpoint}.txt")
    with open(index_path, "w") as f:
        for pickle_file in sorted(pickle_files):
            class_name = pickle_file.replace('.pkl', '')
            f.write(f"{class_name}:{os.path.join(activations_dir, pickle_file)}\n")
    
    print(f"Created master activations index at {index_path} with {len(pickle_files)} entries")
    return True

def main():
    parser = argparse.ArgumentParser(description="Super memory-efficient recovery script for SAE activations/latents")
    parser.add_argument("--temp_dir", type=str, required=True, help="Directory containing temporary tensor files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the final dictionary")
    parser.add_argument("--hookpoint", type=str, required=True, help="Hookpoint name for filename")
    parser.add_argument("--file_type", type=str, choices=["activations", "latents"], required=True, 
                        help="Type of file to recover: 'activations' or 'latents'")
    parser.add_argument("--combine_only", action="store_true", help="Skip individual processing and only combine existing files")
    parser.add_argument("--process_only", action="store_true", help="Only process individual files, don't combine into dictionary")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of classes to process (for testing)")
    args = parser.parse_args()
    
    # Check if temp directory exists
    if not os.path.exists(args.temp_dir):
        print(f"Error: Temporary directory {args.temp_dir} does not exist.")
        return
    
    # Process individual files
    if not args.combine_only:
        # Find all temporary files of the requested type
        temp_files = [f for f in os.listdir(args.temp_dir) if f.endswith(f"_{args.file_type}.pt")]
        
        if not temp_files:
            print(f"No temporary {args.file_type} files found in {args.temp_dir}")
            return
        
        print(f"Found {len(temp_files)} temporary {args.file_type} files.")
        
        # Clear reference file
        refs_file = os.path.join(args.save_dir, f"cls_{args.file_type}_dict_{args.hookpoint}.pkl.refs")
        if os.path.exists(refs_file):
            # Backup existing file
            os.rename(refs_file, f"{refs_file}.bak")
        
        # Process limited number of files if specified
        if args.limit:
            temp_files = temp_files[:args.limit]
            print(f"Limited to processing {args.limit} files")
        
        # Process each class completely separately
        for temp_file in temp_files:
            # Extract class name from filename
            class_name = temp_file.split('_')[0]
            file_path = os.path.join(args.temp_dir, temp_file)
            
            process_individual_class(class_name, file_path, args.save_dir, args.hookpoint, args.file_type)
            
            # Force memory cleanup after each class
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)  # Short pause between processing classes
    
    # Combine individual files into dictionary (skipped for activations)
    if not args.process_only:
        combine_individual_files_to_dict(args.save_dir, args.hookpoint, args.file_type)
        
        # For activations, create a master index file instead
        if args.file_type == "activations":
            create_master_activations_index(args.save_dir, args.hookpoint)

if __name__ == "__main__":
    # Initialize psutil for memory monitoring
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not installed. Memory usage reporting will be disabled.")
        def print_memory_usage():
            pass
    
    main()