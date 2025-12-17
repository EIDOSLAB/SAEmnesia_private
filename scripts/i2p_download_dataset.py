#!/usr/bin/env python3
"""
Script to download AIML-TUDA/i2p dataset from Hugging Face
and save it to /leonardo_scratch/fast/IscrC_SAOU/
"""

from datasets import load_dataset
from datasets.config import HF_DATASETS_CACHE
import os

def download_i2p_dataset():
    """Download and save the AIML-TUDA/i2p dataset"""
    
    # Define the save directory
    base_dir = "/leonardo_scratch/fast/IscrC_SAOU"
    save_dir = os.path.join(base_dir, "i2p_dataset")
    cache_dir = os.path.join(base_dir, "hf_cache")
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading AIML-TUDA/i2p dataset...")
    print(f"Save directory: {save_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Default HF cache: {HF_DATASETS_CACHE}")
    
    try:
        # Load the dataset with streaming to avoid disk space check
        print("\nAttempting to load dataset with streaming first...")
        dataset = load_dataset(
            "AIML-TUDA/i2p",
            cache_dir=cache_dir,
            download_mode="force_redownload"
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Save the dataset to disk
        print(f"\nSaving dataset to {save_dir}...")
        dataset.save_to_disk(save_dir)
        
        print(f"\nDataset saved successfully!")
        print(f"\nDataset splits available:")
        for split in dataset.keys():
            print(f"  - {split}: {len(dataset[split])} examples")
            
        # Clean up cache if desired
        print(f"\nCache files are in: {cache_dir}")
        print(f"You can delete the cache directory after verifying the download.")
            
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nTroubleshooting steps:")
        print("  1. Check your home directory quota: quota -s")
        print("  2. Try setting: export HF_HOME=/leonardo_scratch/fast/IscrC_SAOU/hf_cache")
        print("  3. Try setting: export TMPDIR=/leonardo_scratch/fast/IscrC_SAOU/tmp")
        print("  4. Check if dataset requires authentication")
        return False
    
    return True

if __name__ == "__main__":
    # Set environment variables before importing
    base_dir = "/leonardo_scratch/fast/IscrC_SAOU"
    os.environ['HF_HOME'] = os.path.join(base_dir, "hf_cache")
    os.environ['HF_DATASETS_CACHE'] = os.path.join(base_dir, "hf_cache")
    os.environ['TMPDIR'] = os.path.join(base_dir, "tmp")
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(base_dir, "hf_cache")
    
    # Create tmp directory
    os.makedirs(os.environ['TMPDIR'], exist_ok=True)
    
    print("=" * 60)
    print("AIML-TUDA/i2p Dataset Downloader")
    print("=" * 60)
    print(f"\nEnvironment variables set:")
    print(f"  HF_HOME: {os.environ['HF_HOME']}")
    print(f"  TMPDIR: {os.environ['TMPDIR']}")
    print()
    
    success = download_i2p_dataset()
    
    if success:
        print("\n✓ Download completed successfully!")
    else:
        print("\n✗ Download failed. Please check the error messages above.")