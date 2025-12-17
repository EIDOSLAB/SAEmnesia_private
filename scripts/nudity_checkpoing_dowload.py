#!/usr/bin/env python3
"""
Script to download SAeUron nudity model from HuggingFace
"""

from huggingface_hub import snapshot_download
import os

# Configuration
MODEL_NAME = "bcywinski/SAeUron_nudity"
OUTPUT_DIR = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity"

def download_model():
    """Download all model files from HuggingFace using snapshot_download"""
    
    # Create parent directory if it doesn't exist
    parent_dir = os.path.dirname(OUTPUT_DIR)
    os.makedirs(parent_dir, exist_ok=True)
    
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("\nThis may take a few minutes depending on model size...")
    
    try:
        # Download entire repository
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=OUTPUT_DIR,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        print(f"\n✓ Model successfully downloaded to {OUTPUT_DIR}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, OUTPUT_DIR)
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  - {rel_path} ({size_mb:.2f} MB)")
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        raise

if __name__ == "__main__":
    download_model()