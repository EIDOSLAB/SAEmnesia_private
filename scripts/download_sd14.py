"""
Download Stable Diffusion 1.4 model locally for offline use.
Run this script on a machine with internet access.
"""
import sys
from diffusers import StableDiffusionPipeline
import torch

def download_model(save_directory):
    """
    Download SD 1.4 and save it locally.
    
    Args:
        save_directory: Path where the model will be saved
    """
    print(f"Downloading Stable Diffusion v1.4 to: {save_directory}")
    print("This may take several minutes depending on your connection...")
    
    try:
        # Download the model
        pipe = StableDiffusionPipeline.from_pretrained(
            # "CompVis/stable-diffusion-v1-4",
            "CompVis/stable-diffusion-v-1-4-original",
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # Save locally
        pipe.save_pretrained(save_directory)
        
        print(f"\n✅ Model successfully downloaded and saved to: {save_directory}")
        print(f"\nTo use this model in your script, set:")
        print(f"--model_name {save_directory}")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_sd14_model.py <save_directory>")
        print("Example: python download_sd14_model.py /path/to/models/sd-v1-4")
        sys.exit(1)
    
    save_dir = sys.argv[1]
    download_model(save_dir)