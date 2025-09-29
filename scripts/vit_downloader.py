import os
import sys
import torch
import timm
import fire

def download_models(save_dir="./cache"):
    """
    Download the required ViT-Large model to a specified directory.
    
    Args:
        save_dir (str): Directory where models will be cached. Default is './cache'
    """
    # Create the save directory if it doesn't exist
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Set PyTorch hub directory
    torch.hub.set_dir(save_dir)
    
    print(f"Cache directory set to: {save_dir}")
    print("Downloading models...")
    
    try:
        # Download the ViT-Large model used in your script
        model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)
        print("Successfully downloaded: vit_large_patch16_224.augreg_in21k")
        
        # Print model info
        print(f"Model type: {type(model)}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)
    
    print("\nChecking cache contents...")
    for root, dirs, files in os.walk(save_dir):
        level = root.replace(save_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\nDone! The model is now cached in: {save_dir}")
    print("\nNote: Your script also requires custom checkpoint files:")
    print("- A style classification checkpoint (style_ckpt)")
    print("- A class classification checkpoint (class_ckpt)")
    print("These need to be provided separately when running your main script.")

if __name__ == "__main__":
    fire.Fire(download_models)