import os
import sys
import torch
import timm
from PIL import Image
from torchvision import transforms
import fire
import glob

# Add your project path
sys.path.append("")
from UnlearnCanvas_resources.const import theme_available


def test_image(image_path: str, style_ckpt: str, top_k: int = 5):
    """
    Test single or multiple images with the style classifier using glob patterns.
    
    Args:
        image_path: Path to image or glob pattern (e.g., "/path/to/*.jpg")
        style_ckpt: Path to the style classifier checkpoint
        top_k: Number of top predictions to show (default: 5)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Expand glob pattern - use recursive=True for better matching
    image_files = glob.glob(image_path, recursive=True)
    
    # If glob returns nothing, try alternative approaches
    if not image_files:
        # Check if it's a single file
        if os.path.exists(image_path) and os.path.isfile(image_path):
            image_files = [image_path]
        else:
            # Try manual pattern matching if glob fails
            dir_path = os.path.dirname(image_path)
            pattern = os.path.basename(image_path)
            
            if os.path.exists(dir_path):
                print(f"⚠️  glob.glob() returned no results, trying manual matching...")
                print(f"   Directory: {dir_path}")
                print(f"   Pattern: {pattern}")
                
                import fnmatch
                all_files = os.listdir(dir_path)
                image_files = [
                    os.path.join(dir_path, f) 
                    for f in all_files 
                    if fnmatch.fnmatch(f, pattern)
                ]
                
                if image_files:
                    print(f"✓ Found {len(image_files)} files using manual matching")
    
    if not image_files:
        print(f"Error: No images found matching pattern: {image_path}")
        print(f"\nDebugging info:")
        print(f"  - Pattern provided: {image_path}")
        print(f"  - Absolute path: {os.path.abspath(image_path)}")
        
        # Try to help debug by checking directory
        dir_path = os.path.dirname(image_path)
        if os.path.exists(dir_path):
            print(f"  - Directory exists: {dir_path}")
            all_files = os.listdir(dir_path)
            print(f"  - Files in directory: {len(all_files)}")
            if all_files:
                print(f"  - Sample files: {all_files[:5]}")
        else:
            print(f"  - Directory does not exist: {dir_path}")
        return
    
    image_files = sorted(image_files)
    print(f"✓ Found {len(image_files)} image(s) matching pattern")
    
    # Initialize model (once for all images)
    print(f"\n✓ Loading model on {device}...")
    style_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)
    
    style_model.head = torch.nn.Linear(1024, len(theme_available)).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(style_ckpt, map_location=device)
        style_model.load_state_dict(checkpoint["model_state_dict"])
        print("✓ Loaded checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    style_model.eval()
    
    # Transform image
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    print("\n" + "="*80)
    print("PROCESSING IMAGES")
    print("="*80)
    
    # Track correct predictions
    total_images = 0
    correct_predictions = 0
    
    # Process each image
    for img_idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"\n[{img_idx}/{len(image_files)}] 📁 {filename}")
        print(f"     Path: {img_path}")
        
        # Extract expected theme from filename (assumes format: Theme_Object_seedX.jpg)
        try:
            expected_theme = filename.split('_')[0]
            if '_' in filename:
                # Handle multi-word themes like "Artist_Sketch"
                parts = filename.replace('.jpg', '').split('_')
                # Find where the object class starts (everything before it is the theme)
                for i in range(len(parts)-2, 0, -1):  # -2 to skip seedX
                    potential_theme = '_'.join(parts[:i+1])
                    if potential_theme in theme_available:
                        expected_theme = potential_theme
                        break
        except:
            expected_theme = None
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"     ❌ Error loading image: {e}")
            continue
        
        total_images += 1
        image_tensor = image_transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = style_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_theme = theme_available[predicted_idx]
            predicted_prob = probabilities[predicted_idx].item()
        
        # Check if prediction is correct
        is_correct = (expected_theme and predicted_theme == expected_theme)
        if is_correct:
            correct_predictions += 1
        
        # Get top-k predictions
        top_k_actual = min(top_k, len(theme_available))
        sorted_indices = torch.argsort(probabilities, descending=True)[:top_k_actual]
        
        print(f"\n     {'✓ CORRECT' if is_correct else '✗ INCORRECT'} - Expected: {expected_theme if expected_theme else 'Unknown'}")
        print(f"     🎯 Top-{top_k_actual} Predictions:")
        print(f"     {'-'*60}")
        
        for rank, idx in enumerate(sorted_indices, 1):
            theme = theme_available[idx]
            prob = probabilities[idx].item()
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            # Mark the predicted, expected, or just rank
            if rank == 1:
                marker = "👈"
            elif expected_theme and theme == expected_theme:
                marker = "⭐"  # Expected theme
            else:
                marker = f"#{rank}"
            
            print(f"     {marker:3s} {theme:20s} {bar} {prob*100:6.2f}%")
        
        # Quick quality check
        entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum().item()
        if predicted_prob < 0.3:
            print(f"     ⚠️  Low confidence (entropy: {entropy:.2f})")
        elif predicted_prob > 0.9:
            print(f"     ✓  High confidence (entropy: {entropy:.2f})")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total images processed: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_images - correct_predictions}")
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    print("="*80)


if __name__ == "__main__":
    fire.Fire(test_image)