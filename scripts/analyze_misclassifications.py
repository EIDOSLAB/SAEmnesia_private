import os
import torch
import timm
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import fire

# Add your path
import sys
sys.path.append("")
from UnlearnCanvas_resources.const import class_available_ordered, theme_available

class_available = class_available_ordered
class_available_full = class_available  # Adjust if needed
theme_available_full = theme_available  # Adjust if needed

def analyze_misclassifications(
    unlearn_class,
    images_dir,
    class_ckpt,
    style_ckpt,
    seed=188,
):
    """
    Analyze what the classifier sees when it should see 'unlearn_class'
    
    Args:
        unlearn_class: The class being unlearned (e.g., "Bears")
        images_dir: Directory containing the generated images (e.g., .../replace_with_neighbor/Bears/)
        class_ckpt: Path to class classifier checkpoint
        style_ckpt: Path to style classifier checkpoint
        seed: Seed used for image generation
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    class_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
    style_model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True).to(device)
    
    class_model.head = torch.nn.Linear(1024, len(class_available_full)).to(device)
    style_model.head = torch.nn.Linear(1024, len(theme_available_full)).to(device)
    
    class_checkpoint = torch.load(class_ckpt, map_location=device, weights_only=False)
    style_checkpoint = torch.load(style_ckpt, map_location=device, weights_only=False)
    
    class_model.load_state_dict(class_checkpoint["model_state_dict"])
    style_model.load_state_dict(style_checkpoint["model_state_dict"])
    
    class_model.eval()
    style_model.eval()
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Results storage
    results = {
        "unlearn_class": unlearn_class,
        "total_images": 0,
        "correctly_classified": 0,
        "misclassifications_by_style": {},
        "detailed_results": []
    }
    
    # For each style, check what the classifier predicts
    for style in theme_available:
        if style == "Seed_Images":
            continue
            
        img_path = os.path.join(images_dir, f"{style}_{unlearn_class}_seed{seed}.jpg")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image_tensor = image_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Class prediction
                class_output = class_model(image_tensor)
                class_probs = torch.nn.functional.softmax(class_output, dim=1)
                class_pred_idx = torch.argmax(class_output, dim=1).item()
                class_pred = class_available[class_pred_idx]
                class_confidence = class_probs[0][class_pred_idx].item()
                
                # Get probability for the target (unlearned) class
                unlearn_class_idx = class_available.index(unlearn_class)
                unlearn_class_prob = class_probs[0][unlearn_class_idx].item()
                
                # Style prediction
                style_output = style_model(image_tensor)
                style_probs = torch.nn.functional.softmax(style_output, dim=1)
                style_pred_idx = torch.argmax(style_output, dim=1).item()
                style_pred = theme_available[style_pred_idx]
                style_confidence = style_probs[0][style_pred_idx].item()
                
                # Get probability for the target style
                style_idx = theme_available.index(style)
                style_prob = style_probs[0][style_idx].item()
                
                # Get top 5 class predictions
                top5_class_probs, top5_class_indices = torch.topk(class_probs[0], 5)
                top5_classes = [(class_available[idx.item()], prob.item()) 
                               for idx, prob in zip(top5_class_indices, top5_class_probs)]
                
                results["total_images"] += 1
                is_correct = (class_pred == unlearn_class)
                
                if is_correct:
                    results["correctly_classified"] += 1
                
                if style not in results["misclassifications_by_style"]:
                    results["misclassifications_by_style"][style] = defaultdict(int)
                
                results["misclassifications_by_style"][style][class_pred] += 1
                
                detail = {
                    "image_path": img_path,
                    "style": style,
                    "expected_class": unlearn_class,
                    "predicted_class": class_pred,
                    "class_confidence": class_confidence,
                    "unlearn_class_probability": unlearn_class_prob,
                    "is_correct": is_correct,
                    "expected_style": style,
                    "predicted_style": style_pred,
                    "style_confidence": style_confidence,
                    "target_style_probability": style_prob,
                    "top5_classes": top5_classes
                }
                
                results["detailed_results"].append(detail)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Print summary
    print("\n" + "="*80)
    print(f"MISCLASSIFICATION ANALYSIS FOR: {unlearn_class}")
    print("="*80)
    print(f"Total images analyzed: {results['total_images']}")
    print(f"Correctly classified as {unlearn_class}: {results['correctly_classified']} ({results['correctly_classified']/results['total_images']*100:.2f}%)")
    print(f"Misclassified: {results['total_images'] - results['correctly_classified']} ({(results['total_images'] - results['correctly_classified'])/results['total_images']*100:.2f}%)")
    
    # Overall misclassification summary
    print("\n" + "-"*80)
    print("WHAT THE CLASSIFIER IS SEEING (Overall):")
    print("-"*80)
    overall_predictions = defaultdict(int)
    for style_preds in results["misclassifications_by_style"].values():
        for pred_class, count in style_preds.items():
            overall_predictions[pred_class] += count
    
    sorted_predictions = sorted(overall_predictions.items(), key=lambda x: x[1], reverse=True)
    for pred_class, count in sorted_predictions:
        percentage = count / results['total_images'] * 100
        marker = " ← TARGET CLASS (FAILED UNLEARNING)" if pred_class == unlearn_class else ""
        print(f"  {pred_class:.<25} {count:>3} times ({percentage:>5.1f}%){marker}")
    
    # Show images that were correctly classified (FAILED unlearning)
    correctly_classified_images = [d for d in results["detailed_results"] if d["is_correct"]]
    if correctly_classified_images:
        print("\n" + "="*80)
        print(f"FAILED UNLEARNING: Images still classified as {unlearn_class}")
        print("="*80)
        print(f"Total: {len(correctly_classified_images)} images ({len(correctly_classified_images)/results['total_images']*100:.1f}%)")
        print("\nStyles where unlearning FAILED:")
        print("-"*80)
        for detail in correctly_classified_images:
            print(f"\n  Style: {detail['style']}")
            print(f"    Image: {detail['image_path']}")
            print(f"    Confidence: {detail['class_confidence']*100:.1f}%")
            print(f"    Top 5 predictions:")
            for cls, prob in detail['top5_classes']:
                marker = " ← FAILED" if cls == unlearn_class else ""
                print(f"      {cls:.<20} {prob*100:>5.1f}%{marker}")
    
    # Per-style breakdown
    print("\n" + "-"*80)
    print("MISCLASSIFICATIONS BY STYLE (Successful Unlearning):")
    print("-"*80)
    
    problematic_styles = []
    for style in sorted(results["misclassifications_by_style"].keys()):
        preds = results["misclassifications_by_style"][style]
        if len(preds) > 0:
            # Find what it was classified as
            pred_class = list(preds.keys())[0]  # Should only be one per style
            count = preds[pred_class]
            
            if pred_class != unlearn_class:
                problematic_styles.append((style, pred_class))
                print(f"  {style:.<30} → Classified as: {pred_class}")
    
    # Show detailed info for misclassified cases (successful unlearning)
    if problematic_styles:
        print("\n" + "-"*80)
        print("DETAILED ANALYSIS OF MISCLASSIFIED IMAGES (Successful Unlearning):")
        print("-"*80)
        
        for detail in results["detailed_results"]:
            if not detail["is_correct"]:
                print(f"\nStyle: {detail['style']}")
                print(f"  Image: {detail['image_path']}")
                print(f"  Predicted as: {detail['predicted_class']} (confidence: {detail['class_confidence']*100:.1f}%)")
                print(f"  {unlearn_class} probability: {detail['unlearn_class_probability']*100:.1f}%")
                print(f"  Top 5 predictions:")
                for cls, prob in detail['top5_classes']:
                    marker = " ← " if cls == unlearn_class else ""
                    print(f"    {cls:.<20} {prob*100:>5.1f}%{marker}")
    
    return results

if __name__ == "__main__":
    fire.Fire(analyze_misclassifications)