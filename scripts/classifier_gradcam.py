#!/usr/bin/env python3
"""
Simple GradCAM for ViT Classifier
Usage: python simple_gradcam.py image.jpg --class_ckpt model.pth
"""

import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# Add your project path
sys.path.append("")
from UnlearnCanvas_resources.const import class_available


def load_model(ckpt_path, device='cuda'):
    """Load your ViT classifier"""
    model = timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=False).to(device)
    model.head = torch.nn.Linear(1024, len(class_available)).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_gradcam(model, image_tensor, target_class=None):
    """Generate proper GradCAM for ViT using gradients of activations"""
    gradients = []
    activations = []
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    # Hook last layer before head
    handle_b = model.blocks[-1].norm1.register_full_backward_hook(backward_hook)
    handle_f = model.blocks[-1].norm1.register_forward_hook(forward_hook)
    
    # Forward pass
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Backward pass for target class
    model.zero_grad()
    output[0, target_class].backward()
    
    # Get gradients and activations
    grads = gradients[0][0, 1:]  # Skip CLS token [num_patches, dim]
    acts = activations[0][0, 1:]  # Skip CLS token [num_patches, dim]
    
    # GradCAM: Global average pooling of gradients as weights
    weights = grads.mean(dim=1)  # [num_patches]
    
    # Weighted combination of activations
    cam = (weights.unsqueeze(1) * acts).sum(dim=0)  # [dim]
    
    # Wait, we need spatial cam. Let me recalculate properly
    # For each spatial location, compute weighted sum across channels
    weights = grads.mean(dim=0)  # [dim] - average gradient per channel
    cam = (acts * weights.unsqueeze(0)).sum(dim=1)  # [num_patches]
    
    # Reshape to spatial grid
    num_patches = cam.shape[0]
    grid_size = int(np.sqrt(num_patches))
    cam = cam.reshape(grid_size, grid_size).cpu().numpy()
    
    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    
    handle_b.remove()
    handle_f.remove()
    
    probs = F.softmax(output, dim=1)[0]
    return cam, target_class, probs


def main(image_path, class_ckpt, output_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    print(f"Loading model from {class_ckpt}...")
    model = load_model(class_ckpt, device)
    
    # Load image
    print(f"Processing {image_path}...")
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get GradCAM
    cam, pred_idx, probs = get_gradcam(model, img_tensor)
    
    # Print predictions
    print(f"\nTop 5 Predictions:")
    top_probs, top_idx = torch.topk(probs, 5)
    for i, (idx, prob) in enumerate(zip(top_idx, top_probs), 1):
        print(f"{i}. {class_available[idx]}: {prob.item()*100:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'Attention: {class_available[pred_idx]}')
    axes[1].axis('off')
    
    # Overlay
    img_resized = np.array(image.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {output_path}")
    else:
        output_path = image_path.replace('.jpg', '_gradcam.png').replace('.png', '_gradcam.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {output_path}")
    
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('--class_ckpt', required=True, help='Path to classifier checkpoint')
    parser.add_argument('--output', help='Output path (optional)')
    args = parser.parse_args()
    
    main(args.image_path, args.class_ckpt, args.output)