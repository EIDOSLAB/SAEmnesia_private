import os

import timm
import torch
from torchvision import transforms

torch.hub.set_dir("cache")
import sys

import fire
from PIL import Image
from tqdm import tqdm

sys.path.append("")
# Import both full and subsample versions
from UnlearnCanvas_resources.const import class_available as class_available_full
from UnlearnCanvas_resources.const import theme_available as theme_available_full
from UnlearnCanvas_resources.const import class_available as class_available
from UnlearnCanvas_resources.const import theme_available as theme_available
# from UnlearnCanvas_resources.const import class_available_subsample as class_available
# from UnlearnCanvas_resources.const import theme_available_subsample as theme_available


def main(
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    cls=None,
    seed=[188],  # [188, 288, 588, 688, 888],
    dry_run=False,
    limit_classes=-1,
    batch_size=32,
):
    """
    Evaluate classification accuracy for decoder-only steering results.
    
    This script works with the new directory structure:
    - Old: percentile_{percentile}_multiplier_{multiplier}/{class}/
    - New: percentile_{percentile}_alpha_{alpha}/{class}/
    
    Args:
        input_dir: Input directory containing generated images (should point to a specific class folder)
        output_dir: Output directory for classification results
        style_ckpt: Path to style classifier checkpoint
        class_ckpt: Path to class classifier checkpoint
        cls: Class name being evaluated (used for output naming)
        seed: List of seeds used during generation
        dry_run: If True, don't save results
        limit_classes: Limit number of classes to evaluate (-1 for all)
        batch_size: Batch size for classification
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dir = os.path.join(input_dir, cls) if cls is not None else input_dir

    # Create folder if not exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize both models with full architecture
    style_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)
    class_model = timm.create_model(
        "vit_large_patch16_224.augreg_in21k", pretrained=True
    ).to(device)

    # First, set head to original full size to load checkpoint
    style_model.head = torch.nn.Linear(1024, len(theme_available_full)).to(device)
    class_model.head = torch.nn.Linear(1024, len(class_available_full)).to(device)

    # Load checkpoints with full size
    style_checkpoint = torch.load(style_ckpt, map_location=device, weights_only=False)
    class_checkpoint = torch.load(class_ckpt, map_location=device, weights_only=False)
    
    style_model.load_state_dict(style_checkpoint["model_state_dict"])
    class_model.load_state_dict(class_checkpoint["model_state_dict"])

    # Now extract only the weights for subsampled themes/classes
    # Get indices of subsampled items in the full lists
    theme_indices = [theme_available_full.index(t) for t in theme_available]
    class_indices = [class_available_full.index(c) for c in class_available]

    # Create new heads with subsampled size
    new_style_head = torch.nn.Linear(1024, len(theme_available)).to(device)
    new_class_head = torch.nn.Linear(1024, len(class_available)).to(device)

    # Extract subsampled weights and biases
    with torch.no_grad():
        new_style_head.weight.data = style_model.head.weight.data[theme_indices, :]
        new_style_head.bias.data = style_model.head.bias.data[theme_indices]
        
        new_class_head.weight.data = class_model.head.weight.data[class_indices, :]
        new_class_head.bias.data = class_model.head.bias.data[class_indices]

    # Replace heads with subsampled versions
    style_model.head = new_style_head
    class_model.head = new_class_head

    style_model.eval()
    class_model.eval()

    # Initialize results dictionaries for both tasks
    style_results = {
        "test_theme": cls if cls is not None else "sd",
        "input_dir": input_dir,
        "loss": {theme: 0.0 for theme in theme_available},
        "acc": {theme: 0.0 for theme in theme_available},
        "pred_loss": {theme: 0.0 for theme in theme_available},
        "misclassified": {
            theme: {other_theme: 0 for other_theme in theme_available}
            for theme in theme_available
        },
    }

    class_results = {
        "test_theme": cls if cls is not None else "sd",
        "input_dir": input_dir,
        "loss": {class_: 0.0 for class_ in class_available},
        "acc": {class_: 0.0 for class_ in class_available},
        "pred_loss": {class_: 0.0 for class_ in class_available},
        "misclassified": {
            class_: {other_class: 0 for other_class in class_available}
            for class_ in class_available
        },
    }

    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels):
            self.image_paths = image_paths
            self.labels = labels

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            try:
                image = Image.open(img_path).convert('RGB')
                image = image_transform(image)
                return image, self.labels[idx]
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a black image as fallback
                return torch.zeros(3, 224, 224), self.labels[idx]

    classes_to_use = (
        class_available[:limit_classes] if limit_classes > 0 else class_available
    )
    style_image_paths = []
    class_image_paths = []
    style_labels = []
    class_labels = []
    style_label_map = {theme: idx for idx, theme in enumerate(theme_available)}
    class_label_map = {class_: idx for idx, class_ in enumerate(classes_to_use)}

    print(f"Processing directory: {input_dir}")
    print(f"Classes to evaluate: {classes_to_use}")

    for theme in theme_available:
        if theme == "Seed_Images":
            continue
        print(f"Collecting images for theme: {theme}")

    # Collect all image paths
    for t, test_theme in enumerate(theme_available):
        if test_theme == "Seed_Images":
            continue
        for s in seed:
            for object_class in classes_to_use:
                img_path = os.path.join(
                    input_dir,
                    f"{test_theme}_{object_class}_seed{s}.jpg",
                )
                if os.path.exists(img_path):
                    style_image_paths.append(img_path)
                    class_image_paths.append(img_path)
                    class_labels.append(class_label_map[object_class])
                    style_labels.append(style_label_map[test_theme])
                else:
                    print(f"Warning: Image not found: {img_path}")
                
                class_image_path = os.path.join(
                    input_dir,
                    f"{object_class}_seed{s}.jpg",
                )
                if os.path.exists(class_image_path) and class_image_path not in class_image_paths:
                    class_image_paths.append(class_image_path)
                    class_labels.append(class_label_map[object_class])

    print(f"Total style images to evaluate: {len(style_image_paths)}")
    print(f"Total class images to evaluate: {len(class_image_paths)}")

    if len(style_image_paths) == 0 or len(class_image_paths) == 0:
        print("Warning: No images found to evaluate!")
        return

    style_dataset = ImageDataset(style_image_paths, style_labels)
    class_dataset = ImageDataset(class_image_paths, class_labels)
    style_dataloader = torch.utils.data.DataLoader(
        style_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    class_dataloader = torch.utils.data.DataLoader(
        class_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print("Evaluating style classification...")
    for batch_images, batch_style_labels in tqdm(style_dataloader, desc="Style eval"):
        batch_images = batch_images.to(device)
        batch_style_labels = batch_style_labels.to(device)

        with torch.no_grad():
            # Style classification
            style_res = style_model(batch_images)
            style_loss = torch.nn.functional.cross_entropy(
                style_res, batch_style_labels, reduction="none"
            )
            style_softmax = torch.nn.functional.softmax(style_res, dim=1)
            style_pred_labels = torch.argmax(style_res, dim=1)
            style_pred_success = style_pred_labels == batch_style_labels
            
            for i in range(len(batch_style_labels)):
                # Update style results
                test_theme = theme_available[batch_style_labels[i].item()]
                style_results["loss"][test_theme] += style_loss[i].item()
                style_results["pred_loss"][test_theme] += style_softmax[i][
                    batch_style_labels[i]
                ].item()
                style_results["acc"][test_theme] += style_pred_success[i].item()
                misclassified_as = theme_available[style_pred_labels[i].item()]
                style_results["misclassified"][test_theme][misclassified_as] += 1
    
    # Normalize accuracy by total samples per theme
    for theme in theme_available:
        total_samples_for_theme = sum(style_results["misclassified"][theme].values())
        if total_samples_for_theme > 0:
            style_results["acc"][theme] = (style_results["acc"][theme] / total_samples_for_theme)

    if not dry_run:
        style_output_path = os.path.join(output_dir, f"{cls}.pth")
        torch.save(style_results, style_output_path)
        print(f"Style results saved to: {style_output_path}")

    print("Evaluating class classification...")
    for batch_images, batch_class_labels in tqdm(class_dataloader, desc="Class eval"):
        batch_images = batch_images.to(device)
        batch_class_labels = batch_class_labels.to(device)

        with torch.no_grad():
            class_res = class_model(batch_images)
            class_loss = torch.nn.functional.cross_entropy(
                class_res, batch_class_labels, reduction="none"
            )
            class_softmax = torch.nn.functional.softmax(class_res, dim=1)
            class_pred_labels = torch.argmax(class_res, dim=1)
            class_pred_success = class_pred_labels == batch_class_labels

            for i in range(len(batch_class_labels)):
                object_class = class_available[batch_class_labels[i].item()]
                class_results["loss"][object_class] += class_loss[i].item()
                class_results["pred_loss"][object_class] += class_softmax[i][
                    batch_class_labels[i]
                ].item()
                class_results["acc"][object_class] += class_pred_success[i].item()
                misclassified_as = class_available[class_pred_labels[i].item()]
                class_results["misclassified"][object_class][misclassified_as] += 1
    
    # Normalize accuracy by total samples per class
    for object_class in class_available:
        total_samples_for_class = sum(class_results["misclassified"][object_class].values())
        if total_samples_for_class > 0:
            class_results["acc"][object_class] = class_results["acc"][object_class] / total_samples_for_class
    
    if not dry_run:
        class_output_path = os.path.join(output_dir, f"{cls}_cls.pth")
        torch.save(class_results, class_output_path)
        print(f"Class results saved to: {class_output_path}")

    # Print summary statistics
    print("\n=== Classification Summary ===")
    print(f"Style accuracy: {sum(style_results['acc'].values()) / len([t for t in theme_available if t != 'Seed_Images']):.4f}")
    print(f"Class accuracy: {sum(class_results['acc'].values()) / len(class_available):.4f}")


if __name__ == "__main__":
    fire.Fire(main)