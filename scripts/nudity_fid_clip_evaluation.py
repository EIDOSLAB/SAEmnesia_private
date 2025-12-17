"""
This script calculates both FID and CLIP scores between real and generated images.
Memory-efficient version with configurable CLIP cache directory.
"""

import multiprocessing
import os
import sys
import warnings

import cv2
import fire
import numpy as np
import torch
from scipy import linalg
from torch import nn
from torchvision.models import inception_v3
from tqdm import tqdm
from PIL import Image
import clip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

torch.hub.set_dir("/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/cache")


def to_cuda(elements):
    """
    Transfers elements to cuda if GPU is available
    Args:
        elements: torch.tensor or torch.nn.module
    Returns:
        elements: same as input on GPU memory, if available
    """
    if torch.cuda.is_available():
        return elements.to("cuda")
    return elements


def load_inception_offline():
    """
    Load Inception V3 model from local cache without downloading
    """
    model_path = "/leonardo_scratch/fast/IscrC_SAOU/inception_v3_google-0cc3c7bd.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please download it first:\n"
            f"wget https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth -O {model_path}"
        )
    
    # Create model with random weights first
    model = inception_v3(weights=None)
    
    # Load the pre-trained weights
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model


class PartialInceptionNetwork(nn.Module):
    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = load_inception_offline()
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        """
        Args:
            x: shape (N, 3, 299, 299) dtype: torch.float32 in range 0-1
        Returns:
            inception activations: torch.tensor, shape: (N, 2048), dtype: torch.float32
        """
        assert x.shape[1:] == (3, 299, 299), (
            "Expected input shape to be: (N,3,299,299)" + ", but got {}".format(x.shape)
        )
        x = x * 2 - 1  # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
        activations = activations.view(x.shape[0], 2048)
        return activations


def get_activations(images, batch_size):
    """
    Calculates activations for last pool layer for all images
    Args:
        images: torch.array shape: (N, 3, 299, 299), dtype: torch.float32
        batch_size: batch size used for inception network
    Returns: 
        np array shape: (N, 2048), dtype: np.float32
    """
    assert images.shape[1:] == (3, 299, 299), (
        "Expected input shape to be: (N,3,299,299)"
        + ", but got {}".format(images.shape)
    )

    num_images = images.shape[0]
    inception_network = PartialInceptionNetwork()
    inception_network = to_cuda(inception_network)
    inception_network.eval()
    n_batches = int(np.ceil(num_images / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in tqdm(range(n_batches), desc="Computing Inception activations"):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = to_cuda(ims)
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (
            ims.shape[0],
            2048,
        ), "Expected output shape to be: {}, but was: {}".format(
            (ims.shape[0], 2048), activations.shape
        )
        inception_activations[start_idx:end_idx, :] = activations
    return inception_activations


def calculate_activation_statistics(images, batch_size):
    """Calculates the statistics used by FID
    Args:
        images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
        batch_size: batch size to use to calculate inception scores
    Returns:
        mu:     mean over all activations from the last pool layer of the inception model
        sigma:  covariance matrix over all activations from the last pool layer
                of the inception model.
    """
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalculated
               on a representative data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalculated on a representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def preprocess_image(im):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        im: np.array, shape: (H, W, 3), dtype: float32 between 0-1 or np.uint8
    Return:
        im: torch.tensor, shape: (3, 299, 299), dtype: torch.float32 between 0-1
    """
    assert im.shape[2] == 3
    assert len(im.shape) == 3
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    assert im.max() <= 1.0
    assert im.min() >= 0.0
    assert im.dtype == torch.float32
    assert im.shape == (3, 299, 299)

    return im


def preprocess_images(images, use_multiprocessing=False):
    """Resizes and shifts the dynamic range of image to 0-1
    Args:
        images: np.array, shape: (N, H, W, 3), dtype: float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
    Return:
        final_images: torch.tensor, shape: (N, 3, 299, 299), dtype: torch.float32 between 0-1
    """
    if use_multiprocessing:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            jobs = []
            for im in images:
                job = pool.apply_async(preprocess_image, (im,))
                jobs.append(job)
            final_images = torch.zeros(images.shape[0], 3, 299, 299)
            for idx, job in enumerate(jobs):
                im = job.get()
                final_images[idx] = im
    else:
        final_images = torch.stack([preprocess_image(im) for im in tqdm(images, desc="Preprocessing images")], dim=0)
    assert final_images.shape == (images.shape[0], 3, 299, 299)
    assert final_images.max() <= 1.0
    assert final_images.min() >= 0.0
    assert final_images.dtype == torch.float32
    return final_images


def calculate_fid(
    images1, images2, use_multiprocessing=False, batch_size=64
):
    """Calculate FID between images1 and images2
    Args:
        images1: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        images2: np.array, shape: (N, H, W, 3), dtype: np.float32 between 0-1 or np.uint8
        use_multiprocessing: If multiprocessing should be used to pre-process the images
        batch_size: batch size used for inception network
    Returns:
        FID (scalar)
    """
    print("\n=== Computing FID Score ===")
    # FIXED: Always preprocess both image sets
    images1 = preprocess_images(images1, use_multiprocessing)
    images2 = preprocess_images(images2, use_multiprocessing)
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    print("mu1 shape:", mu1.shape, "sigma1 shape:", sigma1.shape)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    print("mu2 shape:", mu2.shape, "sigma2 shape:", sigma2.shape)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def calculate_clip_score(images, captions, batch_size=64, model_name="ViT-B/32", clip_cache_dir=None):
    """Calculate CLIP score between images and their captions
    Args:
        images: list of PIL Images or np.array, shape: (N, H, W, 3)
        captions: list of strings, length N
        batch_size: batch size for processing
        model_name: CLIP model to use (e.g., "ViT-B/32", "ViT-L/14")
        clip_cache_dir: directory to cache CLIP models
    Returns:
        mean_clip_score: float, average CLIP score
        clip_scores: list of CLIP scores for each image-caption pair
    """
    print("\n=== Computing CLIP Score ===")
    
    # Set CLIP cache directory if provided
    if clip_cache_dir is not None:
        os.environ['CLIP_CACHE_DIR'] = clip_cache_dir
        print(f"Using CLIP cache directory: {clip_cache_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    
    # Convert numpy arrays to PIL Images if needed
    if isinstance(images, np.ndarray):
        pil_images = []
        for img in images:
            # Convert to uint8 if needed
            if img.dtype == np.float32 or img.dtype == np.float64:
                img_uint8 = (img * 255).astype(np.uint8)
            else:
                img_uint8 = img
            
            # Ensure RGB format
            if len(img_uint8.shape) == 2:
                img_uint8 = np.stack([img_uint8] * 3, axis=-1)
            elif img_uint8.shape[2] == 4:
                img_uint8 = img_uint8[:, :, :3]
            
            pil_images.append(Image.fromarray(img_uint8))
        images = pil_images
    
    assert len(images) == len(captions), "Number of images must match number of captions"
    
    clip_scores = []
    n_batches = int(np.ceil(len(images) / batch_size))
    
    for batch_idx in tqdm(range(n_batches), desc="Computing CLIP scores"):
        start_idx = batch_size * batch_idx
        end_idx = min(batch_size * (batch_idx + 1), len(images))
        
        batch_images = images[start_idx:end_idx]
        batch_captions = captions[start_idx:end_idx]
        
        # Preprocess images
        image_inputs = torch.stack([preprocess(img) for img in batch_images]).to(device)
        
        # Tokenize captions
        text_inputs = clip.tokenize(batch_captions, truncate=True).to(device)
        
        # Get features
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity (CLIP score)
            similarity = (image_features * text_features).sum(dim=-1)
            clip_scores.extend(similarity.cpu().numpy())
    
    mean_clip_score = np.mean(clip_scores)
    return mean_clip_score, clip_scores


def load_images_from_directory(directory, max_images=None, target_size=256):
    """Load all images from a directory and resize to target size
    Args:
        directory: path to directory containing images
        max_images: optional limit on number of images to load
        target_size: resize all images to (target_size, target_size)
    Returns:
        images: np.array of shape (N, target_size, target_size, 3)
        image_paths: list of image paths
    """
    print(f"\nLoading images from: {directory}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in valid_extensions:
                image_paths.append(os.path.join(root, file))
    
    image_paths.sort()
    
    if max_images is not None:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {directory}")
    
    # Load all images and resize to target_size
    images = np.zeros((len(image_paths), target_size, target_size, 3), dtype=np.uint8)
    for idx, path in enumerate(tqdm(image_paths, desc="Loading images")):
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Failed to load image {path}, skipping...")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize to target size
        img = cv2.resize(img, (target_size, target_size))
        images[idx] = img
    
    print(f"Loaded images shape: {images.shape}")
    return images, image_paths


def extract_captions_from_paths(image_paths, caption_prefix=""):
    """Generate captions from image filenames
    Args:
        image_paths: list of image paths
        caption_prefix: optional prefix for captions
    Returns:
        captions: list of caption strings
    """
    captions = []
    for path in image_paths:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(path))[0]
        # Use filename as caption (you can customize this)
        caption = f"{caption_prefix}{filename.replace('_', ' ')}"
        captions.append(caption)
    return captions


def main(
    generated_dir,
    reference_dir,
    output_dir=None,
    batch_size=64,
    multiprocessing=False,
    max_images=None,
    compute_fid=True,
    compute_clip=False,
    captions_file=None,
    clip_model="ViT-B/32",
    target_size=256,
    clip_cache_dir=None
):
    """
    Calculate FID and optionally CLIP scores between generated and reference images
    
    Args:
        generated_dir: Path to directory containing generated images
        reference_dir: Path to directory containing reference/real images
        output_dir: Path to save results (optional)
        batch_size: Batch size for processing
        multiprocessing: Use multiprocessing for image preprocessing
        max_images: Limit number of images to process (for testing)
        compute_fid: Whether to compute FID score
        compute_clip: Whether to compute CLIP score
        captions_file: Path to JSON file with captions (for CLIP score)
        clip_model: CLIP model variant to use
        target_size: Resize all images to this size (default: 256)
        clip_cache_dir: Directory to cache CLIP models (default: ~/.cache/clip)
    """
    
    print(f"\n{'='*70}")
    print("FID and CLIP Score Evaluation")
    print(f"{'='*70}")
    print(f"Generated images: {generated_dir}")
    print(f"Reference images: {reference_dir}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Batch size: {batch_size}")
    print(f"Compute FID: {compute_fid}")
    print(f"Compute CLIP: {compute_clip}")
    print(f"{'='*70}\n")
    
    # Load images
    generated_images, gen_paths = load_images_from_directory(generated_dir, max_images, target_size)
    reference_images, ref_paths = load_images_from_directory(reference_dir, max_images, target_size)
    
    results = {}
    
    # Compute FID
    if compute_fid:
        fid_score = calculate_fid(
            reference_images,
            generated_images,
            multiprocessing,
            batch_size
        )
        results['fid'] = float(fid_score)
        print(f"\n{'='*70}")
        print(f"FID Score: {fid_score:.4f}")
        print(f"{'='*70}\n")
    
    # Compute CLIP score
    if compute_clip:
        # Load or generate captions
        if captions_file is not None:
            import json
            with open(captions_file, 'r') as f:
                captions_data = json.load(f)
            
            # Extract captions based on format
            if isinstance(captions_data, list):
                if isinstance(captions_data[0], dict):
                    captions = [item['caption'] for item in captions_data]
                else:
                    captions = captions_data
            else:
                captions = list(captions_data.values())
            
            # Match number of captions to number of images
            captions = captions[:len(generated_images)]
        else:
            # Generate captions from filenames
            print("No captions file provided, using filenames as captions")
            captions = extract_captions_from_paths(gen_paths)
        
        mean_clip_score, clip_scores = calculate_clip_score(
            generated_images,
            captions,
            batch_size,
            clip_model,
            clip_cache_dir=clip_cache_dir
        )
        results['clip_score'] = float(mean_clip_score)
        results['clip_scores_per_image'] = [float(s) for s in clip_scores]
        
        print(f"\n{'='*70}")
        print(f"Mean CLIP Score: {mean_clip_score:.4f}")
        print(f"Std CLIP Score: {np.std(clip_scores):.4f}")
        print(f"{'='*70}\n")
    
    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        if compute_fid:
            fid_path = os.path.join(output_dir, "fid_score.pth")
            torch.save(results['fid'], fid_path)
            print(f"Saved FID score to: {fid_path}")
        
        if compute_clip:
            clip_path = os.path.join(output_dir, "clip_scores.pth")
            torch.save(results, clip_path)
            print(f"Saved CLIP scores to: {clip_path}")
        
        # Also save as JSON for easy reading
        import json
        json_path = os.path.join(output_dir, "scores.json")
        with open(json_path, 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'clip_scores_per_image'}, f, indent=2)
        print(f"Saved summary to: {json_path}")
    
    return results


if __name__ == "__main__":
    fire.Fire(main)