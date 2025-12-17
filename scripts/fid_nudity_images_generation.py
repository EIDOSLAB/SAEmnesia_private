import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae


class COCOImageGenerator:
    """Generate images from COCO captions with optional SAE reconstruction"""
    
    def __init__(
        self,
        sd_path: str,
        output_dir: str,
        device: str = "cuda",
        target_hookpoint: str = "unet.up_blocks.1.attentions.1"
    ):
        self.sd_path = sd_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.target_hookpoint = target_hookpoint
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Stable Diffusion model
        print("Loading Stable Diffusion model...")
        self.pipe = HookedStableDiffusionPipeline.from_pretrained(
            sd_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        
        # Remove safety features
        self.pipe.safety_checker = None
        for attr in ['feature_extractor', 'safety_checker']:
            if hasattr(self.pipe, attr):
                setattr(self.pipe, attr, None)
        
        self.pipe = self.pipe.to(device)
        
        print(f"✓ Model loaded on {device}")
        print(f"✓ Safety checker disabled: {self.pipe.safety_checker is None}")
    
    def load_sae(self, sae_path: Path, checkpoint_type: str = "best"):
        """Load SAE model from checkpoint"""
        checkpoint_dir = sae_path / checkpoint_type
        
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Look for hookpoint-specific subdirectory
        hookpoint_subdir = checkpoint_dir / self.target_hookpoint
        
        if hookpoint_subdir.exists():
            sae_checkpoint_path = hookpoint_subdir
        else:
            subdirs = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                sae_checkpoint_path = subdirs[0]
            else:
                raise ValueError(
                    f"Could not find SAE checkpoint for hookpoint {self.target_hookpoint}"
                )
        
        print(f"Loading SAE from {sae_checkpoint_path}...")
        sae_model = Sae.load_from_disk(sae_checkpoint_path, device=self.device)
        sae_model = sae_model.eval().to(dtype=torch.float16)
        sae_model.cfg.batch_topk = False
        sae_model.cfg.sample_topk = False
        
        print(f"✓ SAE loaded successfully")
        return sae_model
    
    def load_captions(self, captions_path: Path):
        """Load captions from JSON or text file (one caption per line)"""
        print(f"Loading captions from {captions_path}...")
        
        # Check file extension to determine format
        if captions_path.suffix.lower() in ['.txt', '.text']:
            # Load as text file - one caption per line
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions = [line.strip() for line in f if line.strip()]
        else:
            # Load as JSON file
            with open(captions_path, 'r') as f:
                captions_data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(captions_data, list):
                # Check if it's a list of strings or list of dicts
                if len(captions_data) > 0:
                    if isinstance(captions_data[0], str):
                        captions = captions_data
                    elif isinstance(captions_data[0], dict):
                        # Extract 'caption' field from each dict
                        captions = [item.get('caption', item.get('text', str(item))) 
                                   for item in captions_data]
                    else:
                        captions = [str(item) for item in captions_data]
                else:
                    captions = []
            elif isinstance(captions_data, dict):
                if 'captions' in captions_data:
                    raw_captions = captions_data['captions']
                elif 'annotations' in captions_data:
                    raw_captions = [ann['caption'] for ann in captions_data['annotations']]
                else:
                    raise ValueError("Unsupported captions JSON format")
                
                # Ensure all captions are strings
                if isinstance(raw_captions, list):
                    captions = [str(c) if not isinstance(c, str) else c 
                               for c in raw_captions]
                else:
                    raise ValueError("Captions must be a list")
            else:
                raise ValueError("Unsupported captions JSON format")
        
        # Verify all captions are strings
        for i, cap in enumerate(captions[:5]):
            print(f"  Sample caption {i}: {cap[:80]}... (type: {type(cap).__name__})")
        
        print(f"✓ Loaded {len(captions)} captions")
        return captions
    
    def generate_baseline_images(
        self,
        captions: list,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42
    ):
        """Generate images WITHOUT SAE (vanilla Stable Diffusion baseline)"""
        
        print(f"\n{'='*70}")
        print(f"Generating {len(captions)} BASELINE images (Vanilla SD)")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")
        print(f"  Mode: Baseline (no SAE)")
        print(f"{'='*70}\n")
        
        successful_count = 0
        failed_count = 0
        
        for idx, caption in enumerate(tqdm(captions, desc="Generating baseline images")):
            output_path = self.output_dir / f"image_{idx:05d}.png"
            
            # Skip if already exists
            if output_path.exists():
                successful_count += 1
                continue
            
            try:
                # Ensure caption is a string
                if not isinstance(caption, str):
                    caption = str(caption)
                
                # Create a new generator for each image with the same seed
                # This ensures reproducibility
                generator = torch.Generator(device="cpu").manual_seed(seed + idx)
                
                with torch.no_grad():
                    # Generate WITHOUT hooks - vanilla SD
                    images = self.pipe(
                        prompt=[caption],
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                    ).images
                    image = images[0]
                
                # Save image
                image.save(output_path)
                successful_count += 1
                
            except Exception as e:
                print(f"\n❌ Error generating baseline image {idx}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                
                # Stop after 5 consecutive errors at the start
                if failed_count >= 5 and successful_count == 0:
                    print("\n⚠️  Multiple consecutive errors. Stopping generation.")
                    break
        
        print(f"\n{'='*70}")
        print(f"Baseline Generation Complete")
        print(f"  Successful: {successful_count}/{len(captions)}")
        print(f"  Failed: {failed_count}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return successful_count, failed_count
    
    def generate_images(
        self,
        captions: list,
        sae_model,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42
    ):
        """Generate images with SAE in reconstruction-only mode"""
        
        print(f"\n{'='*70}")
        print(f"Generating {len(captions)} images with SAE reconstruction")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")
        print(f"  Hookpoint: {self.target_hookpoint}")
        print(f"{'='*70}\n")
        
        # Create reconstruction-only hook ONCE for all images
        reconstruction_hook = hooks.SAEReconstructHook(sae=sae_model)
        
        successful_count = 0
        failed_count = 0
        
        for idx, caption in enumerate(tqdm(captions, desc="Generating SAE images")):
            output_path = self.output_dir / f"image_{idx:05d}.png"
            
            # Skip if already exists
            if output_path.exists():
                successful_count += 1
                continue
            
            try:
                # Ensure caption is a string
                if not isinstance(caption, str):
                    caption = str(caption)
                
                # Create a new generator for each image with the same seed
                # This ensures reproducibility
                generator = torch.Generator(device="cpu").manual_seed(seed + idx)
                
                with torch.no_grad():
                    # Generate with SAE reconstruction hook
                    images = self.pipe.run_with_hooks(
                        prompt=[caption],
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        position_hook_dict={self.target_hookpoint: reconstruction_hook},
                    )
                    image = images[0]
                
                # Save image
                image.save(output_path)
                successful_count += 1
                
            except Exception as e:
                print(f"\n❌ Error generating SAE image {idx}: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                
                # Stop after 5 consecutive errors at the start
                if failed_count >= 5 and successful_count == 0:
                    print("\n⚠️  Multiple consecutive errors. Stopping generation.")
                    break
        
        print(f"\n{'='*70}")
        print(f"SAE Generation Complete")
        print(f"  Successful: {successful_count}/{len(captions)}")
        print(f"  Failed: {failed_count}")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        return successful_count, failed_count


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate COCO images with optional SAE reconstruction"
    )
    parser.add_argument("--captions_path", type=str, required=True,
                       help="Path to captions file (JSON or text file with one caption per line)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save generated images")
    parser.add_argument("--sd_path", type=str, required=True,
                       help="Path to Stable Diffusion model")
    parser.add_argument("--mode", type=str, default="sae",
                       choices=["sae", "baseline"],
                       help="Generation mode: 'sae' for SAE-modified or 'baseline' for vanilla SD")
    parser.add_argument("--sae_path", type=str, default=None,
                       help="Path to SAE checkpoint directory (required for SAE mode)")
    parser.add_argument("--target_hookpoint", type=str,
                       default="unet.up_blocks.1.attentions.1",
                       help="Target hookpoint for SAE")
    parser.add_argument("--checkpoint_type", type=str, default="best",
                       choices=["best", "last"],
                       help="Which checkpoint to use")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "sae" and args.sae_path is None:
        parser.error("--sae_path is required when mode is 'sae'")
    
    # Initialize generator
    print(f"\n{'='*70}")
    print(f"COCO Image Generation - Mode: {args.mode.upper()}")
    print(f"{'='*70}\n")
    
    generator = COCOImageGenerator(
        sd_path=args.sd_path,
        output_dir=args.output_dir,
        device=args.device,
        target_hookpoint=args.target_hookpoint
    )
    
    # Load captions
    captions = generator.load_captions(Path(args.captions_path))
    
    # Generate images based on mode
    if args.mode == "baseline":
        # Generate baseline images (vanilla SD)
        successful, failed = generator.generate_baseline_images(
            captions=captions,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
    else:
        # Load SAE and generate with SAE reconstruction
        sae_model = generator.load_sae(
            Path(args.sae_path),
            checkpoint_type=args.checkpoint_type
        )
        
        successful, failed = generator.generate_images(
            captions=captions,
            sae_model=sae_model,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
    
    if successful > 0:
        print("✅ Generation completed successfully!")
    else:
        print("❌ Generation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()