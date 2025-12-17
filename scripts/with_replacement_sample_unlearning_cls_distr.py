import os
import pickle
import sys

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.with_replacement_hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import (
    class_available,
    theme_available,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    class_params_path,
    use_replacement_map=True,  # NEW: Enable/disable replacement mapping
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/class20/",
    start_from=0,
):
    """
    Main function for concept unlearning with automatic replacement mapping.
    
    Args:
        pipe_checkpoint: Path to the Stable Diffusion checkpoint
        hookpoint: Which layer to hook (e.g., "up.1.1")
        class_latents_path: Path to the pickle file with class latents
        sae_checkpoint: Path to the SAE checkpoint
        class_params_path: Path to class parameters (percentile and multiplier for each class)
        use_replacement_map: If True, uses REPLACEMENT_MAP to replace unlearned objects.
                           If False, standard unlearning (white stains).
        seed: Random seed
        steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        output_dir: Directory to save generated images
        start_from: Index to start from (for resuming interrupted runs)
    """
    
    # ============================================================================
    # REPLACEMENT MAP: Define what to replace each unlearned object with
    # ============================================================================
    REPLACEMENT_MAP = {
        "Dogs": "Cats",
        "Cats": "Dogs",
        "Bears": "Rabbits",
        "Rabbits": "Bears",
        "Birds": "Butterfly",
        "Butterfly": "Birds",
        "Horses": "Fishes",
        "Fishes": "Horses",
        "Flame": "Flowers",
        "Flowers": "Flame",
        "Frogs": "Jellyfish",
        "Jellyfish": "Frogs",
        "Human": "Statues",
        "Statues": "Human",
        "Architectures": "Towers",
        "Towers": "Architectures",
        "Trees": "Waterfalls",
        "Waterfalls": "Trees",
        "Sandwiches": "Sea",
        "Sea": "Sandwiches",
        # Add more mappings as needed, or set to None for no replacement:
        # "Dogs": None,  # This would give white stains for Dogs
    }
    # ============================================================================
    
    accelerator = Accelerator()
    device = accelerator.device

    # Load the diffusion model
    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    # Enable xformers if available
    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. "
                    "If you observe problems during training, please update xFormers "
                    "to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers "
                    "for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    # Set random seeds
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Load SAE
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    # Load class latents dictionary
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)

    # Load class parameters (percentile and multiplier for each class)
    class_params = torch.load(class_params_path)

    # Get available themes (excluding Seed_Images)
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    
    # Get list of classes to process
    classes_to_process = class_available[:]
    
    # Validate replacement map if enabled
    if use_replacement_map and accelerator.is_main_process:
        print(f"=" * 80)
        print(f"REPLACEMENT MAP MODE ENABLED")
        print(f"=" * 80)
        print("\nReplacement mappings:")
        for unlearn_obj, replace_obj in REPLACEMENT_MAP.items():
            if replace_obj is None:
                print(f"  {unlearn_obj:15s} → [white stain - no replacement]")
            else:
                print(f"  {unlearn_obj:15s} → {replace_obj}")
        print(f"=" * 80)
        
        # Validate all replacement objects exist
        for unlearn_obj, replace_obj in REPLACEMENT_MAP.items():
            if replace_obj is not None:
                if replace_obj not in class_available:
                    print(f"WARNING: Replacement '{replace_obj}' for '{unlearn_obj}' not in class_available!")
                if replace_obj not in class_latents_dict:
                    print(f"WARNING: Replacement '{replace_obj}' for '{unlearn_obj}' not in class_latents_dict!")
                if replace_obj not in class_params:
                    print(f"WARNING: Replacement '{replace_obj}' for '{unlearn_obj}' not in class_params!")
        print()
    elif accelerator.is_main_process:
        print("Standard unlearning mode (no replacement)")
    
    if accelerator.is_main_process:
        print(f"Starting from index {start_from}, processing {len(classes_to_process)} classes")
        print(f"Classes to process: {classes_to_process}\n")
    
    # Progress bar for classes
    progress_bar = tqdm(
        classes_to_process,
        total=len(classes_to_process),
        disable=not accelerator.is_main_process,
        initial=0,
    )
    
    for class_to_unlearn in progress_bar:
        # Determine replacement object for this class
        if use_replacement_map:
            replacement_object = REPLACEMENT_MAP.get(class_to_unlearn, None)
        else:
            replacement_object = None
        
        # Update progress bar description
        if accelerator.is_main_process:
            unlearn_msg = f"Unlearning {class_to_unlearn}"
            if replacement_object:
                unlearn_msg += f" → {replacement_object}"
            else:
                unlearn_msg += f" → [no replacement]"
            progress_bar.set_description(unlearn_msg)
        
        # Create output directory for this class
        if replacement_object:
            output_subdir = f"{class_to_unlearn}_to_{replacement_object}"
        else:
            output_subdir = f"{class_to_unlearn}"
        
        output_path = os.path.join(output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        
        # Check if this class was already completed (for resuming)
        if start_from > 0:
            expected_files = len(theme_avail) * len(class_available) + len(class_available)
            existing_files = len([f for f in os.listdir(output_path) if f.endswith(f'_seed{seed}.jpg')])
            
            if existing_files >= expected_files and accelerator.is_main_process:
                print(f"Skipping {class_to_unlearn} - already completed ({existing_files} files found)")
                continue
        
        # Process each theme
        for test_theme in theme_avail:
            input_classes = []
            input_themes = []
            
            # Create all class-theme pairs for evaluation
            # This includes all classes with the current theme, plus all classes without theme
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            
            # Split pairs across multiple GPUs if using distributed training
            with accelerator.split_between_processes(
                class_theme_pairs
            ) as local_classes_themes:
                # Generate prompts for this process
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                
                # Set up the unlearning hook with automatic replacement
                steering_hooks = {}
                steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                    concept_to_unlearn=[class_to_unlearn],
                    percentile=class_params[class_to_unlearn]["percentile"],
                    multiplier=class_params[class_to_unlearn]["multiplier"],
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=class_latents_dict,
                    sae=sae,
                    steps=steps,
                    preserve_error=True,
                    seed=seed,
                    replacement_concept=replacement_object,  # Automatically selected from map
                    replacement_params=class_params.get(replacement_object) if replacement_object else None,
                )
                
                # Generate images with the hook
                with torch.no_grad():
                    images = model.run_with_hooks(
                        prompt=local_prompts,
                        generator=generator,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        position_hook_dict=steering_hooks,
                    )
                
                # Collect class and theme info for saving
                for object_class, theme in local_classes_themes:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            
            # Wait for all processes to finish
            accelerator.wait_for_everyone()
            
            # Gather results from all processes
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            
            # Save images (only on main process)
            if accelerator.is_main_process:
                for img, object_class, theme in zip(
                    images, input_classes, input_themes
                ):
                    if theme == "":
                        # No style - just object
                        if replacement_object:
                            filename = f"{object_class}_replaced_with_{replacement_object}_seed{seed}.jpg"
                        else:
                            filename = f"{object_class}_seed{seed}.jpg"
                        img.save(os.path.join(output_path, filename))
                    else:
                        # With style
                        if replacement_object:
                            filename = f"{theme}_{object_class}_replaced_with_{replacement_object}_seed{seed}.jpg"
                        else:
                            filename = f"{theme}_{object_class}_seed{seed}.jpg"
                        img.save(os.path.join(output_path, filename))
        
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)