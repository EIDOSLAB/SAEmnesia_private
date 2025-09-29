import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

from UnlearnCanvas_resources.const import class_available, theme_available

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
    cls_latents_path,  # Changed from style_latents_path to match your file naming
    sae_checkpoint,
    class_params_path,  # Added parameter for class-specific parameters
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/object_sequential/",  # Updated output dir
):
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    # Load class/object latents
    with open(
        cls_latents_path,
        "rb",
    ) as f:
        cls_latents_dict = pickle.load(f)

    # Load class-specific parameters (percentile and multiplier for each object)
    class_params = torch.load(class_params_path)

    # Define sequential objects to unlearn (example sequence)
    sequential_objects_to_unlearn = [
        ["Bears"],
        ["Bears", "Cats"],
        ["Bears", "Cats", "Flowers"],
        ["Bears", "Cats", "Flowers", "Frogs"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls"],
    ]
    
    progress_bar = tqdm(
        sequential_objects_to_unlearn,
        total=len(sequential_objects_to_unlearn),
        disable=not accelerator.is_main_process,
    )
    
    for objects_to_unlearn in progress_bar:
        if accelerator.is_main_process:
            progress_bar.set_description(f"Unlearning {objects_to_unlearn}")
        
        # Create output directory name based on the objects being unlearned
        output_path = os.path.join(
            output_dir,
            f"{'_'.join(objects_to_unlearn)}",
        )
        os.makedirs(output_path, exist_ok=True)
        
        # Test on all object classes
        for test_class in class_available:
            input_classes = []
            input_themes = []
            
            # Create pairs of (test_class, theme) for all themes, plus (test_class, no_theme)
            class_theme_pairs = [(test_class, theme) for theme in theme_available if theme != "Seed_Images"] + [
                (test_class, "")
            ]
            
            with accelerator.split_between_processes(
                class_theme_pairs
            ) as local_classes_themes:
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                
                # Create steering hooks with object-specific parameters
                steering_hooks = {}
                
                # Get percentile and multiplier from the first object to unlearn
                # (assuming all objects in the sequence use the same parameters)
                first_object = objects_to_unlearn[0]
                percentile = class_params[first_object]["percentile"]
                multiplier = class_params[first_object]["multiplier"]
                
                steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                    concept_to_unlearn=objects_to_unlearn,  # List of objects to unlearn
                    percentile=percentile,  # Use class-specific percentile
                    multiplier=multiplier,  # Use class-specific multiplier
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=cls_latents_dict,  # Using class latents
                    sae=sae,
                    steps=steps,
                    preserve_error=True,
                )
                
                with torch.no_grad():
                    images = model.run_with_hooks(
                        prompt=local_prompts,
                        generator=generator,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        position_hook_dict=steering_hooks,
                    )
                
                for object_class, theme in local_classes_themes:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            
            if accelerator.is_main_process:
                for img, object_class, theme in zip(
                    images, input_classes, input_themes
                ):
                    if theme == "":
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}.jpg",
                            )
                        )
                    else:
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}.jpg",
                            )
                        )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)