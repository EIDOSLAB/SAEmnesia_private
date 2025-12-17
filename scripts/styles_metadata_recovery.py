#!/usr/bin/env python3
"""
Recover style labels from cached activations by reconstructing the original prompt order.
"""
import sys
import os
sys.path.append("/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning")

from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from collections import Counter
import shutil
import tempfile
import gc

# Import the theme and class lists
from UnlearnCanvas_resources.const import class_available, theme_available

def reconstruct_prompts_and_labels(class_start=0, class_end=20, seed=42):
    """Reconstruct the exact prompt order that was used during caching."""
    all_prompts = []
    all_object_labels = []
    all_style_labels = []
    
    for class_avail in class_available[class_start:class_end]:
        prompt_file = f"UnlearnCanvas_resources/anchor_prompts/finetune_prompts/sd_prompt_{class_avail}.txt"
        
        with open(prompt_file, 'r') as f:
            for prompt in f:
                prompt = prompt.strip()
                prompt = prompt if not prompt.endswith(".") else prompt[:-1]
                
                # Add styled prompts (50 styles)
                for theme in theme_available:
                    theme_prompt = f"{prompt} in {theme.replace('_', ' ')} style."
                    all_prompts.append(theme_prompt)
                    all_object_labels.append(class_avail)
                    all_style_labels.append(theme)
                
                # Add plain prompt without style
                all_prompts.append(prompt + ".")
                all_object_labels.append(class_avail)
                all_style_labels.append("none")
    
    # Create dataset and shuffle with same seed
    temp_ds = Dataset.from_dict({
        "caption": all_prompts,
        "object_label": all_object_labels,
        "style_label": all_style_labels
    })
    temp_ds = temp_ds.shuffle(seed)
    
    return temp_ds

def fix_style_labels_in_dataset(dataset_path, reconstructed_labels, 
                                 num_inference_steps=4, cache_every_n_timesteps=1):
    """Fix the style labels in a cached dataset."""
    
    print(f"  Loading dataset from {dataset_path}")
    ds = Dataset.load_from_disk(str(dataset_path))
    
    print(f"    Current dataset size: {len(ds)}")
    
    # Calculate how many samples per prompt
    samples_per_prompt = num_inference_steps // cache_every_n_timesteps
    print(f"    Samples per prompt: {samples_per_prompt}")
    
    # Build a lookup: object -> list of (style, count)
    # by filtering the reconstructed labels for this object
    concept_name = Path(dataset_path).name
    concept_labels = [r for r in reconstructed_labels if r["object_label"] == concept_name]
    
    print(f"    Found {len(concept_labels)} prompts for concept '{concept_name}'")
    
    # Create new labels based on the reconstructed order
    new_style_labels = []
    new_object_labels = []
    
    for i in range(len(ds)):
        # Each prompt generates samples_per_prompt samples
        # So sample i corresponds to prompt (i // samples_per_prompt)
        current_prompt_idx = i // samples_per_prompt
        
        if current_prompt_idx < len(concept_labels):
            new_style_labels.append(concept_labels[current_prompt_idx]["style_label"])
            new_object_labels.append(concept_labels[current_prompt_idx]["object_label"])
        else:
            # Fallback if something is off
            new_style_labels.append("none")
            new_object_labels.append(concept_name)
    
    # Verify counts
    style_dist = Counter(new_style_labels)
    print(f"    New style distribution (top 5): {dict(list(style_dist.most_common(5)))}")
    print(f"    Total unique styles: {len(style_dist)}")
    
    # Update the dataset - create a new modified dataset
    ds = ds.remove_columns(["style_label", "object_label"])
    ds = ds.add_column("style_label", new_style_labels)
    ds = ds.add_column("object_label", new_object_labels)
    
    return ds

def recover_all_styles(base_path, hook_name, class_start=0, class_end=20, 
                       seed=42, num_inference_steps=4, cache_every_n_timesteps=1,
                       dry_run=False, skip_backup=False):
    """Recover style labels for all concept directories."""
    
    print("=" * 70)
    print("STYLE RECOVERY SCRIPT")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Base path: {base_path}")
    print(f"  Hook name: {hook_name}")
    print(f"  Class range: {class_start} to {class_end}")
    print(f"  Seed: {seed}")
    print(f"  Num inference steps: {num_inference_steps}")
    print(f"  Cache every N timesteps: {cache_every_n_timesteps}")
    print(f"  Dry run: {dry_run}")
    print(f"  Skip backup: {skip_backup}")
    print()
    
    # Reconstruct the original prompt order
    print("Step 1: Reconstructing original prompt order...")
    reconstructed = reconstruct_prompts_and_labels(class_start, class_end, seed)
    print(f"  ✓ Reconstructed {len(reconstructed)} total prompts")
    
    # Show distribution
    obj_dist = Counter(reconstructed["object_label"])
    style_dist = Counter(reconstructed["style_label"])
    print(f"  ✓ Objects: {len(obj_dist)} unique")
    print(f"  ✓ Styles: {len(style_dist)} unique")
    print()
    
    hook_path = Path(base_path) / hook_name
    
    if not hook_path.exists():
        print(f"❌ Error: Hook path does not exist: {hook_path}")
        return
    
    concept_dirs = [d for d in hook_path.iterdir() 
                   if d.is_dir() and d.name != "metadata" and not d.name.endswith("_backup")]
    
    print(f"Step 2: Found {len(concept_dirs)} concept directories to process")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
    
    for concept_dir in tqdm(concept_dirs, desc="Processing concepts"):
        concept_name = concept_dir.name
        print(f"\n{'='*70}")
        print(f"Processing: {concept_name}")
        print(f"{'='*70}")
        
        try:
            # Filter reconstructed labels for this object
            concept_labels = [
                {
                    "object_label": r["object_label"],
                    "style_label": r["style_label"],
                    "caption": r["caption"]
                }
                for r in reconstructed 
                if r["object_label"] == concept_name
            ]
            
            print(f"  Found {len(concept_labels)} prompts for {concept_name}")
            
            if len(concept_labels) == 0:
                print(f"  ⚠️  Warning: No prompts found for {concept_name}, skipping")
                continue
            
            # Show expected style distribution
            expected_styles = Counter(r["style_label"] for r in concept_labels)
            print(f"  Expected styles: {len(expected_styles)} unique")
            print(f"  Top 3 expected: {dict(list(expected_styles.most_common(3)))}")
            
            if not dry_run:
                # Fix the dataset
                fixed_ds = fix_style_labels_in_dataset(
                    concept_dir, 
                    concept_labels,
                    num_inference_steps,
                    cache_every_n_timesteps
                )
                
                # Save to a temporary location first
                temp_dir = concept_dir.parent / f".tmp_{concept_name}"
                print(f"  💾 Saving to temporary location...")
                fixed_ds.save_to_disk(str(temp_dir))
                
                # Explicitly delete the dataset object and force garbage collection
                del fixed_ds
                gc.collect()
                
                # Create backup if requested
                backup_path = concept_dir.parent / f"{concept_name}_backup"
                if not skip_backup and not backup_path.exists():
                    print(f"  📦 Creating backup (moving original)...")
                    shutil.move(str(concept_dir), str(backup_path))
                elif backup_path.exists():
                    print(f"  ℹ️  Backup already exists, removing old dataset...")
                    shutil.rmtree(concept_dir)
                elif skip_backup:
                    print(f"  ⚠️  Skipping backup, removing old dataset...")
                    shutil.rmtree(concept_dir)
                
                # Move the fixed dataset to the original location
                print(f"  📦 Moving fixed dataset to final location...")
                shutil.move(str(temp_dir), str(concept_dir))
                
                # Verify
                check_ds = Dataset.load_from_disk(str(concept_dir))
                final_style_dist = Counter(check_ds["style_label"])
                print(f"  ✅ Fixed! Found {len(final_style_dist)} unique styles")
                print(f"     Top 5 styles: {dict(list(final_style_dist.most_common(5)))}")
                
                del check_ds
                gc.collect()
            else:
                # Just show what would happen
                print(f"  [DRY RUN] Would update {concept_dir}")
                
        except Exception as e:
            print(f"  ❌ Error processing {concept_name}: {e}")
            import traceback
            traceback.print_exc()
            # Clean up temp directory if it exists
            temp_dir = concept_dir.parent / f".tmp_{concept_name}"
            if temp_dir.exists():
                print(f"  🧹 Cleaning up temporary directory...")
                shutil.rmtree(temp_dir)
            continue
    
    print()
    print("=" * 70)
    if dry_run:
        print("DRY RUN COMPLETE - No changes were made")
        print("Run again without --dry-run to apply changes")
    else:
        print("🎉 STYLE RECOVERY COMPLETE!")
        if not skip_backup:
            print(f"   Backups saved in {hook_path}/*_backup/")
    print("=" * 70)

if __name__ == "__main__":
    # Configuration matching your caching parameters
    BASE_PATH = "/leonardo_scratch/fast/IscrC_SAOU/cassano/finetuning_activations/sdxl_objects"
    HOOK_NAME = "unet.up_blocks.0.attentions.1"
    CLASS_START = 0
    CLASS_END = 20
    SEED = 42
    NUM_INFERENCE_STEPS = 4  # SDXL-Turbo uses 4 steps
    CACHE_EVERY_N_TIMESTEPS = 1
    
    # Check for flags
    dry_run = "--dry-run" in sys.argv
    skip_backup = "--skip-backup" in sys.argv
    
    recover_all_styles(
        BASE_PATH, 
        HOOK_NAME,
        CLASS_START,
        CLASS_END,
        SEED,
        NUM_INFERENCE_STEPS,
        CACHE_EVERY_N_TIMESTEPS,
        dry_run=dry_run,
        skip_backup=skip_backup
    )