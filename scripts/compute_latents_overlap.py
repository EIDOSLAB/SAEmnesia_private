import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from packaging import version
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

sys.path.append("..")

import fire

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


class FeatureCapturingHook:
    """Hook that captures the selected features from the existing SAEMaskedUnlearningHook"""
    def __init__(self, original_hook):
        self.original_hook = original_hook
        self.captured_features = {}
        self.timestep_counter = 0
        
    def __call__(self, module, input, output):
        # Call the original hook to get its behavior
        result = self.original_hook(module, input, output)
        
        # Try to extract which features were selected by examining the hook's internal state
        timestep = self.timestep_counter % self.original_hook.steps
        
        # The SAEMaskedUnlearningHook should have internal logic for feature selection
        # We can try to access its selected features or reconstruct them
        try:
            # If the hook has stored selected features, capture them
            if hasattr(self.original_hook, 'selected_features'):
                self.captured_features[timestep] = self.original_hook.selected_features.copy()
            elif hasattr(self.original_hook, 'feature_mask'):
                # If it has a feature mask, find which features are masked
                mask = self.original_hook.feature_mask
                if mask is not None:
                    selected_indices = torch.where(mask)[0].cpu().numpy().tolist()
                    self.captured_features[timestep] = selected_indices
            else:
                # Fallback: try to manually compute the features using the same logic
                self.captured_features[timestep] = self._compute_selected_features(timestep)
        except Exception as e:
            print(f"Error capturing features at timestep {timestep}: {e}")
            self.captured_features[timestep] = []
            
        self.timestep_counter += 1
        return result
    
    def _compute_selected_features(self, timestep):
        """Manually compute which features would be selected using the hook's parameters"""
        try:
            concept_name = self.original_hook.concept_to_unlearn[0]
            if concept_name in self.original_hook.concept_latents_dict:
                concept_latents = self.original_hook.concept_latents_dict[concept_name]
                if timestep in concept_latents:
                    # This is where we'd compute feature importance, but we have the shape issue
                    # For now, return empty list
                    return []
            return []
        except:
            return []


def analyze_concept_with_existing_hook(model, sae, concept_latents_dict, class_params, concept, hookpoint, steps, seed, guidance_scale):
    """
    Analyze features for a concept by wrapping the existing SAEMaskedUnlearningHook
    """
    print(f"Analyzing features for concept: {concept}")
    
    if concept not in class_params:
        print(f"Warning: {concept} not found in class_params")
        return {}
    
    if concept not in concept_latents_dict:
        print(f"Warning: {concept} not found in concept_latents_dict")
        return {}
    
    # Create the original unlearning hook
    original_hook = hooks.SAEMaskedUnlearningHook(
        concept_to_unlearn=[concept],
        percentile=class_params[concept]["percentile"],
        multiplier=class_params[concept]["multiplier"],
        feature_importance_fn=compute_feature_importance,
        concept_latents_dict=concept_latents_dict,
        sae=sae,
        steps=steps,
        preserve_error=True,
    )
    
    # Wrap it with our capturing hook
    capturing_hook = FeatureCapturingHook(original_hook)
    
    # Test with concept-specific prompts
    test_prompts = [f"An image of {concept}."]
    
    # Create hook dictionary for the model
    hook_dict = {hookpoint: capturing_hook}
    
    # Generate images to trigger feature tracking
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    with torch.no_grad():
        _ = model.run_with_hooks(
            prompt=test_prompts,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            position_hook_dict=hook_dict,
        )
    
    print(f"Captured features across {len(capturing_hook.captured_features)} timesteps for {concept}")
    
    return capturing_hook.captured_features.copy()


def analyze_offline_feature_importance(concept_latents_dict, class_params, sae, concept, steps):
    """
    Analyze features offline by directly computing importance scores from stored latents
    """
    print(f"Computing offline feature importance for: {concept}")
    
    if concept not in class_params:
        print(f"Warning: {concept} not found in class_params")
        return {}
    
    if concept not in concept_latents_dict:
        print(f"Warning: {concept} not found in concept_latents_dict")
        return {}
    
    concept_latents = concept_latents_dict[concept]
    percentile = class_params[concept]["percentile"]
    
    selected_features_per_timestep = {}
    
    print(f"Found {len(concept_latents)} timesteps for {concept}")
    
    for timestep in range(steps):
        if timestep in concept_latents:
            try:
                # Get the stored concept activations
                concept_acts = concept_latents[timestep]
                
                print(f"Timestep {timestep}: concept_acts shape = {concept_acts.shape}, type = {type(concept_acts)}")
                
                # For offline analysis, we can try a simpler approach
                # If the stored activations are already processed through SAE, 
                # we might be able to directly analyze them
                
                if isinstance(concept_acts, torch.Tensor):
                    # Convert to numpy for percentile calculation
                    acts_np = concept_acts.cpu().numpy()
                    
                    # If this is already feature activations (1D), use directly
                    if len(acts_np.shape) == 1:
                        # These might already be SAE feature activations
                        threshold = np.percentile(acts_np, percentile)
                        selected_features = np.where(acts_np > threshold)[0].tolist()
                        selected_features_per_timestep[timestep] = selected_features
                        print(f"  Found {len(selected_features)} features above {percentile}th percentile")
                    
                    # If this is 2D [timesteps, features], extract the specific timestep
                    elif len(acts_np.shape) == 2:  # [100, 20480] - timesteps x features
                        print(f"  Processing 2D activations: {acts_np.shape[0]} timesteps x {acts_np.shape[1]} features")
                        
                        # Check if we have the right timestep
                        if timestep < acts_np.shape[0]:
                            # Get feature activations for this specific timestep
                            timestep_features = acts_np[timestep]  # [20480] features for this timestep
                            
                            # Calculate how many features we should select based on percentile
                            n_features = len(timestep_features)
                            n_select = max(1, int(np.ceil(n_features * (100 - percentile) / 100)))
                            
                            # Use top-k selection instead of threshold-based selection
                            top_k_indices = np.argsort(timestep_features)[-n_select:]  # Get indices of top-k features
                            selected_features = top_k_indices.tolist()
                            selected_features_per_timestep[timestep] = selected_features
                            
                            # Get the actual threshold for reporting
                            threshold = timestep_features[top_k_indices[0]]  # Lowest value among selected features
                            
                            print(f"  Timestep {timestep}: Selected top {n_select} features (percentile {percentile})")
                            print(f"  Threshold: {threshold:.6f}, Max: {timestep_features.max():.6f}")
                            print(f"  Selected features: {selected_features}")
                            print(f"  Feature activation stats: min={timestep_features.min():.6f}, max={timestep_features.max():.6f}, mean={timestep_features.mean():.6f}")
                            
                        else:
                            print(f"  Timestep {timestep} out of range for stored data (max: {acts_np.shape[0]-1})")
                            selected_features_per_timestep[timestep] = []
                    
                    # If this is spatial activations, we need to process them
                    elif len(acts_np.shape) == 4:  # [batch, height, width, channels]
                        print(f"  Processing spatial activations of shape {acts_np.shape}")
                        
                        # Flatten spatial dimensions for SAE processing
                        batch_size, height, width, channels = acts_np.shape
                        acts_flat = acts_np.reshape(-1, channels)  # [batch*height*width, channels]
                        
                        # Convert back to tensor for SAE
                        acts_tensor = torch.from_numpy(acts_flat).to(sae.cfg.device).to(sae.dtype)
                        
                        # Encode with SAE to get feature activations
                        with torch.no_grad():
                            feature_acts = sae.encode(acts_tensor)  # [batch*height*width, n_features]
                            
                        # Average across spatial positions
                        avg_feature_acts = feature_acts.mean(dim=0).cpu().numpy()  # [n_features]
                        
                        # Select features based on percentile
                        threshold = np.percentile(avg_feature_acts, percentile)
                        selected_features = np.where(avg_feature_acts > threshold)[0].tolist()
                        selected_features_per_timestep[timestep] = selected_features
                        
                        print(f"  Found {len(selected_features)} features above {percentile}th percentile")
                    
                    else:
                        print(f"  Unsupported activation shape: {acts_np.shape}")
                        selected_features_per_timestep[timestep] = []
                
                else:
                    print(f"  Unsupported activation type: {type(concept_acts)}")
                    selected_features_per_timestep[timestep] = []
                    
            except Exception as e:
                print(f"Error processing timestep {timestep}: {e}")
                selected_features_per_timestep[timestep] = []
        else:
            selected_features_per_timestep[timestep] = []
    
    print(f"Processed {len(selected_features_per_timestep)} timesteps for {concept}")
    return selected_features_per_timestep


def analyze_cat_dog_overlap(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    class_params_path,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="feature_overlap_analysis/",
    offline_mode=True,
):
    """
    Analyze feature overlap between cats and dogs across timesteps
    """
    accelerator = Accelerator()
    device = accelerator.device

    # Load SAE and data components (we might not need the full model for offline analysis)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    
    with open(class_latents_path, "rb") as f:
        class_latents_dict = pickle.load(f)

    class_params = torch.load(class_params_path, weights_only=False)
    
    os.makedirs(output_dir, exist_ok=True)

    # Find cat and dog concepts in the data
    possible_cat_names = ["cats", "cat", "Cats", "Cat"]
    possible_dog_names = ["dogs", "dog", "Dogs", "Dog"]
    
    cat_concept = None
    dog_concept = None
    
    # Find cat concept
    for name in possible_cat_names:
        if name in class_latents_dict and name in class_params:
            cat_concept = name
            break
    
    # Find dog concept  
    for name in possible_dog_names:
        if name in class_latents_dict and name in class_params:
            dog_concept = name
            break
    
    if not cat_concept or not dog_concept:
        print("Available concepts in class_latents_dict:", list(class_latents_dict.keys()))
        print("Available concepts in class_params:", list(class_params.keys()))
        print(f"Cat concept found: {cat_concept}")
        print(f"Dog concept found: {dog_concept}")
        
        if not cat_concept:
            print("Could not find cat concept. Please check concept names.")
            return
        if not dog_concept:
            print("Could not find dog concept. Please check concept names.")
            return

    print(f"Analyzing overlap between '{cat_concept}' and '{dog_concept}' in offline mode")

    # Analyze features for each concept using offline method
    results = {}
    
    for concept in [cat_concept, dog_concept]:
        results[concept] = analyze_offline_feature_importance(
            class_latents_dict, class_params, sae, concept, steps
        )

    # Analyze and visualize overlap
    if len(results) >= 2:
        analyze_and_plot_overlap(results, cat_concept, dog_concept, steps, output_dir)
    else:
        print("Could not collect data for both concepts")

    return results


def analyze_and_plot_overlap(results, cat_concept, dog_concept, steps, output_dir):
    """Analyze and visualize feature overlap between concepts"""
    
    cat_features = results[cat_concept]
    dog_features = results[dog_concept]
    
    # Compute overlap for each timestep
    overlap_counts = []
    overlap_features = []
    total_cat_features = []
    total_dog_features = []
    
    for timestep in range(steps):
        cat_set = set(cat_features.get(timestep, []))
        dog_set = set(dog_features.get(timestep, []))
        
        overlap = cat_set.intersection(dog_set)
        overlap_counts.append(len(overlap))
        overlap_features.append(list(overlap))
        total_cat_features.append(len(cat_set))
        total_dog_features.append(len(dog_set))
    
    # Print statistics
    print(f"\n=== FEATURE OVERLAP ANALYSIS ===")
    print(f"Analyzing overlap between {cat_concept} and {dog_concept}")
    print(f"Across {steps} timesteps")
    
    total_overlaps = sum(overlap_counts)
    avg_cat_features = np.mean(total_cat_features) if total_cat_features else 0
    avg_dog_features = np.mean(total_dog_features) if total_dog_features else 0
    
    print(f"\nOverall Statistics:")
    print(f"Total overlapping feature instances: {total_overlaps}")
    print(f"Average {cat_concept} features per timestep: {avg_cat_features:.2f}")
    print(f"Average {dog_concept} features per timestep: {avg_dog_features:.2f}")
    print(f"Average overlap per timestep: {np.mean(overlap_counts):.2f}")
    print(f"Max overlap in single timestep: {np.max(overlap_counts) if overlap_counts else 0}")
    
    # Find timesteps with high overlap
    high_overlap_timesteps = [(t, count) for t, count in enumerate(overlap_counts) if count > 0]
    if high_overlap_timesteps:
        print(f"\nTimesteps with feature overlap (showing first 20):")
        for t, count in high_overlap_timesteps[:20]:
            features = overlap_features[t]
            print(f"  Timestep {t}: {count} overlapping features {features}")
    else:
        print("\nNo overlapping features found across timesteps.")
    
    # Show some example timesteps with features
    print(f"\nExample feature counts per timestep:")
    for t in range(min(10, steps)):
        cat_count = len(cat_features.get(t, []))
        dog_count = len(dog_features.get(t, []))
        if cat_count > 0 or dog_count > 0:
            print(f"  Timestep {t}: {cat_concept}={cat_count}, {dog_concept}={dog_count}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature counts per timestep
    plt.subplot(2, 2, 1)
    timesteps = list(range(steps))
    plt.plot(timesteps, total_cat_features, 'b-', label=f'{cat_concept} features', linewidth=2)
    plt.plot(timesteps, total_dog_features, 'r-', label=f'{dog_concept} features', linewidth=2)
    plt.plot(timesteps, overlap_counts, 'g-', label='Overlapping features', linewidth=2)
    plt.title('Feature Counts Across Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Number of Features')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Overlap counts
    plt.subplot(2, 2, 2)
    plt.plot(timesteps, overlap_counts, 'g-', linewidth=2)
    plt.title(f'Feature Overlap Between {cat_concept.title()} and {dog_concept.title()}')
    plt.xlabel('Timestep')
    plt.ylabel('Number of Overlapping Features')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Overlap percentage
    plt.subplot(2, 2, 3)
    overlap_percentages = []
    for i in range(steps):
        if total_cat_features[i] > 0:
            pct = (overlap_counts[i] / total_cat_features[i]) * 100
        else:
            pct = 0
        overlap_percentages.append(pct)
    
    plt.plot(timesteps, overlap_percentages, 'purple', linewidth=2)
    plt.title(f'Overlap as % of {cat_concept.title()} Features')
    plt.xlabel('Timestep')
    plt.ylabel('Overlap Percentage (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of overlap counts
    plt.subplot(2, 2, 4)
    if overlap_counts and max(overlap_counts) > 0:
        plt.hist(overlap_counts, bins=max(1, max(overlap_counts) + 1), alpha=0.7, color='green')
    else:
        plt.hist([0], bins=1, alpha=0.7, color='green')
    plt.title('Distribution of Overlap Counts')
    plt.xlabel('Number of Overlapping Features')
    plt.ylabel('Frequency (Timesteps)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'feature_overlap_{cat_concept}_vs_{dog_concept}_offline.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_data = {
        'timesteps': timesteps,
        'overlap_counts': overlap_counts,
        'overlap_features': overlap_features,
        'cat_feature_counts': total_cat_features,
        'dog_feature_counts': total_dog_features,
        'cat_concept': cat_concept,
        'dog_concept': dog_concept,
        'statistics': {
            'total_overlaps': total_overlaps,
            'avg_cat_features': avg_cat_features,
            'avg_dog_features': avg_dog_features,
            'avg_overlap': np.mean(overlap_counts),
            'max_overlap': np.max(overlap_counts) if overlap_counts else 0
        }
    }
    
    results_path = os.path.join(output_dir, f'overlap_analysis_{cat_concept}_vs_{dog_concept}_offline.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\nResults saved to:")
    print(f"  Plot: {plot_path}")
    print(f"  Data: {results_path}")


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    class_params_path,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="feature_overlap_analysis/",
    offline_mode=True,
    test_percentiles=None,  # Allow testing different percentiles
):
    """
    Main function to run the feature overlap analysis
    """
    if test_percentiles is not None:
        # Test multiple percentiles to find reasonable thresholds
        print(f"Testing percentiles: {test_percentiles}")
        
        # Load data first
        with open(class_latents_path, "rb") as f:
            class_latents_dict = pickle.load(f)
        
        class_params = torch.load(class_params_path, weights_only=False)
        
        # Find concepts
        possible_cat_names = ["cats", "cat", "Cats", "Cat"]
        possible_dog_names = ["dogs", "dog", "Dogs", "Dog"]
        
        cat_concept = None
        dog_concept = None
        
        for name in possible_cat_names:
            if name in class_latents_dict and name in class_params:
                cat_concept = name
                break
        
        for name in possible_dog_names:
            if name in class_latents_dict and name in class_params:
                dog_concept = name
                break
        
        if cat_concept and dog_concept:
            # Test different percentiles on a few timesteps
            test_timesteps = [0, 25, 50, 75, 99]
            
            print(f"\nTesting percentiles on timesteps {test_timesteps} for {cat_concept}:")
            
            for timestep in test_timesteps:
                if timestep in class_latents_dict[cat_concept]:
                    concept_acts = class_latents_dict[cat_concept][timestep].cpu().numpy()
                    if timestep < concept_acts.shape[0]:
                        timestep_features = concept_acts[timestep]
                        
                        print(f"\nTimestep {timestep}:")
                        print(f"  Feature stats: min={timestep_features.min():.6f}, max={timestep_features.max():.6f}, mean={timestep_features.mean():.6f}")
                        
                        for percentile in test_percentiles:
                            threshold = np.percentile(timestep_features, percentile)
                            selected_features = np.where(timestep_features > threshold)[0]
                            print(f"  {percentile:6.2f}th percentile: threshold={threshold:.6f}, features={len(selected_features)}")
        
        return
    
    return analyze_cat_dog_overlap(
        pipe_checkpoint=pipe_checkpoint,
        hookpoint=hookpoint,
        class_latents_path=class_latents_path,
        sae_checkpoint=sae_checkpoint,
        class_params_path=class_params_path,
        seed=seed,
        steps=steps,
        guidance_scale=guidance_scale,
        output_dir=output_dir,
        offline_mode=offline_mode,
    )


if __name__ == "__main__":
    fire.Fire(main)