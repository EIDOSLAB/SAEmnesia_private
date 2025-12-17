"""
Collect activations from a diffusion model for nudity detection and save them to a file.
Organizes activations into 'nudity' and 'non_nudity' directories.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def run():
    from SAE.nudity_finetuning_activations_runner import CacheActivationsRunner
    from SAE.config import FineTuningCacheActivationsRunnerConfig
    from simple_parsing import parse
    
    print("Running nudity detection activation caching")
    print("Activations will be organized into 'nudity' and 'non_nudity' directories")
    
    # Parse configuration
    args = parse(FineTuningCacheActivationsRunnerConfig)
    
    # Validate that prompts_path is set
    if not hasattr(args, 'prompts_path') or args.prompts_path is None:
        print("ERROR: prompts_path must be specified in the config")
        print("This should point to your file containing ~30k COCO captions + nudity prompts")
        sys.exit(1)
    
    if not os.path.exists(args.prompts_path):
        print(f"ERROR: prompts file not found at: {args.prompts_path}")
        sys.exit(1)
    
    print(f"Reading prompts from: {args.prompts_path}")
    print(f"Saving activations to: {args.new_cached_activations_path}")
    print(f"Hook names: {args.hook_names}")
    
    # Run the cache activations runner
    runner = CacheActivationsRunner(args)
    datasets = runner.run()
    
    print("\n✅ Activation caching complete!")
    print(f"Activations saved in:")
    print(f"  - {args.new_cached_activations_path}/<hook_name>/nudity/")
    print(f"  - {args.new_cached_activations_path}/<hook_name>/non_nudity/")

if __name__ == "__main__":
    run()