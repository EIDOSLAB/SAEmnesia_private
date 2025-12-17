"""
Collect activations from a diffusion model for a given hookpoint and save them to a file.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def run():
    # Check if mode is specified in command line arguments
    if len(sys.argv) > 1 and '--mode' in sys.argv:
        mode_index = sys.argv.index('--mode')
        if mode_index + 1 < len(sys.argv):
            mode = sys.argv[mode_index + 1]
        else:
            mode = 'normal'
    else:
        mode = 'normal'
    
    # Check for metadata generation flag
    generate_metadata = '--generate-metadata' in sys.argv
    
    print(f"Running in {mode} mode")
    if generate_metadata:
        print("Will generate metadata after caching")
    
    # Import based on mode
    if mode == 'finetuning':
        from SAE.fine_tuning_cache_activations_runner_unlearn_canvas import CacheActivationsRunner
        from SAE.config import FineTuningCacheActivationsRunnerConfig
        from simple_parsing import parse
        
        print("About to parse FineTuningCacheActivationsRunnerConfig")
        args = parse(FineTuningCacheActivationsRunnerConfig)
        
        print(f"Args parsed. organization_type = {args.organization_type}")
        print(f"All args: {vars(args)}")
        
        # Run the cache activations runner
        runner = CacheActivationsRunner(args)
        datasets = runner.run()
        
        # Generate metadata after caching (automatic in finetuning mode or if flag is set)
        if generate_metadata or mode == 'finetuning':
            print("\n🔄 Generating metadata for object+style querying...")
            runner.generate_activation_metadata()
            print("✅ Metadata generation complete!")
    
    elif mode == 'dual_object':
        # NEW: Dual object mode for two-object removal training
        from SAE.dual_object_cache_activations_runner_unlearn_canvas import CacheActivationsRunner
        from SAE.config import FineTuningCacheActivationsRunnerConfig
        from simple_parsing import parse
        
        print("🎯 Running in DUAL OBJECT mode")
        print("About to parse FineTuningCacheActivationsRunnerConfig")
        args = parse(FineTuningCacheActivationsRunnerConfig)
        
        print(f"Args parsed. organization_type = {args.organization_type}")
        print(f"All args: {vars(args)}")
        
        # Run the dual object cache activations runner
        runner = CacheActivationsRunner(args)
        datasets = runner.run()
        
        # Generate metadata after caching (automatic in dual_object mode or if flag is set)
        if generate_metadata or mode == 'dual_object':
            print("\n🔄 Generating metadata for dual-object querying...")
            runner.generate_activation_metadata()
            print("✅ Metadata generation complete!")
    
    elif mode == 'metadata':
        # Metadata-only mode for existing cached activations
        from SAE.fine_tuning_cache_activations_runner_unlearn_canvas import ActivationMetadataGenerator
        from SAE.config import FineTuningCacheActivationsRunnerConfig
        from simple_parsing import parse
        
        print("Parsing config for metadata generation...")
        args = parse(FineTuningCacheActivationsRunnerConfig)
        
        if args.new_cached_activations_path is None:
            print("❌ Error: new_cached_activations_path must be specified for metadata mode")
            sys.exit(1)
        
        print(f"Generating metadata for: {args.new_cached_activations_path}")
        generator = ActivationMetadataGenerator(args.new_cached_activations_path)
        generator.generate_metadata(args.hook_names)
        
        print("✅ Metadata generation complete!")
        return  # Exit early, don't run the normal caching
    
    elif mode == 'dual_object_metadata':
        # NEW: Metadata-only mode for dual-object cached activations
        from SAE.dual_object_cache_activations_runner_unlearn_canvas import ActivationMetadataGenerator
        from SAE.config import FineTuningCacheActivationsRunnerConfig
        from simple_parsing import parse
        
        print("Parsing config for dual-object metadata generation...")
        args = parse(FineTuningCacheActivationsRunnerConfig)
        
        if args.new_cached_activations_path is None:
            print("❌ Error: new_cached_activations_path must be specified for dual_object_metadata mode")
            sys.exit(1)
        
        print(f"Generating dual-object metadata for: {args.new_cached_activations_path}")
        generator = ActivationMetadataGenerator(args.new_cached_activations_path)
        generator.generate_metadata(args.hook_names)
        
        print("✅ Dual-object metadata generation complete!")
        return  # Exit early, don't run the normal caching
    
    else:
        from SAE.cache_activations_runner_unlearn_canvas import CacheActivationsRunner
        from SAE.config import CacheActivationsRunnerConfig
        from simple_parsing import parse
        args = parse(CacheActivationsRunnerConfig)
        
        print("About to run CacheActivationsRunner")
        
        # Run the cache activations runner
        runner = CacheActivationsRunner(args)
        datasets = runner.run()
        
        # Generate metadata if flag is set
        if generate_metadata:
            print("\n🔄 Generating metadata for object+style querying...")
            runner.generate_activation_metadata()
            print("✅ Metadata generation complete!")

if __name__ == "__main__":
    run()