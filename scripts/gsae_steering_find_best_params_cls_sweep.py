from collections import defaultdict
from pathlib import Path
import fire
import torch


def find_best_parameters(
    percentiles: list[float], alphas: list[float], base_path: str
):
    """
    Analyze metric files to find best parameters maximizing average of UA and IRA metrics
    for each class separately.
    
    Updated for decoder-only steering approach using alpha values instead of multipliers.
    
    Args:
        percentiles: List of percentiles to analyze
        alphas: List of alpha (steering factor) values to analyze
        base_path: Base path to the results directory
    """
    # Initialize dictionary to store best results per class
    best_params = defaultdict(
        lambda: {"best_avg": float("-inf"), "params": None, "metrics": None}
    )
    
    for percentile in percentiles:
        for alpha in alphas:
            # Construct path using alpha instead of multiplier
            result_path = (
                Path(base_path)
                / f"percentile_{percentile}_alpha_{alpha}"
                / "class_metrics.pth"
            )
            
            if not result_path.exists():
                print(f"Warning: File not found: {result_path}")
                continue
            
            # Load metrics
            metrics = torch.load(result_path)
            
            # Analyze each class
            for class_name, class_metrics in metrics.items():
                # Calculate average of UA and IRA for this class
                avg_metrics = (class_metrics["UA"] + class_metrics["IRA"]) / 2
                
                if avg_metrics > best_params[class_name]["best_avg"]:
                    best_params[class_name]["best_avg"] = avg_metrics
                    best_params[class_name]["params"] = (percentile, alpha)
                    best_params[class_name]["metrics"] = class_metrics
    
    # Print results and save parameters
    print("\nBest parameters for each class (maximizing average of UA and IRA):")
    print("-" * 80)
    
    # Create dictionary for saving parameters
    params_dict = {}
    
    for class_name, data in best_params.items():
        print(f"\nClass: {class_name}")
        percentile, alpha = data["params"]
        print(f"Parameters: percentile={percentile}, alpha={alpha}")
        print(f"UA:  {data['metrics']['UA']:.4f}")
        print(f"IRA: {data['metrics']['IRA']:.4f}")
        print(f"Average: {data['best_avg']:.4f}")
        
        # Save with 'alpha' key instead of 'multiplier'
        params_dict[class_name] = {"percentile": percentile, "alpha": alpha}
    
    # Save parameters
    save_path = Path(base_path) / "class_params.pth"
    torch.save(params_dict, save_path)
    print(f"\n✓ Parameters saved to: {save_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    avg_ua = sum(data["metrics"]["UA"] for data in best_params.values()) / len(best_params)
    avg_ira = sum(data["metrics"]["IRA"] for data in best_params.values()) / len(best_params)
    avg_combined = sum(data["best_avg"] for data in best_params.values()) / len(best_params)
    print(f"Average UA across all classes:  {avg_ua:.4f}")
    print(f"Average IRA across all classes: {avg_ira:.4f}")
    print(f"Average combined metric:        {avg_combined:.4f}")
    
    # Count alpha distribution
    alpha_counts = defaultdict(int)
    for data in best_params.values():
        _, alpha = data["params"]
        alpha_counts[alpha] += 1
    
    print(f"\nAlpha value distribution:")
    for alpha in sorted(alpha_counts.keys()):
        print(f"  alpha={alpha:6.1f}: {alpha_counts[alpha]:2d} classes")


if __name__ == "__main__":
    fire.Fire(find_best_parameters)