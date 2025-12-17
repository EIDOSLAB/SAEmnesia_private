from collections import defaultdict
from pathlib import Path
import fire
import torch


def find_best_parameters(
    percentiles: list[float], 
    multipliers: list[float], 
    base_path: str,
    ccm_weight: float = 1.0,
    ua_weight: float = 1.0,
    ira_weight: float = 1.0,
    selection_strategy: str = "weighted_average"
):
    """
    Analyze metric files to find best parameters for each class.
    
    Args:
        percentiles: List of percentiles to analyze
        multipliers: List of multipliers to analyze
        base_path: Base path to the results directory
        ccm_weight: Weight for CCM metric (default: 1.0)
        ua_weight: Weight for UA metric (default: 1.0)
        ira_weight: Weight for IRA metric (default: 1.0)
        selection_strategy: Strategy for selecting best parameters
            - "weighted_average": Weighted average of UA, IRA, and CCM
            - "min_threshold": Ensure UA and IRA meet thresholds, then maximize CCM
            - "pareto": Find Pareto optimal solutions
    """
    # Initialize dictionary to store best results per class
    best_params = defaultdict(
        lambda: {
            "best_score": float("-inf"), 
            "params": None, 
            "metrics": None,
            "selection_info": None
        }
    )
    
    # Store all results for Pareto analysis if needed
    all_results = defaultdict(list)
    
    for percentile in percentiles:
        for multiplier in multipliers:
            # Construct path
            result_path = (
                Path(base_path)
                / f"percentile_{percentile}_multiplier_{multiplier}"
                / "class_metrics.pth"
            )
            
            if not result_path.exists():
                print(f"Warning: File not found: {result_path}")
                continue
            
            # Load metrics
            metrics = torch.load(result_path)
            
            # Analyze each class
            for class_name, class_metrics in metrics.items():
                ua = class_metrics["UA"]
                ira = class_metrics["IRA"]
                ccm = class_metrics["CCM"]
                
                # Store for potential Pareto analysis
                all_results[class_name].append({
                    "params": (percentile, multiplier),
                    "UA": ua,
                    "IRA": ira,
                    "CCM": ccm,
                    "metrics": class_metrics
                })
                
                # Calculate score based on selection strategy
                if selection_strategy == "weighted_average":
                    # Weighted average of all three metrics
                    score = (ua_weight * ua + ira_weight * ira + ccm_weight * ccm) / (ua_weight + ira_weight + ccm_weight)
                    selection_info = f"Weighted avg (UA:{ua_weight}, IRA:{ira_weight}, CCM:{ccm_weight})"
                    
                elif selection_strategy == "min_threshold":
                    # Ensure UA and IRA meet minimum thresholds, then maximize CCM
                    ua_threshold = 80.0  # Can be made configurable
                    ira_threshold = 80.0  # Can be made configurable
                    
                    if ua >= ua_threshold and ira >= ira_threshold:
                        # If thresholds are met, score is primarily CCM with bonuses for high UA/IRA
                        score = ccm + (ua - ua_threshold) * 0.1 + (ira - ira_threshold) * 0.1
                        selection_info = f"Threshold met (UA≥{ua_threshold}, IRA≥{ira_threshold}), CCM optimized"
                    else:
                        # If thresholds not met, heavily penalize
                        score = (ua + ira) / 2 - 100  # Large penalty
                        selection_info = f"Threshold NOT met (UA≥{ua_threshold}, IRA≥{ira_threshold})"
                        
                elif selection_strategy == "pareto":
                    # For Pareto, we'll do a second pass after collecting all results
                    # For now, just use weighted average as placeholder
                    score = (ua + ira + ccm) / 3
                    selection_info = "Pareto analysis (pending)"
                    
                else:
                    raise ValueError(f"Unknown selection strategy: {selection_strategy}")
                
                # Update best if this is better
                if score > best_params[class_name]["best_score"]:
                    best_params[class_name]["best_score"] = score
                    best_params[class_name]["params"] = (percentile, multiplier)
                    best_params[class_name]["metrics"] = class_metrics
                    best_params[class_name]["selection_info"] = selection_info
    
    # If Pareto strategy, perform Pareto analysis
    if selection_strategy == "pareto":
        best_params = find_pareto_optimal(all_results, best_params)
    
    # Print results and save parameters
    print("\n" + "="*100)
    print(f"BEST PARAMETERS - Strategy: {selection_strategy}")
    print("="*100)
    
    # Create dictionary for saving parameters
    params_dict = {}
    
    for class_name, data in best_params.items():
        print(f"\nClass: {class_name}")
        percentile, multiplier = data["params"]
        print(f"  Parameters: percentile={percentile}, multiplier={multiplier}")
        print(f"  UA:  {data['metrics']['UA']:.2f}%")
        print(f"  IRA: {data['metrics']['IRA']:.2f}%")
        print(f"  CCM: {data['metrics']['CCM']:.2f}%")
        print(f"    └─ Max Confidence: {data['metrics']['max_confidence']:.2f}%")
        print(f"    └─ Norm. Entropy:  {data['metrics']['normalized_entropy']:.2f}%")
        print(f"    └─ Top-2 Gap:      {data['metrics']['top2_gap']:.2f}%")
        print(f"  Score: {data['best_score']:.2f}")
        print(f"  Selection: {data['selection_info']}")
        
        params_dict[class_name] = {
            "percentile": percentile, 
            "multiplier": multiplier,
            "UA": data['metrics']['UA'],
            "IRA": data['metrics']['IRA'],
            "CCM": data['metrics']['CCM'],
            "score": data['best_score'],
            "selection_strategy": selection_strategy
        }
    
    # Save parameters
    strategy_suffix = f"_{selection_strategy}" if selection_strategy != "weighted_average" else ""
    save_path = Path(base_path) / f"class_params{strategy_suffix}.pth"
    torch.save(params_dict, save_path)
    print("\n" + "="*100)
    print(f"Parameters saved to: {save_path}")
    print("="*100 + "\n")


def find_pareto_optimal(all_results, best_params):
    """
    Find Pareto optimal solutions for each class.
    A solution is Pareto optimal if no other solution is better in all objectives.
    """
    for class_name, results in all_results.items():
        if not results:
            continue
            
        # Find Pareto front
        pareto_front = []
        for i, result_i in enumerate(results):
            is_dominated = False
            for j, result_j in enumerate(results):
                if i == j:
                    continue
                # Check if result_j dominates result_i
                # (better in all three metrics)
                if (result_j["UA"] >= result_i["UA"] and 
                    result_j["IRA"] >= result_i["IRA"] and 
                    result_j["CCM"] >= result_i["CCM"] and
                    (result_j["UA"] > result_i["UA"] or 
                     result_j["IRA"] > result_i["IRA"] or 
                     result_j["CCM"] > result_i["CCM"])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(result_i)
        
        # Among Pareto optimal solutions, choose the one with highest average
        best_in_pareto = max(pareto_front, 
                            key=lambda x: (x["UA"] + x["IRA"] + x["CCM"]) / 3)
        
        best_params[class_name]["best_score"] = (best_in_pareto["UA"] + best_in_pareto["IRA"] + best_in_pareto["CCM"]) / 3
        best_params[class_name]["params"] = best_in_pareto["params"]
        best_params[class_name]["metrics"] = best_in_pareto["metrics"]
        best_params[class_name]["selection_info"] = f"Pareto optimal ({len(pareto_front)} solutions in front)"
    
    return best_params


if __name__ == "__main__":
    fire.Fire(find_best_parameters)