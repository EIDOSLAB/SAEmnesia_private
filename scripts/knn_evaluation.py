"""
K-nearest neighbors classification on SAE features.
Section 5.2.1 from the paper.
"""
import os
import fire
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, List
import tqdm


def load_activations(activations_path: str, hookpoint: str) -> Dict[str, torch.Tensor]:
    """Load pre-computed SAE activations from disk."""
    filename = f"cls_latents_dict_{hookpoint}.pkl"
    filepath = os.path.join(activations_path, filename)
    
    print(f"Loading activations from: {filepath}")
    with open(filepath, "rb") as f:
        cls_latents_dict = pickle.load(f)
    
    print(f"Loaded {len(cls_latents_dict)} classes")
    first_class = list(cls_latents_dict.keys())[0]
    print(f"Activation shape for '{first_class}': {cls_latents_dict[first_class].shape}")
    
    return cls_latents_dict


def select_features_score_based(
    cls_latents_dict: Dict[str, torch.Tensor],
    timestep_idx: int,
    num_features_per_class: int = 2
) -> np.ndarray:
    """Select features based on importance scores (Eq. 4 from paper)."""
    class_names = sorted(cls_latents_dict.keys())
    num_classes = len(class_names)
    
    all_acts_list = []
    class_acts_dict = {}
    
    for class_name in class_names:
        acts = cls_latents_dict[class_name][:, timestep_idx, :]  # [num_samples, num_features]
        all_acts_list.append(acts)
        class_acts_dict[class_name] = acts
    
    all_acts = torch.cat(all_acts_list, dim=0)
    
    feature_scores = []
    
    for class_idx, class_name in enumerate(class_names):
        class_acts = class_acts_dict[class_name]
        
        # Get other classes
        other_acts = []
        for other_name in class_names:
            if other_name != class_name:
                other_acts.append(class_acts_dict[other_name])
        other_acts = torch.cat(other_acts, dim=0)
        
        class_mean = class_acts.mean(dim=0)
        other_mean = other_acts.mean(dim=0)
        
        # Score: difference (avoids division issues)
        scores = class_mean - other_mean
        
        # Filter dead features
        epsilon = 1e-8
        active_mask = class_mean > epsilon
        scores = scores * active_mask.float()
        
        # Get top features
        top_values, top_indices = torch.topk(scores, k=num_features_per_class)
        
        for i in range(num_features_per_class):
            feature_scores.append((
                top_values[i].item(),
                top_indices[i].item(),
                class_idx
            ))
    
    # Sort and select ensuring diversity
    feature_scores.sort(reverse=True, key=lambda x: x[0])
    
    selected_features = []
    class_counts = {i: 0 for i in range(num_classes)}
    
    for score, feat_idx, class_idx in feature_scores:
        if class_counts[class_idx] < num_features_per_class:
            if feat_idx not in selected_features:
                selected_features.append(feat_idx)
                class_counts[class_idx] += 1
    
    return np.array(sorted(selected_features))


def evaluate_knn_at_timestep(
    cls_latents_dict: Dict[str, torch.Tensor],
    timestep_idx: int,
    feature_selection: str = "all",
    n_neighbors: int = 5,
    seed: int = 42
) -> float:
    """Evaluate k-NN classification at a specific timestep."""
    class_names = sorted(cls_latents_dict.keys())
    num_classes = len(class_names)
    
    features_list = []
    labels_list = []
    
    for class_idx, class_name in enumerate(class_names):
        acts = cls_latents_dict[class_name][:, timestep_idx, :]
        features_list.append(acts.numpy())
        labels_list.append(np.full(acts.shape[0], class_idx))
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    num_features = features.shape[1]
    
    if feature_selection == "score_based":
        selected_indices = select_features_score_based(
            cls_latents_dict, timestep_idx, num_features_per_class=2
        )
        features = features[:, selected_indices]
    elif feature_selection == "random":
        rng = np.random.RandomState(seed)
        num_to_select = min(num_classes * 2, num_features)
        selected_indices = rng.choice(num_features, size=num_to_select, replace=False)
        features = features[:, selected_indices]
    
    # Train/test split
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []
    
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        rng.shuffle(class_indices)
        
        split_point = int(0.8 * len(class_indices))
        train_indices.extend(class_indices[:split_point])
        test_indices.extend(class_indices[split_point:])
    
    X_train = features[train_indices]
    y_train = labels[train_indices]
    X_test = features[test_indices]
    y_test = labels[test_indices]
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    return accuracy_score(y_test, y_pred) * 100


def evaluate_across_timesteps(
    activations_path: str,
    hookpoint: str,
    num_timesteps: int = 100
) -> Dict[str, List[float]]:
    """Evaluate k-NN across all timesteps."""
    cls_latents_dict = load_activations(activations_path, hookpoint)
    
    num_classes = len(cls_latents_dict)
    random_baseline = 100.0 / num_classes
    
    results = {
        "score_based": [],
        "all_features": [],
        "random_features": [],
        "random_baseline": [random_baseline] * num_timesteps
    }
    
    print(f"\nEvaluating k-NN across {num_timesteps} timesteps...")
    
    for timestep_idx in tqdm.tqdm(range(num_timesteps)):
        acc_score = evaluate_knn_at_timestep(
            cls_latents_dict, timestep_idx, feature_selection="score_based"
        )
        results["score_based"].append(acc_score)
        
        acc_all = evaluate_knn_at_timestep(
            cls_latents_dict, timestep_idx, feature_selection="all"
        )
        results["all_features"].append(acc_all)
        
        acc_random = evaluate_knn_at_timestep(
            cls_latents_dict, timestep_idx, feature_selection="random"
        )
        results["random_features"].append(acc_random)
    
    return results


def plot_results(results: Dict[str, List[float]], save_path: str = None):
    """Plot Figure 4 from paper."""
    num_timesteps = len(results["score_based"])
    timesteps = list(range(num_timesteps, 0, -1))
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(timesteps, results["score_based"], 
             label="Top-1 single feature", linewidth=4)
    plt.plot(timesteps, results["all_features"], 
             label="All features", linewidth=4)
    plt.plot(timesteps, results["random_features"], 
             label="Random features", linewidth=4)
    plt.plot(timesteps, results["random_baseline"], 
             label="Random guess", linewidth=4, linestyle='--')
    
    # Invert x-axis to show timestep from 100 to 1
    plt.gca().invert_xaxis()
    
    # Set y-axis limits and ticks
    plt.ylim(0, 100)
    y_ticks = [0, 5, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    plt.yticks(y_ticks, fontsize=16)
    plt.xticks(fontsize=16)
    
    plt.xlabel("Timestep", fontsize=20)
    plt.ylabel("Accuracy [%]", fontsize=20)
    
    # The legend has to be outside and below the plot, two columns
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               fontsize=20, ncol=2)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")


def main(
    activations_path: str = None,
    hookpoint: str = None,
    num_timesteps: int = 100,
    results_save_path: str | None = None,
    plot_save_path: str | None = None,
    precomputed_results_path: str | None = None
):
    """Run k-NN evaluation.
    
    Args:
        activations_path: Path to the activations directory (required if not using precomputed results)
        hookpoint: Hookpoint name (required if not using precomputed results)
        num_timesteps: Number of timesteps to evaluate
        results_save_path: Path to save the results pickle file
        plot_save_path: Path to save the plot
        precomputed_results_path: Path to precomputed results pickle file. If specified, 
                                  skips computation and only generates plot.
    """
    if precomputed_results_path:
        # Load precomputed results
        print(f"Loading precomputed results from: {precomputed_results_path}")
        with open(precomputed_results_path, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded results with {len(results['score_based'])} timesteps")
    else:
        # Compute results
        if activations_path is None or hookpoint is None:
            raise ValueError("activations_path and hookpoint are required when not using precomputed results")
        
        results = evaluate_across_timesteps(activations_path, hookpoint, num_timesteps)
        
        if results_save_path:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            with open(results_save_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Results saved to: {results_save_path}")
    
    plot_results(results, save_path=plot_save_path)
    
    print("\n" + "="*60)
    for method in ["score_based", "all_features", "random_features"]:
        accs = results[method]
        print(f"{method:20s}: Mean={np.mean(accs):.2f}%, Range=[{np.min(accs):.2f}, {np.max(accs):.2f}]%")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(main)