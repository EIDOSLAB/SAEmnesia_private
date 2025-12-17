"""
K-nearest neighbors classification on SAE features.
Section 5.2.1 from the paper.
Modified for style classification.
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
    filename = f"style_latents_dict_{hookpoint}.pkl"
    filepath = os.path.join(activations_path, filename)
    
    print(f"Loading activations from: {filepath}")
    with open(filepath, "rb") as f:
        style_latents_dict = pickle.load(f)
    
    print(f"Loaded {len(style_latents_dict)} styles")
    first_style = list(style_latents_dict.keys())[0]
    print(f"Activation shape for '{first_style}': {style_latents_dict[first_style].shape}")
    
    return style_latents_dict


def select_features_score_based(
    style_latents_dict: Dict[str, torch.Tensor],
    timestep_idx: int,
    num_features_per_style: int = 2
) -> np.ndarray:
    """Select features based on importance scores (Eq. 4 from paper)."""
    style_names = sorted(style_latents_dict.keys())
    num_styles = len(style_names)
    
    all_acts_list = []
    style_acts_dict = {}
    
    for style_name in style_names:
        acts = style_latents_dict[style_name][:, timestep_idx, :]  # [num_samples, num_features]
        all_acts_list.append(acts)
        style_acts_dict[style_name] = acts
    
    all_acts = torch.cat(all_acts_list, dim=0)
    
    feature_scores = []
    
    for style_idx, style_name in enumerate(style_names):
        style_acts = style_acts_dict[style_name]
        
        # Get other styles
        other_acts = []
        for other_name in style_names:
            if other_name != style_name:
                other_acts.append(style_acts_dict[other_name])
        other_acts = torch.cat(other_acts, dim=0)
        
        style_mean = style_acts.mean(dim=0)
        other_mean = other_acts.mean(dim=0)
        
        # Score: difference (avoids division issues)
        scores = style_mean - other_mean
        
        # Filter dead features
        epsilon = 1e-8
        active_mask = style_mean > epsilon
        scores = scores * active_mask.float()
        
        # Get top features
        top_values, top_indices = torch.topk(scores, k=num_features_per_style)
        
        for i in range(num_features_per_style):
            feature_scores.append((
                top_values[i].item(),
                top_indices[i].item(),
                style_idx
            ))
    
    # Sort and select ensuring diversity
    feature_scores.sort(reverse=True, key=lambda x: x[0])
    
    selected_features = []
    style_counts = {i: 0 for i in range(num_styles)}
    
    for score, feat_idx, style_idx in feature_scores:
        if style_counts[style_idx] < num_features_per_style:
            if feat_idx not in selected_features:
                selected_features.append(feat_idx)
                style_counts[style_idx] += 1
    
    return np.array(sorted(selected_features))


def evaluate_knn_at_timestep(
    style_latents_dict: Dict[str, torch.Tensor],
    timestep_idx: int,
    feature_selection: str = "all",
    n_neighbors: int = 5,
    seed: int = 42
) -> float:
    """Evaluate k-NN classification at a specific timestep."""
    style_names = sorted(style_latents_dict.keys())
    num_styles = len(style_names)
    
    features_list = []
    labels_list = []
    
    for style_idx, style_name in enumerate(style_names):
        acts = style_latents_dict[style_name][:, timestep_idx, :]
        features_list.append(acts.numpy())
        labels_list.append(np.full(acts.shape[0], style_idx))
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    num_features = features.shape[1]
    
    if feature_selection == "score_based":
        selected_indices = select_features_score_based(
            style_latents_dict, timestep_idx, num_features_per_style=2
        )
        features = features[:, selected_indices]
    elif feature_selection == "random":
        rng = np.random.RandomState(seed)
        num_to_select = min(num_styles * 2, num_features)
        selected_indices = rng.choice(num_features, size=num_to_select, replace=False)
        features = features[:, selected_indices]
    
    # Train/test split
    rng = np.random.RandomState(seed)
    train_indices = []
    test_indices = []
    
    for style_idx in range(num_styles):
        style_mask = labels == style_idx
        style_indices = np.where(style_mask)[0]
        rng.shuffle(style_indices)
        
        split_point = int(0.8 * len(style_indices))
        train_indices.extend(style_indices[:split_point])
        test_indices.extend(style_indices[split_point:])
    
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
    style_latents_dict = load_activations(activations_path, hookpoint)
    
    num_styles = len(style_latents_dict)
    random_baseline = 100.0 / num_styles
    
    results = {
        "score_based": [],
        "all_features": [],
        "random_features": [],
        "random_baseline": [random_baseline] * num_timesteps
    }
    
    print(f"\nEvaluating k-NN across {num_timesteps} timesteps...")
    
    for timestep_idx in tqdm.tqdm(range(num_timesteps)):
        acc_score = evaluate_knn_at_timestep(
            style_latents_dict, timestep_idx, feature_selection="score_based"
        )
        results["score_based"].append(acc_score)
        
        acc_all = evaluate_knn_at_timestep(
            style_latents_dict, timestep_idx, feature_selection="all"
        )
        results["all_features"].append(acc_all)
        
        acc_random = evaluate_knn_at_timestep(
            style_latents_dict, timestep_idx, feature_selection="random"
        )
        results["random_features"].append(acc_random)
    
    return results


def plot_results(results: Dict[str, List[float]], save_path: str = None):
    """Plot Figure 4 from paper (modified for styles)."""
    num_timesteps = len(results["score_based"])
    timesteps = list(range(num_timesteps, 0, -1))
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(timesteps, results["score_based"], 
             label="Score-based selection", linewidth=2)
    plt.plot(timesteps, results["all_features"], 
             label="All features", linewidth=2)
    plt.plot(timesteps, results["random_features"], 
             label="Random features", linewidth=2)
    plt.plot(timesteps, results["random_baseline"], 
             label="Random guess", linewidth=2, linestyle='--')
    
    # Invert x-axis to show timestep from 100 to 1
    plt.gca().invert_xaxis()
    
    # Set y-axis limits and ticks
    plt.ylim(0, 100)
    y_ticks = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    plt.yticks(y_ticks)
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=12)
    plt.title("Style classification with k-nearest neighbors\nbased on SAE feature activations", fontsize=13)
    
    # Place legend in upper right to avoid overlapping with plot lines
    plt.legend(fontsize=11, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")


def main(
    activations_path: str,
    hookpoint: str,
    num_timesteps: int = 100,
    results_save_path: str = None,
    plot_save_path: str = None
):
    """Run k-NN evaluation."""
    results = evaluate_across_timesteps(activations_path, hookpoint, num_timesteps)
    
    if results_save_path:
        os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
        with open(results_save_path, "wb") as f:
            pickle.dump(results, f)
    
    plot_results(results, save_path=plot_save_path)
    
    print("\n" + "="*60)
    for method in ["score_based", "all_features", "random_features"]:
        accs = results[method]
        print(f"{method:20s}: Mean={np.mean(accs):.2f}%, Range=[{np.min(accs):.2f}, {np.max(accs):.2f}]%")
    print("="*60)


if __name__ == "__main__":
    fire.Fire(main)