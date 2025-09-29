#!/usr/bin/env python
"""
Analyze SAE checkpoint: score distributions, latent assignments, entropy, and activations for themes.

This script loads a trained SAE, evaluates the importance scores for style themes,
calculates entropy of activations, and logs metrics using wandb.

Memory-optimized version to handle large models and datasets.
"""

import os
import sys
import pickle
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from collections import defaultdict
import seaborn as sns
from tqdm.auto import tqdm
import json

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import SAE and utils - this might need to be adjusted based on your import structure
try:
    from SAE.sae import Sae
except ImportError:
    print("Warning: Unable to import Sae directly, will attempt to use manual loading")
    Sae = None


class SAEThemeLatentAnalyzer:
    """
    Analyzer for SAE models to evaluate theme-specific latent assignments and distributions.
    
    This analyzer:
    1. Loads raw activations for different themes
    2. Calculates feature importance scores for each theme
    3. Computes entropy of activations and latent distributions
    4. Logs metrics and visualizations using wandb
    """
    def __init__(
        self,
        checkpoint_path,
        style_activations_path,
        device="cuda",
        output_dir="sae-theme-analysis",
        wandb_project="sae_theme_latent_analyzer",
        run_name=None,
        seed=42,
        batch_size=8,  # Added smaller default batch size
        max_samples=1000,  # Added max samples limit
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.style_activations_path = Path(style_activations_path)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.wandb_project = wandb_project
        self.run_name = run_name or f"sae_theme_latent_analysis_{int(time.time())}"
        self.seed = seed
        self.batch_size = batch_size  # Store batch size
        self.max_samples = max_samples  # Store max samples
        
        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Will be populated in initialize() methods
        self.style_activations_dict = {}
        self.sae = None
        self.scores = {}
        self.distributions = {}
        self.entropy = {}
        self.theme_to_latent = {}
        
        # Initialize everything
        self.load_style_activations()
        self.initialize_sae()
        self.initialize_wandb()
    
    def load_style_activations(self):
        """
        Load the style activations dictionary from a pickle file.
        """
        print(f"Loading style activations from {self.style_activations_path}")
        try:
            with open(self.style_activations_path, "rb") as f:
                self.style_activations_dict = pickle.load(f)
            
            # Limit the number of samples per theme to save memory
            limited_activations_dict = {}
            for theme, activations in self.style_activations_dict.items():
                if len(activations) > self.max_samples:
                    print(f"  Limiting theme '{theme}' from {len(activations)} to {self.max_samples} samples")
                    # Use random sampling to get a representative subset
                    indices = np.random.choice(len(activations), self.max_samples, replace=False)
                    limited_activations_dict[theme] = activations[indices]
                else:
                    limited_activations_dict[theme] = activations
            
            self.style_activations_dict = limited_activations_dict
            
            # Print summary of loaded data
            print("\nStyle Activations Dictionary Summary:")
            for theme, activations in self.style_activations_dict.items():
                print(f"  - Theme '{theme}': Shape {activations.shape}")
        except Exception as e:
            raise ValueError(f"Failed to load style activations: {e}")
    
    def initialize_sae(self):
        """
        Load SAE model from checkpoint.
        """
        print(f"Loading SAE model from {self.checkpoint_path}")
        
        try:
            # First, try loading directly from the SAE class method
            if Sae is not None:
                try:
                    self.sae = Sae.load_from_disk(self.checkpoint_path, device=self.device)
                except Exception as e:
                    print(f"Error loading with Sae.load_from_disk: {e}")
                    print("Attempting manual loading instead...")
                    self.sae = None
            
            # If that fails, try manual loading
            if self.sae is None:
                # Load config
                config_path = self.checkpoint_path / "cfg.json"
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                
                # Load tensor file
                weights_path = self.checkpoint_path / "sae.safetensors"
                
                if not weights_path.exists():
                    raise FileNotFoundError(f"Could not find weights file at {weights_path}")
                
                # Create a minimal SAE structure with required attributes for our analysis
                class MinimalSAE:
                    def __init__(self, cfg, device):
                        self.cfg = type('SaeConfig', (), cfg)
                        self.device = device
                        
                        # Calculate d_sae if not present
                        if not hasattr(self.cfg, 'd_sae'):
                            if hasattr(self.cfg, 'd_in') and hasattr(self.cfg, 'expansion_factor'):
                                self.cfg.d_sae = self.cfg.d_in * self.cfg.expansion_factor
                            else:
                                raise ValueError("Config missing required attributes: d_in and/or expansion_factor")
                        
                        # Set num_latents from d_sae if not present
                        if not hasattr(self.cfg, 'num_latents') or self.cfg.num_latents == 0:
                            self.num_latents = self.cfg.d_sae
                        else:
                            self.num_latents = self.cfg.num_latents
                            
                        # Load tensor file using safetensors
                        try:
                            from safetensors import safe_open
                            with safe_open(weights_path, framework="pt", device=device) as f:
                                tensors = {key: f.get_tensor(key) for key in f.keys()}
                                
                                # Extract encoder/decoder weights
                                if "encoder.weight" in tensors and "decoder.weight" in tensors:
                                    self.encode_mat = tensors["encoder.weight"]
                                    self.decode_mat = tensors["decoder.weight"]
                                else:
                                    # Try alternate names
                                    possible_encoder_keys = [k for k in tensors.keys() if "encode" in k.lower() and "weight" in k.lower()]
                                    possible_decoder_keys = [k for k in tensors.keys() if "decode" in k.lower() and "weight" in k.lower()]
                                    
                                    if possible_encoder_keys and possible_decoder_keys:
                                        self.encode_mat = tensors[possible_encoder_keys[0]]
                                        self.decode_mat = tensors[possible_decoder_keys[0]]
                                    else:
                                        raise ValueError(f"Could not find encoder/decoder weights in {list(tensors.keys())}")
                                
                                # Extract biases if present
                                if "encoder.bias" in tensors:
                                    self.encode_bias = tensors["encoder.bias"]
                                else:
                                    self.encode_bias = torch.zeros(self.num_latents, device=device)
                                    
                                if "decoder.bias" in tensors or "pre.bias" in tensors:
                                    key = "decoder.bias" if "decoder.bias" in tensors else "pre.bias"
                                    self.decode_bias = tensors[key]
                                else:
                                    self.decode_bias = torch.zeros(self.cfg.d_in, device=device)
                        except ImportError:
                            print("safetensors not found, trying torch.load...")
                            tensors = torch.load(weights_path, map_location=device)
                            if hasattr(tensors, "state_dict"):
                                tensors = tensors.state_dict()
                            
                            # Extract weights with similar logic as above
                            if "encoder.weight" in tensors and "decoder.weight" in tensors:
                                self.encode_mat = tensors["encoder.weight"]
                                self.decode_mat = tensors["decoder.weight"]
                            else:
                                raise ValueError(f"Could not find encoder/decoder weights")
                
                    def pre_acts(self, x):
                        """Compute pre-activations for input activations."""
                        # Remove batch dimension if needed (activations have shape [batch_size, d_in])
                        if len(x.shape) == 2:
                            # Modified to manually apply ReLU here instead of using torch.nn.functional
                            # to avoid compatibility issues
                            result = (x @ self.encode_mat.t()) + self.encode_bias
                            return result.clamp(min=0)  # Manual ReLU
                        elif len(x.shape) == 3:
                            # For 3D tensors [batch_size, spatial_dim, d_in]
                            reshape_flag = True
                            orig_shape = x.shape
                            x = x.reshape(-1, x.shape[-1])
                            pre_acts = (x @ self.encode_mat.t()) + self.encode_bias
                            pre_acts = pre_acts.clamp(min=0)  # Manual ReLU
                            return pre_acts.reshape(orig_shape[0], orig_shape[1], -1)
                        else:
                            raise ValueError(f"Unexpected activation shape: {x.shape}")
                
                    def select_topk(self, pre_acts):
                        """Select top-k activations."""
                        k = getattr(self.cfg, 'k', 32)  # Default to 32 if not specified
                        
                        # Handle 2D and 3D inputs
                        if len(pre_acts.shape) == 2:
                            # Get top-k values and indices
                            topk_values, topk_indices = torch.topk(pre_acts, k, dim=1)
                            # Apply ReLU
                            topk_values = topk_values.clamp(min=0)  # Manual ReLU
                            
                            # Create sparse representation
                            acts = torch.zeros_like(pre_acts)
                            
                            # Scatter values
                            batch_indices = torch.arange(pre_acts.shape[0], device=pre_acts.device).unsqueeze(1).expand_as(topk_indices)
                            acts[batch_indices, topk_indices] = topk_values
                            
                            return acts, topk_indices
                        elif len(pre_acts.shape) == 3:
                            # Reshape to 2D, apply topk, then reshape back
                            batch_size, spatial_dim, feature_dim = pre_acts.shape
                            pre_acts_2d = pre_acts.reshape(-1, feature_dim)
                            
                            # Get top-k values and indices
                            topk_values, topk_indices = torch.topk(pre_acts_2d, k, dim=1)
                            # Apply ReLU
                            topk_values = topk_values.clamp(min=0)  # Manual ReLU
                            
                            # Create sparse representation
                            acts_2d = torch.zeros_like(pre_acts_2d)
                            
                            # Scatter values
                            batch_indices = torch.arange(pre_acts_2d.shape[0], device=pre_acts.device).unsqueeze(1).expand_as(topk_indices)
                            acts_2d[batch_indices, topk_indices] = topk_values
                            
                            # Reshape back
                            acts = acts_2d.reshape(batch_size, spatial_dim, feature_dim)
                            topk_indices = topk_indices.reshape(batch_size, spatial_dim, k)
                            
                            return acts, topk_indices
                        else:
                            raise ValueError(f"Unexpected activation shape: {pre_acts.shape}")
                
                    def decode(self, acts):
                        """Decode activations back to input space."""
                        return acts @ self.decode_mat + self.decode_bias
                    
                    def eval(self):
                        """Set model to evaluation mode."""
                        return self
                
                # Create instance of our minimal SAE class
                self.sae = MinimalSAE(cfg, self.device)
                print("Successfully loaded SAE using manual implementation.")
                
            # Ensure necessary attributes exist
            if not hasattr(self.sae, 'num_latents'):
                if hasattr(self.sae, 'encode_mat'):
                    self.sae.num_latents = self.sae.encode_mat.shape[0]
                elif hasattr(self.sae.cfg, 'd_sae'):
                    self.sae.num_latents = self.sae.cfg.d_sae
                else:
                    raise ValueError("Could not determine num_latents")
            
            expansion_factor = getattr(self.sae.cfg, 'expansion_factor', None)
            print(f"Loaded SAE with {self.sae.num_latents} latents, expansion factor {expansion_factor}")
        except Exception as e:
            print(f"Could not load SAE from {self.checkpoint_path}: {e}")
            raise

    def initialize_wandb(self):
        """
        Initialize weights and biases for logging in offline mode.
        """
        if WANDB_AVAILABLE:
            # Create directory for wandb logs
            wandb_dir = os.path.join(self.output_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            
            # Set environment variable to run wandb in offline mode
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = wandb_dir
            
            config = {
                "checkpoint_path": str(self.checkpoint_path),
                "style_activations_path": str(self.style_activations_path),
                "seed": self.seed,
                "num_latents": self.sae.num_latents,
                "batch_size": self.batch_size,
                "max_samples": self.max_samples,
            }
            
            wandb.init(
                project=self.wandb_project,
                name=self.run_name,
                config=config,
                dir=wandb_dir
            )
            
            # Log available themes
            wandb.config.update({"themes": list(self.style_activations_dict.keys())})
            
            print(f"Initialized wandb logging in OFFLINE mode")
            print(f"Logs will be stored in: {wandb_dir}")
            print(f"Project: '{self.wandb_project}', Run: '{self.run_name}'")
            print(f"To sync these logs later, run: `wandb sync {wandb_dir}`")
        else:
            print("Wandb not available, skipping wandb initialization")
    
    def process_in_batches(self, activations, batch_size=None):
        """
        Process activations in batches to avoid memory issues
        
        Args:
            activations: Activations to process
            batch_size: Batch size to use, defaults to self.batch_size
            
        Returns:
            all_pre_acts: Pre-activations for all inputs
        """
        batch_size = batch_size or self.batch_size
        all_pre_acts = []
        
        # Convert to tensor if needed
        if not isinstance(activations, torch.Tensor):
            is_tensor = False
        else:
            is_tensor = True
        
        # Process in batches
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size]
            if not is_tensor:
                batch = torch.from_numpy(batch).to(self.device)
            else:
                batch = batch.to(self.device)
            
            # Get pre-activations
            with torch.no_grad():
                batch_pre_acts = self.sae.pre_acts(batch)
                all_pre_acts.append(batch_pre_acts.cpu())  # Move to CPU to save GPU memory
            
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Combine results if there were multiple batches
        if len(all_pre_acts) > 1:
            return torch.cat(all_pre_acts, dim=0)
        else:
            return all_pre_acts[0]
    
    def compute_score(self, theme, timestep=None, theme_activations=None, non_theme_activations=None):
        """
        Compute importance score for a theme at a specific timestep as defined in the paper.
        
        Args:
            theme: The theme to compute score for
            timestep: The timestep to use (None for all timesteps)
            theme_activations: Optional pre-computed activations for the theme
            non_theme_activations: Optional pre-computed activations for other themes
            
        Returns:
            scores: Array of scores for each latent
        """
        # Collect activations for the theme
        if theme_activations is None:
            if theme not in self.style_activations_dict:
                raise ValueError(f"Theme '{theme}' not found in activations dict")
                
            if timestep is not None:
                theme_acts = self.style_activations_dict[theme][timestep:timestep+1]
            else:
                theme_acts = self.style_activations_dict[theme]
                
            # Get pre-activations (latent features) in batches
            theme_activations = self.process_in_batches(theme_acts)
        
        # Collect activations for non-theme samples
        if non_theme_activations is None:
            non_theme_acts = []
            
            # To avoid memory issues, limit samples per theme for non-theme activations
            max_samples_per_theme = min(self.max_samples // (len(self.style_activations_dict) - 1), 200)
            
            for other_theme, acts in self.style_activations_dict.items():
                if other_theme != theme:
                    if timestep is not None:
                        non_theme_acts.append(acts[timestep:timestep+1])
                    else:
                        # If acts is too large, sample a subset
                        if len(acts) > max_samples_per_theme:
                            indices = np.random.choice(len(acts), max_samples_per_theme, replace=False)
                            non_theme_acts.append(acts[indices])
                        else:
                            non_theme_acts.append(acts)
            
            if non_theme_acts:
                non_theme_acts = np.concatenate(non_theme_acts)
                # Process in batches
                non_theme_activations = self.process_in_batches(non_theme_acts)
            else:
                raise ValueError("No non-theme activations found")
        
        # Calculate mean activations
        mean_theme_acts = torch.mean(theme_activations, dim=0)
        mean_non_theme_acts = torch.mean(non_theme_activations, dim=0)
        
        # Move to CPU to save memory
        mean_theme_acts = mean_theme_acts.cpu()
        mean_non_theme_acts = mean_non_theme_acts.cpu()
        
        # Compute normalized score as defined in the paper
        sum_theme_acts = torch.sum(mean_theme_acts) + 1e-10
        sum_non_theme_acts = torch.sum(mean_non_theme_acts) + 1e-10
        
        normalized_theme_acts = mean_theme_acts / sum_theme_acts
        normalized_non_theme_acts = mean_non_theme_acts / sum_non_theme_acts
        
        scores = normalized_theme_acts - normalized_non_theme_acts
        return scores.numpy()
    
    def compute_entropy(self, activations):
        """
        Compute entropy of activation distribution
        
        Args:
            activations: Tensor of activations [batch_size, num_latents]
            
        Returns:
            entropy: Entropy value
        """
        # Sum activations across batch to get latent distribution
        dist = torch.sum(activations, dim=0)
        # Normalize to get probability distribution
        dist = dist / (torch.sum(dist) + 1e-10)
        # Compute entropy
        entropy = -torch.sum(dist * torch.log2(dist + 1e-10))
        return entropy.item()
    
    def compute_theme_latent_distribution(self, theme):
        """
        Compute the distribution of latent activations for a theme.

        Args:
            theme: The theme to analyze

        Returns:
            dict: Distribution statistics
        """
        activations = self.style_activations_dict[theme]
        
        self.sae.eval()
        with torch.no_grad():
            # Process activations in batches to avoid memory issues
            all_pre_acts = []
            all_top_indices = []
            
            for i in range(0, len(activations), self.batch_size):
                batch = activations[i:i+self.batch_size]
                if not isinstance(batch, torch.Tensor):
                    batch = torch.from_numpy(batch).to(self.device)
                else:
                    batch = batch.to(self.device)
                
                # Get pre-activations
                pre_acts = self.sae.pre_acts(batch)
                
                # Get top-k indices
                acts, top_indices = self.sae.select_topk(pre_acts)
                
                all_pre_acts.append(pre_acts.cpu())  # Move to CPU to save memory
                all_top_indices.append(top_indices.cpu())
                
                # Clear GPU cache
                torch.cuda.empty_cache()
            
            # Combine results
            all_pre_acts = torch.cat(all_pre_acts, dim=0)
            all_top_indices = torch.cat(all_top_indices, dim=0)
            
            # Count frequency of each latent
            latent_counts = torch.zeros(self.sae.num_latents, dtype=torch.float)
            
            # Process in chunks for memory efficiency
            for i in range(0, all_top_indices.shape[0], 1000):
                end = min(i + 1000, all_top_indices.shape[0])
                chunk = all_top_indices[i:end]
                
                # Flatten and count
                flat_indices = chunk.flatten()
                for idx in range(self.sae.num_latents):
                    latent_counts[idx] += (flat_indices == idx).sum().item()
            
            # Normalize to get distribution
            total_counts = torch.sum(latent_counts)
            dist = latent_counts / total_counts if total_counts > 0 else latent_counts
            
            # Get dominant latent
            dominant_latent = torch.argmax(dist).item()
            
            # Calculate entropy as a measure of concentration
            entropy = -torch.sum(dist * torch.log2(dist + 1e-10))
            
            # Calculate average and std of activations for each latent
            # Use memory-efficient approach
            mean_acts = torch.zeros(self.sae.num_latents)
            std_acts = torch.zeros(self.sae.num_latents)
            
            # Process in chunks
            chunk_size = 1000
            num_chunks = (all_pre_acts.shape[0] + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, all_pre_acts.shape[0])
                chunk = all_pre_acts[start_idx:end_idx]
                
                # Update running mean and variance
                chunk_mean = torch.mean(chunk, dim=0)
                chunk_std = torch.std(chunk, dim=0)
                
                # Simple approximation - could be improved with Welford's algorithm
                if i == 0:
                    mean_acts = chunk_mean
                    std_acts = chunk_std
                else:
                    weight = (end_idx - start_idx) / end_idx
                    mean_acts = (1 - weight) * mean_acts + weight * chunk_mean
                    std_acts = (1 - weight) * std_acts + weight * chunk_std
            
            # Store distribution statistics
            stats = {
                "distribution": dist.numpy(),
                "dominant_latent": dominant_latent,
                "dominance_score": dist[dominant_latent].item(),
                "entropy": entropy.item(),
                "mean_activations": mean_acts.numpy(),
                "std_activations": std_acts.numpy(),
            }
            
            return stats
    
    def assign_themes_to_latents(self):
        """
        Assign each theme to a specific latent based on activation scores.
            
        Returns:
            theme_to_latent: Dictionary mapping themes to latent indices
        """
        print(f"\nAssigning themes to latents...")
        theme_to_latent = {}
        latent_assignments = set()
        
        # Process each theme to find its most responsive latent
        for theme in self.style_activations_dict.keys():
            # Compute scores for this theme
            scores = self.compute_score(theme)
            
            # Find the latent with highest score
            latent_idx = np.argmax(scores)
            
            # Avoid duplicate assignments by finding the next best if needed
            while latent_idx in latent_assignments and len(latent_assignments) < self.sae.num_latents:
                scores[latent_idx] = -float('inf')
                latent_idx = np.argmax(scores)
            
            # Assign this latent to the theme
            theme_to_latent[theme] = latent_idx
            latent_assignments.add(latent_idx)
            
            print(f"  Theme '{theme}' assigned to latent {latent_idx}")
            
            # Clear memory
            torch.cuda.empty_cache()
        
        return theme_to_latent
    
    def analyze_timestep_scores(self, theme, timesteps=range(100)):
        """
        Analyze scores across specified timesteps for a theme.
        
        Args:
            theme: The theme to analyze
            timesteps: Range of timesteps to analyze
            
        Returns:
            scores_per_timestep: List of scores for each timestep
        """
        scores_per_timestep = []
        
        for t in tqdm(timesteps, desc=f"Analyzing scores for {theme}"):
            scores = self.compute_score(theme, timestep=t)
            scores_per_timestep.append(scores)
            
            # Clear memory
            torch.cuda.empty_cache()
        
        return np.array(scores_per_timestep)
    
    def plot_score_distribution(self, theme=None):
        """
        Plot distribution of scores for a theme or all themes.
        
        Args:
            theme: Specific theme to plot or None for all themes
        """
        if theme:
            themes = [theme]
        else:
            themes = list(self.scores.keys())
        
        plt.figure(figsize=(12, 8))
        
        for theme in themes:
            scores = self.scores[theme].flatten()
            sns.histplot(scores, kde=True, label=theme, alpha=0.5)
        
        plt.title("Distribution of Feature Importance Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save to file
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, f"score_distribution{'_'+theme if theme else ''}.png"))
        
        if WANDB_AVAILABLE:
            wandb.log({f"score_distribution{'_'+theme if theme else ''}": wandb.Image(plt)})
        
        plt.close()
    
    def plot_theme_latent_heatmap(self):
        """
        Plot heatmap of theme-to-latent assignments.
        """
        themes = list(self.theme_to_latent.keys())
        num_themes = len(themes)
        
        # Create matrix of assignments
        matrix = np.zeros((num_themes, self.sae.num_latents))
        
        for i, theme in enumerate(themes):
            latent_idx = self.theme_to_latent[theme]
            matrix[i, latent_idx] = 1
        
        plt.figure(figsize=(20, max(8, num_themes * 0.3)))
        sns.heatmap(matrix, 
                    yticklabels=themes, 
                    xticklabels=range(self.sae.num_latents), 
                    cmap='viridis',
                    cbar_kws={'label': 'Assignment'})
        
        plt.title("Theme to Latent Assignments")
        plt.xlabel("Latent Index")
        plt.ylabel("Theme")
        
        # Save to file
        plt.savefig(os.path.join(self.output_dir, "theme_latent_heatmap.png"))
        
        if WANDB_AVAILABLE:
            wandb.log({"theme_latent_heatmap": wandb.Image(plt)})
        
        plt.close()
    
    def analyze_theme_latents(self):
        """
        Perform analysis of theme latents and log results.
        """
        print("\nAnalyzing theme latents...")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Assign themes to latents
        self.theme_to_latent = self.assign_themes_to_latents()
        
        # Save theme-to-latent mapping
        with open(os.path.join(self.output_dir, "theme_to_latent.pkl"), "wb") as f:
            pickle.dump(self.theme_to_latent, f)
        
        # Compute distributions for all themes
        for theme in tqdm(self.style_activations_dict.keys(), desc="Computing distributions"):
            self.distributions[theme] = self.compute_theme_latent_distribution(theme)
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Compute scores for all themes
        all_scores = []
        for theme in tqdm(self.style_activations_dict.keys(), desc="Computing scores"):
            scores = self.compute_score(theme)
            self.scores[theme] = scores
            all_scores.append(scores)
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        # Combine all scores and compute percentiles
        all_scores = np.concatenate([s.reshape(1, -1) for s in all_scores], axis=0)
        percentiles = {
            '99.999th': np.percentile(all_scores, 99.999),
            '99.99th': np.percentile(all_scores, 99.99),
            '99.9th': np.percentile(all_scores, 99.9),
            '99th': np.percentile(all_scores, 99),
            '95th': np.percentile(all_scores, 95),
            '90th': np.percentile(all_scores, 90),
            '75th': np.percentile(all_scores, 75),
            '50th': np.percentile(all_scores, 50),
            '25th': np.percentile(all_scores, 25),
            '10th': np.percentile(all_scores, 10),
            '5th': np.percentile(all_scores, 5),
            '1st': np.percentile(all_scores, 1),
        }
        
        # Log percentiles
        print("\nScore Percentiles:")
        for p, value in percentiles.items():
            print(f"  {p}: {value:.6f}")
        
        # Compute average and std of scores for each theme
        theme_score_stats = {}
        for theme, scores in self.scores.items():
            theme_score_stats[theme] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores),
            }
        
        # Log theme score stats
        print("\nTheme Score Statistics:")
        for theme, stats in theme_score_stats.items():
            print(f"  {theme}:")
            for metric, value in stats.items():
                print(f"    {metric}: {value:.6f}")
        
        # Log entropy values
        print("\nTheme Entropy Values:")
        for theme, dist in self.distributions.items():
            print(f"  {theme}: {dist['entropy']:.6f}")
        
        # Plot score distributions
        self.plot_score_distribution()
        
        # Plot theme-latent heatmap
        self.plot_theme_latent_heatmap()
        
        # Log statistics to wandb
        if WANDB_AVAILABLE:
            # Log percentiles
            wandb.log({"percentiles": wandb.Table(
                columns=["Percentile", "Value"],
                data=[[p, v] for p, v in percentiles.items()]
            )})
            
            # Log theme score stats
            wandb.log({"theme_score_stats": wandb.Table(
                columns=["Theme", "Mean", "Std", "Max", "Min"],
                data=[[theme, stats['mean'], stats['std'], stats['max'], stats['min']] 
                    for theme, stats in theme_score_stats.items()]
            )})
            
            # Log entropy values
            wandb.log({"theme_entropy": wandb.Table(
                columns=["Theme", "Entropy"],
                data=[[theme, dist['entropy']] for theme, dist in self.distributions.items()]
            )})
            
            # Log latent assignment table
            wandb.log({"theme_latent_assignment": wandb.Table(
                columns=["Theme", "Latent"],
                data=[[theme, latent] for theme, latent in self.theme_to_latent.items()]
            )})
            
            # Log dominant latent scores
            wandb.log({"dominant_latent_scores": wandb.Table(
                columns=["Theme", "Dominant Latent", "Dominance Score"],
                data=[[theme, dist['dominant_latent'], dist['dominance_score']] 
                    for theme, dist in self.distributions.items()]
            )})
            
            # Log latent activation histograms for each theme - doing this in batches to avoid memory issues
            for theme, dist in self.distributions.items():
                # Only log histogram data, not the entire distribution
                wandb.log({
                    f"latent_distribution/{theme}": wandb.Histogram(np.clip(dist["distribution"], 0, 1)),
                    f"latent_activation/mean/{theme}": wandb.Histogram(dist["mean_activations"]),
                    f"latent_activation/std/{theme}": wandb.Histogram(dist["std_activations"])
                })

    def finish(self):
        """
        Properly finish the analysis, closing wandb and cleaning up resources.
        """
        print("Finalizing analysis...")
        
        # Close wandb if it's being used
        if WANDB_AVAILABLE and wandb.run is not None:
            print("Finishing wandb run...")
            wandb.finish()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("Analysis finished successfully!")

def main():
    """
    Main entry point for the SAE Theme Latent Analyzer.
    """
    parser = argparse.ArgumentParser(description="Analyze SAE models for theme-specific latent assignments.")
    
    # Required parameters
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to the SAE checkpoint directory"
    )
    parser.add_argument(
        "--style_activations_path", 
        type=str, 
        required=True, 
        help="Path to the style activations dictionary pickle file"
    )
    
    # Optional parameters
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="sae-theme-analysis", help="Directory to save analysis outputs")
    parser.add_argument("--wandb_project", type=str, default="sae_theme_latent_analyzer", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="Name for this analysis run")
    parser.add_argument("--analyze_timesteps", action="store_true", help="Analyze trends across timesteps")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps to analyze")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing activations")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to use per theme")
    parser.add_argument("--memory_efficient", action="store_true", help="Use more memory-efficient processing")
    
    args = parser.parse_args()
    
    # If memory_efficient flag is set, adjust batch size and max_samples automatically
    if args.memory_efficient:
        args.batch_size = min(args.batch_size, 4)  # Smaller batch size
        args.max_samples = min(args.max_samples, 500)  # Fewer samples
        print(f"Memory-efficient mode: batch_size={args.batch_size}, max_samples={args.max_samples}")
    
    # Create and run the analyzer
    analyzer = SAEThemeLatentAnalyzer(
        checkpoint_path=args.checkpoint_path,
        style_activations_path=args.style_activations_path,
        device=args.device,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        seed=args.seed,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    
    try:
        # Configure PyTorch for memory efficiency
        torch.cuda.empty_cache()
        
        # If using CUDA, try setting memory allocation strategy to reduce fragmentation
        if args.device == "cuda" and torch.cuda.is_available():
            # Set environment variable if not already set
            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")
        
        # Analyze theme latents
        try:
            analyzer.analyze_theme_latents()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("\n❌ CUDA out of memory during theme latent analysis.")
                print("Try running again with smaller batch_size and max_samples values:")
                print("  --batch_size 4 --max_samples 500 --memory_efficient")
                raise
            else:
                raise
        
        # Analyze timestep trends if requested
        if args.analyze_timesteps:
            try:
                analyzer.analyze_timestep_trends(num_timesteps=args.num_timesteps)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("\n❌ CUDA out of memory during timestep analysis. Skipping this part.")
                    # Continue with the rest of the analysis
                else:
                    raise
        
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Analysis failed with error: {e}")
        raise
    finally:
        # Clean up
        analyzer.finish()


if __name__ == "__main__":
    main()