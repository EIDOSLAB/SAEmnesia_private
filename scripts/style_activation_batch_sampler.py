import os
import sys
import pickle
import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.optim import Adam
from tqdm.auto import tqdm
from pathlib import Path
from transformers import get_scheduler
from collections import defaultdict
import json
import csv
import time

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Import SAE and utils
from SAE.sae import Sae
from SAE.utils import geometric_median
from SAE.unlearning_utils import compute_feature_importance, get_percentile_threshold

# Import constants
sys.path.append("..")
from UnlearnCanvas_resources.const import class_available, theme_available


class StyleActivationBatchSampler:
    """
    Batch sampler that creates batches containing raw activations from themes.
    Supports both single-theme batches and mixed-theme batches.
    """
    def __init__(self, style_activations_dict, batch_size=32, seed=42, mixed_batches=False):
        self.style_activations_dict = style_activations_dict
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.mixed_batches = mixed_batches
        
        # Available themes
        self.themes = list(style_activations_dict.keys())
        
        # Print the number of activations for each theme
        print("Style activations dictionary contains:")
        for theme in self.themes:
            num_activations = style_activations_dict[theme].shape[0]
            print(f"  - {theme}: {num_activations} activation vectors")
    
    def get_batch(self, concept=None):
        """
        Create a batch containing activations from a specific theme or multiple themes.
        
        Args:
            theme: Optional; specific theme to get activations for. If None:
                  - If mixed_batches=True: creates a batch with mixed themes
                  - If mixed_batches=False: randomly selects a single theme
            
        Returns:
            batch_data: Tensor of activations
            batch_themes: List of themes for each sample in the batch
        """
        theme = concept
        if theme is not None:
            # Specific theme requested
            all_activations = self.style_activations_dict[theme]
            sample_size = min(self.batch_size, all_activations.shape[0])
            indices = torch.randperm(all_activations.shape[0])[:sample_size]
            batch_data = all_activations[indices].float()
            batch_themes = [theme] * sample_size
            
        elif self.mixed_batches:
            # Create a mixed batch with samples from multiple themes
            batch_data = []
            batch_themes = []
            
            # Calculate samples per theme (at least 1 sample per theme, distribute remaining evenly)
            samples_per_theme = {}
            num_themes = len(self.themes)
            base_samples = self.batch_size // num_themes
            extra_samples = self.batch_size % num_themes
            
            for i, theme in enumerate(self.themes):
                # Add an extra sample to some themes to use the full batch size
                samples_per_theme[theme] = base_samples + (1 if i < extra_samples else 0)
            
            # Get samples for each theme
            for theme, num_samples in samples_per_theme.items():
                if num_samples > 0:
                    all_activations = self.style_activations_dict[theme]
                    # Don't sample more than we have
                    num_samples = min(num_samples, all_activations.shape[0])
                    indices = torch.randperm(all_activations.shape[0])[:num_samples]
                    theme_samples = all_activations[indices].float()
                    
                    batch_data.append(theme_samples)
                    batch_themes.extend([theme] * num_samples)
            
            # Concatenate all samples
            batch_data = torch.cat(batch_data, dim=0)
            
            # Shuffle the batch to mix themes
            shuffle_indices = torch.randperm(batch_data.shape[0])
            batch_data = batch_data[shuffle_indices]
            batch_themes = [batch_themes[i] for i in shuffle_indices]
            
        else:
            # Random single theme (original behavior)
            theme = self.rng.choice(self.themes)
            all_activations = self.style_activations_dict[theme]
            sample_size = min(self.batch_size, all_activations.shape[0])
            indices = torch.randperm(all_activations.shape[0])[:sample_size]
            batch_data = all_activations[indices].float()
            batch_themes = [theme] * sample_size
            
        return batch_data, batch_themes