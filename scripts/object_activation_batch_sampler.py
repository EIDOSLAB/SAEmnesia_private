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
from UnlearnCanvas_resources.const import class_available


class ObjectActivationBatchSampler:
    """
    Batch sampler that creates batches containing raw activations from objects.
    Supports both single-object batches and mixed-object batches.
    """
    def __init__(self, object_activations_dict, batch_size=32, seed=42, mixed_batches=False):
        self.object_activations_dict = object_activations_dict
        self.batch_size = batch_size
        self.seed = seed
        self.rng = random.Random(seed)
        self.mixed_batches = mixed_batches
        
        # Available object classes
        self.objects = list(object_activations_dict.keys())
        
        # Print the number of activations for each object class
        print("Object activations dictionary contains:")
        for obj_class in self.objects:
            num_activations = object_activations_dict[obj_class].shape[0]
            print(f"  - {obj_class}: {num_activations} activation vectors")
    
    def get_batch(self, concept=None):
        """
        Create a batch containing activations from a specific object class or multiple object classes.
    
        Args:
            concept: Optional; specific object class to get activations for. If None:
                  - If mixed_batches=True: creates a batch with mixed object classes
                  - If mixed_batches=False: randomly selects a single object class
            
        Returns:
            batch_data: Tensor of activations
            batch_objects: List of object classes for each sample in the batch
        """
    
        obj_class = concept
    
        # Check if we're using on-demand loading
        using_on_demand = hasattr(self.object_activations_dict, 'get_activation')
    
        # Helper function to get activations for a class
        def get_class_activations(class_name):
            if using_on_demand:
                # Using the on-demand loading approach
                activations = self.object_activations_dict.get_activation(class_name)
                # Handle both direct tensor return and dictionary return formats
                if isinstance(activations, dict) and class_name in activations:
                    return activations[class_name]
                return activations
            else:
                # Traditional dictionary approach
                return self.object_activations_dict[class_name]
    
        # Helper function to check if a class is available
        def is_class_available(class_name):
            if using_on_demand:
                # For on-demand, check if the class is in the paths dictionary
                if hasattr(self.object_activations_dict, 'concept_activation_paths'):
                    return class_name in self.object_activations_dict.concept_activation_paths
                # Fallback check - try to get activations
                activations = get_class_activations(class_name)
                return activations is not None
            else:
                # For traditional dict, simply check if the key exists
                return class_name in self.object_activations_dict
    
        # Get available classes
        available_classes = [cls for cls in self.objects if is_class_available(cls)]
    
        if len(available_classes) == 0:
            raise ValueError("No object classes available for sampling. Check your activations dictionary.")
    
        if obj_class is not None:
            # Specific object class requested - check if available
            if not is_class_available(obj_class):
                raise ValueError(f"Requested object class '{obj_class}' is not available")
                
            # Get activations for the requested class
            all_activations = get_class_activations(obj_class)
            if all_activations is None:
                raise ValueError(f"Could not retrieve activations for object class '{obj_class}'")
                
            sample_size = min(self.batch_size, all_activations.shape[0])
            indices = torch.randperm(all_activations.shape[0])[:sample_size]
            batch_data = all_activations[indices].float()
            batch_objects = [obj_class] * sample_size
            
        elif self.mixed_batches:
            # Create a mixed batch with samples from multiple object classes
            batch_data = []
            batch_objects = []
            
            # Calculate samples per object class (at least 1 sample per class, distribute remaining evenly)
            samples_per_class = {}
            num_classes = len(available_classes)
            
            if num_classes == 0:
                raise ValueError("No object classes available for mixed batch")
                
            base_samples = self.batch_size // num_classes
            extra_samples = self.batch_size % num_classes
            
            for i, obj_class in enumerate(available_classes):
                # Add an extra sample to some classes to use the full batch size
                samples_per_class[obj_class] = base_samples + (1 if i < extra_samples else 0)
            
            # Get samples for each object class
            for obj_class, num_samples in samples_per_class.items():
                if num_samples > 0:
                    all_activations = get_class_activations(obj_class)
                    if all_activations is None or len(all_activations) == 0:
                        print(f"Warning: No activations available for '{obj_class}'. Skipping.")
                        continue
                        
                    # Don't sample more than we have
                    num_samples = min(num_samples, all_activations.shape[0])
                    indices = torch.randperm(all_activations.shape[0])[:num_samples]
                    class_samples = all_activations[indices].float()
                    
                    batch_data.append(class_samples)
                    batch_objects.extend([obj_class] * num_samples)
            
            # Check if we got any samples
            if len(batch_data) == 0:
                raise ValueError("Could not retrieve any valid samples for batch")
                
            # Concatenate all samples
            batch_data = torch.cat(batch_data, dim=0)
            
            # Shuffle the batch to mix object classes
            shuffle_indices = torch.randperm(batch_data.shape[0])
            batch_data = batch_data[shuffle_indices]
            batch_objects = [batch_objects[i] for i in shuffle_indices]
            
        else:
            # Random single object class
            obj_class = self.rng.choice(available_classes)
            all_activations = get_class_activations(obj_class)
            
            if all_activations is None:
                raise ValueError(f"Could not retrieve activations for randomly selected object class '{obj_class}'")
                
            sample_size = min(self.batch_size, all_activations.shape[0])
            indices = torch.randperm(all_activations.shape[0])[:sample_size]
            batch_data = all_activations[indices].float()
            batch_objects = [obj_class] * sample_size
            
        return batch_data, batch_objects