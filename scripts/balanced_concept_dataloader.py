import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import numpy as np
import json
from pathlib import Path

class MetadataBalancedDataset(Dataset):
    """
    Ultra-fast balanced dataset that uses pre-existing metadata instead of scanning samples.
    Reads folder structure and JSON annotations to build concept mappings instantly.
    """
    
    def __init__(self, original_dataset, object_to_latent, style_to_latent, 
                 batch_size, balance_ratio=0.5, seed=42, activations_dir=None, hookpoint=None):
        self.original_dataset = original_dataset
        self.object_to_latent = object_to_latent
        self.style_to_latent = style_to_latent
        self.batch_size = batch_size
        self.balance_ratio = balance_ratio
        self.seed = seed
        self.activations_dir = activations_dir
        self.hookpoint = hookpoint
        
        # Get all concepts we care about
        self.all_concepts = set(object_to_latent.keys()) | set(style_to_latent.keys())
        print(f"Building balanced dataset for {len(self.all_concepts)} concepts using metadata")
        
        # Build concept mappings from metadata (fast!)
        if activations_dir and hookpoint:
            self._build_concept_indices_from_metadata()
        else:
            # Fallback to the optimized scanning method
            print("No metadata path provided, falling back to sample scanning...")
            self._build_concept_indices_fallback()
        
        # Pre-generate balanced batches
        self._generate_balanced_batches()
        
    def _build_concept_indices_from_metadata(self):
        """Build concept indices by directly reading the properly labeled dataset."""
        self.concept_indices = defaultdict(list)
        self.non_concept_indices = defaultdict(list)
        
        print("Building concept indices from properly labeled dataset...")
        
        # Sample the dataset to build concept mappings
        total_samples = len(self.original_dataset)
        sample_size = min(50000, total_samples)
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
        
        print(f"Sampling {sample_size:,} samples to build concept mapping...")
        
        for i, sample_idx in enumerate(sample_indices):
            if i % 10000 == 0:
                print(f"  Progress: {i:,}/{sample_size:,}")
                
            try:
                sample = self.original_dataset[sample_idx]
                object_label = sample['object_label']
                style_label = sample['style_label']
                
                for concept in self.all_concepts:
                    has_concept = (
                        (concept in self.object_to_latent and object_label == concept) or
                        (concept in self.style_to_latent and style_label == concept)
                    )
                    
                    if has_concept:
                        self.concept_indices[concept].append(sample_idx)
                    else:
                        self.non_concept_indices[concept].append(sample_idx)
                        
            except Exception as e:
                continue
            
        print(f"✅ Concept mapping completed from dataset sampling")

    def _build_concept_indices_fallback(self):
        """Fallback method using sample scanning (still optimized)."""
        self.concept_indices = defaultdict(list)
        self.non_concept_indices = defaultdict(list)
        
        dataset_size = len(self.original_dataset)
        print(f"Fallback: Analyzing sample distribution ({dataset_size:,} samples)...")
        
        # Sample a subset for analysis if dataset is very large
        max_samples = min(50000, dataset_size)
        if max_samples < dataset_size:
            print(f"Sampling {max_samples:,} samples for analysis")
            sample_indices = np.random.choice(dataset_size, max_samples, replace=False)
        else:
            sample_indices = range(dataset_size)
        
        for i, sample_idx in enumerate(sample_indices):
            if i % 10000 == 0:
                print(f"  Processed {i:,}/{len(sample_indices):,} samples...")
            
            try:
                sample = self.original_dataset[int(sample_idx)]
                object_label = sample['object_label']
                style_label = sample['style_label']
                
                for concept in self.all_concepts:
                    has_concept = False
                    
                    if concept in self.object_to_latent and object_label == concept:
                        has_concept = True
                    elif concept in self.style_to_latent and style_label == concept:
                        has_concept = True
                    
                    if has_concept:
                        self.concept_indices[concept].append(int(sample_idx))
                    else:
                        self.non_concept_indices[concept].append(int(sample_idx))
                        
            except Exception as e:
                continue
        
        print("Fallback analysis completed.")
    
    def _generate_balanced_batches(self):
        """Pre-generate all balanced batches."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.batch_indices = []
        total_samples = len(self.original_dataset)
        num_batches = total_samples // self.batch_size
        
        print(f"Generating {num_batches:,} balanced batches...")
        
        # Create a cycle through concepts
        concept_list = list(self.all_concepts)
        
        for batch_idx in range(num_batches):
            if batch_idx % 1000 == 0 and batch_idx > 0:
                print(f"  Generated {batch_idx:,}/{num_batches:,} batches...")
            
            # Select primary concept for this batch (cycle through)
            primary_concept = concept_list[batch_idx % len(concept_list)]
            
            batch_indices = self._create_balanced_batch(primary_concept)
            
            if len(batch_indices) == self.batch_size:
                self.batch_indices.extend(batch_indices)
            else:
                # Pad incomplete batches
                remaining = self.batch_size - len(batch_indices)
                all_indices = list(range(total_samples))
                additional = random.sample(all_indices, min(remaining, len(all_indices)))
                batch_indices.extend(additional)
                self.batch_indices.extend(batch_indices[:self.batch_size])
        
        print(f"Generated {len(self.batch_indices):,} total samples in balanced batches")
    
    def _create_balanced_batch(self, primary_concept):
        """Create a single balanced batch focused on primary_concept."""
        target_positive = int(self.batch_size * self.balance_ratio)
        target_negative = self.batch_size - target_positive
        
        batch_indices = []
        
        # Get available samples
        pos_candidates = self.concept_indices[primary_concept].copy()
        neg_candidates = self.non_concept_indices[primary_concept].copy()
        
        # Shuffle
        random.shuffle(pos_candidates)
        random.shuffle(neg_candidates)
        
        # Add positive samples
        if len(pos_candidates) >= target_positive:
            batch_indices.extend(pos_candidates[:target_positive])
        else:
            batch_indices.extend(pos_candidates)
            # Fill remainder with samples from other concepts
            remaining = target_positive - len(pos_candidates)
            other_concepts = [c for c in self.all_concepts if c != primary_concept]
            for other_concept in other_concepts:
                if remaining <= 0:
                    break
                other_pos = self.concept_indices[other_concept]
                take = min(remaining, len(other_pos))
                if take > 0:
                    batch_indices.extend(random.sample(other_pos, take))
                    remaining -= take
        
        # Add negative samples
        if len(neg_candidates) >= target_negative:
            batch_indices.extend(neg_candidates[:target_negative])
        else:
            batch_indices.extend(neg_candidates)
            remaining = target_negative - len(neg_candidates)
            if remaining > 0:
                all_indices = list(range(len(self.original_dataset)))
                available = [i for i in all_indices if i not in batch_indices]
                if len(available) >= remaining:
                    batch_indices.extend(random.sample(available, remaining))
                else:
                    batch_indices.extend(available)
        
        return batch_indices
    
    def __len__(self):
        return len(self.batch_indices)
    
    def __getitem__(self, idx):
        original_idx = self.batch_indices[idx]
        return self.original_dataset[original_idx]