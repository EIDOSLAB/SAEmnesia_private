#!/usr/bin/env python
"""
Load activation dictionaries and optimize SAE models to assign specific latents to concepts
while maintaining reconstruction quality.

This script processes raw activations from different concepts, assigns specific latent neurons
to each concept based on pre-computed scores from JSON files, and finetunes the SAE to maintain 
this assignment through cross-entropy loss.

Enhanced version that handles both objects and styles with separate latent assignments.
Added from-scratch training capability.

EDITED VERSION: Uses only two loss components:
1. Reconstruction loss: (x̂ - x)² / x²
2. BCE loss: Applied to sigmoid-activated latents (after TopK + Sigmoid)
"""
import os
import sys
import json
import glob

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.g_sae import Sae, SaeConfig
import torch
import numpy as np
from pathlib import Path
import random
import pyarrow as pa
import pyarrow.parquet as pq
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets, load_from_disk
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_datasets_from_category_dirs_with_styles(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load datasets from concept directories with recovered style information.
    """
    datasets = []
    print(f"Loading datasets with recovered styles from {base_dirs} for hookpoint {hookpoint}")

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        hookpoint_dir = base_path / hookpoint
        
        if not hookpoint_dir.exists():
            print(f"❌ Hookpoint directory does not exist: {hookpoint_dir}")
            continue
        
        # Load the recovered style metadata
        metadata_path = hookpoint_dir / "metadata" / "recovered_object_to_style_index.json"
        if not metadata_path.exists():
            print(f"❌ No recovered metadata found at {metadata_path}")
            print("   Run the style recovery first!")
            continue
        
        with open(metadata_path, 'r') as f:
            object_style_index = json.load(f)
        
        print(f"✅ Loaded recovered metadata with {len(object_style_index)} objects")
        
        concept_subdirs = [d for d in hookpoint_dir.iterdir() if d.is_dir() and d.name != 'metadata']
        
        for concept_dir in concept_subdirs:
            concept_name = concept_dir.name
            
            if (concept_dir / "dataset_info.json").exists():
                print(f"  Loading concept '{concept_name}' with style recovery...")
                
                # Load the dataset
                dataset = HFDataset.load_from_disk(str(concept_dir), keep_in_memory=False)
                print(f"    Original dataset: {len(dataset)} samples")
                
                # Check if this object is in our recovered metadata
                if concept_name not in object_style_index:
                    print(f"    ⚠️  No style recovery data for '{concept_name}', using 'none' style")
                    # Fallback: assign all to 'none' style
                    dataset = dataset.remove_columns(["object_label", "style_label"] if "object_label" in dataset.column_names else ["style_label"] if "style_label" in dataset.column_names else [])
                    dataset = dataset.add_column("object_label", [concept_name] * len(dataset))
                    dataset = dataset.add_column("style_label", ["none"] * len(dataset))
                    datasets.append(dataset)
                    continue
                
                # Create samples with proper style labels using recovered metadata
                style_datasets = []
                total_recovered_samples = 0
                
                for style_name, style_entries in object_style_index[concept_name].items():
                    for entry in style_entries:
                        start_idx, end_idx = entry["sample_range"]
                        sample_count = entry["sample_count"]
                        confidence = entry.get("recovery_confidence", "unknown")
                        
                        print(f"      {style_name}: samples {start_idx}-{end_idx-1} ({sample_count} samples, confidence: {confidence})")
                        
                        # Extract samples for this style
                        try:
                            style_samples = dataset.select(range(start_idx, end_idx))
                            
                            # Remove existing labels and add correct ones
                            if "object_label" in style_samples.column_names:
                                style_samples = style_samples.remove_columns(["object_label"])
                            if "style_label" in style_samples.column_names:
                                style_samples = style_samples.remove_columns(["style_label"])
                            
                            # Add correct labels
                            style_samples = style_samples.add_column("object_label", [concept_name] * len(style_samples))
                            style_samples = style_samples.add_column("style_label", [style_name] * len(style_samples))
                            
                            style_datasets.append(style_samples)
                            total_recovered_samples += len(style_samples)
                            
                        except Exception as e:
                            print(f"        ❌ Error extracting {style_name} samples: {e}")
                            continue
                
                if style_datasets:
                    # Combine all style datasets for this object
                    combined_dataset = concatenate_datasets(style_datasets)
                    print(f"    ✅ Combined dataset: {len(combined_dataset)} samples ({total_recovered_samples} recovered)")
                    
                    # Set format
                    combined_dataset.set_format(
                        type="torch",
                        columns=["activations", "timestep", "object_label", "style_label"],
                        dtype=dtype,
                    )
                    
                    datasets.append(combined_dataset)
                else:
                    print(f"    ❌ No valid style samples recovered for '{concept_name}'")

    if not datasets:
        raise ValueError(f"No valid datasets found for hookpoint {hookpoint}")

    final_dataset = concatenate_datasets(datasets)
    print(f"\n✅ Final combined dataset: {len(final_dataset)} samples")
    
    # Print style distribution summary
    unique_objects = set(final_dataset["object_label"])
    unique_styles = set(final_dataset["style_label"])
    print(f"   Objects: {len(unique_objects)} ({list(unique_objects)[:5]}...)")
    print(f"   Styles: {len(unique_styles)} ({list(unique_styles)[:5]}...)")
    
    return final_dataset

class SAEConceptLatentOptimizer:
    """
    Optimizer for SAE models to assign specific latents to concepts while maintaining reconstruction quality.
    
    This optimizer:
    1. Loads raw activations for different concepts with both object and style labels
    2. Assigns each concept (object/style) to a specific latent neuron based on pre-computed scores from JSON files
    3. Fine-tunes the SAE to maintain reconstruction while encouraging concept-specific latent assignments
    
    Uses only TWO loss components:
    - Reconstruction loss: (x̂ - x)² / x²
    - BCE loss: Applied to sigmoid-activated latents (after TopK + Sigmoid)
    """
    def __init__(
        self,
        checkpoint_path,
        activations_dir,
        object_scores_json_path,
        style_scores_json_path,
        device="cuda",
        learning_rate=5e-6,
        num_epochs=5,
        reconstruction_weight=1.0,
        cross_entropy_weight=1.0,
        batch_size=32,
        save_dir="sae-concept-latent-optimized",
        seed=42,
        validation_split=0.2,
        mixed_batches=True,
        mixed_precision=False,
        world_size=1,
        rank=0,
        gradient_accumulation_steps=1,
        use_float16=False,
        activation_column="activations",
        patience=5,
        resume=False,
        from_scratch=False
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.activations_dir = Path(activations_dir)
        self.object_scores_json_path = Path(object_scores_json_path)
        self.style_scores_json_path = Path(style_scores_json_path)
        self.device = torch.device(device)
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.reconstruction_weight = reconstruction_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.seed = seed
        self.validation_split = validation_split
        self.mixed_batches = mixed_batches
        self.mixed_precision = mixed_precision
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_float16 = use_float16
        self.dtype = torch.float16 if use_float16 else torch.float32
        self.activation_column = activation_column
        self.patience = patience
        self.resume = resume
        self.from_scratch = from_scratch

        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 1

        if use_float16:
            self.mixed_precision = False  # Disable mixed precision when using float16
            self.scaler = None  # No scaler needed since we're already in float16
        else:
            self.mixed_precision = mixed_precision
            self.scaler = torch.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Will be populated in initialize methods
        self.saes = {}
        self.optimizers = {}
        self.object_to_latent = {}
        self.style_to_latent = {}
        self.object_scores_data = None
        self.style_scores_data = None

        # Initialize everything
        self.load_scores_data()
        self.initialize_saes()
        self.initialize_datasets_with_styles()
        self.initialize_wandb()

    def find_latest_checkpoint(self, hook_name):
        """Find the latest checkpoint for resume."""
        current_path = self.save_dir / "current" / hook_name

        if current_path.exists() and (current_path / "cfg.json").exists():
            # Try to load training state to get the epoch
            training_state_path = current_path / "training_state.pt"
            if training_state_path.exists():
                try:
                    training_state = torch.load(training_state_path, map_location=self.device)
                    epoch = training_state.get('epoch', 0)
                    return epoch, current_path
                except:
                    pass
            return 0, current_path

        return None, None

    def load_checkpoint_state(self, hook_name, checkpoint_path):
        """
        Load SAE model and optimizer state from checkpoint.
        
        Args:
            hook_name: Name of the hook/layer
            checkpoint_path: Path to the checkpoint directory
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            print(f"Loading checkpoint for {hook_name} from {checkpoint_path}")
            
            # Load the SAE model
            sae = Sae.load_from_disk(checkpoint_path, device=self.device)
            sae = sae.to(dtype=self.dtype)
            self.saes[hook_name] = sae
            
            # Create optimizer (we'll load state if available)
            optimizer = Adam(
                [{"params": sae.parameters(), "lr": self.lr}],
                eps=1e-8
            )
            
            # Try to load optimizer state
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists():
                try:
                    optimizer_state = torch.load(optimizer_path, map_location=self.device)
                    optimizer.load_state_dict(optimizer_state)
                    print(f"  ✅ Loaded optimizer state for {hook_name}")
                except Exception as e:
                    print(f"  ⚠️  Could not load optimizer state: {e}")
            
            self.optimizers[hook_name] = optimizer
            
            # Try to load training state (best loss, patience counter)
            training_state_path = checkpoint_path / "training_state.pt"
            if training_state_path.exists():
                try:
                    training_state = torch.load(training_state_path, map_location=self.device)
                    self.best_val_loss = training_state.get('best_val_loss', float('inf'))
                    self.patience_counter = training_state.get('patience_counter', 0)
                    print(f"  ✅ Loaded training state - best_val_loss: {self.best_val_loss:.6f}, patience: {self.patience_counter}")
                except Exception as e:
                    print(f"  ⚠️  Could not load training state: {e}")
            
            print(f"✅ Successfully loaded checkpoint for {hook_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint for {hook_name}: {e}")
            return False

    def save_best_model(self, sae, hook_name, epoch, optimizer=None):
        """Save the best model checkpoint."""
        best_path = self.save_dir / "best" / hook_name
        os.makedirs(best_path, exist_ok=True)

        try:
            sae.save_to_disk(best_path)

            if optimizer is not None:
                torch.save(optimizer.state_dict(), best_path / "optimizer.pt")

            training_state = {
                'epoch': epoch,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
            }
            torch.save(training_state, best_path / "training_state.pt")

            print(f"Saved best model at epoch {epoch} to {best_path}")

        except Exception as e:
            print(f"Error saving best model: {e}")

    def save_current_checkpoint(self, sae, hook_name, epoch, optimizer=None):
        """Save current training state for resume (overwrites previous)."""
        current_path = self.save_dir / "current" / hook_name
        os.makedirs(current_path, exist_ok=True)

        try:
            sae.save_to_disk(current_path)

            if optimizer is not None:
                torch.save(optimizer.state_dict(), current_path / "optimizer.pt")

            training_state = {
                'epoch': epoch,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
            }
            torch.save(training_state, current_path / "training_state.pt")

        except Exception as e:
            print(f"Error saving current checkpoint: {e}")

    def normalize_concept_name(self, name):
        """Convert between underscore and space formats for concept names."""
        return name.replace('_', ' ')

    def find_concept_in_scores(self, concept_name, scores):
        """Find concept in scores dict, trying both original and normalized names."""
        # Try original name first
        if concept_name in scores:
            return concept_name, scores[concept_name]

        # Try with underscores replaced by spaces
        normalized_name = self.normalize_concept_name(concept_name)
        if normalized_name in scores:
            return normalized_name, scores[normalized_name]

        # Try with spaces replaced by underscores
        underscore_name = concept_name.replace(' ', '_')
        if underscore_name in scores:
            return underscore_name, scores[underscore_name]

        return None, None

    def load_scores_data(self):
        """Load both object and style scores from separate JSON files."""
        print(f"Loading object scores from {self.object_scores_json_path}")
        print(f"Loading style scores from {self.style_scores_json_path}")
        
        # Load object scores - skip if from_scratch and file doesn't exist
        if not self.object_scores_json_path.exists():
            if self.from_scratch:
                print(f"⚠️  Object scores file not found, but training from scratch - will use random assignment")
                self.object_scores_data = None
            else:
                raise FileNotFoundError(f"Object scores JSON file not found: {self.object_scores_json_path}")
        else:
            with open(self.object_scores_json_path, 'r') as f:
                self.object_scores_data = json.load(f)
            print(f"✅ Loaded object scores:")
            print(f"  Concept type: {self.object_scores_data.get('concept_type', 'unknown')}")
            print(f"  Number of concepts: {len(self.object_scores_data.get('scores', {}))}")
        
        # Load style scores - skip if from_scratch and file doesn't exist
        if not self.style_scores_json_path.exists():
            if self.from_scratch:
                print(f"⚠️  Style scores file not found, but training from scratch - will use random assignment")
                self.style_scores_data = None
            else:
                raise FileNotFoundError(f"Style scores JSON file not found: {self.style_scores_json_path}")
        else:
            with open(self.style_scores_json_path, 'r') as f:
                self.style_scores_data = json.load(f)
            print(f"✅ Loaded style scores:")
            print(f"  Concept type: {self.style_scores_data.get('concept_type', 'unknown')}")
            print(f"  Number of concepts: {len(self.style_scores_data.get('scores', {}))}")

    def initialize_datasets_with_styles(self):
        """Dataset initialization with recovered style information."""
        print("Initializing datasets with recovered style information...")

        hookpoint_names = list(self.saes.keys())

        dataset_dict = {}
        if not self.world_size > 1 or self.rank == 0:
            for hookpoint in hookpoint_names:
                # Use the function with style recovery
                dataset = load_datasets_from_category_dirs_with_styles(
                    [str(self.activations_dir)], 
                    hookpoint, 
                    self.dtype
                )

                # Apply numpy shuffling
                print(f"Applying numpy-based shuffling to {len(dataset)} samples...")
                indices = np.arange(len(dataset))
                np.random.seed(self.seed)
                np.random.shuffle(indices)
                dataset = dataset.select(indices)
                print(f"✅ Applied numpy shuffling for {len(dataset)} samples")

                dataset_dict[hookpoint] = dataset
                print(f"Completed loading for {hookpoint}: {len(dataset)} samples")

        # DDP synchronization
        if self.world_size > 1:
            dist.barrier()
            if self.rank != 0:
                for hookpoint in hookpoint_names:
                    dataset = load_datasets_from_category_dirs_with_styles(
                        [str(self.activations_dir)], 
                        hookpoint, 
                        self.dtype
                    )
                    indices = np.arange(len(dataset))
                    np.random.seed(self.seed)
                    np.random.shuffle(indices)
                    dataset = dataset.select(indices)
                    dataset = dataset.shard(self.world_size, self.rank)
                    dataset_dict[hookpoint] = dataset

        # Create data loaders with dual labels
        self._create_dual_data_loaders(dataset_dict)
        print("\n✅ Dataset initialization with styles completed!")

    def _create_dual_data_loaders(self, dataset_dict):
        """DataLoader creation with both object and style labels."""
        hookpoint, dataset = next(iter(dataset_dict.items()))

        total_size = len(dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size

        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))

        def dual_label_collate_fn(batch):
            """Collate function that handles both object and style labels."""
            activations = torch.stack([item['activations'] for item in batch])
            object_labels = [item['object_label'] for item in batch]
            style_labels = [item['style_label'] for item in batch]
            
            return activations, object_labels, style_labels

        # Handle distributed training properly
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, seed=self.seed)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, seed=self.seed)
            train_shuffle = False
            val_shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True
            val_shuffle = False

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=train_shuffle, sampler=train_sampler,
            num_workers=2, pin_memory=True, collate_fn=dual_label_collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=val_shuffle, sampler=val_sampler,
            num_workers=2, pin_memory=True, collate_fn=dual_label_collate_fn
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def _assign_single_concept_to_latent(self, concept_name, concept_scores, model_num_latents, latent_assignments):
        """Helper method to assign a single concept to a latent."""

        # Handle both 2D (timestep x latent) and 1D (latent) score arrays
        if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
            # 2D: Average across timesteps first
            avg_scores = np.mean(concept_scores, axis=0)
            print(f"  Using averaged scores for {concept_name}: shape {len(avg_scores)}")
        else:
            # 1D: Already averaged or single values
            avg_scores = concept_scores
            print(f"  Using direct scores for {concept_name}: shape {len(avg_scores)}")

        # Find the highest scoring latent that's not already assigned
        sorted_scores = sorted(enumerate(avg_scores), key=lambda x: x[1], reverse=True)

        for latent_idx, score in sorted_scores:
            if latent_idx < model_num_latents and latent_idx not in latent_assignments:
                print(f"    → Assigned latent {latent_idx} with avg score {score:.6f}")
                return latent_idx

        print(f"❌ No available latent found for concept '{concept_name}'")
        return None

    def _assign_concepts_randomly(self, hook_name):
        """Random assignment when training from scratch."""
        sae = self.saes[hook_name]
        model = sae.module if hasattr(sae, 'module') else sae
        model_num_latents = model.num_latents

        # Get unique objects and styles from your data
        sample_batch = next(iter(self.train_loader))
        _, object_labels, style_labels = sample_batch
        unique_objects = set(object_labels)
        unique_styles = set([s for s in style_labels if s != "none"])

        print(f"Random assignment for {len(unique_objects)} objects and {len(unique_styles)} styles")
        print(f"Available latents: {model_num_latents}")

        # Create random assignments
        import random
        available_latents = list(range(model_num_latents))
        random.shuffle(available_latents)

        object_to_latent = {}
        style_to_latent = {}
        latent_idx = 0

        # Assign objects first (priority)
        for obj in sorted(unique_objects):
            if latent_idx < len(available_latents):
                assigned_latent = available_latents[latent_idx]
                object_to_latent[obj] = assigned_latent
                print(f"  Random object '{obj}' → latent {assigned_latent}")
                latent_idx += 1
            else:
                # Fallback to random assignment with possible conflicts
                assigned_latent = random.randint(0, model_num_latents - 1)
                object_to_latent[obj] = assigned_latent
                print(f"  Random object '{obj}' → latent {assigned_latent} (conflict possible)")

        # Assign styles
        for style in sorted(unique_styles):
            if latent_idx < len(available_latents):
                assigned_latent = available_latents[latent_idx]
                style_to_latent[style] = assigned_latent
                print(f"  Random style '{style}' → latent {assigned_latent}")
                latent_idx += 1
            else:
                # Fallback to random assignment with possible conflicts
                assigned_latent = random.randint(0, model_num_latents - 1)
                style_to_latent[style] = assigned_latent
                print(f"  Random style '{style}' → latent {assigned_latent} (conflict possible)")

        print(f"\nRandom assignment completed:")
        print(f"  Objects: {len(object_to_latent)} assigned")
        print(f"  Styles: {len(style_to_latent)} assigned")
        print(f"  Latents used: {latent_idx}/{model_num_latents}")

        return object_to_latent, style_to_latent

    def _assign_concepts_from_scores(self, hook_name):
        """Original score-based assignment logic."""
        if self.object_scores_data is None or self.style_scores_data is None:
            raise RuntimeError("Object or style scores data not loaded.")

        object_scores = self.object_scores_data.get('scores', {})
        style_scores = self.style_scores_data.get('scores', {})
        
        sae = self.saes[hook_name]
        model = sae.module if hasattr(sae, 'module') else sae
        model_num_latents = model.num_latents

        # Get unique objects and styles from your data
        sample_batch = next(iter(self.train_loader))
        _, object_labels, style_labels = sample_batch
        unique_objects = set(object_labels)
        unique_styles = set(style_labels)

        print(f"Found {len(unique_objects)} unique objects: {list(unique_objects)[:5]}...")
        print(f"Found {len(unique_styles)} unique styles: {list(unique_styles)[:5]}...")

        # Helper function to get averaged scores
        def get_averaged_scores(concept_scores):
            if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
                # 2D: Average across timesteps
                return np.mean(concept_scores, axis=0)
            else:
                # 1D: Already averaged
                return concept_scores

        # Collect ALL concepts with their best scores and priority
        concept_priorities = []
        concepts_without_scores = []

        # Add objects (with priority boost)
        for concept_name in unique_objects:
            found_name, concept_scores = self.find_concept_in_scores(concept_name, object_scores)
            if found_name is not None:
                avg_scores = get_averaged_scores(concept_scores)
                best_score = max(avg_scores)
                concept_priorities.append((concept_name, best_score + 1.0, 'object', avg_scores))
            else:
                concepts_without_scores.append((concept_name, 'object'))

        # Add styles (no priority boost)
        for concept_name in unique_styles:
            if concept_name != "none":
                found_name, concept_scores = self.find_concept_in_scores(concept_name, style_scores)
                if found_name is not None:
                    avg_scores = get_averaged_scores(concept_scores)
                    best_score = max(avg_scores)
                    concept_priorities.append((concept_name, best_score, 'style', avg_scores))
                else:
                    concepts_without_scores.append((concept_name, 'style'))

        # Sort by priority score (highest first)
        concept_priorities.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🎯 Priority-based assignment (top scores get first choice):")
        print(f"Concepts with scores: {len(concept_priorities)}")
        print(f"Concepts without scores: {len(concepts_without_scores)}")

        # Assign in priority order
        object_to_latent = {}
        style_to_latent = {}
        latent_assignments = set()

        for concept_name, priority_score, concept_type, avg_scores in concept_priorities:
            # Use the existing helper function
            latent_idx = self._assign_single_concept_to_latent(
                concept_name, avg_scores, model_num_latents, latent_assignments
            )

            if latent_idx is not None:
                if concept_type == 'object':
                    object_to_latent[concept_name] = latent_idx
                else:  # style
                    style_to_latent[concept_name] = latent_idx
                    
                latent_assignments.add(latent_idx)
                actual_priority = priority_score - (1.0 if concept_type == 'object' else 0.0)
                score = avg_scores[latent_idx]
                print(f"✅ {concept_type.title()} '{concept_name}' → latent {latent_idx} "
                      f"(score: {score:.6f}, priority: {actual_priority:.6f})")
            else:
                concepts_without_scores.append((concept_name, concept_type))
                print(f"❌ {concept_type.title()} '{concept_name}' - no available latents")

        # FALLBACK: Assign remaining concepts to unused latents
        if concepts_without_scores:
            print(f"\n🔄 FALLBACK: Assigning {len(concepts_without_scores)} remaining concepts...")

            # Find unused latents
            unused_latents = [i for i in range(model_num_latents) if i not in latent_assignments]
            print(f"Available unused latents: {len(unused_latents)}")

            if len(unused_latents) >= len(concepts_without_scores):
                # Simple assignment: one unused latent per unassigned concept
                # Sort concepts_without_scores to prioritize objects
                concepts_without_scores.sort(key=lambda x: 0 if x[1] == 'object' else 1)

                for i, (concept_name, concept_type) in enumerate(concepts_without_scores):
                    if i < len(unused_latents):
                        latent_idx = unused_latents[i]
                        if concept_type == 'object':
                            object_to_latent[concept_name] = latent_idx
                        else:
                            style_to_latent[concept_name] = latent_idx
                        latent_assignments.add(latent_idx)
                        print(f"🔄 Fallback {concept_type} '{concept_name}' → latent {latent_idx}")
                    else:
                        print(f"⚠️  No latent available for {concept_type} '{concept_name}'")

            elif unused_latents:
                # More concepts than unused latents - use round-robin on unused latents
                concepts_without_scores.sort(key=lambda x: 0 if x[1] == 'object' else 1)

                for i, (concept_name, concept_type) in enumerate(concepts_without_scores):
                    latent_idx = unused_latents[i % len(unused_latents)]

                    # Check if this latent is already assigned in fallback (avoid conflicts)
                    if concept_name not in object_to_latent and concept_name not in style_to_latent:
                        if concept_type == 'object':
                            object_to_latent[concept_name] = latent_idx
                        else:
                            style_to_latent[concept_name] = latent_idx
                        print(f"🔄 Fallback {concept_type} '{concept_name}' → latent {latent_idx} (shared)")
            else:
                print("⚠️  No unused latents available for fallback assignment!")
                # Last resort: assign to random latents (will conflict with existing assignments)
                import random
                for concept_name, concept_type in concepts_without_scores:
                    latent_idx = random.randint(0, model_num_latents - 1)
                    if concept_type == 'object':
                        object_to_latent[concept_name] = latent_idx
                    else:
                        style_to_latent[concept_name] = latent_idx
                    print(f"🎲 Random fallback {concept_type} '{concept_name}' → latent {latent_idx} (CONFLICT LIKELY)")

        # Summary statistics
        assigned_objects = len(object_to_latent)
        assigned_styles = len(style_to_latent)
        total_assigned = assigned_objects + assigned_styles
        unique_latents_used = len(set(list(object_to_latent.values()) + list(style_to_latent.values())))

        print(f"\n📊 ASSIGNMENT SUMMARY:")
        print(f"Objects: {assigned_objects}/{len(unique_objects)} assigned")
        print(f"Styles: {assigned_styles}/{len([s for s in unique_styles if s != 'none'])} assigned") 
        print(f"Total concepts: {total_assigned} assigned")
        print(f"Latents used: {unique_latents_used}/{model_num_latents} ({unique_latents_used/model_num_latents*100:.1f}%)")

        # Check for conflicts (multiple concepts assigned to same latent)
        all_assignments = {}
        for concept, latent in object_to_latent.items():
            if latent not in all_assignments:
                all_assignments[latent] = []
            all_assignments[latent].append(f"object:{concept}")
        for concept, latent in style_to_latent.items():
            if latent not in all_assignments:
                all_assignments[latent] = []
            all_assignments[latent].append(f"style:{concept}")

        conflicts = {latent: concepts for latent, concepts in all_assignments.items() if len(concepts) > 1}
        if conflicts:
            print(f"⚠️  CONFLICTS DETECTED ({len(conflicts)} latents with multiple concepts):")
            for latent, concepts in list(conflicts.items())[:5]:  # Show first 5 conflicts
                print(f"   Latent {latent}: {concepts}")
            if len(conflicts) > 5:
                print(f"   ... and {len(conflicts) - 5} more conflicts")
        else:
            print("✅ No conflicts - each latent assigned to at most one concept")

        return object_to_latent, style_to_latent

    def assign_concepts_to_latents_from_scores(self, hook_name):
        """
        Assign both objects and styles to specific latents.
        Enhanced to support random assignment when training from scratch.
        """
        print(f"\nAssigning objects AND styles to latents for {hook_name}...")

        # Check if training from scratch
        if hasattr(self, 'from_scratch') and self.from_scratch:
            print("Training from scratch - using random assignment...")
            return self._assign_concepts_randomly(hook_name)
        
        # Original score-based assignment logic
        print("Using pre-computed scores for assignment...")
        return self._assign_concepts_from_scores(hook_name)

    @staticmethod
    def setup_distributed(rank, world_size):
        """Initialize the distributed environment based on environment variables set by torchrun."""
        # When using torchrun, these environment variables should already be set
        if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
            # If torchrun didn't set them (unlikely), set defaults
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
        
        # Print distributed info for debugging
        print(f"Initializing process group with rank={rank}, world_size={world_size}")
        print(f"MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Initialize process group using environment variables
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)

    def print_epoch_summary(self, epoch, hook_name, train_losses, val_losses, 
                        train_distributions, val_distributions, object_to_latent, style_to_latent):
        """
        Print a comprehensive, easy-to-read summary for each epoch.
        """
        print(f"\n" + "="*80)
        print(f"EPOCH {epoch} SUMMARY - {hook_name}")
        print(f"="*80)

        # 1. LOSS BREAKDOWN
        print(f"\n📊 LOSS BREAKDOWN:")
        print(f"{'Metric':<20} {'Training':<12} {'Validation':<12} {'Difference':<12}")
        print(f"-" * 56)

        train_diff = val_losses['total_loss'] - train_losses['total_loss']
        recon_diff = val_losses['recon_loss'] - train_losses['recon_loss']
        ce_diff = val_losses['ce_loss'] - train_losses['ce_loss']

        print(f"{'Total Loss':<20} {train_losses['total_loss']:<12.6f} {val_losses['total_loss']:<12.6f} {train_diff:>+12.6f}")
        print(f"{'Reconstruction':<20} {train_losses['recon_loss']:<12.6f} {val_losses['recon_loss']:<12.6f} {recon_diff:>+12.6f}")
        print(f"{'Cross Entropy':<20} {train_losses['ce_loss']:<12.6f} {val_losses['ce_loss']:<12.6f} {ce_diff:>+12.6f}")

        # Early stopping info
        print(f"\n🛑 EARLY STOPPING INFO:")
        print(f"Best validation loss so far: {self.best_val_loss:.6f}")
        print(f"Current patience counter: {self.patience_counter}/{self.patience}")
        if val_losses['total_loss'] < self.best_val_loss:
            print(f"✅ New best validation loss!")
        else:
            print(f"⚠️  No improvement in validation loss")

        # 2. CONCEPT ASSIGNMENT SUCCESS RATES
        print(f"\n🎯 CONCEPT ASSIGNMENT SUCCESS:")
        combined_concept_to_latent = {**object_to_latent, **style_to_latent}

        # Calculate success rates
        train_correct = sum(1 for concept, stats in train_distributions.items() 
                           if combined_concept_to_latent.get(concept) == stats["dominant_latent"])
        train_total = len(train_distributions)
        train_success_rate = (train_correct / train_total * 100) if train_total > 0 else 0

        val_correct = sum(1 for concept, stats in val_distributions.items() 
                         if combined_concept_to_latent.get(concept) == stats["dominant_latent"])
        val_total = len(val_distributions)
        val_success_rate = (val_correct / val_total * 100) if val_total > 0 else 0

        print(f"Training:   {train_correct:>2}/{train_total:<2} concepts correct ({train_success_rate:>6.1f}%)")
        print(f"Validation: {val_correct:>2}/{val_total:<2} concepts correct ({val_success_rate:>6.1f}%)")

        # 3. DETAILED CONCEPT TABLE
        print(f"\n📋 CONCEPT ASSIGNMENT DETAILS:")
        print(f"{'Concept':<15} {'Type':<8} {'Assigned':<8} {'Train Dom.':<10} {'Train Score':<11} {'Val Dom.':<9} {'Val Score':<10} {'Status':<8}")
        print(f"-" * 90)

        # Get all concepts
        all_concepts = set(train_distributions.keys()) | set(val_distributions.keys())

        for concept in sorted(all_concepts):
            concept_type = "object" if concept in object_to_latent else ("style" if concept in style_to_latent else "unknown")
            assigned_latent = combined_concept_to_latent.get(concept, -1)

            # Training stats
            train_stats = train_distributions.get(concept, {})
            train_dominant = train_stats.get("dominant_latent", -1)
            train_score = train_stats.get("dominance_score", 0.0)

            # Validation stats
            val_stats = val_distributions.get(concept, {})
            val_dominant = val_stats.get("dominant_latent", -1)
            val_score = val_stats.get("dominance_score", 0.0)

            # Status
            train_match = "✓" if assigned_latent == train_dominant else "✗"
            val_match = "✓" if assigned_latent == val_dominant else "✗"
            status = f"{train_match}/{val_match}"

            print(f"{concept:<15} {concept_type:<8} {assigned_latent:<8} {train_dominant:<10} {train_score:<11.4f} {val_dominant:<9} {val_score:<10.4f} {status:<8}")

        # 4. IMPROVEMENT INDICATORS
        print(f"\n📈 PROGRESS INDICATORS:")
        if epoch > 1:
            # You can store previous epoch metrics and compare here
            print(f"🔄 Compared to previous epoch: (implement if storing previous metrics)")

        # Overfitting check
        if val_losses['total_loss'] > train_losses['total_loss'] * 1.2:
            print(f"⚠️  WARNING: Potential overfitting detected (val_loss > 1.2 * train_loss)")
        elif val_success_rate < train_success_rate - 10:
            print(f"⚠️  WARNING: Validation concept success significantly lower than training")
        else:
            print(f"✅ Training appears healthy")

        print(f"\n" + "="*80 + "\n")

    def print_initial_concept_assignments(self, object_to_latent, style_to_latent, hook_name):
        """
        Print the initial concept-to-latent assignments clearly for both objects and styles.
        """
        print(f"\n" + "="*70)
        print(f"INITIAL CONCEPT ASSIGNMENTS - {hook_name}")
        print(f"="*70)
        print(f"{'Concept':<20} {'Type':<8} {'Assigned Latent':<15} {'Avg Score':<15}")
        print(f"-" * 58)

        # Get scores for display
        object_scores = self.object_scores_data.get('scores', {}) if self.object_scores_data else {}
        style_scores = self.style_scores_data.get('scores', {}) if self.style_scores_data else {}

        # Print object assignments
        for concept, latent_idx in sorted(object_to_latent.items()):
            # Get the original score for this assignment
            score = "N/A"
            if not self.from_scratch:
                found_name, concept_scores = self.find_concept_in_scores(concept, object_scores)

                if found_name is not None:
                    # Handle both 2D (timestep x latent) and 1D (latent) score arrays
                    if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
                        # 2D: Average across timesteps
                        avg_scores = np.mean(concept_scores, axis=0)
                    else:
                        # 1D: Already averaged
                        avg_scores = concept_scores

                    # Get the score for the assigned latent
                    if 0 <= latent_idx < len(avg_scores):
                        score = f"{avg_scores[latent_idx]:.6f}"
            else:
                score = "Random"

            print(f"{concept:<20} {'object':<8} {latent_idx:<15} {score:<15}")

        # Print style assignments
        for concept, latent_idx in sorted(style_to_latent.items()):
            # Get the original score for this assignment
            score = "N/A"
            if not self.from_scratch:
                found_name, concept_scores = self.find_concept_in_scores(concept, style_scores)

                if found_name is not None:
                    # Handle both 2D (timestep x latent) and 1D (latent) score arrays
                    if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
                        # 2D: Average across timesteps
                        avg_scores = np.mean(concept_scores, axis=0)
                    else:
                        # 1D: Already averaged
                        avg_scores = concept_scores

                    # Get the score for the assigned latent
                    if 0 <= latent_idx < len(avg_scores):
                        score = f"{avg_scores[latent_idx]:.6f}"
            else:
                score = "Random"

            print(f"{concept:<20} {'style':<8} {latent_idx:<15} {score:<15}")

        print(f"\nTotal objects: {len(object_to_latent)}")
        print(f"Total styles: {len(style_to_latent)}")
        print(f"Total concepts: {len(object_to_latent) + len(style_to_latent)}")
        print(f"="*70 + "\n")

    def print_latent_distribution_summary(self, distributions, object_to_latent, style_to_latent, epoch=None, is_validation=False):
        """
        Simplified version - the detailed output is now in print_epoch_summary.
        """
        dataset_type = "Validation" if is_validation else "Training"
        combined_concept_to_latent = {**object_to_latent, **style_to_latent}

        total_concepts = len(distributions)
        correct_concepts = sum(1 for concept, stats in distributions.items() 
                              if combined_concept_to_latent.get(concept) == stats["dominant_latent"])
        success_rate = correct_concepts / total_concepts if total_concepts > 0 else 0

        print(f"{dataset_type} concept assignment: {correct_concepts}/{total_concepts} ({success_rate:.1%})")

    def get_latent_distribution_statistics(self, sae, data_loader, object_to_latent, style_to_latent):
        """
        Calculate latent distribution statistics from sigmoid-activated sparse representations.
        Works with the paper's SAE architecture that outputs sparse activations.
        """
        model = sae.module if hasattr(sae, 'module') else sae
        model.eval()

        distributions = {}
        concept_activations = {}  # Store sparse activations per concept
        combined_concept_to_latent = {**object_to_latent, **style_to_latent}

        print("Calculating latent distribution statistics from sparse activations...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_idx >= 3:  # Limited for efficiency
                    break
                    
                activations, object_labels, style_labels = batch_data

                try:
                    activations = activations.to(self.device, dtype=self.dtype)

                    # Get SAE output - this returns top_acts and top_indices
                    # top_acts are sigmoid-activated values, top_indices are their positions
                    output = model(activations)
                    top_acts = output.latent_acts  # Sigmoid(TopK(h))
                    top_indices = output.latent_indices  # Indices of top-k
                    
                    # Handle reshaping if needed
                    if len(top_acts.shape) == 3:
                        # [batch, seq, k] - average over sequence
                        batch_size = top_acts.shape[0]
                        # We need to aggregate the sparse activations across sequence
                        # For each sample, collect all activated latents across the sequence
                    elif len(top_acts.shape) == 2:
                        batch_size = len(object_labels)
                    else:
                        print(f"  Unexpected shape: {top_acts.shape}")
                        continue
                    
                    # For each sample, aggregate its activated latents
                    for i, (obj_concept, style_concept) in enumerate(zip(object_labels, style_labels)):
                        # Get activations for this sample
                        if len(top_acts.shape) == 3:
                            sample_acts = top_acts[i]  # [seq, k]
                            sample_indices = top_indices[i]  # [seq, k]
                            # Flatten across sequence
                            sample_acts = sample_acts.reshape(-1)
                            sample_indices = sample_indices.reshape(-1)
                        else:
                            sample_acts = top_acts[i]  # [k]
                            sample_indices = top_indices[i]  # [k]
                        
                        # Store activations for object concept
                        if obj_concept not in concept_activations:
                            concept_activations[obj_concept] = {'acts': [], 'indices': []}
                        concept_activations[obj_concept]['acts'].append(sample_acts.cpu())
                        concept_activations[obj_concept]['indices'].append(sample_indices.cpu())
                        
                        # Store activations for style concept (if not "none")
                        if style_concept != "none":
                            if style_concept not in concept_activations:
                                concept_activations[style_concept] = {'acts': [], 'indices': []}
                            concept_activations[style_concept]['acts'].append(sample_acts.cpu())
                            concept_activations[style_concept]['indices'].append(sample_indices.cpu())
                        
                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Calculate statistics for each concept from sparse activations
        for concept, data in concept_activations.items():
            if concept not in combined_concept_to_latent:
                continue
                
            try:
                # Aggregate all activations for this concept into a dense representation
                # This tells us which latents are most activated for this concept
                num_latents = model.num_latents
                latent_sums = torch.zeros(num_latents)
                latent_counts = torch.zeros(num_latents)
                
                for acts, indices in zip(data['acts'], data['indices']):
                    for act, idx in zip(acts, indices):
                        if 0 <= idx < num_latents:
                            latent_sums[idx] += act
                            latent_counts[idx] += 1
                
                # Calculate average activation per latent
                latent_avg = torch.zeros(num_latents)
                mask = latent_counts > 0
                latent_avg[mask] = latent_sums[mask] / latent_counts[mask]
                
                # Find dominant latent (most frequently/strongly activated)
                dominant_latent = torch.argmax(latent_avg).item()
                dominance_score = latent_avg[dominant_latent].item()
                
                # Calculate entropy over the sparse distribution
                # Normalize to get probabilities
                prob_dist = latent_avg / (latent_avg.sum() + 1e-10)
                entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10)).item()
                
                distributions[concept] = {
                    "dominant_latent": dominant_latent,
                    "dominance_score": dominance_score,
                    "entropy": entropy
                }
                    
            except Exception as e:
                print(f"  Error processing concept {concept}: {e}")
                continue
                
        return distributions

    def compute_reconstruction_loss(self, sae, activations):
        """Compute reconstruction loss using normalized MSE: (x̂ - x)² / x²"""
        model = sae.module if hasattr(sae, 'module') else sae

        original_shape = activations.shape

        # Ensure activations are in the shape the model expects: [batch, seq, features]
        if len(activations.shape) == 2:
            # [batch, features] - add seq dimension
            batch_size, features = activations.shape
            activations = activations.unsqueeze(1)  # [batch, 1, features]
            seq_len = 1
        elif len(activations.shape) == 3:
            # [batch, seq, features] - already correct
            batch_size, seq_len, features = activations.shape
        else:
            print(f"  Recon Error: Cannot handle activations shape: {activations.shape}")
            return torch.tensor(1.0, device=self.device, dtype=self.dtype), None, None

        try:
            # Forward pass through SAE - expects [batch, seq, features]
            output = model(activations)

            # SAE outputs are in flattened form [batch*seq, features]
            # We need to reshape them back to [batch, seq, features]
            sae_out = output.sae_out
            top_acts = output.latent_acts
            top_indices = output.latent_indices

            # Reshape sae_out from [batch*seq, features] back to [batch, seq, features]
            sae_out = sae_out.reshape(batch_size, seq_len, features)

            # Also reshape top_acts and top_indices for consistency
            # top_acts shape: [batch*seq, k] -> [batch, seq, k]
            # top_indices shape: [batch*seq, k] -> [batch, seq, k]
            if len(top_acts.shape) == 2:
                k = top_acts.shape[1]
                top_acts = top_acts.reshape(batch_size, seq_len, k)
                top_indices = top_indices.reshape(batch_size, seq_len, k)

            e = (sae_out - activations).float()

            # Compute normalized MSE: (x̂ - x)² / x²
            eps = 1e-8
            squared_diff = e ** 2
            squared_original = activations.float() ** 2 + eps
            loss = torch.mean(squared_diff / squared_original)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Recon Warning: NaN/Inf in loss")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype), top_acts, top_indices

            return loss, top_acts, top_indices

        except Exception as e:
            print(f"  Recon Error: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(1.0, device=self.device, dtype=self.dtype), None, None

    def compute_cross_entropy_loss(self, top_acts, top_indices, object_labels, style_labels, object_to_latent, style_to_latent):
        """
        Binary cross-entropy loss applied to sigmoid-activated latents.
        OPTIMIZED: Uses vectorized operations instead of loops.
        """
        try:
            model = list(self.saes.values())[0]
            if hasattr(model, 'module'):
                model = model.module
            num_latents = model.num_latents
            
            batch_size = len(object_labels)
            
            # Handle input shapes - should now be [batch, seq, k] from reconstruction loss
            if len(top_acts.shape) == 3:
                batch_size_actual, seq_length, k = top_acts.shape
            elif len(top_acts.shape) == 2:
                # Fallback for [batch, k] case
                batch_size_actual, k = top_acts.shape
                seq_length = 1
                top_acts = top_acts.unsqueeze(1)  # [batch, 1, k]
                top_indices = top_indices.unsqueeze(1)
            else:
                print(f"  CE Loss Error: Unexpected shape: {top_acts.shape}")
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # VECTORIZED sparse-to-dense conversion
            # Flatten batch and seq dimensions: [batch, seq, k] -> [batch*seq, k]
            top_acts_flat = top_acts.reshape(-1, k)
            top_indices_flat = top_indices.reshape(-1, k)
            total_samples = top_acts_flat.shape[0]
            
            # FIX: Create dense representation with matching dtype
            dense_acts = torch.zeros(total_samples, num_latents, 
                                    device=self.device, dtype=top_acts_flat.dtype)  # Changed from self.dtype
            
            # Use advanced indexing for vectorized assignment
            sample_indices = torch.arange(total_samples, device=self.device).unsqueeze(1).expand(-1, k)
            valid_mask = (top_indices_flat >= 0) & (top_indices_flat < num_latents)
            
            dense_acts[sample_indices[valid_mask], top_indices_flat[valid_mask]] = top_acts_flat[valid_mask]
            
            # Average over sequence dimension: [batch*seq, num_latents] -> [batch, num_latents]
            dense_acts = dense_acts.reshape(batch_size, seq_length, num_latents).mean(dim=1)
            
            # Get all assigned latent indices
            all_assigned_latents = set()
            all_assigned_latents.update(object_to_latent.values())
            all_assigned_latents.update(style_to_latent.values())
            all_assigned_latents = sorted(list(all_assigned_latents))
            
            if len(all_assigned_latents) == 0:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # FIX: Create target tensor with matching dtype
            target_tensor = torch.zeros(batch_size, num_latents, device=self.device, dtype=dense_acts.dtype)
            
            # Vectorized target assignment
            for i, (object_concept, style_concept) in enumerate(zip(object_labels, style_labels)):
                if object_concept in object_to_latent:
                    object_latent = object_to_latent[object_concept]
                    if 0 <= object_latent < num_latents:
                        target_tensor[i, object_latent] = 1.0
                
                if style_concept != "none" and style_concept in style_to_latent:
                    style_latent = style_to_latent[style_concept]
                    if 0 <= style_latent < num_latents:
                        target_tensor[i, style_latent] = 1.0
            
            # Create mask for assigned latents only (vectorized)
            assigned_latents_tensor = torch.tensor(all_assigned_latents, device=self.device, dtype=torch.long)
            loss_mask = torch.zeros(batch_size, num_latents, device=self.device, dtype=torch.bool)
            loss_mask[:, assigned_latents_tensor] = True
            
            if loss_mask.sum() == 0:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # Disable autocast only for BCE calculation
            with torch.amp.autocast('cuda', enabled=False):
                # FIX: Convert to float32 for BCE calculation
                dense_acts_float = dense_acts.float()
                target_tensor_float = target_tensor.float()
                bce_loss = F.binary_cross_entropy(dense_acts_float, target_tensor_float, reduction='none')
            
            # Average loss over assigned latent positions
            ce_loss = bce_loss[loss_mask].mean()
            
            # FIX: Convert back to original dtype if needed
            if ce_loss.dtype != self.dtype:
                ce_loss = ce_loss.to(dtype=self.dtype)
            
            return ce_loss
            
        except Exception as e:
            print(f"  CE Loss Error: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
    def evaluate_losses(self, sae, hook_name, object_to_latent, style_to_latent, is_validation=False):
        """
        Evaluate the losses for either training or validation set.
        Updated to work with sparse SAE outputs.
        """
        # Choose the appropriate loader
        loader = self.val_loader if is_validation else self.train_loader

        # Set model to evaluation mode
        if hasattr(sae, 'module'):
            sae.module.eval()
        else:
            sae.eval()

        # Track losses
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        ce_loss_sum = 0.0
        num_batches = 0

        # Evaluate on a limited number of batches for efficiency
        max_batches = 5

        dataset_type = "validation" if is_validation else "training"
        print(f"Evaluating {dataset_type} losses for {hook_name}...")

        # Evaluate
        with torch.no_grad():
            for batch_idx, (activations, object_labels, style_labels) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                
                print(f"  Processing batch {batch_idx + 1}/{max_batches}...")

                try:
                    activations = activations.to(self.device, dtype=self.dtype)

                    # Compute losses - now returns top_acts and top_indices
                    recon_loss, top_acts, top_indices = self.compute_reconstruction_loss(sae, activations)
                    
                    if top_acts is None or top_indices is None:
                        print(f"  Skipping batch {batch_idx} due to reconstruction error")
                        continue
                    
                    ce_loss = self.compute_cross_entropy_loss(
                        top_acts, top_indices, object_labels, style_labels, 
                        object_to_latent, style_to_latent
                    )

                    # Combined loss
                    total_loss = (
                        self.reconstruction_weight * recon_loss +
                        self.cross_entropy_weight * ce_loss
                    )

                    # Accumulate losses
                    total_loss_sum += total_loss.item()
                    recon_loss_sum += recon_loss.item()
                    ce_loss_sum += ce_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        # Calculate averages
        if num_batches > 0:
            avg_total_loss = total_loss_sum / num_batches
            avg_recon_loss = recon_loss_sum / num_batches
            avg_ce_loss = ce_loss_sum / num_batches
        else:
            print(f"  WARNING: No batches processed for {dataset_type}")
            avg_total_loss = avg_recon_loss = avg_ce_loss = 0.0

        print(f"  Completed {dataset_type} evaluation")

        return {
            "total_loss": avg_total_loss,
            "recon_loss": avg_recon_loss,
            "ce_loss": avg_ce_loss,
        }

    def check_early_stopping(self, val_loss, epoch, sae, hook_name, optimizer):
        """Check early stopping and save best model when validation improves."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"✅ New best validation loss: {self.best_val_loss:.6f}")

            # Save the best model
            if isinstance(sae, DDP):
                self.save_best_model(sae.module, hook_name, epoch, optimizer)
            else:
                self.save_best_model(sae, hook_name, epoch, optimizer)

            return False
        else:
            self.patience_counter += 1
            print(f"⚠️  No improvement in validation loss. Patience: {self.patience_counter}/{self.patience}")

            if self.patience_counter >= self.patience:
                print(f"🛑 Early stopping triggered after {self.patience} epochs without improvement")
                return True

            return False
    
    def train(self):
        """
        Train the SAE models to assign specific latents to concepts.
        Updated to work with the paper's SAE architecture (Sigmoid + TopK).
        
        LOSS COMPONENTS:
        1. Reconstruction loss: (x̂ - x)² / x²
        2. BCE loss: Applied to sigmoid-activated latents f = Sigmoid(TopK(h))
        """
        # Create save directory
        if self.rank == 0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Process each SAE model
        for hook_name, sae in self.saes.items():
            if self.rank == 0:
                print(f"\nTraining SAE model for {hook_name}")
                if self.resume and self.start_epoch > 1:
                    print(f"Resuming training from epoch {self.start_epoch}")

            # Assign concepts to latents
            if self.rank == 0:
                object_to_latent, style_to_latent = self.assign_concepts_to_latents_from_scores(hook_name)
                self.object_to_latent[hook_name] = object_to_latent
                self.style_to_latent[hook_name] = style_to_latent
                self.print_initial_concept_assignments(object_to_latent, style_to_latent, hook_name)

                if self.world_size > 1:
                    # Broadcast mappings
                    object_list = [self.object_to_latent[hook_name]]
                    style_list = [self.style_to_latent[hook_name]]
                    dist.broadcast_object_list(object_list, src=0)
                    dist.broadcast_object_list(style_list, src=0)
            else:
                # Other ranks receive mappings
                object_list = [None]
                style_list = [None]
                dist.broadcast_object_list(object_list, src=0)
                dist.broadcast_object_list(style_list, src=0)
                self.object_to_latent[hook_name] = object_list[0]
                self.style_to_latent[hook_name] = style_list[0]
            
            if self.world_size > 1:
                dist.barrier()
            
            # Compute initial losses
            if self.rank == 0 and self.start_epoch == 1:
                train_losses = self.evaluate_losses(sae, hook_name, self.object_to_latent[hook_name], self.style_to_latent[hook_name], is_validation=False)
                val_losses = self.evaluate_losses(sae, hook_name, self.object_to_latent[hook_name], self.style_to_latent[hook_name], is_validation=True)
                
                print("\n=== Initial Losses ===")
                print(f"  Training   - Total: {train_losses['total_loss']:.6f}, Recon: {train_losses['recon_loss']:.6f}, CE: {train_losses['ce_loss']:.6f}")
                print(f"  Validation - Total: {val_losses['total_loss']:.6f}, Recon: {val_losses['recon_loss']:.6f}, CE: {val_losses['ce_loss']:.6f}")
                
                if WANDB_AVAILABLE:
                    initial_metrics = {
                        f"{hook_name}/initial/train/total_loss": train_losses['total_loss'],
                        f"{hook_name}/initial/train/recon_loss": train_losses['recon_loss'],
                        f"{hook_name}/initial/train/ce_loss": train_losses['ce_loss'],
                        f"{hook_name}/initial/val/total_loss": val_losses['total_loss'],
                        f"{hook_name}/initial/val/recon_loss": val_losses['recon_loss'],
                        f"{hook_name}/initial/val/ce_loss": val_losses['ce_loss'],
                    }
                    wandb.log(initial_metrics)
        
            # Training loop
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                if self.rank == 0:
                    print(f"\nEpoch {epoch}/{self.num_epochs}")
                    print(f"Training {hook_name}...")

                sae.train()
                optimizer = self.optimizers[hook_name]
                
                object_to_latent = self.object_to_latent[hook_name]
                style_to_latent = self.style_to_latent[hook_name]
                
                # Set epoch for sampler
                if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                
                # Track losses
                total_loss_sum = 0.0
                recon_loss_sum = 0.0
                ce_loss_sum = 0.0
                num_batches = 0
                
                data_iter = self.train_loader
                if self.rank == 0:
                    data_iter = tqdm(data_iter, desc="Batches")

                # Training loop
                for batch_idx, (activations, object_labels, style_labels) in enumerate(data_iter):
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    activations = activations.to(self.device)

                    # Mixed precision or standard training
                    if self.mixed_precision and torch.cuda.is_available() and not self.use_float16:
                        with torch.amp.autocast('cuda'):
                            recon_loss, top_acts, top_indices = self.compute_reconstruction_loss(sae, activations)
                            
                            if top_acts is None or top_indices is None:
                                continue

                            ce_loss = self.compute_cross_entropy_loss(
                                top_acts, top_indices,
                                object_labels, style_labels,
                                object_to_latent, style_to_latent
                            )

                            total_loss = (
                                self.reconstruction_weight * recon_loss +
                                self.cross_entropy_weight * ce_loss
                            )
                        
                        optimizer.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        # Standard precision
                        recon_loss, top_acts, top_indices = self.compute_reconstruction_loss(sae, activations)
                        
                        if top_acts is None or top_indices is None:
                            continue

                        ce_loss = self.compute_cross_entropy_loss(
                            top_acts, top_indices,
                            object_labels, style_labels,
                            object_to_latent, style_to_latent
                        )

                        total_loss = (
                            self.reconstruction_weight * recon_loss +
                            self.cross_entropy_weight * ce_loss
                        )
                        
                        optimizer.zero_grad()

                        if torch.isnan(total_loss).any():
                            print(f"WARNING: NaN detected in loss, skipping backward")
                            continue
                        
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    
                    # Accumulate losses
                    total_loss_sum += total_loss.item()
                    recon_loss_sum += recon_loss.item()
                    ce_loss_sum += ce_loss.item()
                    num_batches += 1

                    # Cleanup
                    del recon_loss, ce_loss, total_loss, top_acts, top_indices

                    if batch_idx % 50 == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Synchronize losses across processes
                if self.world_size > 1:
                    loss_tensor = torch.tensor(
                        [total_loss_sum, recon_loss_sum, ce_loss_sum, num_batches],
                        dtype=torch.float32, device=self.device
                    )
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    total_loss_sum = loss_tensor[0].item()
                    recon_loss_sum = loss_tensor[1].item()
                    ce_loss_sum = loss_tensor[2].item()
                    num_batches = int(loss_tensor[3].item())
                
                if num_batches > 0:
                    avg_total_loss = total_loss_sum / num_batches
                    avg_recon_loss = recon_loss_sum / num_batches
                    avg_ce_loss = ce_loss_sum / num_batches
                else:
                    print(f"WARNING: No batches processed in epoch {epoch}")
                    continue
                
                # Evaluation and logging (rank 0 only)
                if self.rank == 0:
                    print(f"\nEpoch {epoch} Training Averages:")
                    print(f"  Total Loss: {avg_total_loss:.6f}")
                    print(f"  Recon Loss: {avg_recon_loss:.6f}")
                    print(f"  CE Loss: {avg_ce_loss:.6f}")
                    
                    # Evaluate
                    train_losses = self.evaluate_losses(sae, hook_name, object_to_latent, style_to_latent, is_validation=False)
                    val_losses = self.evaluate_losses(sae, hook_name, object_to_latent, style_to_latent, is_validation=True)
                    
                    print(f"\n=== End of Epoch {epoch} Losses ===")
                    print(f"  Training   - Total: {train_losses['total_loss']:.6f}, Recon: {train_losses['recon_loss']:.6f}, CE: {train_losses['ce_loss']:.6f}")
                    print(f"  Validation - Total: {val_losses['total_loss']:.6f}, Recon: {val_losses['recon_loss']:.6f}, CE: {val_losses['ce_loss']:.6f}")
                    
                    # Early stopping
                    should_stop = self.check_early_stopping(val_losses['total_loss'], epoch, sae, hook_name, optimizer)

                    # Save current checkpoint
                    if isinstance(sae, DDP):
                        self.save_current_checkpoint(sae.module, hook_name, epoch, optimizer)
                    else:
                        self.save_current_checkpoint(sae, hook_name, epoch, optimizer)

                    if should_stop:
                        print(f"🛑 Early stopping triggered at epoch {epoch}")
                        break
                    
                    # Distribution statistics
                    train_distributions = self.get_latent_distribution_statistics(
                        sae if not isinstance(sae, DDP) else sae.module,
                        self.train_loader,
                        object_to_latent,
                        style_to_latent
                    )
                    val_distributions = self.get_latent_distribution_statistics(
                        sae if not isinstance(sae, DDP) else sae.module,
                        self.val_loader,
                        object_to_latent,
                        style_to_latent
                    )
                    
                    self.print_latent_distribution_summary(
                        train_distributions, object_to_latent, style_to_latent,
                        epoch=epoch, is_validation=False
                    )
                    self.print_latent_distribution_summary(
                        val_distributions, object_to_latent, style_to_latent,
                        epoch=epoch, is_validation=True
                    )
                    
                    # Wandb logging
                    if WANDB_AVAILABLE:
                        combined_concept_to_latent = {**object_to_latent, **style_to_latent}
                        
                        metrics = {
                            f"{hook_name}/train/total_loss": train_losses['total_loss'],
                            f"{hook_name}/train/recon_loss": train_losses['recon_loss'],
                            f"{hook_name}/train/ce_loss": train_losses['ce_loss'],
                            f"{hook_name}/val/total_loss": val_losses['total_loss'],
                            f"{hook_name}/val/recon_loss": val_losses['recon_loss'],
                            f"{hook_name}/val/ce_loss": val_losses['ce_loss'],
                            f"{hook_name}/best_val_loss": self.best_val_loss,
                            f"{hook_name}/patience_counter": self.patience_counter,
                            "epoch": epoch
                        }
                        
                        # Success rates
                        train_success = sum(1 for c, s in train_distributions.items() 
                                           if combined_concept_to_latent.get(c) == s["dominant_latent"])
                        train_success_rate = train_success / len(train_distributions) if train_distributions else 0
    
                        val_success = sum(1 for c, s in val_distributions.items() 
                                         if combined_concept_to_latent.get(c) == s["dominant_latent"])
                        val_success_rate = val_success / len(val_distributions) if val_distributions else 0
    
                        metrics.update({
                            f"{hook_name}/train/concept_success_rate": train_success_rate,
                            f"{hook_name}/val/concept_success_rate": val_success_rate,
                        })
                        
                        wandb.log(metrics)
                    
                    # Print epoch summary
                    self.print_epoch_summary(
                        epoch, hook_name, train_losses, val_losses,
                        train_distributions, val_distributions, object_to_latent, style_to_latent
                    )
                    
                    if should_stop:
                        break
                
                # Synchronize processes
                if self.world_size > 1:
                    dist.barrier()
        
        if self.rank == 0:
            if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf'):
                print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
            else:
                print("\nTraining completed successfully!")

    def _create_sae_from_scratch(self, hook_name):
        """Create a new SAE model from scratch."""
        try:
            # Default SAE configuration
            cfg = {
                "expansion_factor": 16,
                "normalize_decoder": True,
                "num_latents": 0,  # Will be calculated from d_in * expansion_factor
                "k": 32,
                "batch_topk": False,
                "sample_topk": False,
                "input_unit_norm": False,
                "multi_topk": False
            }
            
            # Create SaeConfig object
            sae_config = SaeConfig(**cfg)
            
            # Create new SAE instance with d_in=1280 (adjust as needed)
            sae = Sae(d_in=1280, cfg=sae_config, device=self.device, dtype=self.dtype)
            sae = sae.to(device=self.device, dtype=self.dtype)
            self.saes[hook_name] = sae
            
            # Create optimizer
            self.optimizers[hook_name] = Adam(
                [{"params": sae.parameters(), "lr": self.lr}],
                eps=1e-8
            )
            print(f"✅ Created SAE from scratch for {hook_name}")
            print(f"   d_in: 1280, num_latents: {sae.num_latents}, expansion_factor: {cfg['expansion_factor']}")
            
        except Exception as e:
            print(f"❌ Could not create SAE from scratch for {hook_name}: {e}")
            raise

    def initialize_saes(self):
        """Load SAE models from checkpoint with resume functionality or create from scratch."""
        print(f"Loading SAE models from {self.checkpoint_path}")
        
        if self.resume:
            print("Resume mode enabled - looking for latest checkpoints...")
        
        # Check if checkpoint path contains an SAE model
        if (self.checkpoint_path / "cfg.json").exists() and (self.checkpoint_path / "sae.safetensors").exists():
            hook_name = self.checkpoint_path.name
            
            if self.from_scratch:
                print(f"Creating SAE from scratch for {hook_name}")
                self._create_sae_from_scratch(hook_name)
                
                if self.resume:
                    latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                    if latest_epoch is not None:
                        print(f"Loading training state from epoch {latest_epoch}")
                        if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                            self.start_epoch = latest_epoch + 1
                return
            
            # Normal loading or resume
            if self.resume:
                latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                if latest_epoch is not None:
                    print(f"Found checkpoint for {hook_name} at epoch {latest_epoch}")
                    if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                        self.start_epoch = latest_epoch + 1
                        return
            
            # Load original model
            try:
                sae = Sae.load_from_disk(self.checkpoint_path, device=self.device)
                sae = sae.to(dtype=self.dtype)
                self.saes[hook_name] = sae
                
                self.optimizers[hook_name] = Adam(
                    [{"params": sae.parameters(), "lr": self.lr}],
                    eps=1e-8
                )
                print(f"Loaded SAE for {hook_name}")
            except Exception as e:
                print(f"Could not load SAE from {self.checkpoint_path}: {e}")
        
        # Handle subdirectories
        if not self.saes:
            for hook_dir in self.checkpoint_path.iterdir():
                if hook_dir.is_dir():
                    hook_name = hook_dir.name
                    
                    if self.from_scratch:
                        print(f"Creating SAE from scratch for {hook_name}")
                        self._create_sae_from_scratch(hook_name)
                        
                        if self.resume:
                            latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                            if latest_epoch is not None:
                                print(f"Loading training state from epoch {latest_epoch}")
                                if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                                    self.start_epoch = max(self.start_epoch, latest_epoch + 1)
                        continue
                    
                    if self.resume:
                        latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                        if latest_epoch is not None:
                            if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                                self.start_epoch = max(self.start_epoch, latest_epoch + 1)
                                continue
                    
                    # Load original model
                    try:
                        sae = Sae.load_from_disk(hook_dir, device=self.device)
                        sae = sae.to(dtype=self.dtype)
                        self.saes[hook_name] = sae
                        
                        self.optimizers[hook_name] = Adam(
                            [{"params": sae.parameters(), "lr": self.lr}],
                            eps=1e-8
                        )
                        print(f"Loaded SAE for {hook_name}")
                    except Exception as e:
                        print(f"Could not load SAE for {hook_name}: {e}")
    
    def initialize_wandb(self):
        """Initialize weights and biases for logging in offline mode."""
        # Only initialize wandb on rank 0 to avoid conflicts in distributed training
        if WANDB_AVAILABLE and self.rank == 0:
            wandb_dir = os.path.join(self.save_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)

            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = wandb_dir

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"sae_dual_concept_optimization_{timestamp}"

            config = {
                "learning_rate": self.lr,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "reconstruction_weight": self.reconstruction_weight,
                "cross_entropy_weight": self.cross_entropy_weight,
                "seed": self.seed,
                "validation_split": self.validation_split,
                "mixed_batches": self.mixed_batches,
                "object_scores_json_path": str(self.object_scores_json_path),
                "style_scores_json_path": str(self.style_scores_json_path),
                "patience": self.patience,
                "resume": self.resume,
                "start_epoch": self.start_epoch,
                "from_scratch": self.from_scratch,
            }

            try:
                wandb.init(
                    project="sae_dual_concept_latent_optimizer",
                    name=run_name,
                    config=config,
                    dir=wandb_dir,
                    settings=wandb.Settings(_service_wait=60)  # Increase timeout
                )

                print(f"Initialized wandb logging in OFFLINE mode")
                print(f"Logs will be stored in: {wandb_dir}")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
        elif self.rank != 0:
            print(f"Rank {self.rank}: Skipping wandb initialization (only rank 0 logs)")


def run_distributed_training(rank, world_size, args):
    # When using torchrun, get rank and world_size from environment
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Using torchrun environment: rank={rank}, world_size={world_size}")

    # Setup distributed training
    SAEConceptLatentOptimizer.setup_distributed(rank, world_size)
    
    # Create optimizer with local rank as device
    device = torch.device(f"cuda:{rank}")
    
    # Empty CUDA cache first
    torch.cuda.empty_cache()
    
    optimizer = SAEConceptLatentOptimizer(
        checkpoint_path=args.checkpoint_path,
        activations_dir=args.activations_dir,
        object_scores_json_path=args.object_scores_json_path,
        style_scores_json_path=args.style_scores_json_path,
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        reconstruction_weight=args.reconstruction_weight,
        cross_entropy_weight=args.cross_entropy_weight,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed,
        validation_split=args.validation_split,
        mixed_batches=args.mixed_batches,
        rank=rank,
        world_size=world_size,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience,
        resume=args.resume,
        from_scratch=args.from_scratch,
    )

    # Update SAE models to DDP
    for hook_name, sae in optimizer.saes.items():
        sae = sae.to(device)
        ddp_model = DDP(sae, device_ids=[rank])
        optimizer.saes[hook_name] = ddp_model
        
        optimizer.optimizers[hook_name] = Adam(
            [{"params": ddp_model.parameters(), "lr": optimizer.lr}],
            eps=1e-8
        )

    # Train
    optimizer.train()

    # Cleanup
    dist.destroy_process_group()

def main():
    """
    Main entry point for the SAE Dual Concept Latent Optimizer.
    
    LOSS CONFIGURATION:
    - Reconstruction loss: (x̂ - x)² / x²
    - BCE loss: Applied to sigmoid-activated latents f = Sigmoid(TopK(h))
    """
    parser = argparse.ArgumentParser(description="Optimize SAE models to assign specific latents to both object and style concepts.")
    
    # Required parameters
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        required=True, 
        help="Path to the SAE checkpoint directory"
    )
    parser.add_argument(
        "--activations_dir", 
        type=str, 
        required=True, 
        help="Path to the concept activations directory with style recovery metadata"
    )
    parser.add_argument(
        "--object_scores_json_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing pre-computed object scores"
    )
    parser.add_argument(
        "--style_scores_json_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing pre-computed style scores"
    )

    parser.add_argument(
        "--activation_column", 
        type=str, 
        default="activations", 
        help="Name of the column containing activations in the dataset"
    )
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for optimization")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--mixed_batches", action="store_true", help="Use batches with mixed concepts for training")
    
    # Loss weights
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--cross_entropy_weight", type=float, default=1.0, help="Weight for cross-entropy loss")
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="sae-dual-concept-optimized", help="Directory to save optimized models")
    
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision (FP16) training")
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for distributed training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 precision for all tensors")
    
    # Early stopping and resume parameters
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument("--from_scratch", action="store_true", help="Start training from scratch without loading any previous checkpoints")

    args = parser.parse_args()
    
    world_size = args.num_gpus
    os.environ['OMP_NUM_THREADS'] = "8"

    if 'LOCAL_RANK' in os.environ:
        rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Running with torchrun: rank={rank}, world_size={world_size}")
    
        run_distributed_training(rank, world_size, args)
    
    else:
        if world_size > 1:
            mp.spawn(
                run_distributed_training,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        else:
            # Create and run the optimizer
            optimizer = SAEConceptLatentOptimizer(
                checkpoint_path=args.checkpoint_path,
                activations_dir=args.activations_dir,
                object_scores_json_path=args.object_scores_json_path,
                style_scores_json_path=args.style_scores_json_path,
                device=args.device,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                reconstruction_weight=args.reconstruction_weight,
                cross_entropy_weight=args.cross_entropy_weight,
                batch_size=args.batch_size,
                save_dir=args.save_dir,
                seed=args.seed,
                validation_split=args.validation_split,
                mixed_batches=args.mixed_batches,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                use_float16=args.use_float16,
                patience=args.patience,
                resume=args.resume,
                from_scratch=args.from_scratch
            )
    
            optimizer.train()
            print("Training completed successfully!")


if __name__ == "__main__":
    main()