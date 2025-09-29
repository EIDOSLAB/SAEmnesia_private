#!/usr/bin/env python
"""
Load activation dictionaries and optimize SAE models to assign specific latents to concepts
while maintaining reconstruction quality.

This script processes raw activations from different concepts, assigns specific latent neurons
to each concept based on pre-computed scores from JSON files, and finetunes the SAE to maintain 
this assignment through binary cross-entropy loss.

Enhanced version that handles both objects and styles with separate latent assignments.
MODIFIED: Computes BCE loss AFTER topk selection instead of before.
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

from SAE.sae import Sae
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

def load_datasets_from_category_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load datasets from concept directories, ensuring correct labels.
    Falls back to single concept mode if no style metadata is found.
    """
    datasets = []
    print(f"Loading datasets from {base_dirs} for hookpoint {hookpoint}")

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        hookpoint_dir = base_path / hookpoint
        
        if not hookpoint_dir.exists():
            print(f"❌ Hookpoint directory does not exist: {hookpoint_dir}")
            continue
        
        # Check if we have style metadata for dual concept mode
        metadata_path = hookpoint_dir / "metadata" / "recovered_object_to_style_index.json"
        has_style_metadata = metadata_path.exists()
        
        if has_style_metadata:
            print(f"✅ Found style metadata - using dual concept mode")
            return load_datasets_from_category_dirs_with_styles(base_dirs, hookpoint, dtype)
        
        print(f"ℹ️  No style metadata found - using single concept mode")
            
        concept_subdirs = [d for d in hookpoint_dir.iterdir() if d.is_dir()]
        
        for concept_dir in concept_subdirs:
            concept_name = concept_dir.name
            
            if (concept_dir / "dataset_info.json").exists():
                print(f"  Loading concept '{concept_name}'...")
                
                # Load the dataset
                dataset = HFDataset.load_from_disk(str(concept_dir), keep_in_memory=False)
                
                print(f"    Original columns: {dataset.column_names}")
                
                # Remove existing labels if they exist
                columns_to_remove = []
                if "object_label" in dataset.column_names:
                    columns_to_remove.append("object_label")
                if "style_label" in dataset.column_names:
                    columns_to_remove.append("style_label")
                
                if columns_to_remove:
                    dataset = dataset.remove_columns(columns_to_remove)
                    print(f"    Removed existing label columns: {columns_to_remove}")
                
                # Add the correct concept label based on directory name
                dataset = dataset.add_column("object_label", [concept_name] * len(dataset))
                dataset = dataset.add_column("style_label", ["none"] * len(dataset))
                print(f"    Added object_label='{concept_name}' and style_label='none'")
                
                # Set format
                dataset.set_format(
                    type="torch",
                    columns=["activations", "timestep", "object_label", "style_label"],
                    dtype=dtype,
                )
                
                datasets.append(dataset)
                print(f"    ✅ Loaded {len(dataset)} samples from '{concept_name}'")

    if not datasets:
        raise ValueError(f"No valid datasets found for hookpoint {hookpoint}")

    return concatenate_datasets(datasets)

class SAEConceptLatentOptimizer:
    """
    Optimizer for SAE models to assign specific latents to concepts while maintaining reconstruction quality.
    
    This optimizer:
    1. Loads raw activations for different concepts with both object and style labels
    2. Assigns each concept (object/style) to a specific latent neuron based on pre-computed scores from JSON files
    3. Fine-tunes the SAE to maintain reconstruction while encouraging concept-specific latent assignments
    
    MODIFIED: Computes BCE loss AFTER topk selection instead of before.
    """
    def __init__(
        self,
        checkpoint_path,
        activations_dir,
        scores_json_path,
        style_scores_json_path=None,
        device="cuda",
        learning_rate=5e-6,
        num_epochs=5,
        reconstruction_weight=1.0,
        cross_entropy_weight=1.0,
        sparsity_weight=0.01,
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
        self.scores_json_path = Path(scores_json_path)
        self.style_scores_json_path = Path(style_scores_json_path) if style_scores_json_path else None
        self.device = torch.device(device)
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.reconstruction_weight = reconstruction_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.sparsity_weight = sparsity_weight
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
        self.scores_data = None
        self.style_scores_data = None
        self.dual_concept_mode = False

        # Initialize everything
        self.load_scores_data()
        self.initialize_saes()
        self.initialize_datasets()
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
        """
        Load the scores data from JSON file(s).
        """
        print(f"Loading object scores from {self.scores_json_path}")
    
        if not self.scores_json_path.exists():
            if self.from_scratch:
                print(f"⚠️  Scores file not found, but training from scratch - will use random assignment")
                self.scores_data = None
                return
            else:
                raise FileNotFoundError(f"Object scores JSON file not found: {self.scores_json_path}")
        
        # ADD THIS MISSING SECTION - Actually load the JSON data!
        try:
            with open(self.scores_json_path, 'r') as f:
                self.scores_data = json.load(f)
            print(f"✅ Loaded object scores: {len(self.scores_data.get('scores', {}))} concepts")
        except Exception as e:
            raise RuntimeError(f"Failed to load object scores from {self.scores_json_path}: {e}")
        
        # Load style scores if provided (for dual concept mode)
        if self.style_scores_json_path:
            print(f"Loading style scores from {self.style_scores_json_path}")
            
            if not self.style_scores_json_path.exists():
                print(f"⚠️  Style scores file not found - falling back to single concept mode")
                self.style_scores_data = None
                self.dual_concept_mode = False
            else:
                try:
                    with open(self.style_scores_json_path, 'r') as f:
                        self.style_scores_data = json.load(f)
                    print(f"✅ Loaded style scores: {len(self.style_scores_data.get('scores', {}))} concepts")
                    self.dual_concept_mode = True
                except Exception as e:
                    print(f"⚠️  Failed to load style scores: {e} - falling back to single concept mode")
                    self.style_scores_data = None
                    self.dual_concept_mode = False
        else:
            print("No style scores provided - using single concept mode")
            self.style_scores_data = None
            self.dual_concept_mode = False

    def initialize_datasets(self):
        """Dataset initialization with numpy shuffling."""
        print("Initializing datasets...")

        hookpoint_names = list(self.saes.keys())

        dataset_dict = {}
        if not self.world_size > 1 or self.rank == 0:
            for hookpoint in hookpoint_names:
                dataset = load_datasets_from_category_dirs(
                    [str(self.activations_dir)], 
                    hookpoint, 
                    self.dtype
                )

                # Use numpy shuffling instead of dataset.shuffle()
                print(f"Applying numpy-based shuffling to {len(dataset)} samples...")

                # Create shuffled indices using numpy
                indices = np.arange(len(dataset))
                np.random.seed(self.seed)
                np.random.shuffle(indices)

                # Use select with shuffled indices - this is memory efficient
                dataset = dataset.select(indices)
                print(f"✅ Applied numpy shuffling for {len(dataset)} samples")

                dataset_dict[hookpoint] = dataset
                print(f"Completed loading for {hookpoint}: {len(dataset)} samples")

        # DDP synchronization
        if self.world_size > 1:
            dist.barrier()

            if self.rank != 0:
                for hookpoint in hookpoint_names:
                    dataset = load_datasets_from_category_dirs(
                        [str(self.activations_dir)], 
                        hookpoint, 
                        self.dtype
                    )

                    # Apply same numpy shuffling for other ranks
                    print(f"Rank {self.rank}: Applying numpy shuffling for {hookpoint}...")
                    indices = np.arange(len(dataset))
                    np.random.seed(self.seed)
                    np.random.shuffle(indices)
                    dataset = dataset.select(indices)

                    dataset = dataset.shard(self.world_size, self.rank)
                    dataset_dict[hookpoint] = dataset

        self._create_dual_data_loaders(dataset_dict)
        print("\n✅ Dataset initialization completed!")

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

    def assign_concepts_to_latents_from_scores(self, hook_name):
        """
        Assign concepts to specific latents using pre-computed scores.
        Supports both single concept mode and dual object-style mode.
        """
        print(f"\nAssigning concepts to latents for {hook_name}...")
        sae = self.saes[hook_name]
        model = sae.module if hasattr(sae, 'module') else sae
        model_num_latents = model.num_latents
        
        # ADD THIS BLOCK FOR FROM_SCRATCH HANDLING
        if self.from_scratch and self.scores_data is None:
            print("Training from scratch with no scores - assigning concepts randomly...")
            
            # Get concept names from the first batch of training data
            sample_batch = next(iter(self.train_loader))
            _, object_labels, style_labels = sample_batch
            unique_objects = set(object_labels)
            unique_styles = set([s for s in style_labels if s != "none"])
            
            all_concepts = list(unique_objects) + list(unique_styles)
            print(f"Found {len(all_concepts)} unique concepts: {all_concepts[:5]}...")
            
            # Create random assignment
            import random
            available_latents = list(range(model_num_latents))
            random.shuffle(available_latents)
            
            object_to_latent = {}
            style_to_latent = {}
            
            for i, concept in enumerate(all_concepts):
                if i < len(available_latents):
                    latent_idx = available_latents[i]
                else:
                    latent_idx = random.randint(0, model_num_latents - 1)
                
                if concept in unique_objects:
                    object_to_latent[concept] = latent_idx
                    print(f"  Randomly assigned object '{concept}' to latent {latent_idx}")
                else:
                    style_to_latent[concept] = latent_idx
                    print(f"  Randomly assigned style '{concept}' to latent {latent_idx}")
            
            print(f"\nCompleted random assignment: {len(object_to_latent)} objects, {len(style_to_latent)} styles")
            return object_to_latent, style_to_latent
        
        # If we don't have scores data but not from_scratch, raise error
        if self.scores_data is None:
            raise RuntimeError("Scores data not loaded and not training from scratch.")
        
        # EXISTING CODE CONTINUES HERE...
        # Get unique objects and styles from data
        sample_batch = next(iter(self.train_loader))
        _, object_labels, style_labels = sample_batch
        unique_objects = set(object_labels)
        unique_styles = set(style_labels)
        print(f"Found {len(unique_objects)} unique objects: {list(unique_objects)[:5]}...")
        print(f"Found {len(unique_styles)} unique styles: {list(unique_styles)[:5]}...")
        if self.dual_concept_mode and self.style_scores_data:
            print(f"🎯 Using DUAL CONCEPT MODE - assigning both objects and styles")
            return self._assign_dual_concepts(hook_name, unique_objects, unique_styles, model_num_latents)
        else:
            print(f"ℹ️  Using SINGLE CONCEPT MODE - assigning only objects")
            return self._assign_single_concepts(hook_name, unique_objects, model_num_latents)

    def _assign_dual_concepts(self, hook_name, unique_objects, unique_styles, model_num_latents):
        """Assign both objects and styles to latents with priority-based assignment."""
        
        object_scores = self.scores_data.get('scores', {})
        style_scores = self.style_scores_data.get('scores', {})

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

        print(f"\n📊 DUAL ASSIGNMENT SUMMARY:")
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

    def _assign_single_concepts(self, hook_name, unique_objects, model_num_latents):
        """Assign only objects to latents (single concept mode)."""
        
        object_scores = self.scores_data.get('scores', {})
        
        print(f"🎯 Single concept assignment for {len(unique_objects)} objects...")

        object_to_latent = {}
        style_to_latent = {}  # Empty in single concept mode
        latent_assignments = set()

        # Collect objects with their scores
        object_priorities = []
        objects_without_scores = []

        for concept_name in unique_objects:
            found_name, concept_scores = self.find_concept_in_scores(concept_name, object_scores)
            if found_name is not None:
                # Handle both 2D (timestep x latent) and 1D (latent) score arrays
                if len(concept_scores) > 0 and isinstance(concept_scores[0], list):
                    avg_scores = np.mean(concept_scores, axis=0)
                else:
                    avg_scores = concept_scores
                
                best_score = max(avg_scores)
                object_priorities.append((concept_name, best_score, avg_scores))
            else:
                objects_without_scores.append(concept_name)

        # Sort by score (highest first)
        object_priorities.sort(key=lambda x: x[1], reverse=True)

        print(f"Objects with scores: {len(object_priorities)}")
        print(f"Objects without scores: {len(objects_without_scores)}")

        # Assign in score order
        for concept_name, best_score, avg_scores in object_priorities:
            latent_idx = self._assign_single_concept_to_latent(
                concept_name, avg_scores, model_num_latents, latent_assignments
            )

            if latent_idx is not None:
                object_to_latent[concept_name] = latent_idx
                latent_assignments.add(latent_idx)
                score = avg_scores[latent_idx]
                print(f"✅ Object '{concept_name}' → latent {latent_idx} (score: {score:.6f})")
            else:
                objects_without_scores.append(concept_name)
                print(f"❌ Object '{concept_name}' - no available latents")

        # Fallback assignment for objects without scores
        if objects_without_scores:
            print(f"\n🔄 FALLBACK: Assigning {len(objects_without_scores)} objects without scores...")
            unused_latents = [i for i in range(model_num_latents) if i not in latent_assignments]
            
            for i, concept_name in enumerate(objects_without_scores):
                if i < len(unused_latents):
                    latent_idx = unused_latents[i]
                    object_to_latent[concept_name] = latent_idx
                    print(f"🔄 Fallback object '{concept_name}' → latent {latent_idx}")
                else:
                    print(f"⚠️  No latent available for object '{concept_name}'")

        print(f"\n📊 SINGLE ASSIGNMENT SUMMARY:")
        print(f"Objects: {len(object_to_latent)}/{len(unique_objects)} assigned")
        print(f"Latents used: {len(set(object_to_latent.values()))}/{model_num_latents}")

        return object_to_latent, style_to_latent

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
        sparsity_diff = val_losses['sparsity_loss'] - train_losses['sparsity_loss']

        print(f"{'Total Loss':<20} {train_losses['total_loss']:<12.6f} {val_losses['total_loss']:<12.6f} {train_diff:>+12.6f}")
        print(f"{'Reconstruction':<20} {train_losses['recon_loss']:<12.6f} {val_losses['recon_loss']:<12.6f} {recon_diff:>+12.6f}")
        print(f"{'Binary CE':<20} {train_losses['ce_loss']:<12.6f} {val_losses['ce_loss']:<12.6f} {ce_diff:>+12.6f}")
        print(f"{'Sparsity':<20} {train_losses['sparsity_loss']:<12.6f} {val_losses['sparsity_loss']:<12.6f} {sparsity_diff:>+12.6f}")

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
        object_scores = self.scores_data.get('scores', {}) if self.scores_data else {}
        style_scores = self.style_scores_data.get('scores', {}) if self.style_scores_data else {}

        # Print object assignments
        for concept, latent_idx in sorted(object_to_latent.items()):
            # Get the original score for this assignment
            score = "N/A"
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

            print(f"{concept:<20} {'object':<8} {latent_idx:<15} {score:<15}")

        # Print style assignments
        for concept, latent_idx in sorted(style_to_latent.items()):
            # Get the original score for this assignment
            score = "N/A"
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
        """Fixed statistics calculation with proper bounds checking."""
        model = sae.module if hasattr(sae, 'module') else sae
        model.eval()

        distributions = {}
        concept_probs = {}
        combined_concept_to_latent = {**object_to_latent, **style_to_latent}

        print("Calculating latent distribution statistics...")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_idx >= 3:  # Very limited for efficiency
                    break
                    
                activations, object_labels, style_labels = batch_data

                try:
                    activations = activations.to(self.device, dtype=self.dtype)

                    # Handle reshaping
                    if len(activations.shape) == 3:
                        original_shape = activations.shape
                        activations = activations.reshape(-1, activations.shape[-1])
                    
                    # Get post-topk activations for statistics
                    pre_acts = model.pre_acts(activations)
                    top_acts, top_indices = model.select_topk(pre_acts)
                    
                    # Reshape back if needed
                    if len(original_shape) == 3:
                        batch_size = len(object_labels)  # Use object_labels length
                        seq_len = original_shape[1]
                        top_acts = top_acts.reshape(batch_size, seq_len, -1)
                        top_acts = top_acts.mean(dim=1)
                    
                    # Convert to full activation vector for analysis
                    full_acts = torch.zeros(top_acts.shape[0], model.num_latents, device=self.device, dtype=self.dtype)
                    for i in range(top_acts.shape[0]):
                        if len(top_indices.shape) == 2:
                            # top_indices is [batch, k]
                            indices = top_indices[i]
                            values = top_acts[i]
                        else:
                            # Handle other shapes if needed
                            indices = top_indices[i] if top_indices.dim() > 1 else top_indices
                            values = top_acts[i] if top_acts.dim() > 1 else top_acts
                        
                        full_acts[i, indices] = values
                    
                    # Use softmax on the full activation vector for probability distribution
                    probs = F.softmax(full_acts, dim=1)
                    
                    # Collect stats for objects and styles separately
                    for i, (obj_concept, style_concept) in enumerate(zip(object_labels, style_labels)):
                        # Object concept
                        if obj_concept not in concept_probs:
                            concept_probs[obj_concept] = []
                        concept_probs[obj_concept].append(probs[i])
                        
                        # Style concept (if not "none")
                        if style_concept != "none":
                            if style_concept not in concept_probs:
                                concept_probs[style_concept] = []
                            concept_probs[style_concept].append(probs[i])
                        
                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    continue
                
        # Calculate statistics for each concept
        for concept, prob_list in concept_probs.items():
            if prob_list and concept in combined_concept_to_latent:
                try:
                    mean_probs = torch.stack(prob_list).mean(dim=0)
                    dominant_latent = torch.argmax(mean_probs).item()
                    
                    # CRITICAL: Validate the dominant latent index
                    if 0 <= dominant_latent < model.num_latents:
                        dominance_score = mean_probs[dominant_latent].item()
                        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item()
                        
                        distributions[concept] = {
                            "dominant_latent": dominant_latent,
                            "dominance_score": dominance_score,
                            "entropy": entropy
                        }
                    else:
                        print(f"  Invalid dominant latent {dominant_latent} for concept {concept}")
                        
                except Exception as e:
                    print(f"  Error processing concept {concept}: {e}")
                    continue
                
        return distributions

    def initialize_saes(self):
        """Load SAE models from checkpoint with resume functionality or create from scratch."""
        print(f"Loading SAE models from {self.checkpoint_path}")
        
        # Check if we should resume from a saved checkpoint
        if self.resume:
            print("Resume mode enabled - looking for latest checkpoints...")
        
        # Check if the checkpoint path itself contains an SAE model
        if (self.checkpoint_path / "cfg.json").exists() and (self.checkpoint_path / "sae.safetensors").exists():
            hook_name = self.checkpoint_path.name
            
            # If from_scratch is True, only try to resume if we're explicitly resuming training
            # (not loading the base model)
            if self.resume and not self.from_scratch:
                # Try to find latest checkpoint first
                latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                if latest_epoch is not None:
                    print(f"Found checkpoint for {hook_name} at epoch {latest_epoch}")
                    if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                        self.start_epoch = latest_epoch + 1
                        print(f"Will resume training from epoch {self.start_epoch}")
                        return
                    else:
                        print(f"Failed to load checkpoint, falling back to original model")
            elif self.resume and self.from_scratch:
                # When from_scratch=True, only resume if we have training checkpoints
                # but always create the model from scratch initially
                latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                if latest_epoch is not None:
                    print(f"Found training checkpoint for {hook_name} at epoch {latest_epoch}")
                    print("Will create model from scratch first, then load training state")
            
            # Load original model or create from scratch
            if self.from_scratch:
                print(f"Creating SAE from scratch for {hook_name}")
                try:
                    # Default SAE configuration (excluding d_in since it's passed separately)
                    cfg = {
                        "expansion_factor": 16,
                        "normalize_decoder": True,
                        "num_latents": 0,  # This will be calculated from d_in * expansion_factor
                        "k": 32,
                        "batch_topk": False,
                        "sample_topk": False,
                        "input_unit_norm": False,
                        "multi_topk": False
                    }
                    
                    # Create SaeConfig object (assuming you need to import SaeConfig)
                    from SAE.sae import SaeConfig
                    sae_config = SaeConfig(**cfg)
                    
                    # Create new SAE instance with d_in and config
                    sae = Sae(d_in=1280, cfg=sae_config, device=self.device, dtype=self.dtype)
                    sae = sae.to(device=self.device, dtype=self.dtype)
                    self.saes[hook_name] = sae
                    
                    # Create optimizer
                    self.optimizers[hook_name] = Adam(
                        [{"params": sae.parameters(), "lr": self.lr}],
                        eps=1e-8
                    )
                    print(f"Created SAE from scratch for {hook_name}")
                    
                    # Now try to resume training state if from_scratch + resume
                    if self.resume:
                        latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                        if latest_epoch is not None:
                            print(f"Loading training state from epoch {latest_epoch}")
                            if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                                self.start_epoch = latest_epoch + 1
                                print(f"Will resume training from epoch {self.start_epoch}")
                            else:
                                print(f"Failed to load training state, starting from epoch 0")
                                
                except Exception as e:
                    print(f"Could not create SAE from scratch for {hook_name}: {e}")
            else:
                try:
                    sae = Sae.load_from_disk(self.checkpoint_path, device=self.device)
                    sae = sae.to(dtype=self.dtype)
                    self.saes[hook_name] = sae
                    
                    # Create optimizer
                    self.optimizers[hook_name] = Adam(
                        [{"params": sae.parameters(), "lr": self.lr}],
                        eps=1e-8
                    )
                    print(f"Loaded SAE for {hook_name}")
                except Exception as e:
                    print(f"Could not load SAE from {self.checkpoint_path}: {e}")
        
        # If we haven't loaded any models yet, try to find them in subdirectories
        if not self.saes:
            # Try to load SAEs from subdirectories
            for hook_dir in self.checkpoint_path.iterdir():
                if hook_dir.is_dir():
                    hook_name = hook_dir.name
                    
                    # Similar logic for subdirectories...
                    if self.resume and not self.from_scratch:
                        # Try to find latest checkpoint first
                        latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                        if latest_epoch is not None:
                            print(f"Found checkpoint for {hook_name} at epoch {latest_epoch}")
                            if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                                self.start_epoch = max(self.start_epoch, latest_epoch + 1)
                                print(f"Will resume training from epoch {self.start_epoch}")
                                continue
                            else:
                                print(f"Failed to load checkpoint, falling back to original model")
                    elif self.resume and self.from_scratch:
                        # When from_scratch=True, only resume if we have training checkpoints
                        # but always create the model from scratch initially
                        latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                        if latest_epoch is not None:
                            print(f"Found training checkpoint for {hook_name} at epoch {latest_epoch}")
                            print("Will create model from scratch first, then load training state")
                    
                    # Load original model or create from scratch
                    if self.from_scratch:
                        print(f"Creating SAE from scratch for {hook_name}")
                        try:
                            # Default SAE configuration (excluding d_in since it's passed separately)
                            cfg = {
                                "expansion_factor": 16,
                                "normalize_decoder": True,
                                "num_latents": 0,  # This will be calculated from d_in * expansion_factor
                                "k": 32,
                                "batch_topk": False,
                                "sample_topk": False,
                                "input_unit_norm": False,
                                "multi_topk": False
                            }
                            
                            # Create SaeConfig object (assuming you need to import SaeConfig)
                            from SAE.sae import SaeConfig
                            sae_config = SaeConfig(**cfg)
                            
                            # Create new SAE instance with d_in and config
                            sae = Sae(d_in=1280, cfg=sae_config, device=self.device, dtype=self.dtype)
                            sae = sae.to(device=self.device, dtype=self.dtype)
                            self.saes[hook_name] = sae
                            
                            # Create optimizer
                            self.optimizers[hook_name] = Adam(
                                [{"params": sae.parameters(), "lr": self.lr}],
                                eps=1e-8
                            )
                            print(f"Created SAE from scratch for {hook_name}")
                            
                            # Now try to resume training state if from_scratch + resume
                            if self.resume:
                                latest_epoch, latest_checkpoint_path = self.find_latest_checkpoint(hook_name)
                                if latest_epoch is not None:
                                    print(f"Loading training state from epoch {latest_epoch}")
                                    if self.load_checkpoint_state(hook_name, latest_checkpoint_path):
                                        self.start_epoch = max(self.start_epoch, latest_epoch + 1)
                                        print(f"Will resume training from epoch {self.start_epoch}")
                                    else:
                                        print(f"Failed to load training state, starting from epoch 0")
                                        
                        except Exception as e:
                            print(f"Could not create SAE from scratch for {hook_name}: {e}")
                    else:
                        try:
                            sae = Sae.load_from_disk(hook_dir, device=self.device)
                            sae = sae.to(dtype=self.dtype)
                            self.saes[hook_name] = sae
                            
                            # Create optimizer
                            self.optimizers[hook_name] = Adam(
                                [{"params": sae.parameters(), "lr": self.lr}],
                                eps=1e-8
                            )
                            print(f"Loaded SAE for {hook_name}")
                        except Exception as e:
                            print(f"Could not load SAE for {hook_name}: {e}")
    
    def initialize_wandb(self):
        """Initialize weights and biases for logging in offline mode with incremental resume support."""
        if WANDB_AVAILABLE:
            # Create directory for wandb logs
            wandb_dir = os.path.join(self.save_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            
            # Set environment variable to run wandb in offline mode
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = wandb_dir
            
            # Check if we're resuming and find existing run
            run_id = None
            run_name = None
            
            if self.resume and self.start_epoch > 1:
                # Try to find existing run ID from previous runs
                run_file = os.path.join(wandb_dir, "run_id.txt")
                if os.path.exists(run_file):
                    try:
                        with open(run_file, 'r') as f:
                            run_id = f.read().strip()
                        print(f"Found existing run ID for resume: {run_id}")
                        run_name = f"sae_dual_concept_optimization_resumed_{run_id}"
                    except Exception as e:
                        print(f"Could not load existing run ID: {e}")
            
            # Create a new run name if not resuming or couldn't find existing
            if run_name is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                mode = "dual" if self.dual_concept_mode else "single"
                run_name = f"sae_{mode}_concept_optimization_{timestamp}"
            
            # Prepare SAE configurations
            sae_configs = {}
            for hook_name, sae in self.saes.items():
                model = sae.module if hasattr(sae, 'module') else sae
                
                # Extract SAE configuration
                sae_config = {
                    "num_latents": getattr(model, 'num_latents', None),
                    "d_in": getattr(model, 'd_in', None),
                    "expansion_factor": getattr(model, 'expansion_factor', None),
                    "dtype": str(getattr(model, 'dtype', None)),
                    "device": str(getattr(model, 'device', None)),
                    "k": getattr(model, 'k', None),  # for TopK SAEs
                    "auxk_alpha": getattr(model, 'auxk_alpha', None),  # auxiliary loss coefficient
                    "dead_threshold": getattr(model, 'dead_threshold', None),
                    "norm_activation": getattr(model, 'norm_activation', None),
                    "scale_sparsity_penalty_by_decoder_norm": getattr(model, 'scale_sparsity_penalty_by_decoder_norm', None),
                    "decoder_bias": getattr(model, 'decoder_bias', None),
                    "normalize_w_dec": getattr(model, 'normalize_w_dec', None),
                }
                
                # Add encoder/decoder weight info if available
                if hasattr(model, 'W_enc') and model.W_enc is not None:
                    sae_config["W_enc_shape"] = list(model.W_enc.shape)
                if hasattr(model, 'W_dec') and model.W_dec is not None:
                    sae_config["W_dec_shape"] = list(model.W_dec.shape)
                if hasattr(model, 'b_enc') and model.b_enc is not None:
                    sae_config["b_enc_shape"] = list(model.b_enc.shape)
                if hasattr(model, 'b_dec') and model.b_dec is not None:
                    sae_config["b_dec_shape"] = list(model.b_dec.shape)
                
                # Add to configs dict
                sae_configs[hook_name] = sae_config
            
            # Add concept assignment info
            concept_assignment_info = {}
            if hasattr(self, 'object_to_latent') and hasattr(self, 'style_to_latent'):
                for hook_name in self.saes.keys():
                    object_mapping = self.object_to_latent.get(hook_name, {})
                    style_mapping = self.style_to_latent.get(hook_name, {})
                    concept_assignment_info[hook_name] = {
                        "num_objects": len(object_mapping),
                        "num_styles": len(style_mapping),
                        "object_names": list(object_mapping.keys()),
                        "style_names": list(style_mapping.keys()),
                        "assigned_object_latents": list(object_mapping.values()),
                        "assigned_style_latents": list(style_mapping.values()),
                        "unique_latents_used": len(set(list(object_mapping.values()) + list(style_mapping.values()))),
                    }
            
            # Add scores data info
            scores_info = {
                "object_scores_json_path": str(self.scores_json_path),
                "dual_concept_mode": self.dual_concept_mode,
            }
            if self.scores_data:
                scores_info.update({
                    "object_concept_type": self.scores_data.get('concept_type', 'unknown'),
                    "object_num_timesteps": self.scores_data.get('num_timesteps', 'unknown'),
                    "num_object_concepts_in_scores": len(self.scores_data.get('scores', {})),
                })
            if self.style_scores_data:
                scores_info.update({
                    "style_scores_json_path": str(self.style_scores_json_path),
                    "style_concept_type": self.style_scores_data.get('concept_type', 'unknown'),
                    "style_num_timesteps": self.style_scores_data.get('num_timesteps', 'unknown'),
                    "num_style_concepts_in_scores": len(self.style_scores_data.get('scores', {})),
                })
            
            config = {
                # Training hyperparameters
                "learning_rate": self.lr,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "reconstruction_weight": self.reconstruction_weight,
                "cross_entropy_weight": self.cross_entropy_weight,
                "sparsity_weight": self.sparsity_weight,
                "seed": self.seed,
                "validation_split": self.validation_split,
                "mixed_batches": self.mixed_batches,
                "patience": self.patience,
                "resume": self.resume,
                "from_scratch": self.from_scratch,
                "start_epoch": self.start_epoch,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "mixed_precision": self.mixed_precision,
                "use_float16": self.use_float16,
                
                # Data paths
                "checkpoint_path": str(self.checkpoint_path),
                "activations_dir": str(self.activations_dir),
                "save_dir": str(self.save_dir),
                
                # System info
                "device": str(self.device),
                "world_size": self.world_size,
                "rank": self.rank,
                
                # SAE configurations
                "sae_configs": sae_configs,
                
                # Concept assignment info
                "concept_assignments": concept_assignment_info,
                
                # Scores data info
                "scores_info": scores_info,
            }
            
            # Initialize wandb with resume capability
            if run_id and self.resume:
                try:
                    wandb.init(
                        project="sae_dual_concept_latent_optimizer",
                        name=run_name,
                        id=run_id,
                        resume="must",
                        config=config,
                        dir=wandb_dir
                    )
                    print(f"Resumed wandb logging with ID: {run_id}")
                except Exception as e:
                    print(f"Could not resume wandb run, starting new: {e}")
                    wandb.init(
                        project="sae_dual_concept_latent_optimizer",
                        name=run_name,
                        config=config,
                        dir=wandb_dir
                    )
                    run_id = wandb.run.id
            else:
                wandb.init(
                    project="sae_dual_concept_latent_optimizer",
                    name=run_name,
                    config=config,
                    dir=wandb_dir
                )
                run_id = wandb.run.id
            
            # Save run ID for future resume
            run_file = os.path.join(wandb_dir, "run_id.txt")
            try:
                with open(run_file, 'w') as f:
                    f.write(run_id)
                print(f"Saved run ID {run_id} for future resume")
            except Exception as e:
                print(f"Could not save run ID: {e}")
            
            mode_str = "DUAL CONCEPT" if self.dual_concept_mode else "SINGLE CONCEPT"
            print(f"Initialized wandb logging in OFFLINE mode ({mode_str})")
            print(f"Logs will be stored in: {wandb_dir}")
            print(f"SAE configurations logged for {len(sae_configs)} hooks: {list(sae_configs.keys())}")

    def compute_reconstruction_loss(self, sae, activations):
        """Fixed reconstruction loss that handles tensor dimensions correctly."""
        # Get the actual model
        model = sae.module if hasattr(sae, 'module') else sae

        # Ensure activations are 2D [batch_size, features]
        if len(activations.shape) == 3:
            # If 3D [batch, time, features], flatten first two dimensions
            batch_size, time_steps, features = activations.shape
            activations = activations.reshape(batch_size * time_steps, features)
        elif len(activations.shape) == 4:
            # If 4D [batch, time, height, width], flatten spatial dimensions
            batch_size, time_steps, height, width = activations.shape
            activations = activations.reshape(batch_size * time_steps, height * width)

        if len(activations.shape) != 2:
            print(f"  Recon Error: Cannot handle activations shape: {activations.shape}")
            return torch.tensor(1.0, device=self.device, dtype=self.dtype), torch.zeros(activations.shape[0], 1000, device=self.device)

        try:
            # Get pre-activations
            pre_acts = model.pre_acts(activations)

            # Check for NaN
            if torch.isnan(pre_acts).any():
                print(f"  Recon Warning: NaN in pre_acts")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype), torch.zeros_like(pre_acts)

            # Get top-k activations
            top_acts, top_indices = model.select_topk(pre_acts)

            # Decode and compute reconstruction loss
            reconstructed = model.decode(top_acts, top_indices)

            # Ensure shapes match for loss computation
            if reconstructed.shape != activations.shape:
                print(f"  Recon Error: Shape mismatch - reconstructed: {reconstructed.shape}, original: {activations.shape}")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype), pre_acts

            loss = F.mse_loss(reconstructed, activations)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Recon Warning: NaN/Inf in loss")
                return torch.tensor(1.0, device=self.device, dtype=self.dtype), pre_acts

            print(f"  Recon Loss: {loss.item():.6f}")
            return loss, (top_acts, top_indices)  # Return topk info instead of pre_acts

        except Exception as e:
            print(f"  Recon Error: {e}")
            print(f"    Input shape: {activations.shape}")
            import traceback
            traceback.print_exc()
            return torch.tensor(1.0, device=self.device, dtype=self.dtype), (torch.zeros(activations.shape[0], 32, device=self.device), torch.zeros(activations.shape[0], 32, dtype=torch.long, device=self.device))

    def compute_binary_cross_entropy_loss(self, topk_info, object_labels, style_labels, object_to_latent, style_to_latent):
        """
        Enhanced binary cross-entropy loss that operates on post-topk activations.
        Uses binary cross-entropy with logits to allow multiple latents to be simultaneously active.
        
        Args:
            topk_info: Tuple of (top_acts, top_indices) from SAE's select_topk method
            object_labels: List of object concept names for this batch
            style_labels: List of style concept names for this batch
            object_to_latent: Mapping from object names to latent indices
            style_to_latent: Mapping from style names to latent indices
            
        Returns:
            Binary cross-entropy loss computed only on assigned latents using post-topk activations
        """
        top_acts, top_indices = topk_info
        
        # print(f"  BCE Loss Debug (Post-TopK): top_acts shape = {top_acts.shape}, top_indices shape = {top_indices.shape}")
        # print(f"  BCE Loss Debug: num objects = {len(object_labels)}, num styles = {len(style_labels)}")
    
        # Handle the case where activations were reshaped from [batch, seq, features] to [batch*seq, features]
        if len(top_acts.shape) == 2:
            batch_times_seq, k = top_acts.shape
            batch_size = len(object_labels)
    
            # Check if we need to reshape back
            if batch_times_seq != batch_size:
                # Calculate sequence length
                seq_length = batch_times_seq // batch_size
                if batch_times_seq == batch_size * seq_length:
                    # Reshape back to [batch, seq, k]
                    top_acts = top_acts.view(batch_size, seq_length, k)
                    top_indices = top_indices.view(batch_size, seq_length, k)
                    # Take mean over sequence dimension for acts, mode for indices
                    top_acts = top_acts.mean(dim=1)  # [batch, k]
                    # For indices, we'll take the first timestep's indices (or implement a more sophisticated strategy)
                    top_indices = top_indices[:, 0, :]  # [batch, k] - taking first timestep
                else:
                    print(f"  BCE Loss Error: Cannot reshape {batch_times_seq} to match batch size {batch_size}")
                    return torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
        elif len(top_acts.shape) == 3:
            # If 3D [batch, seq, k], take mean over sequence for acts
            top_acts = top_acts.mean(dim=1)
            top_indices = top_indices[:, 0, :]  # Take first timestep's indices
    
        if len(top_acts.shape) != 2:
            print(f"  BCE Loss Error: Unexpected top_acts shape after processing: {top_acts.shape}")
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
        batch_size, k = top_acts.shape
    
        # Ensure batch size matches
        if batch_size != len(object_labels) or batch_size != len(style_labels):
            print(f"  BCE Loss Error: Batch size mismatch")
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
        # Get the actual model to know num_latents
        if hasattr(self, 'saes'):
            hook_name = list(self.saes.keys())[0]  # Get first hook for num_latents
            model = self.saes[hook_name].module if hasattr(self.saes[hook_name], 'module') else self.saes[hook_name]
            num_latents = model.num_latents
        else:
            print(f"  BCE Loss Error: Cannot determine num_latents")
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Create target tensor for all samples
        # Shape: [batch_size, k] - binary targets indicating which of the top-k latents should be active
        target_mask = torch.zeros(batch_size, k, device=self.device, dtype=torch.float32)
        
        valid_samples = 0
        
        for i, (object_concept, style_concept) in enumerate(zip(object_labels, style_labels)):
            has_targets = False
            sample_indices = top_indices[i]  # [k] - the latent indices for this sample
            
            # Check if object latent is in this sample's top-k
            if object_concept in object_to_latent:
                object_latent = object_to_latent[object_concept]
                if 0 <= object_latent < num_latents:
                    # Find if this latent is in the top-k for this sample
                    mask = (sample_indices == object_latent)
                    if mask.any():
                        target_mask[i, mask] = 1.0
                        has_targets = True
            
            # Check if style latent is in this sample's top-k
            if style_concept != "none" and style_concept in style_to_latent:
                style_latent = style_to_latent[style_concept]
                if 0 <= style_latent < num_latents:
                    # Find if this latent is in the top-k for this sample
                    mask = (sample_indices == style_latent)
                    if mask.any():
                        target_mask[i, mask] = 1.0
                        has_targets = True
            
            if has_targets:
                valid_samples += 1
    
        if valid_samples == 0:
            print(f"  BCE Loss: No valid targets found in top-k activations")
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
        # Use binary cross-entropy with logits on the top-k activations
        # top_acts are already the selected activations, we want to encourage the target ones to be high
        bce_loss = F.binary_cross_entropy_with_logits(top_acts, target_mask, reduction='none')
        
        # Only compute loss for samples and positions that have targets
        valid_mask = target_mask > 0  # Only positions where we have target latents
        
        if valid_mask.sum() == 0:
            print(f"  BCE Loss: No valid target positions in top-k")
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Average loss only over valid target positions
        final_loss = bce_loss[valid_mask].mean()
        
        print(f"  BCE Loss (Post-TopK): {final_loss.item():.6f} (from {valid_samples} samples, {valid_mask.sum().item()} target positions in top-k)")
        return final_loss
    
    def compute_sparsity_loss(self, topk_info):
        """
        Compute L1 sparsity regularization on top-k activations.

        Args:
            topk_info: Tuple of (top_acts, top_indices) from select_topk

        Returns:
            loss: The sparsity loss
        """
        top_acts, top_indices = topk_info
        
        # Check for NaN values
        if torch.isnan(top_acts).any():
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Clip extremely large values to prevent overflow
        clipped_acts = torch.clamp(top_acts, -100, 100)

        # Use a more stable formulation
        sparsity = torch.mean(torch.abs(clipped_acts))

        # Prevent NaN return
        if torch.isnan(sparsity) or torch.isinf(sparsity):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return sparsity
    
    def evaluate_losses(self, sae, hook_name, object_to_latent, style_to_latent, is_validation=False):
        """
        Evaluate the losses for either training or validation set.
        Fixed version with better memory management and progress tracking.
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
        sparsity_loss_sum = 0.0
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
                
                # Print progress
                print(f"  Processing batch {batch_idx + 1}/{max_batches}...")

                try:
                    activations = activations.to(self.device, dtype=self.dtype)

                    # Compute losses
                    recon_loss, topk_info = self.compute_reconstruction_loss(sae, activations)
                    ce_loss = self.compute_binary_cross_entropy_loss(topk_info, object_labels, style_labels, object_to_latent, style_to_latent)
                    sparsity_loss = self.compute_sparsity_loss(topk_info)

                    # Combined loss
                    total_loss = (
                        self.reconstruction_weight * recon_loss +
                        self.cross_entropy_weight * ce_loss +
                        self.sparsity_weight * sparsity_loss
                    )

                    # Accumulate losses
                    total_loss_sum += total_loss.item()
                    recon_loss_sum += recon_loss.item()
                    ce_loss_sum += ce_loss.item()
                    sparsity_loss_sum += sparsity_loss.item()
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
            avg_sparsity_loss = sparsity_loss_sum / num_batches
        else:
            print(f"  WARNING: No batches processed for {dataset_type}")
            avg_total_loss = avg_recon_loss = avg_ce_loss = avg_sparsity_loss = 0.0

        print(f"  Completed {dataset_type} evaluation")

        # Return losses
        return {
            "total_loss": avg_total_loss,
            "recon_loss": avg_recon_loss,
            "ce_loss": avg_ce_loss,
            "sparsity_loss": avg_sparsity_loss
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
        Train the SAE models to assign specific latents to concepts using distributed training.
        This method handles both single-GPU and multi-GPU (distributed) training.
        Enhanced with dual object-style concept assignment, resume functionality and early stopping.
        MODIFIED: Uses post-topk BCE loss computation.
        """
        # Create save directory if it doesn't exist
        if self.rank == 0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # Process each SAE model
        for hook_name, sae in self.saes.items():
            if self.rank == 0:
                print(f"\nTraining SAE model for {hook_name}")
                if self.resume and self.start_epoch > 1:
                    print(f"Resuming training from epoch {self.start_epoch}")

            # Assign concepts to latents based on pre-computed scores (only on rank 0)
            if self.rank == 0:
                object_to_latent, style_to_latent = self.assign_concepts_to_latents_from_scores(hook_name)
                self.object_to_latent[hook_name] = object_to_latent
                self.style_to_latent[hook_name] = style_to_latent
                self.print_initial_concept_assignments(object_to_latent, style_to_latent, hook_name)

                if self.world_size > 1:
                    # Broadcast mappings to all processes
                    object_list = [self.object_to_latent[hook_name]]
                    style_list = [self.style_to_latent[hook_name]]
                    dist.broadcast_object_list(object_list, src=0)
                    dist.broadcast_object_list(style_list, src=0)
            else:
                # Other ranks receive the mappings
                object_list = [None]
                style_list = [None]
                dist.broadcast_object_list(object_list, src=0)
                dist.broadcast_object_list(style_list, src=0)
                self.object_to_latent[hook_name] = object_list[0]
                self.style_to_latent[hook_name] = style_list[0]
            
            # Make sure all processes have the mappings before continuing
            if self.world_size > 1:
                dist.barrier()
            
            # Compute initial losses (only on rank 0 for logging purposes)
            if self.rank == 0 and self.start_epoch == 1:
                train_losses = self.evaluate_losses(sae, hook_name, self.object_to_latent[hook_name], self.style_to_latent[hook_name], is_validation=False)
                val_losses = self.evaluate_losses(sae, hook_name, self.object_to_latent[hook_name], self.style_to_latent[hook_name], is_validation=True)
                
                print("\n=== Initial Losses ===")
                print(f"  Training   - Total: {train_losses['total_loss']:.6f}, Recon: {train_losses['recon_loss']:.6f}, BCE: {train_losses['ce_loss']:.6f}")
                print(f"  Validation - Total: {val_losses['total_loss']:.6f}, Recon: {val_losses['recon_loss']:.6f}, BCE: {val_losses['ce_loss']:.6f}")
                
                # Log initial metrics to wandb
                if WANDB_AVAILABLE:
                    initial_metrics = {
                        f"{hook_name}/initial/train/total_loss": train_losses['total_loss'],
                        f"{hook_name}/initial/train/recon_loss": train_losses['recon_loss'],
                        f"{hook_name}/initial/train/bce_loss": train_losses['ce_loss'],
                        f"{hook_name}/initial/val/total_loss": val_losses['total_loss'],
                        f"{hook_name}/initial/val/recon_loss": val_losses['recon_loss'],
                        f"{hook_name}/initial/val/bce_loss": val_losses['ce_loss'],
                    }
                    wandb.log(initial_metrics)
        
            # Training loop - start from self.start_epoch
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                if self.rank == 0:
                    print(f"\nEpoch {epoch}/{self.num_epochs}")
                    print(f"Training {hook_name}...")

                sae.train()
                optimizer = self.optimizers[hook_name]
                
                # Get the concept-to-latent mappings for this hook
                object_to_latent = self.object_to_latent[hook_name]
                style_to_latent = self.style_to_latent[hook_name]
                
                # Set epoch for train sampler (for distributed training)
                if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)
                
                # Track losses
                total_loss_sum = 0.0
                recon_loss_sum = 0.0
                ce_loss_sum = 0.0
                sparsity_loss_sum = 0.0
                num_batches = 0
                
                data_iter = self.train_loader
                if self.rank == 0:
                    data_iter = tqdm(data_iter, desc="Batches")

                # Process batches
                for batch_idx, (activations, object_labels, style_labels) in enumerate(data_iter):
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    activations = activations.to(self.device)
                    original_batch_size = activations.size(0)

                    # Mixed precision training
                    if self.mixed_precision and torch.cuda.is_available() and not self.use_float16:
                        with torch.amp.autocast('cuda'):
                            recon_loss, topk_info = self.compute_reconstruction_loss(sae, activations)

                            # Binary cross-entropy loss for concept-specific latents (AFTER topk)
                            ce_loss = self.compute_binary_cross_entropy_loss(
                                topk_info, 
                                object_labels, 
                                style_labels,
                                object_to_latent,
                                style_to_latent
                            )

                            # Sparsity loss
                            sparsity_loss = self.compute_sparsity_loss(topk_info)

                            # Combined loss
                            total_loss = (
                                self.reconstruction_weight * recon_loss +
                                self.cross_entropy_weight * ce_loss +
                                self.sparsity_weight * sparsity_loss
                            )
                        
                        # Optimization step with mixed precision
                        optimizer.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        # Standard precision training
                        recon_loss, topk_info = self.compute_reconstruction_loss(sae, activations)

                        # Binary cross-entropy loss for concept-specific latents (AFTER topk)
                        ce_loss = self.compute_binary_cross_entropy_loss(
                            topk_info, 
                            object_labels, 
                            style_labels,
                            object_to_latent,
                            style_to_latent
                        )

                        # Sparsity loss
                        sparsity_loss = self.compute_sparsity_loss(topk_info)

                        # Combined loss
                        total_loss = (
                            self.reconstruction_weight * recon_loss +
                            self.cross_entropy_weight * ce_loss +
                            self.sparsity_weight * sparsity_loss
                        )
                        
                        optimizer.zero_grad()

                        # Check for NaN in loss
                        if torch.isnan(total_loss).any():
                            print(f"WARNING: NaN detected in loss, skipping backward")
                            continue
                        
                        total_loss.backward()

                        # Gradient clipping to prevent explosion
                        torch.nn.utils.clip_grad_norm_(sae.parameters() if not isinstance(sae, DDP) else sae.module.parameters(), 1.0)

                        # Gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    
                    # Accumulate losses for logging
                    total_loss_sum += total_loss.item()
                    recon_loss_sum += recon_loss.item()
                    ce_loss_sum += ce_loss.item()
                    sparsity_loss_sum += sparsity_loss.item()
                    num_batches += 1

                    del recon_loss, ce_loss, sparsity_loss, total_loss

                    if 'topk_info' in locals():
                        del topk_info

                    # Force garbage collection every 50 batches
                    if batch_idx % 50 == 0:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                
                # Synchronize loss statistics across processes (for distributed training)
                if self.world_size > 1:
                    # Create tensors with loss values
                    loss_tensor = torch.tensor(
                        [total_loss_sum, recon_loss_sum, ce_loss_sum, sparsity_loss_sum, num_batches],
                        dtype=torch.float32, device=self.device
                    )
                    
                    # All-reduce to get the sum across all processes
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    
                    # Unpack the reduced values
                    total_loss_sum = loss_tensor[0].item()
                    recon_loss_sum = loss_tensor[1].item()
                    ce_loss_sum = loss_tensor[2].item()
                    sparsity_loss_sum = loss_tensor[3].item()
                    num_batches = int(loss_tensor[4].item())
                
                if num_batches > 0:
                    avg_total_loss = total_loss_sum / num_batches
                    avg_recon_loss = recon_loss_sum / num_batches
                    avg_ce_loss = ce_loss_sum / num_batches
                    avg_sparsity_loss = sparsity_loss_sum / num_batches
                else:
                    print(f"WARNING: No batches processed in epoch {epoch}")
                    avg_total_loss = avg_recon_loss = avg_ce_loss = avg_sparsity_loss = 0.0
                    continue  # Skip to next epoch
                
                # Print training statistics (only on rank 0)
                if self.rank == 0:
                    print(f"\nEpoch {epoch} Training Averages:")
                    print(f"  Total Loss: {avg_total_loss:.6f}")
                    print(f"  Recon Loss: {avg_recon_loss:.6f}")
                    print(f"  BCE Loss (Post-TopK): {avg_ce_loss:.6f}")
                    print(f"  Sparsity Loss: {avg_sparsity_loss:.6f}")
                    
                    # Evaluate on validation set
                    train_losses = self.evaluate_losses(sae, hook_name, object_to_latent, style_to_latent, is_validation=False)
                    val_losses = self.evaluate_losses(sae, hook_name, object_to_latent, style_to_latent, is_validation=True)
                    
                    print(f"\n=== End of Epoch {epoch} Losses ===")
                    print(f"  Training   - Total: {train_losses['total_loss']:.6f}, Recon: {train_losses['recon_loss']:.6f}, BCE: {train_losses['ce_loss']:.6f}")
                    print(f"  Validation - Total: {val_losses['total_loss']:.6f}, Recon: {val_losses['recon_loss']:.6f}, BCE: {val_losses['ce_loss']:.6f}")
                    
                    # Check for early stopping and save best model
                    should_stop = self.check_early_stopping(val_losses['total_loss'], epoch, sae, hook_name, optimizer)

                    # Always save current checkpoint for resume capability
                    if isinstance(sae, DDP):
                        self.save_current_checkpoint(sae.module, hook_name, epoch, optimizer)
                    else:
                        self.save_current_checkpoint(sae, hook_name, epoch, optimizer)

                    # Check if we should stop early
                    if should_stop:
                        print(f"🛑 Early stopping triggered at epoch {epoch}")
                        break

                    # Calculate latent distribution statistics
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
                    
                    # Print distribution summaries
                    self.print_latent_distribution_summary(
                        train_distributions, 
                        object_to_latent,
                        style_to_latent,
                        epoch=epoch, 
                        is_validation=False
                    )
                    self.print_latent_distribution_summary(
                        val_distributions, 
                        object_to_latent,
                        style_to_latent,
                        epoch=epoch, 
                        is_validation=True
                    )
                    
                    # Log metrics to wandb
                    if WANDB_AVAILABLE:
                        # Calculate actual step for incremental logging
                        step = epoch if not self.resume else (self.start_epoch - 1) + (epoch - self.start_epoch + 1)
                        
                        combined_concept_to_latent = {**object_to_latent, **style_to_latent}
                        
                        metrics = {
                            f"{hook_name}/train/total_loss": train_losses['total_loss'],
                            f"{hook_name}/train/recon_loss": train_losses['recon_loss'],
                            f"{hook_name}/train/bce_loss": train_losses['ce_loss'],
                            f"{hook_name}/val/total_loss": val_losses['total_loss'],
                            f"{hook_name}/val/recon_loss": val_losses['recon_loss'],
                            f"{hook_name}/val/bce_loss": val_losses['ce_loss'],
                            f"{hook_name}/best_val_loss": self.best_val_loss,
                            f"{hook_name}/patience_counter": self.patience_counter,
                            "epoch": epoch,
                            "global_step": step
                        }
                        
                        # Calculate and log success rates
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
                        
                        wandb.log(metrics, step=step)
                    
                    # Print comprehensive epoch summary
                    self.print_epoch_summary(
                        epoch, hook_name, train_losses, val_losses,
                        train_distributions, val_distributions, object_to_latent, style_to_latent
                    )
                    
                    # Check if we should stop early
                    if should_stop:
                        print(f"🛑 Early stopping triggered at epoch {epoch}")
                        break
                
                # Synchronize processes before starting the next epoch
                if self.world_size > 1:
                    dist.barrier()
        
        if self.rank == 0:
            if hasattr(self, 'best_val_loss') and self.best_val_loss != float('inf'):
                print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.6f}")
            else:
                print("\nTraining completed successfully!")


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
        scores_json_path=args.scores_json_path,
        style_scores_json_path=getattr(args, 'style_scores_json_path', None),
        device=device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        reconstruction_weight=args.reconstruction_weight,
        cross_entropy_weight=args.cross_entropy_weight,
        sparsity_weight=args.sparsity_weight,
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
        from_scratch=getattr(args, 'from_scratch', False)
    )

    # Update the SAE models to be DDP models
    for hook_name, sae in optimizer.saes.items():
        # Move to device
        sae = sae.to(device)
        # Wrap with DDP
        ddp_model = DDP(sae, device_ids=[rank])
        optimizer.saes[hook_name] = ddp_model
        
        # Update optimizer to point to new model parameters
        optimizer.optimizers[hook_name] = Adam(
            [{"params": ddp_model.parameters(), "lr": optimizer.lr}],
            eps=1e-8
        )

    # Train the models
    optimizer.train()

    # Cleanup
    dist.destroy_process_group()

def main():
    """
    Main entry point for the Enhanced SAE Concept Latent Optimizer with dual object-style support.
    MODIFIED: Uses post-topk BCE loss computation.
    """
    parser = argparse.ArgumentParser(description="Optimize SAE models to assign specific latents to concepts with optional dual object-style support (Post-TopK BCE version).")
    
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
        help="Path to the concept activations directory"
    )
    parser.add_argument(
        "--scores_json_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing pre-computed object scores"
    )
    
    # Optional style scores for dual concept mode
    parser.add_argument(
        "--style_scores_json_path", 
        type=str, 
        default=None,
        help="Path to the JSON file containing pre-computed style scores (enables dual concept mode)"
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
    parser.add_argument("--cross_entropy_weight", type=float, default=1.0, help="Weight for binary cross-entropy loss")
    parser.add_argument("--sparsity_weight", type=float, default=0.01, help="Weight for sparsity regularization")
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="sae-concept-latent-optimized", help="Directory to save optimized models")
    
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
    
        # With torchrun, run the training function directly
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
            # Create and run the optimizer with updated parameters
            optimizer = SAEConceptLatentOptimizer(
                checkpoint_path=args.checkpoint_path,
                activations_dir=args.activations_dir,
                scores_json_path=args.scores_json_path,
                style_scores_json_path=args.style_scores_json_path,
                device=args.device,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                reconstruction_weight=args.reconstruction_weight,
                cross_entropy_weight=args.cross_entropy_weight,
                sparsity_weight=args.sparsity_weight,
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