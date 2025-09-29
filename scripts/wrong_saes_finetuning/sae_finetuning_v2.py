#!/usr/bin/env python
"""
Load activation dictionaries and optimize SAE models to assign specific latents to concepts
while maintaining reconstruction quality.

This script processes raw activations from different concepts, assigns specific latent neurons
to each concept based on pre-computed scores from a JSON file, and finetunes the SAE to maintain 
this assignment through binary cross-entropy loss.
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

from SAE.sae import Sae, SaeConfig
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

def load_datasets_from_category_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load datasets from concept directories, ensuring correct labels.
    """
    datasets = []
    print(f"Loading datasets from {base_dirs} for hookpoint {hookpoint}")

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        hookpoint_dir = base_path / hookpoint
        
        if not hookpoint_dir.exists():
            print(f"❌ Hookpoint directory does not exist: {hookpoint_dir}")
            continue
            
        concept_subdirs = [d for d in hookpoint_dir.iterdir() if d.is_dir()]
        
        for concept_dir in concept_subdirs:
            concept_name = concept_dir.name
            
            if (concept_dir / "dataset_info.json").exists():
                print(f"  Loading concept '{concept_name}'...")
                
                # Load the dataset
                dataset = HFDataset.load_from_disk(str(concept_dir), keep_in_memory=False)
                
                print(f"    Original columns: {dataset.column_names}")
                
                # Remove existing object_label if it exists
                if "object_label" in dataset.column_names:
                    dataset = dataset.remove_columns(["object_label"])
                    print(f"    Removed existing object_label column")
                
                # Add the correct concept label based on directory name
                dataset = dataset.add_column("object_label", [concept_name] * len(dataset))
                print(f"    Added object_label column with value '{concept_name}'")
                
                # Set format
                dataset.set_format(
                    type="torch",
                    columns=["activations", "timestep", "object_label"],
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
    1. Loads raw activations for different concepts
    2. Assigns each concept to a specific latent neuron based on pre-computed scores from JSON file
    3. Fine-tunes the SAE to maintain reconstruction while encouraging concept-specific latent assignments
    """
    def __init__(
        self,
        checkpoint_path,
        activations_dir,
        scores_json_path,
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
        self.concept_to_latent = {}
        self.scores_data = None

        # Initialize everything - MODIFIED SEQUENCE
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

    def load_scores_data(self):
        """
        Load the scores data from JSON file.
        """
        print(f"Loading scores data from {self.scores_json_path}")
        
        if not self.scores_json_path.exists():
            if self.from_scratch:
                print(f"⚠️  Scores file not found, but training from scratch - will use random assignment")
                self.scores_data = None
                return
            else:
                raise FileNotFoundError(f"Scores JSON file not found: {self.scores_json_path}")
        
        try:
            with open(self.scores_json_path, 'r') as f:
                self.scores_data = json.load(f)
            
            print(f"✅ Loaded scores data:")
            print(f"  Concept type: {self.scores_data.get('concept_type', 'unknown')}")
            print(f"  Number of timesteps: {self.scores_data.get('num_timesteps', 'unknown')}")
            print(f"  Number of concepts: {len(self.scores_data.get('scores', {}))}")
            
            # Print concept names
            concept_names = list(self.scores_data.get('scores', {}).keys())
            print(f"  Concepts: {concept_names[:5]}{'...' if len(concept_names) > 5 else ''}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading scores JSON file: {e}")

    def assign_concepts_to_latents_from_scores(self, hook_name):
        """
        Assign each concept to a specific latent using pre-computed scores from JSON file.
        If training from scratch, assigns concepts randomly to latents.
        """
        print(f"\nAssigning concepts to latents for {hook_name}...")

        # Get the SAE model to check number of latents
        sae = self.saes[hook_name]
        model = sae.module if hasattr(sae, 'module') else sae
        model_num_latents = model.num_latents

        print(f"SAE model has {model_num_latents} latents")

        # If training from scratch, assign concepts randomly
        if hasattr(self, 'from_scratch') and self.from_scratch:
            print("Training from scratch - assigning concepts randomly to latents...")

            # When training from scratch, we need to get concept names
            if self.scores_data is None:
                print("No scores data available - getting concept names from dataset...")

                # Check if we have data loaders available
                if not hasattr(self, 'train_loader') or self.train_loader is None:
                    print("⚠️  Train loader not available yet. Will assign concepts during training.")
                    # Return empty mapping for now - this will be populated during first training batch
                    return {}

                # Get concept names from the first batch of training data
                try:
                    sample_batch = next(iter(self.train_loader))
                    _, concepts = sample_batch
                    concept_names = list(set(concepts))
                    print(f"Found {len(concept_names)} unique concepts in dataset: {concept_names}")
                except Exception as e:
                    print(f"Could not get concepts from data loader: {e}")
                    print("Will assign concepts during training.")
                    return {}
            else:
                # Use concept names from scores data
                scores = self.scores_data.get('scores', {})
                concept_names = list(scores.keys())
                print(f"Found {len(concept_names)} concepts from scores data")

            # Create random assignment ensuring no duplicates
            import random
            available_latents = list(range(model_num_latents))
            random.shuffle(available_latents)

            concept_to_latent = {}

            for i, concept_name in enumerate(concept_names):
                if i < len(available_latents):
                    latent_idx = available_latents[i]
                    concept_to_latent[concept_name] = latent_idx
                    print(f"  Randomly assigned '{concept_name}' to latent {latent_idx}")
                else:
                    # If we have more concepts than latents, assign randomly with possible duplicates
                    latent_idx = random.randint(0, model_num_latents - 1)
                    concept_to_latent[concept_name] = latent_idx
                    print(f"  Randomly assigned '{concept_name}' to latent {latent_idx} (with possible duplicates)")

            print(f"\nCompleted random assignment: {len(concept_to_latent)} concepts assigned")
            print(f"Unique latents used: {len(set(concept_to_latent.values()))}")

            return concept_to_latent

        # Original logic for score-based assignment requires scores data
        if self.scores_data is None:
            raise RuntimeError("Scores data not loaded and not training from scratch.")

        # ... rest of existing score-based logic unchanged ...


        scores = self.scores_data.get('scores', {})
        if not scores:
            raise RuntimeError("No scores found in the JSON file.")

        num_timesteps = self.scores_data.get('num_timesteps', 100)
        print(f"Expected number of timesteps: {num_timesteps}")

        concept_to_latent = {}
        latent_assignments = set()

        print(f"Processing {len(scores)} concepts from scores data...")

        for concept_idx, (concept_name, concept_timestep_scores) in enumerate(scores.items()):
            try:
                print(f"  [{concept_idx+1}/{len(scores)}] Processing '{concept_name}'")

                if not concept_timestep_scores or not isinstance(concept_timestep_scores, list):
                    print(f"    Warning: No valid timestep scores for concept '{concept_name}', skipping")
                    continue
                
                print(f"    Found {len(concept_timestep_scores)} timesteps")

                # Validate number of timesteps
                if len(concept_timestep_scores) != num_timesteps:
                    print(f"    Warning: Expected {num_timesteps} timesteps, but found {len(concept_timestep_scores)}")

                # Convert to numpy array: shape should be [timesteps, neurons]
                timestep_scores = np.array(concept_timestep_scores)

                if len(timestep_scores.shape) == 1:
                    print(f"    ERROR: Expected 2D array [timesteps, neurons], but got 1D array of length {len(timestep_scores)}")
                    print(f"    This suggests the JSON structure might be different than expected")
                    continue
                elif len(timestep_scores.shape) != 2:
                    print(f"    ERROR: Expected 2D array [timesteps, neurons], but got shape {timestep_scores.shape}")
                    continue
                
                num_actual_timesteps, num_neurons = timestep_scores.shape
                print(f"    Timestep scores shape: [{num_actual_timesteps}, {num_neurons}]")

                # Check if number of neurons matches the model
                if num_neurons != model_num_latents:
                    print(f"    ERROR: Number of neurons in scores ({num_neurons}) doesn't match model latents ({model_num_latents})")
                    if num_neurons < model_num_latents:
                        print(f"    Padding neuron scores with zeros to match model size")
                        # Pad with very negative values so they won't be selected
                        padded_scores = np.full((num_actual_timesteps, model_num_latents), -float('inf'))
                        padded_scores[:, :num_neurons] = timestep_scores
                        timestep_scores = padded_scores
                        num_neurons = model_num_latents
                    else:
                        print(f"    Truncating neuron scores to match model size")
                        timestep_scores = timestep_scores[:, :model_num_latents]
                        num_neurons = model_num_latents

                    print(f"    Adjusted timestep scores shape: {timestep_scores.shape}")

                # Calculate average score across timesteps for each neuron
                average_scores = np.mean(timestep_scores, axis=0)  # Shape: [neurons]

                print(f"    Average scores shape: {average_scores.shape}")
                print(f"    Score range: [{average_scores.min():.6f}, {average_scores.max():.6f}]")

                # Find the neuron with the highest average score
                best_latent_idx = np.argmax(average_scores)

                # Validate the index
                if best_latent_idx >= len(average_scores) or best_latent_idx >= model_num_latents:
                    print(f"    ERROR: Invalid best_latent_idx {best_latent_idx} for array size {len(average_scores)} and model size {model_num_latents}")
                    continue
                
                best_score = average_scores[best_latent_idx]

                print(f"    Best neuron: {best_latent_idx} with average score: {best_score:.6f}")

                # Additional validation - ensure the index is within model bounds
                if best_latent_idx < 0 or best_latent_idx >= model_num_latents:
                    print(f"    ERROR: Best latent index {best_latent_idx} is out of model bounds [0, {model_num_latents})")
                    continue
                
                # Handle conflicts (multiple concepts assigned to same latent)
                original_idx = best_latent_idx
                original_score = best_score
                attempts = 0
                max_attempts = 10

                while best_latent_idx in latent_assignments and attempts < max_attempts:
                    # Mark this neuron as used and find the next best
                    average_scores[best_latent_idx] = -float('inf')
                    best_latent_idx = np.argmax(average_scores)

                    # Validate the new index
                    if best_latent_idx >= len(average_scores) or best_latent_idx >= model_num_latents:
                        print(f"    ERROR: Invalid conflict resolution index {best_latent_idx}")
                        break
                    
                    best_score = average_scores[best_latent_idx]
                    attempts += 1

                    print(f"    Conflict detected, trying neuron {best_latent_idx} with average score {best_score:.6f}")

                if attempts >= max_attempts:
                    print(f"    Warning: Could not resolve conflict for '{concept_name}' after {max_attempts} attempts")
                    print(f"    Using original assignment: neuron {original_idx}")
                    best_latent_idx = original_idx
                    best_score = original_score

                # Final validation before assignment
                if best_latent_idx < 0 or best_latent_idx >= model_num_latents:
                    print(f"    ERROR: Final index {best_latent_idx} is invalid for model with {model_num_latents} latents, skipping concept")
                    continue
                
                # Assign the concept to the selected latent
                concept_to_latent[concept_name] = best_latent_idx
                latent_assignments.add(best_latent_idx)

                print(f"    ✅ Assigned '{concept_name}' to latent {best_latent_idx} (avg score: {best_score:.6f})")

            except Exception as e:
                print(f"    Error processing concept '{concept_name}': {e}")
                import traceback
                traceback.print_exc()
                continue
            
        print(f"\nCompleted assignment: {len(concept_to_latent)} concepts assigned")
        print(f"Unique latents used: {len(latent_assignments)}")
        print(f"Model has {model_num_latents} total latents")

        # Validate all final assignments
        invalid_assignments = []
        for concept, latent_idx in concept_to_latent.items():
            if latent_idx < 0 or latent_idx >= model_num_latents:
                invalid_assignments.append((concept, latent_idx))

        if invalid_assignments:
            print(f"\nWARNING: Found {len(invalid_assignments)} invalid assignments:")
            for concept, latent_idx in invalid_assignments:
                print(f"  {concept} -> {latent_idx} (invalid for model with {model_num_latents} latents)")

            # Remove invalid assignments
            for concept, _ in invalid_assignments:
                del concept_to_latent[concept]

            print(f"Removed invalid assignments. Final count: {len(concept_to_latent)} concepts")

        # Print summary of assignments
        if concept_to_latent:
            print(f"\nAssignment summary:")
            sorted_assignments = sorted(concept_to_latent.items(), key=lambda x: x[1])
            for concept, latent_idx in sorted_assignments[:10]:  # Show first 10
                # Calculate the average score for display
                avg_score = "N/A"
                if concept in scores and len(scores[concept]) > 0:
                    try:
                        concept_timestep_scores = np.array(scores[concept])
                        if len(concept_timestep_scores.shape) == 2 and latent_idx < concept_timestep_scores.shape[1]:
                            avg_score = f"{np.mean(concept_timestep_scores[:, latent_idx]):.6f}"
                    except:
                        pass

                print(f"  {concept:20} -> Latent {latent_idx:4} (avg score: {avg_score})")

            if len(sorted_assignments) > 10:
                print(f"  ... and {len(sorted_assignments) - 10} more")

        return concept_to_latent

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

    def initialize_datasets(self):
        """Dataset initialization with numpy shuffling (your working approach)."""
        print("Initializing datasets with numpy shuffling...")

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

        self._create_simple_data_loaders(dataset_dict)
        print("\n✅ Dataset initialization completed!")

    def print_epoch_summary(self, epoch, hook_name, train_losses, val_losses, 
                        train_distributions, val_distributions, concept_to_latent):
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

        # Calculate success rates
        train_correct = sum(1 for concept, stats in train_distributions.items() 
                           if concept_to_latent.get(concept) == stats["dominant_latent"])
        train_total = len(train_distributions)
        train_success_rate = (train_correct / train_total * 100) if train_total > 0 else 0

        val_correct = sum(1 for concept, stats in val_distributions.items() 
                         if concept_to_latent.get(concept) == stats["dominant_latent"])
        val_total = len(val_distributions)
        val_success_rate = (val_correct / val_total * 100) if val_total > 0 else 0

        print(f"Training:   {train_correct:>2}/{train_total:<2} concepts correct ({train_success_rate:>6.1f}%)")
        print(f"Validation: {val_correct:>2}/{val_total:<2} concepts correct ({val_success_rate:>6.1f}%)")

        # 3. DETAILED CONCEPT TABLE
        print(f"\n📋 CONCEPT ASSIGNMENT DETAILS:")
        print(f"{'Concept':<15} {'Assigned':<8} {'Train Dom.':<10} {'Train Score':<11} {'Val Dom.':<9} {'Val Score':<10} {'Status':<8}")
        print(f"-" * 82)

        # Get all concepts
        all_concepts = set(train_distributions.keys()) | set(val_distributions.keys())

        for concept in sorted(all_concepts):
            assigned_latent = concept_to_latent.get(concept, -1)

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

            print(f"{concept:<15} {assigned_latent:<8} {train_dominant:<10} {train_score:<11.4f} {val_dominant:<9} {val_score:<10.4f} {status:<8}")

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


    def print_initial_concept_assignments(self, concept_to_latent, hook_name):
        """
        Print the initial concept-to-latent assignments clearly.
        """
        print(f"\n" + "="*60)
        print(f"INITIAL CONCEPT ASSIGNMENTS - {hook_name}")
        print(f"="*60)
        print(f"{'Concept':<20} {'Assigned Latent':<15} {'Score':<15}")
        print(f"-" * 50)

        # Get scores for display
        scores = self.scores_data.get('scores', {}) if self.scores_data else {}

        for concept, latent_idx in sorted(concept_to_latent.items()):
            # Get the original score for this assignment
            score = "N/A"
            if not self.from_scratch and concept in scores and latent_idx < len(scores[concept]):
                score = f"{scores[concept][latent_idx]:.6f}"
            elif self.from_scratch:
                score = "Random"

            print(f"{concept:<20} {latent_idx:<15} {score:<15}")

        print(f"\nTotal concepts: {len(concept_to_latent)}")
        print(f"="*60 + "\n")


    # REPLACE the existing print_latent_distribution_summary method with this simplified version:
    def print_latent_distribution_summary(self, distributions, concept_to_latent, epoch=None, is_validation=False):
        """
        Simplified version - the detailed output is now in print_epoch_summary.
        """
        # This method can now be much simpler since we have the detailed print_epoch_summary
        dataset_type = "Validation" if is_validation else "Training"

        total_concepts = len(distributions)
        correct_concepts = sum(1 for concept, stats in distributions.items() 
                              if concept_to_latent.get(concept) == stats["dominant_latent"])
        success_rate = correct_concepts / total_concepts if total_concepts > 0 else 0

        print(f"{dataset_type} concept assignment: {correct_concepts}/{total_concepts} ({success_rate:.1%})")

    
    def _create_simple_data_loaders(self, dataset_dict):
        """DataLoader creation with proper distributed sampler handling."""
        hookpoint, dataset = next(iter(dataset_dict.items()))

        total_size = len(dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size

        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))

        def simple_collate_fn(batch):
            activations = torch.stack([item['activations'] for item in batch])
            concepts = [item['object_label'] for item in batch]
            return activations, concepts

        # Handle distributed training properly
        if self.world_size > 1:
            # Use DistributedSampler for multi-GPU training
            train_sampler = DistributedSampler(
                train_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=True,
                seed=self.seed
            )
            val_sampler = DistributedSampler(
                val_dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False,
                seed=self.seed
            )

            # When using DistributedSampler, don't use shuffle in DataLoader
            train_shuffle = False
            val_shuffle = False
        else:
            # Single GPU - use regular sampler
            train_sampler = None
            val_sampler = None
            train_shuffle = True
            val_shuffle = False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=val_shuffle,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=simple_collate_fn
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def get_latent_distribution_statistics(self, sae, data_loader, concept_to_latent):
        """Fixed statistics calculation with proper bounds checking."""
        model = sae.module if hasattr(sae, 'module') else sae
        model.eval()
        
        distributions = {}
        concept_probs = {}
        
        print("Calculating latent distribution statistics...")
        
        with torch.no_grad():
            for batch_idx, (activations, concepts) in enumerate(data_loader):
                if batch_idx >= 3:  # Very limited for efficiency
                    break
                    
                try:
                    activations = activations.to(self.device, dtype=self.dtype)
                    
                    # Handle reshaping
                    if len(activations.shape) == 3:
                        original_shape = activations.shape
                        activations = activations.reshape(-1, activations.shape[-1])
                    
                    pre_acts = model.pre_acts(activations)
                    
                    # Reshape back if needed
                    if len(original_shape) == 3:
                        batch_size = len(concepts)
                        seq_len = original_shape[1]
                        pre_acts = pre_acts.reshape(batch_size, seq_len, -1)
                        pre_acts = pre_acts.mean(dim=1)
                    
                    # CRITICAL: Check dimensions
                    if pre_acts.shape[1] != model.num_latents:
                        print(f"  Skipping batch - dimension mismatch: {pre_acts.shape[1]} vs {model.num_latents}")
                        continue
                    
                    probs = F.softmax(pre_acts, dim=1)
                    
                    for i, concept in enumerate(concepts):
                        if concept not in concept_probs:
                            concept_probs[concept] = []
                        concept_probs[concept].append(probs[i])
                        
                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    continue
                
        # Calculate statistics for each concept
        for concept, prob_list in concept_probs.items():
            if prob_list and concept in concept_to_latent:
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

    def check_model_gradients(self, sae, hook_name):
        """Check if model parameters still require gradients."""
        model = sae.module if hasattr(sae, 'module') else sae

        params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())

        if params_requiring_grad == 0:
            print(f"CRITICAL ERROR: No parameters require gradients in {hook_name}!")
            # Force all parameters to require gradients
            for param in model.parameters():
                param.requires_grad_(True)
            return False
        elif params_requiring_grad != total_params:
            print(f"WARNING: Only {params_requiring_grad}/{total_params} parameters require gradients in {hook_name}")
            # Force all parameters to require gradients
            for param in model.parameters():
                param.requires_grad_(True)
            return False

        return True

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
                    
                    # If from_scratch is True, only try to resume if we're explicitly resuming training
                    # (not loading the base model)
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

        for hook_name, sae in self.saes.items():
            model = sae.module if hasattr(sae, 'module') else sae

            # CRITICAL FIX: Ensure all parameters require gradients
            params_requiring_grad = 0
            params_total = 0

            for name, param in model.named_parameters():
                params_total += 1
                if param.requires_grad:
                    params_requiring_grad += 1
                else:
                    print(f"WARNING: Parameter {name} does not require gradients! Fixing...")
                    param.requires_grad_(True)

            print(f"SAE {hook_name}: {params_requiring_grad}/{params_total} parameters require gradients")

            # Additional check: ensure model is in training mode
            model.train()

            # Verify the model can compute gradients
            try:
                # Test forward pass with dummy data
                dummy_input = torch.randn(2, model.d_in, device=self.device, dtype=self.dtype, requires_grad=True)
                dummy_pre_acts = model.pre_acts(dummy_input)
                dummy_loss = dummy_pre_acts.sum()
                dummy_loss.backward()
                print(f"✅ Gradient test passed for {hook_name}")
            except Exception as e:
                print(f"❌ Gradient test failed for {hook_name}: {e}")
                # Force all parameters to require gradients
                for param in model.parameters():
                    param.requires_grad_(True)
            finally:
                # Clear gradients from test
                model.zero_grad()
    
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
                        run_name = f"sae_concept_latent_optimization_resumed_{run_id}"
                    except Exception as e:
                        print(f"Could not load existing run ID: {e}")
            
            # Create a new run name if not resuming or couldn't find existing
            if run_name is None:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"sae_concept_latent_optimization_{timestamp}"
            
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
            if hasattr(self, 'concept_to_latent'):
                for hook_name, concept_mapping in self.concept_to_latent.items():
                    concept_assignment_info[hook_name] = {
                        "num_concepts": len(concept_mapping),
                        "concept_names": list(concept_mapping.keys()),
                        "assigned_latents": list(concept_mapping.values()),
                        "unique_latents_used": len(set(concept_mapping.values())),
                    }
            
            # Add scores data info
            scores_info = {}
            if self.scores_data:
                scores_info = {
                    "concept_type": self.scores_data.get('concept_type', 'unknown'),
                    "num_timesteps": self.scores_data.get('num_timesteps', 'unknown'),
                    "num_concepts_in_scores": len(self.scores_data.get('scores', {})),
                    "scores_json_path": str(self.scores_json_path),
                }
            
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
                "start_epoch": self.start_epoch,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "mixed_precision": self.mixed_precision,
                "use_float16": self.use_float16,
                
                # Data paths
                "checkpoint_path": str(self.checkpoint_path),
                "activations_dir": str(self.activations_dir),
                "scores_json_path": str(self.scores_json_path),
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
                        project="sae_concept_latent_optimizer",
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
                        project="sae_concept_latent_optimizer",
                        name=run_name,
                        config=config,
                        dir=wandb_dir
                    )
                    run_id = wandb.run.id
            else:
                wandb.init(
                    project="sae_concept_latent_optimizer",
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
            
            print(f"Initialized wandb logging in OFFLINE mode")
            print(f"Logs will be stored in: {wandb_dir}")
            print(f"SAE configurations logged for {len(sae_configs)} hooks: {list(sae_configs.keys())}")
            
            # Log a summary table of SAE configurations
            if len(sae_configs) > 0:
                print("\nSAE Configuration Summary:")
                print(f"{'Hook':<20} {'Latents':<8} {'d_in':<6} {'Expansion':<10} {'TopK':<6}")
                print("-" * 50)
                for hook_name, config in sae_configs.items():
                    # Safely handle None values
                    num_latents = config.get('num_latents')
                    d_in = config.get('d_in')
                    k_value = config.get('k')
                    expansion = config.get('expansion_factor')
                    
                    # Convert None to 'N/A' string
                    num_latents_str = str(num_latents) if num_latents is not None else 'N/A'
                    d_in_str = str(d_in) if d_in is not None else 'N/A'
                    k_str = str(k_value) if k_value is not None else 'N/A'
                    
                    # Calculate expansion if not available but num_latents and d_in are
                    if expansion is None and num_latents is not None and d_in is not None and d_in != 0:
                        expansion_str = f"{num_latents / d_in:.1f}"
                    elif expansion is not None:
                        expansion_str = str(expansion)
                    else:
                        expansion_str = 'N/A'
                    
                    print(f"{hook_name:<20} {num_latents_str:<8} {d_in_str:<6} {expansion_str:<10} {k_str:<6}")

    def compute_reconstruction_loss(self, sae, activations):
        """Fixed reconstruction loss that handles tensor dimensions correctly and maintains gradients."""
        # Get the actual model
        model = sae.module if hasattr(sae, 'module') else sae
    
        # Ensure activations are 2D [batch_size, features]
        if len(activations.shape) == 3:
            # If 3D [batch, time, features], take mean over time or flatten
            batch_size, time_steps, features = activations.shape
            activations = activations.reshape(batch_size * time_steps, features)
        elif len(activations.shape) == 4:
            # If 4D [batch, time, height, width], flatten spatial dimensions
            batch_size, time_steps, height, width = activations.shape
            activations = activations.reshape(batch_size * time_steps, height * width)
    
        if len(activations.shape) != 2:
            print(f"  Recon Error: Cannot handle activations shape: {activations.shape}")
            # Return a zero loss that maintains gradients
            dummy_loss = (activations.sum() * 0.0).mean()
            dummy_pre_acts = torch.zeros(activations.shape[0], 1000, device=activations.device, dtype=activations.dtype, requires_grad=True)
            return dummy_loss, dummy_pre_acts
    
        try:
            # Get pre-activations
            pre_acts = model.pre_acts(activations)
    
            # Check for NaN
            if torch.isnan(pre_acts).any():
                print(f"  Recon Warning: NaN in pre_acts")
                # Return a zero loss that maintains gradients
                dummy_loss = (pre_acts.sum() * 0.0).mean()
                return dummy_loss, pre_acts
    
            # Get top-k activations
            top_acts, top_indices = model.select_topk(pre_acts)
    
            # Decode and compute reconstruction loss
            reconstructed = model.decode(top_acts, top_indices)
    
            # Ensure shapes match for loss computation
            if reconstructed.shape != activations.shape:
                print(f"  Recon Error: Shape mismatch - reconstructed: {reconstructed.shape}, original: {activations.shape}")
                # Return a zero loss that maintains gradients
                dummy_loss = (pre_acts.sum() * 0.0).mean()
                return dummy_loss, pre_acts
    
            loss = F.mse_loss(reconstructed, activations)
    
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Recon Warning: NaN/Inf in loss")
                # Return a zero loss that maintains gradients
                dummy_loss = (pre_acts.sum() * 0.0).mean()
                return dummy_loss, pre_acts
    
            print(f"  Recon Loss: {loss.item():.6f}")
            return loss, pre_acts
    
        except Exception as e:
            print(f"  Recon Error: {e}")
            print(f"    Input shape: {activations.shape}")
            # Return a zero loss that maintains gradients
            dummy_loss = (activations.sum() * 0.0).mean()
            dummy_pre_acts = torch.zeros(activations.shape[0], 1000, device=activations.device, dtype=activations.dtype, requires_grad=True)
            return dummy_loss, dummy_pre_acts
    
    
    def compute_binary_cross_entropy_loss(self, pre_acts, concepts, concept_to_latent):
        """
        Compute binary cross-entropy loss BEFORE topk operation.
        Only applies loss to assigned latents - unassigned latents are not influenced.
        Fixed to always maintain gradients.
        """
        # Handle the case where pre_acts were reshaped from [batch, seq, features] to [batch*seq, features]
        if len(pre_acts.shape) == 2:
            batch_times_seq, num_latents = pre_acts.shape
            batch_size = len(concepts)
    
            # Check if we need to reshape back
            if batch_times_seq != batch_size:
                # Calculate sequence length
                seq_length = batch_times_seq // batch_size
                if batch_times_seq == batch_size * seq_length:
                    # Reshape back to [batch, seq, latents]
                    pre_acts = pre_acts.view(batch_size, seq_length, num_latents)
                    # Take mean over sequence dimension
                    pre_acts = pre_acts.mean(dim=1)  # [batch, latents]
                else:
                    print(f"  BCE Loss Error: Cannot reshape {batch_times_seq} to match batch size {batch_size}")
                    # Return a zero tensor that maintains gradients
                    return (pre_acts.sum() * 0.0).mean()
    
        elif len(pre_acts.shape) == 3:
            # If 3D [batch, seq, latents], take mean over sequence
            pre_acts = pre_acts.mean(dim=1)
    
        if len(pre_acts.shape) != 2:
            print(f"  BCE Loss Error: Unexpected pre_acts shape: {pre_acts.shape}")
            # Return a zero tensor that maintains gradients
            return (pre_acts.sum() * 0.0).mean()
    
        batch_size, num_latents = pre_acts.shape
    
        # Ensure batch size matches concepts
        if batch_size != len(concepts):
            print(f"  BCE Loss Error: Batch size mismatch - pre_acts: {batch_size}, concepts: {len(concepts)}")
            # Return a zero tensor that maintains gradients
            return (pre_acts.sum() * 0.0).mean()
    
        # Create binary targets and mask for assigned latents only
        # Shape: [batch_size, num_latents]
        targets = torch.zeros(batch_size, num_latents, device=pre_acts.device, dtype=pre_acts.dtype)
        loss_mask = torch.zeros(batch_size, num_latents, device=pre_acts.device, dtype=torch.bool)
        
        valid_samples = 0
        
        for i, concept in enumerate(concepts):
            target_latent = concept_to_latent.get(concept)
            if target_latent is not None and 0 <= target_latent < num_latents:
                # Set target to 1 for the assigned latent
                targets[i, target_latent] = 1.0
                # Enable loss computation for this latent
                loss_mask[i, target_latent] = True
                valid_samples += 1
    
        if valid_samples == 0:
            print(f"  BCE Loss: No valid concept assignments found")
            # Return a zero tensor that maintains gradients
            return (pre_acts.sum() * 0.0).mean()
    
        # Use binary cross entropy with logits directly (more stable)
        bce_loss = F.binary_cross_entropy_with_logits(pre_acts, targets, reduction='none')
        
        # Apply mask to only compute loss on assigned latents
        masked_loss = bce_loss * loss_mask.float()
        
        # Average over valid (masked) positions
        total_loss = masked_loss.sum()
        total_valid_positions = loss_mask.sum().float()
        
        if total_valid_positions > 0:
            final_loss = total_loss / total_valid_positions
            print(f"  BCE Loss: {final_loss.item():.6f} (from {total_valid_positions.item():.0f} assigned latents)")
            return final_loss
        else:
            # Return a zero tensor that maintains gradients
            return (pre_acts.sum() * 0.0).mean()
    
    
    def compute_sparsity_loss(self, pre_acts):
        """
        Compute L1 sparsity regularization on pre-activations.
        Fixed to always maintain gradients.
        """
        # Check for NaN values
        if torch.isnan(pre_acts).any():
            print(f"  Sparsity Warning: NaN in pre_acts")
            # Return a zero tensor that maintains gradients  
            return (pre_acts.sum() * 0.0).mean()
    
        # Clip extremely large values to prevent overflow
        clipped_pre_acts = torch.clamp(pre_acts, -100, 100)
    
        # Use a more stable formulation
        sparsity = torch.mean(torch.abs(clipped_pre_acts))
    
        # Prevent NaN return
        if torch.isnan(sparsity) or torch.isinf(sparsity):
            print(f"  Sparsity Warning: NaN/Inf in sparsity loss")
            # Return a zero tensor that maintains gradients
            return (pre_acts.sum() * 0.0).mean()
    
        return sparsity
    
    def evaluate_losses(self, sae, hook_name, concept_to_latent, is_validation=False):
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
            for batch_idx, (activations, concepts) in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                
                # Print progress
                print(f"  Processing batch {batch_idx + 1}/{max_batches}...")

                try:
                    activations = activations.to(self.device, dtype=self.dtype)

                    # Compute losses
                    recon_loss, pre_acts = self.compute_reconstruction_loss(sae, activations)
                    ce_loss = self.compute_binary_cross_entropy_loss(pre_acts, concepts, concept_to_latent)
                    sparsity_loss = self.compute_sparsity_loss(pre_acts)

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
        Enhanced with resume functionality and early stopping.
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
                self.concept_to_latent[hook_name] = self.assign_concepts_to_latents_from_scores(hook_name)
                self.print_initial_concept_assignments(self.concept_to_latent[hook_name], hook_name)

                if self.world_size > 1:
                    # Broadcast concept_to_latent to all processes
                    object_list = [self.concept_to_latent[hook_name]]
                    dist.broadcast_object_list(object_list, src=0)
            else:
                # Other ranks receive the concept_to_latent mapping
                object_list = [None]
                dist.broadcast_object_list(object_list, src=0)
                self.concept_to_latent[hook_name] = object_list[0]
            
            # Make sure all processes have the mapping before continuing
            if self.world_size > 1:
                dist.barrier()
            
            # Compute initial losses (only on rank 0 for logging purposes)
            if self.rank == 0 and self.start_epoch == 1:
                train_losses = self.evaluate_losses(sae, hook_name, self.concept_to_latent[hook_name], is_validation=False)
                val_losses = self.evaluate_losses(sae, hook_name, self.concept_to_latent[hook_name], is_validation=True)
                
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
                
                # Get the concept-to-latent mapping for this hook
                concept_to_latent = self.concept_to_latent[hook_name]
                
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
                for batch_idx, batch_data in enumerate(data_iter):
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    activations, concept_labels = batch_data
                    activations = activations.to(self.device)

                    original_batch_size = activations.size(0)

                    # Mixed precision training
                    if self.mixed_precision and torch.cuda.is_available() and not self.use_float16:
                        with torch.amp.autocast('cuda'):
                            recon_loss, pre_acts = self.compute_reconstruction_loss(sae, activations)

                            # 2. Binary Cross-entropy loss for concept-specific latent (BEFORE topk)
                            ce_loss = self.compute_binary_cross_entropy_loss(
                                pre_acts, 
                                concept_labels, 
                                concept_to_latent
                            )

                            # 3. Sparsity loss
                            sparsity_loss = self.compute_sparsity_loss(pre_acts)

                            # Combined loss
                            total_loss = (
                                self.reconstruction_weight * recon_loss +
                                self.cross_entropy_weight * ce_loss +
                                self.sparsity_weight * sparsity_loss
                            )
                        
                        # Optimization step with mixed precision
                        optimizer.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        if not self.check_model_gradients(sae, hook_name):
                            print(f"Fixed gradient requirements for {hook_name}")
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                            self.scaler.step(optimizer)
                            self.scaler.update()
                            optimizer.zero_grad(set_to_none=True)  # More memory efficient
                    else:
                        # Standard precision training
                        recon_loss, pre_acts = self.compute_reconstruction_loss(sae, activations)

                        # 2. Binary Cross-entropy loss for concept-specific latent (BEFORE topk)
                        ce_loss = self.compute_binary_cross_entropy_loss(
                            pre_acts, 
                            concept_labels, 
                            concept_to_latent
                        )

                        # 3. Sparsity loss
                        sparsity_loss = self.compute_sparsity_loss(pre_acts)

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

                        # Gradient clipping to prevent explosion - ADD THIS
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

                    if 'pre_acts' in locals():
                        del pre_acts

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
                    print(f"  BCE Loss: {avg_ce_loss:.6f}")
                    print(f"  Sparsity Loss: {avg_sparsity_loss:.6f}")
                    
                    # Evaluate on validation set
                    train_losses = self.evaluate_losses(sae, hook_name, concept_to_latent, is_validation=False)
                    val_losses = self.evaluate_losses(sae, hook_name, concept_to_latent, is_validation=True)
                    
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
                        self.train_loader,  # Always use train_loader
                        concept_to_latent
                    )
                    val_distributions = self.get_latent_distribution_statistics(
                        sae if not isinstance(sae, DDP) else sae.module,
                        self.val_loader,    # Always use val_loader
                        concept_to_latent
                    )
                    
                    # Print distribution summaries
                    self.print_latent_distribution_summary(
                        train_distributions, 
                        concept_to_latent, 
                        epoch=epoch, 
                        is_validation=False
                    )
                    self.print_latent_distribution_summary(
                        val_distributions, 
                        concept_to_latent, 
                        epoch=epoch, 
                        is_validation=True
                    )
                    
                    # Log metrics to wandb with incremental step
                    if WANDB_AVAILABLE:
                        # Calculate actual step for incremental logging
                        step = epoch if not self.resume else (self.start_epoch - 1) + (epoch - self.start_epoch + 1)
                        
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
                                           if concept_to_latent.get(c) == s["dominant_latent"])
                        train_success_rate = train_success / len(train_distributions) if train_distributions else 0
    
                        val_success = sum(1 for c, s in val_distributions.items() 
                                         if concept_to_latent.get(c) == s["dominant_latent"])
                        val_success_rate = val_success / len(val_distributions) if val_distributions else 0
    
                        metrics.update({
                            f"{hook_name}/train/concept_success_rate": train_success_rate,
                            f"{hook_name}/val/concept_success_rate": val_success_rate,
                        })
                        
                        wandb.log(metrics, step=step)
                    
                    
                    # Print comprehensive epoch summary
                    self.print_epoch_summary(
                        epoch, hook_name, train_losses, val_losses,
                        train_distributions, val_distributions, concept_to_latent
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
        from_scratch=args.from_scratch,
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
    Main entry point for the SAE Concept Latent Optimizer.
    """
    parser = argparse.ArgumentParser(description="Optimize SAE models to assign specific latents to concepts.")
    
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
        help="Path to the concept activations dictionary pickle file"
    )
    parser.add_argument(
        "--scores_json_path", 
        type=str, 
        required=True, 
        help="Path to the JSON file containing pre-computed concept scores"
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