#!/usr/bin/env python
"""
Optimize SAE models to assign specific latents to nudity vs non-nudity concepts.

This script processes activations from nudity and non-nudity samples, assigns specific 
latent neurons to each concept, and finetunes the SAE to maintain this assignment.
"""
import os
import sys
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from SAE.sae import Sae, SaeConfig
import torch
import numpy as np
from pathlib import Path
import random
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm
from datasets import Dataset as HFDataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_nudity_datasets(base_dirs, hookpoint, dtype=torch.float32):
    """Load datasets from nudity and non_nudity directories.
    
    This function ALWAYS overwrites the nudity_label column based on the
    directory name, ignoring any pre-existing labels in the saved datasets.
    """
    datasets = []
    print(f"Loading nudity datasets from {base_dirs} for hookpoint {hookpoint}")

    for base_dir in base_dirs:
        base_path = Path(base_dir)
        hookpoint_dir = base_path / hookpoint
        
        if not hookpoint_dir.exists():
            print(f"❌ Hookpoint directory does not exist: {hookpoint_dir}")
            continue
        
        # Look for 'nudity' and 'non_nudity' subdirectories
        for category in ['nudity', 'non_nudity']:
            category_dir = hookpoint_dir / category
            
            if not category_dir.exists():
                print(f"⚠️  Category directory not found: {category_dir}")
                continue
                
            if (category_dir / "dataset_info.json").exists():
                print(f"  Loading '{category}' dataset...")
                
                dataset = HFDataset.load_from_disk(str(category_dir), keep_in_memory=False)
                print(f"    Loaded {len(dataset)} samples")
                
                # CRITICAL FIX: Always remove and recreate the label column
                # The saved datasets have incorrect labels, so we ignore them
                # and assign labels based on directory name
                if "nudity_label" in dataset.column_names:
                    print(f"    ⚠️  Found existing nudity_label column - removing it")
                    dataset = dataset.remove_columns(["nudity_label"])
                
                print(f"    ✅ Adding nudity_label='{category}' to all {len(dataset)} samples")
                dataset = dataset.add_column("nudity_label", [category] * len(dataset))
                
                # Verify the labels were added correctly
                sample_labels = dataset["nudity_label"][:min(5, len(dataset))]
                unique_sample_labels = set(sample_labels)
                print(f"    Verification - unique labels in first 5 samples: {unique_sample_labels}")
                
                if len(unique_sample_labels) != 1 or category not in unique_sample_labels:
                    print(f"    ❌ ERROR: Label verification failed!")
                else:
                    print(f"    ✅ Label verification passed")
                
                datasets.append(dataset)
            else:
                print(f"  ❌ No dataset_info.json found in {category_dir}")

    if not datasets:
        raise ValueError(f"No valid datasets found for hookpoint {hookpoint}")

    print(f"\n📦 Concatenating {len(datasets)} datasets...")
    final_dataset = concatenate_datasets(datasets)
    print(f"✅ Final combined dataset: {len(final_dataset)} samples")
    
    # Print distribution with detailed debugging
    print(f"\n📊 Label Distribution:")
    all_labels = final_dataset["nudity_label"]
    unique_labels = set(all_labels)
    print(f"   Unique labels found: {unique_labels}")
    
    nudity_count = sum(1 for label in all_labels if label == "nudity")
    non_nudity_count = sum(1 for label in all_labels if label == "non_nudity")
    other_count = len(all_labels) - nudity_count - non_nudity_count
    
    print(f"   ✅ 'nudity': {nudity_count:,} samples ({100*nudity_count/len(all_labels):.2f}%)")
    print(f"   ✅ 'non_nudity': {non_nudity_count:,} samples ({100*non_nudity_count/len(all_labels):.2f}%)")
    
    if other_count > 0:
        print(f"   ⚠️  Other/Unknown: {other_count} samples")
    
    # Critical sanity checks
    if nudity_count == 0:
        raise ValueError("❌ FATAL: No nudity samples found! Cannot train binary classifier.")
    if non_nudity_count == 0:
        raise ValueError("❌ FATAL: No non-nudity samples found! Cannot train binary classifier.")
    
    # Check class imbalance
    imbalance_ratio = max(nudity_count, non_nudity_count) / min(nudity_count, non_nudity_count)
    if imbalance_ratio > 100:
        print(f"   ⚠️  WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        print(f"   Consider using class weights or sampling strategies")
    
    return final_dataset

class SAENudityOptimizer:
    """Optimizer for SAE models to distinguish nudity from non-nudity content."""
    
    def __init__(
        self,
        checkpoint_path,
        activations_dir,
        scores_json_path=None,
        device="cuda",
        learning_rate=5e-6,
        num_epochs=5,
        reconstruction_weight=1.0,
        cross_entropy_weight=1.0,
        sparsity_weight=0.01,
        batch_size=32,
        save_dir="sae-nudity-optimized",
        seed=42,
        validation_split=0.2,
        mixed_precision=False,
        world_size=1,
        rank=0,
        gradient_accumulation_steps=1,
        use_float16=False,
        patience=5,
        resume=False,
        from_scratch=False,
        pos_class_weight=1.0,
        max_val_batches=20
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.activations_dir = Path(activations_dir)
        self.scores_json_path = Path(scores_json_path) if scores_json_path else None
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
        self.mixed_precision = mixed_precision
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_float16 = use_float16
        self.dtype = torch.float16 if use_float16 else torch.float32
        self.patience = patience
        self.resume = resume
        self.from_scratch = from_scratch
        self.pos_class_weight = pos_class_weight
        self.max_val_batches = max_val_batches

        # Early stopping variables
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 1

        if use_float16:
            self.mixed_precision = False
            self.scaler = None
        else:
            self.mixed_precision = mixed_precision
            self.scaler = torch.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None

        # Set random seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.saes = {}
        self.optimizers = {}
        self.concept_to_latent = {}  # Maps 'nudity' and 'non_nudity' to latents
        self.scores_data = None

        # Initialize
        if self.scores_json_path and self.scores_json_path.exists():
            self.load_scores_data()
        elif not self.from_scratch:
            print("⚠️  No scores file provided - will use random assignment")
            
        self.initialize_saes()
        self.initialize_datasets()
        self.initialize_wandb()

    def load_scores_data(self):
        """Load pre-computed scores from JSON file."""
        print(f"Loading scores from {self.scores_json_path}")
        with open(self.scores_json_path, 'r') as f:
            self.scores_data = json.load(f)
        print(f"✅ Loaded scores for {len(self.scores_data.get('scores', {}))} concepts")

    def initialize_datasets(self):
        """Initialize datasets with nudity labels - OPTIMIZED."""
        print("Initializing nudity datasets...")

        hookpoint_names = list(self.saes.keys())
        dataset_dict = {}

        # LOAD ONLY ONCE - all ranks load the same data
        for hookpoint in hookpoint_names:
            dataset = load_nudity_datasets(
                [str(self.activations_dir)], 
                hookpoint,
                dtype=self.dtype
            )

            # Shuffle with consistent seed across all ranks
            print(f"Shuffling {len(dataset)} samples...")
            indices = np.arange(len(dataset))
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            dataset = dataset.select(indices)

            dataset_dict[hookpoint] = dataset
            print(f"Completed loading for {hookpoint}: {len(dataset)} samples")

        # DDP synchronization - just wait for all to finish loading
        if self.world_size > 1:
            dist.barrier()

        self._create_data_loaders(dataset_dict)
        print("\n✅ Dataset initialization completed!")


    def _create_data_loaders(self, dataset_dict):
        """Create data loaders with nudity labels - OPTIMIZED."""
        hookpoint, dataset = next(iter(dataset_dict.items()))

        total_size = len(dataset)
        val_size = int(total_size * self.validation_split)
        train_size = total_size - val_size

        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, total_size))

        def nudity_collate_fn(batch):
            """Collate function for nudity labels - OPTIMIZED."""
            # Extract all activations first
            activations_list = [item['activations'] for item in batch]

            # Convert to tensor in one operation if needed
            if not isinstance(activations_list[0], torch.Tensor):
                activations = torch.tensor(np.array(activations_list), dtype=self.dtype)
            else:
                activations = torch.stack(activations_list)

            nudity_labels = [item['nudity_label'] for item in batch]
            return activations, nudity_labels

        # INCREASED num_workers for better I/O performance
        import torch.multiprocessing as mp
        num_workers = min(8, max(2, mp.cpu_count() // self.world_size))
        # num_workers = 16

        if self.world_size > 1:
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
            train_shuffle = False
            val_shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            train_shuffle = True
            val_shuffle = False

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=train_shuffle, 
            sampler=train_sampler,
            num_workers=num_workers,  # INCREASED from 2
            pin_memory=True, 
            collate_fn=nudity_collate_fn,
            prefetch_factor=4,  # ADDED prefetching
            persistent_workers=True if num_workers > 0 else False  # ADDED to keep workers alive
        )

        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=val_shuffle, 
            sampler=val_sampler,
            num_workers=num_workers,  # INCREASED from 2
            pin_memory=True, 
            collate_fn=nudity_collate_fn,
            prefetch_factor=4,  # ADDED prefetching
            persistent_workers=True if num_workers > 0 else False  # ADDED to keep workers alive
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def assign_concepts_to_latents(self, hook_name):
        """Assign nudity and non_nudity to specific latents."""
        print(f"\nAssigning nudity concepts to latents for {hook_name}...")

        sae = self.saes[hook_name]
        model = sae.module if hasattr(sae, 'module') else sae
        model_num_latents = model.num_latents

        concept_to_latent = {}

        # Random assignment or score-based
        if self.from_scratch or self.scores_data is None:
            print("Using random assignment...")
            # Assign two random latents
            available_latents = list(range(model_num_latents))
            random.shuffle(available_latents)
            
            concept_to_latent['nudity'] = available_latents[0]
            concept_to_latent['non_nudity'] = available_latents[1]
            
            print(f"  'nudity' → latent {available_latents[0]}")
            print(f"  'non_nudity' → latent {available_latents[1]}")
        else:
            print("Using score-based assignment...")
            scores = self.scores_data.get('scores', {})
            
            # Assign based on highest scores
            for concept in ['nudity', 'non_nudity']:
                if concept in scores:
                    concept_scores = scores[concept]
                    if isinstance(concept_scores[0], list):
                        avg_scores = np.mean(concept_scores, axis=0)
                    else:
                        avg_scores = concept_scores
                    
                    best_latent = np.argmax(avg_scores)
                    concept_to_latent[concept] = int(best_latent)
                    print(f"  '{concept}' → latent {best_latent} (score: {avg_scores[best_latent]:.6f})")
                else:
                    latent = random.randint(0, model_num_latents - 1)
                    concept_to_latent[concept] = latent
                    print(f"  '{concept}' → latent {latent} (random - no scores)")

        return concept_to_latent

    def compute_reconstruction_loss(self, sae, activations):
        """Compute reconstruction loss."""
        model = sae.module if hasattr(sae, 'module') else sae

        if len(activations.shape) == 3:
            batch_size, time_steps, features = activations.shape
            activations = activations.reshape(batch_size * time_steps, features)

        try:
            pre_acts = model.pre_acts(activations)
            top_acts, top_indices = model.select_topk(pre_acts)
            reconstructed = model.decode(top_acts, top_indices)
            loss = F.mse_loss(reconstructed, activations)
            return loss, pre_acts
        except Exception as e:
            print(f"  Recon Error: {e}")
            return torch.tensor(1.0, device=self.device, dtype=self.dtype), torch.zeros(activations.shape[0], 1000, device=self.device)

    def compute_cross_entropy_loss(self, pre_acts, nudity_labels, concept_to_latent):
        """Compute weighted binary cross-entropy loss for nudity classification."""
        if len(pre_acts.shape) == 2:
            batch_times_seq, num_latents = pre_acts.shape
            batch_size = len(nudity_labels)

            if batch_times_seq != batch_size:
                seq_length = batch_times_seq // batch_size
                if batch_times_seq == batch_size * seq_length:
                    pre_acts = pre_acts.view(batch_size, seq_length, num_latents)
                    pre_acts = pre_acts.mean(dim=1)
                else:
                    return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        elif len(pre_acts.shape) == 3:
            pre_acts = pre_acts.mean(dim=1)

        batch_size, num_latents = pre_acts.shape

        # Create target mask
        target_mask = torch.zeros(batch_size, num_latents, device=self.device, dtype=torch.float32)

        # NEW: Track which samples are nudity for weighting
        nudity_mask = torch.zeros(batch_size, num_latents, device=self.device, dtype=torch.float32)

        valid_samples = 0
        for i, label in enumerate(nudity_labels):
            if label in concept_to_latent:
                latent_idx = concept_to_latent[label]
                if 0 <= latent_idx < num_latents:
                    target_mask[i, latent_idx] = 1.0
                    # Mark nudity samples for upweighting
                    if label == 'nudity':
                        nudity_mask[i, latent_idx] = 1.0
                    valid_samples += 1

        if valid_samples == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Compute BCE loss per element
        bce_loss = F.binary_cross_entropy_with_logits(pre_acts, target_mask, reduction='none')

        # NEW: Apply class weights
        # Weight nudity samples more heavily
        weights = torch.ones_like(bce_loss)
        weights = weights + (self.pos_class_weight - 1.0) * nudity_mask

        # Apply weights
        weighted_bce_loss = bce_loss * weights

        valid_mask = target_mask > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        ce_loss = weighted_bce_loss[valid_mask].mean()
        return ce_loss

    def compute_sparsity_loss(self, pre_acts):
        """Compute L1 sparsity loss."""
        if torch.isnan(pre_acts).any():
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        clipped_pre_acts = torch.clamp(pre_acts, -100, 100)
        sparsity = torch.mean(torch.abs(clipped_pre_acts))
        if torch.isnan(sparsity) or torch.isinf(sparsity):
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        return sparsity

    def initialize_saes(self):
        """Load or create SAE models with proper DDP wrapping."""
        if self.rank == 0:
            print(f"Initializing SAE models from {self.checkpoint_path}")

        # Handle single model
        if (self.checkpoint_path / "cfg.json").exists():
            hook_name = self.checkpoint_path.name

            if self.from_scratch:
                self._create_sae_from_scratch(hook_name)
            else:
                sae = Sae.load_from_disk(self.checkpoint_path, device=self.device)
                sae = sae.to(dtype=self.dtype)

                # ===== WRAP IN DDP IF MULTI-GPU =====
                if self.world_size > 1:
                    sae = DDP(sae, device_ids=[self.device.index], 
                             output_device=self.device.index,
                             find_unused_parameters=False)

                self.saes[hook_name] = sae
                self.optimizers[hook_name] = Adam([{"params": sae.parameters(), "lr": self.lr}], eps=1e-8)

                if self.rank == 0:
                    print(f"Loaded SAE for {hook_name}")

                # Try to load training state for resume
                if self.resume and self.rank == 0:
                    self._load_training_state(hook_name)
        else:
            # Handle subdirectories
            for hook_dir in self.checkpoint_path.iterdir():
                if hook_dir.is_dir():
                    hook_name = hook_dir.name

                    if self.from_scratch:
                        self._create_sae_from_scratch(hook_name)
                    else:
                        sae = Sae.load_from_disk(hook_dir, device=self.device)
                        sae = sae.to(dtype=self.dtype)

                        # ===== WRAP IN DDP IF MULTI-GPU =====
                        if self.world_size > 1:
                            sae = DDP(sae, device_ids=[self.device.index], 
                                     output_device=self.device.index,
                                     find_unused_parameters=False)

                        self.saes[hook_name] = sae
                        self.optimizers[hook_name] = Adam([{"params": sae.parameters(), "lr": self.lr}], eps=1e-8)

                        if self.rank == 0:
                            print(f"Loaded SAE for {hook_name}")

                        # Try to load training state for resume
                        if self.resume and self.rank == 0:
                            self._load_training_state(hook_name)

        # Synchronize all processes
        if self.world_size > 1:
            dist.barrier()


    def _create_sae_from_scratch(self, hook_name):
        """Create a new SAE from scratch with DDP wrapping."""
        cfg = {
            "expansion_factor": 16,
            "normalize_decoder": True,
            "num_latents": 0,
            "k": 32,
            "batch_topk": True,
            "sample_topk": False,
            "input_unit_norm": False,
            "multi_topk": False
        }

        sae_config = SaeConfig(**cfg)
        sae = Sae(d_in=1280, cfg=sae_config, device=self.device, dtype=self.dtype)
        sae = sae.to(device=self.device, dtype=self.dtype)

        # ===== WRAP IN DDP IF MULTI-GPU =====
        if self.world_size > 1:
            sae = DDP(sae, device_ids=[self.device.index], 
                     output_device=self.device.index,
                     find_unused_parameters=False)

        self.saes[hook_name] = sae
        self.optimizers[hook_name] = Adam([{"params": sae.parameters(), "lr": self.lr}], eps=1e-8)

        if self.rank == 0:
            print(f"✅ Created SAE from scratch for {hook_name}")

    def _load_training_state(self, hook_name):
        """Load training state for resume."""
        checkpoint_path = self.save_dir / hook_name / "last" / "training_state.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            self.start_epoch = state.get('epoch', 1) + 1
            self.best_val_loss = state.get('best_val_loss', float('inf'))
            self.patience_counter = state.get('patience_counter', 0)
            print(f"✅ Resumed from epoch {self.start_epoch - 1}, best_val_loss: {self.best_val_loss:.6f}")
        else:
            print(f"⚠️  No training state found at {checkpoint_path}, starting from epoch 1")

    def _save_training_state(self, hook_name, epoch, val_loss):
        """Save training state for resume."""
        state = {
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'val_loss': val_loss
        }
        checkpoint_path = self.save_dir / hook_name / "last" / "training_state.json"
        os.makedirs(checkpoint_path.parent, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)

    def save_checkpoint(self, hook_name, sae, epoch, val_loss, is_best=False):
        """Save checkpoint (best or last)."""
        if is_best:
            save_path = self.save_dir / hook_name / "best"
            print(f"💾 Saving best model (val_loss: {val_loss:.6f}) to {save_path}")
        else:
            save_path = self.save_dir / hook_name / "last"
            print(f"💾 Saving last model (epoch {epoch}) to {save_path}")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save SAE model
        if isinstance(sae, DDP):
            sae.module.save_to_disk(save_path)
        else:
            sae.save_to_disk(save_path)
        
        # Save training state (only for 'last' checkpoint)
        if not is_best:
            self._save_training_state(hook_name, epoch, val_loss)

    def initialize_wandb(self):
        """Initialize wandb in offline mode with class weight tag."""
        if WANDB_AVAILABLE:
            wandb_dir = os.path.join(self.save_dir, "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = wandb_dir

            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # NEW: Create run name with class weight
            weight_tag = f"posweight{self.pos_class_weight:.1f}".replace('.', 'p')
            run_name = f"sae_nudity_{weight_tag}_{timestamp}"

            config = {
                "learning_rate": self.lr,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "from_scratch": self.from_scratch,
                "pos_class_weight": self.pos_class_weight,  # NEW: Log the weight
                "reconstruction_weight": self.reconstruction_weight,
                "cross_entropy_weight": self.cross_entropy_weight,
                "sparsity_weight": self.sparsity_weight,
            }

            # NEW: Add tags
            tags = [
                f"pos_weight_{self.pos_class_weight}",
                f"ce_weight_{self.cross_entropy_weight}",
                f"batch_size_{self.batch_size}",
            ]

            wandb.init(
                project="sae_nudity_optimizer", 
                name=run_name, 
                config=config, 
                tags=tags,  # NEW: Add tags
                dir=wandb_dir
            )
            print(f"Initialized wandb in OFFLINE mode at {wandb_dir}")
            print(f"Run name: {run_name}")
            print(f"Tags: {tags}")

    def train(self):
        """Train the SAE to distinguish nudity from non-nudity."""
        if self.rank == 0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        for hook_name, sae in self.saes.items():
            if self.rank == 0:
                print(f"\nTraining SAE for {hook_name}")

            # Assign concepts
            if self.rank == 0:
                concept_to_latent = self.assign_concepts_to_latents(hook_name)
                self.concept_to_latent[hook_name] = concept_to_latent

                if self.world_size > 1:
                    concept_list = [self.concept_to_latent[hook_name]]
                    dist.broadcast_object_list(concept_list, src=0)
            else:
                concept_list = [None]
                dist.broadcast_object_list(concept_list, src=0)
                self.concept_to_latent[hook_name] = concept_list[0]

            if self.world_size > 1:
                dist.barrier()

            # Training loop
            for epoch in range(self.start_epoch, self.num_epochs + 1):
                if self.rank == 0:
                    print(f"\nEpoch {epoch}/{self.num_epochs}")

                sae.train()
                optimizer = self.optimizers[hook_name]
                concept_to_latent = self.concept_to_latent[hook_name]

                if self.world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

                total_loss_sum = 0.0
                num_batches = 0

                data_iter = self.train_loader
                if self.rank == 0:
                    data_iter = tqdm(data_iter, desc="Batches")

                for batch_idx, (activations, nudity_labels) in enumerate(data_iter):
                    activations = activations.to(self.device)

                    recon_loss, pre_acts = self.compute_reconstruction_loss(sae, activations)
                    ce_loss = self.compute_cross_entropy_loss(pre_acts, nudity_labels, concept_to_latent)
                    sparsity_loss = self.compute_sparsity_loss(pre_acts)

                    total_loss = (
                        self.reconstruction_weight * recon_loss +
                        self.cross_entropy_weight * ce_loss +
                        self.sparsity_weight * sparsity_loss
                    )

                    optimizer.zero_grad()
                    if not (torch.isnan(total_loss).any() or torch.isinf(total_loss).any()):
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)

                    total_loss_sum += total_loss.item()
                    num_batches += 1

                # ========== FIXED: ALL RANKS PARTICIPATE IN VALIDATION ==========
                # Synchronize before validation
                if self.world_size > 1:
                    dist.barrier()

                if num_batches > 0:
                    # Compute average train loss across all ranks
                    if self.world_size > 1:
                        loss_tensor = torch.tensor([total_loss_sum, num_batches], 
                                                  dtype=torch.float32, device=self.device)
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                        total_loss_sum = loss_tensor[0].item()
                        num_batches = int(loss_tensor[1].item())

                    avg_train_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0

                    if self.rank == 0:
                        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.6f}")
                        print("Running validation...")

                    # ALL RANKS run validation
                    try:
                        val_loss = self.validate(hook_name, sae, concept_to_latent)

                        if self.rank == 0:
                            print(f"Epoch {epoch} Validation Loss: {val_loss:.6f}")

                    except Exception as e:
                        if self.rank == 0:
                            print(f"⚠️  Validation failed with error: {e}")
                            print("Skipping validation for this epoch...")
                        val_loss = float('inf')

                    # Synchronize after validation
                    if self.world_size > 1:
                        dist.barrier()

                    # Only rank 0 saves checkpoints and logs
                    if self.rank == 0:
                        # Save last checkpoint
                        self.save_checkpoint(hook_name, sae, epoch, val_loss, is_best=False)

                        # Save best checkpoint if improved
                        if val_loss < self.best_val_loss:
                            print(f"✨ New best validation loss: {val_loss:.6f} (previous: {self.best_val_loss:.6f})")
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.save_checkpoint(hook_name, sae, epoch, val_loss, is_best=True)
                        else:
                            self.patience_counter += 1
                            print(f"No improvement. Patience: {self.patience_counter}/{self.patience}")

                            if self.patience_counter >= self.patience:
                                print(f"Early stopping triggered after {epoch} epochs")
                                # Broadcast early stopping decision to all ranks
                                if self.world_size > 1:
                                    dist.barrier()
                                break
                            
                        # Log to wandb
                        if WANDB_AVAILABLE:
                            wandb.log({
                                "epoch": epoch,
                                "train_loss": avg_train_loss,
                                "val_loss": val_loss,
                                "best_val_loss": self.best_val_loss
                            })

                # Final synchronization for the epoch
                if self.world_size > 1:
                    dist.barrier()

            if self.rank == 0:
                print("\n✅ Training completed!")
                print(f"Best model saved at: {self.save_dir / hook_name / 'best'}")
                print(f"Last model saved at: {self.save_dir / hook_name / 'last'}")


    def validate(self, hook_name, sae, concept_to_latent):
        """Run validation and return average loss with proper DDP synchronization."""
        sae.eval()
        total_loss_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (activations, nudity_labels) in enumerate(self.val_loader):  # ADD batch_idx
                # ADD THESE 2 LINES:
                if self.max_val_batches and batch_idx >= self.max_val_batches:
                    break
                
                activations = activations.to(self.device)
                recon_loss, pre_acts = self.compute_reconstruction_loss(sae, activations)
                ce_loss = self.compute_cross_entropy_loss(pre_acts, nudity_labels, concept_to_latent)
                sparsity_loss = self.compute_sparsity_loss(pre_acts)
                total_loss = (
                    self.reconstruction_weight * recon_loss +
                    self.cross_entropy_weight * ce_loss +
                    self.sparsity_weight * sparsity_loss
                )
                total_loss_sum += total_loss.item()
                num_batches += 1

        # Synchronize validation metrics across all GPUs
        if self.world_size > 1:
            # Convert to tensors for all_reduce
            loss_tensor = torch.tensor([total_loss_sum, num_batches], 
                                       dtype=torch.float32, device=self.device)
            # Sum across all ranks
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Extract synchronized values
            total_loss_sum = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())

        avg_val_loss = total_loss_sum / num_batches if num_batches > 0 else float('inf')
        return avg_val_loss

    @staticmethod
    def setup_distributed(rank, world_size):
        """Initialize distributed training."""
        if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def main():
    parser = argparse.ArgumentParser(description="Optimize SAE for nudity detection")
    
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to SAE checkpoint")
    parser.add_argument("--activations_dir", type=str, required=True, help="Path to activations directory")
    parser.add_argument("--scores_json_path", type=str, default=None, help="Path to pre-computed scores JSON")
    parser.add_argument("--max_val_batches", type=int, default=20, help="Max validation batches per epoch (0 for all)")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split fraction")
    
    # Loss weights
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--cross_entropy_weight", type=float, default=1.0, help="Cross-entropy loss weight")
    parser.add_argument("--sparsity_weight", type=float, default=0.01, help="Sparsity loss weight")
    parser.add_argument("--pos_class_weight", type=float, default=1.0, help="Weight for positive (nudity) class to handle imbalance")  # NEW
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="sae-nudity-optimized", help="Save directory")
    
    # Advanced parameters
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_float16", action="store_true", help="Use float16 for all tensors")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch")
    
    # DDP parameters (auto-set by torchrun)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # ===== FIX: Properly detect distributed training =====
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # Initialize distributed training
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"🚀 Distributed training initialized: {world_size} GPUs")
            print(f"   Rank: {rank}, Local Rank: {local_rank}")
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        local_rank = 0
        print("🚀 Single GPU training")
    
    # Set device based on local_rank
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # Only print from rank 0
    if rank == 0:
        print(f"Configuration:")
        print(f"  - World Size: {world_size}")
        print(f"  - Rank: {rank}")
        print(f"  - Device: {device}")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Effective Batch Size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
    
    optimizer = SAENudityOptimizer(
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
        mixed_precision=args.mixed_precision,
        use_float16=args.use_float16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        patience=args.patience,
        resume=args.resume,
        from_scratch=args.from_scratch,
        world_size=world_size,  # Pass actual world_size
        rank=rank,  # Pass actual rank
        pos_class_weight=args.pos_class_weight,
        max_val_batches=args.max_val_batches
    )
    
    optimizer.train()
    
    if rank == 0:
        print("Training completed successfully!")
    
    # Clean up distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()