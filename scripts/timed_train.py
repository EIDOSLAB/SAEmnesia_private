"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
import time
import json
from collections import defaultdict
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from simple_parsing import parse

from SAE.config import TrainConfig
from SAE.trainer import SaeTrainer

# Global timing tracker
TIMING_STATS = defaultdict(list)
BENCHMARK_MODE = os.environ.get('BENCHMARK_MODE', '0') == '1'
BENCHMARK_STEPS = int(os.environ.get('BENCHMARK_STEPS', '100'))


@dataclass
class RunConfig(TrainConfig):
    mixed_precision: str = "no"

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    seed: int = 42
    """Random seed for shuffling the dataset."""
    device: str = "cuda"
    num_epochs: int = 1


def load_datasets_from_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load and concatenate datasets from multiple directories.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)

    Returns:
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")

    for base_dir in base_dirs:
        dataset = Dataset.load_from_disk(
            os.path.join(base_dir, hookpoint), keep_in_memory=False
        )

        # Set format for each dataset
        dataset.set_format(
            type="torch",
            columns=["activations", "timestep"],
            dtype=dtype,
        )

        datasets.append(dataset)

    # Concatenate all datasets
    return concatenate_datasets(datasets)


def save_benchmark_results(save_dir, hookpoint, batch_size):
    """Save benchmark timing results to JSON."""
    if not TIMING_STATS:
        return
    
    results = {
        'script': 'train.py (baseline)',
        'hook_name': hookpoint,
        'batch_size': batch_size,
        'num_steps_measured': len(TIMING_STATS['total_step_time']),
        'timing_ms': {
            'avg_total_step': np.mean(TIMING_STATS['total_step_time']) * 1000,
            'std_total_step': np.std(TIMING_STATS['total_step_time']) * 1000,
            'avg_forward': np.mean(TIMING_STATS['forward_time']) * 1000,
            'avg_backward': np.mean(TIMING_STATS['backward_time']) * 1000,
            'avg_optimizer': np.mean(TIMING_STATS['optimizer_time']) * 1000,
        },
        'steps_per_second': 1.0 / np.mean(TIMING_STATS['total_step_time']),
        'memory_gb': {
            'peak': max(TIMING_STATS['memory_allocated']) if TIMING_STATS['memory_allocated'] else 0,
            'avg': np.mean(TIMING_STATS['memory_allocated']) if TIMING_STATS['memory_allocated'] else 0,
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, f'benchmark_results_{hookpoint}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS SAVED: {output_file}")
    print(f"{'='*70}")
    print(json.dumps(results, indent=2))
    print(f"{'='*70}\n")


# Monkey-patch the SaeTrainer.fit method to add benchmark timing
_original_fit = SaeTrainer.fit

def instrumented_fit(self):
    """Instrumented version of the fit method with benchmark timing."""
    if not BENCHMARK_MODE:
        # If not in benchmark mode, use original fit
        return _original_fit(self)
    
    # Benchmark mode: instrument the training loop
    print("\n🔬 BENCHMARK MODE ENABLED")
    print(f"   Will measure {BENCHMARK_STEPS} steps after 10 warmup steps")
    
    # Use Tensor Cores even for fp32 matmuls
    torch.set_float32_matmul_precision("high")

    rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    ddp = dist.is_initialized() and not self.cfg.distribute_modules

    device = torch.device(self.cfg.device)
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    dataloaders = {
        hook: DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor,
        )
        for hook, ds in self.dataset_dict.items()
    }
    
    maybe_wrapped = {}
    
    print(f"Starting benchmark with batch_size={self.batch_size}")
    
    for batch_idx, batch_dict in enumerate(zip(*dataloaders.values())):
        # Early exit after measuring enough steps
        if batch_idx >= BENCHMARK_STEPS + 10:
            print(f"\nBenchmark complete after {batch_idx} steps")
            break
        
        # Start timing after warmup
        if batch_idx >= 10:
            step_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Prepare hidden states
        hidden_dict = {}
        for hook, batch in zip(dataloaders.keys(), batch_dict):
            hidden_dict[hook] = batch["activations"]
        
        if self.cfg.distribute_modules:
            hidden_dict = self.scatter_hiddens(hidden_dict)
        
        # Process each hook
        for name, hiddens in zip(self.local_hookpoints(), hidden_dict.values()):
            raw = self.saes[name]
            
            # Initialize decoder bias on first iteration
            if batch_idx == 0:
                hiddens_input = hiddens.view(-1, hiddens.shape[-1])
                from SAE.utils import geometric_median
                median = geometric_median(self.maybe_all_cat(hiddens_input))
                median = median.to(raw.device)
                raw.b_dec.data = median.to(raw.dtype)
            
            # Wrap with DDP if needed
            if not maybe_wrapped:
                from torch.nn.parallel import DistributedDataParallel as DDP
                maybe_wrapped = (
                    {name: DDP(sae, device_ids=[dist.get_rank()]) 
                     for name, sae in self.saes.items()}
                    if ddp else self.saes
                )
            
            # Normalize decoder
            if raw.cfg.normalize_decoder:
                raw.set_decoder_norm_to_unit_norm()
            
            wrapped = maybe_wrapped[name]
            
            # Forward pass timing
            if batch_idx >= 10:
                forward_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Forward pass
            hiddens = hiddens.to(device)
            out = wrapped(hiddens, dead_mask=None)
            loss = out.fvu  # Simple loss for baseline
            
            if batch_idx >= 10:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                TIMING_STATS['forward_time'].append(time.time() - forward_start)
            
            # Backward pass timing
            if batch_idx >= 10:
                backward_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            loss.backward()
            
            if batch_idx >= 10:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                TIMING_STATS['backward_time'].append(time.time() - backward_start)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        
        # Optimizer step timing
        if batch_idx >= 10:
            opt_start = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        if self.cfg.sae.normalize_decoder:
            for sae in self.saes.values():
                sae.remove_gradient_parallel_to_decoder_directions()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
        if batch_idx >= 10:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            TIMING_STATS['optimizer_time'].append(time.time() - opt_start)
        
        # Total step time and memory
        if batch_idx >= 10:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            TIMING_STATS['total_step_time'].append(time.time() - step_start)
            if torch.cuda.is_available():
                TIMING_STATS['memory_allocated'].append(
                    torch.cuda.memory_allocated() / 1024**3
                )
        
        # Progress update
        if batch_idx >= 10 and (batch_idx - 10) % 10 == 0:
            avg_time = np.mean(list(TIMING_STATS['total_step_time'])[-10:]) * 1000
            print(f"  Step {batch_idx}: {avg_time:.2f} ms/step")
    
    # Save results
    if rank_zero:
        hookpoint = list(self.saes.keys())[0]
        save_dir = f"sae-ckpts/{self.cfg.wandb_project}/{self.cfg.run_name}" if self.cfg.run_name else f"sae-ckpts/{self.cfg.wandb_project}"
        save_benchmark_results(save_dir, hookpoint, self.batch_size)
    
    print("\n✅ Benchmark complete!")

# Apply the monkey patch
SaeTrainer.fit = instrumented_fit


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"_{args.dataset_path[0].split('/')[-2]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    print(f"Training in {dtype=}")
    
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    dataset_dict = {}
    if not ddp or rank == 0:
        for hookpoint in args.hookpoints:
            if len(args.dataset_path) > 1:
                dataset = load_datasets_from_dirs(args.dataset_path, hookpoint, dtype)
            else:
                dataset = Dataset.load_from_disk(
                    os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False
                )
            dataset.set_format(
                type="torch",
                columns=["activations", "timestep"],
                dtype=dtype,
            )
            dataset = dataset.shuffle(args.seed)
            if limit := args.max_examples:
                dataset = dataset.select(range(limit))
            dataset_dict[hookpoint] = dataset
            print(f"Loaded dataset for {hookpoint}")
    # NOTE: DDP not tested so far
    if ddp:
        dist.barrier()
        if rank != 0:
            for hookpoint in args.hookpoints:
                dataset = Dataset.load_from_disk(
                    os.path.join(args.dataset_path, hookpoint), keep_in_memory=False
                )
                dataset.set_format(
                    type="torch",
                    columns=["activations", "timestep"],
                    dtype=dtype,
                )
                dataset = dataset.shuffle(args.seed)
                dataset = dataset.shard(dist.get_world_size(), rank)
                dataset_dict[hookpoint] = dataset
                print(f"Loaded dataset for {hookpoint}")

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)

        trainer.fit()


if __name__ == "__main__":
    run()