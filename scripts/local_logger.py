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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

print("WandB available:", WANDB_AVAILABLE)

class LocalLogger:
    """Simple logger that saves metrics to CSV and JSON files when wandb is not available."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the metrics file
        self.metrics_file = self.log_dir / "metrics.csv"
        self.metrics_keys = set()
        
        # Create metrics file with header if it doesn't exist
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "step", "metric", "value"])
        
        # Config will be saved as JSON
        self.config = {}
        
        print(f"Local logging enabled. Saving logs to {self.log_dir}")
    
    def init(self, project=None, name=None, config=None):
        """Initialize the logger with project and run information."""
        if config:
            self.config = config
            # Save config to JSON file
            with open(self.log_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
        
        # Save project and run info
        run_info = {
            "project": project,
            "name": name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.log_dir / "run_info.json", 'w') as f:
            json.dump(run_info, f, indent=2)
        
        return self
    
    def log(self, metrics, step=None):
        """Log metrics to CSV file."""
        timestamp = time.time()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for metric, value in metrics.items():
                # Skip if the value is not a number
                if not isinstance(value, (int, float)):
                    continue
                
                # Add to keys set for summary
                self.metrics_keys.add(metric)
                
                # Write to CSV
                writer.writerow([timestamp, step, metric, value])
    
    def finish(self):
        """Finish logging and create a summary file."""
        # Create a summary of all metrics
        summary = {key: None for key in self.metrics_keys}
        
        # Save the summary
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Logging finished. All logs saved to {self.log_dir}")


    def initialize_logger(self):
        """Initialize the logging system (wandb or local)."""
        config = {
            "checkpoint_path": str(self.checkpoint_path),
            "style_activations_path": str(self.style_activations_path),
            "learning_rate": self.lr,
            "num_epochs": self.num_epochs,
            "reconstruction_weight": self.reconstruction_weight,
            "initial_cross_entropy_weight": self.initial_cross_entropy_weight,
            "final_cross_entropy_weight": self.final_cross_entropy_weight,
            "sparsity_weight": self.sparsity_weight,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "validation_split": self.validation_split,
        }

        if self.log_to_wandb:
            try:
                os.environ['WANDB_MODE'] = 'offline'  # Set to offline mode
                wandb.init(
                    project=self.wandb_project,
                    name=self.run_name,
                    config=config,
                    dir=self.log_dir  # Specify directory for logs
                )
                print("Initialized wandb in offline mode")
                self.logger = wandb
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
                print("Falling back to local logging")
                self.log_to_wandb = False

    def log_metrics(self, metrics, step=None):
        """Log metrics using the appropriate logger."""
        if self.log_to_wandb:
            wandb.log(metrics, step=step)
        elif self.local_log and self.logger:
            self.logger.log(metrics, step=step)

            # Also print some key metrics to console occasionally
            if step % (self.eval_freq * 10) == 0:
                print(f"\nStep {step} metrics:")
                for k, v in metrics.items():
                    if any(key in k for key in ["total_loss", "ce_loss", "recon_loss", "learning_rate"]):
                        print(f"  {k}: {v}")
                print("")