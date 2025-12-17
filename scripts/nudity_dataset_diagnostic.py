#!/usr/bin/env python
"""
Quick diagnostic script to check what's in the datasets
"""
from datasets import load_from_disk
from pathlib import Path

base_dir = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/activations"
hookpoint = "unet.up_blocks.1.attentions.1"

for category in ['nudity', 'non_nudity']:
    category_dir = Path(base_dir) / hookpoint / category
    
    print(f"\n{'='*60}")
    print(f"Checking: {category_dir}")
    print(f"{'='*60}")
    
    if not category_dir.exists():
        print(f"❌ Directory does not exist!")
        continue
    
    dataset = load_from_disk(str(category_dir))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")
    
    # Check if nudity_label exists
    if "nudity_label" in dataset.column_names:
        labels = dataset["nudity_label"]
        unique_labels = set(labels)
        print(f"✅ nudity_label column exists")
        print(f"   Unique values: {unique_labels}")
        for label in unique_labels:
            count = sum(1 for l in labels if l == label)
            print(f"   '{label}': {count} samples")
    else:
        print(f"❌ No nudity_label column found")
    
    # Check first few samples
    print(f"\nFirst 3 samples:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}:")
        for key in sample.keys():
            if key == "activations":
                print(f"    {key}: shape {len(sample[key]) if isinstance(sample[key], list) else 'N/A'}")
            else:
                print(f"    {key}: {sample[key]}")