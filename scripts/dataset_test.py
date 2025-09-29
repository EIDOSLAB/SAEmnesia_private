from datasets import load_dataset, Dataset
import pyarrow as pa

# Load a single arrow file to inspect
dataset = Dataset.from_file("/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects/unet.up_blocks.1.attentions.1/data-00000-of-00650.arrow")

# Check available columns
print(dataset.column_names)

# Check if object_label is included
if "object_label" in dataset.column_names:
    # Get unique objects
    unique_objects = set(dataset["object_label"])
    print(f"Objects in this file: {unique_objects}")