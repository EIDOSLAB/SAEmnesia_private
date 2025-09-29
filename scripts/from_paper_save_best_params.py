import torch
from pathlib import Path

# Base path for saving parameters
base_path = "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron"  # Change this to your desired directory

# Input data - class names, percentiles, and multipliers
# data = [
#     ("Architectures", 99.999, 0.0),
#     ("Bears", 99.999, 0.0),
#     ("Birds", 99.999, 0.0),
#     ("Butterfly", 99.99, 0.0),
#     ("Cats", 99.999, 0.0),
#     ("Dogs", 99.999, 0.05.0),
#     ("Fishes", 99.995, 0.0),
#     ("Flame", 99.995, 0.0.0),
#     ("Flowers", 99.99, 0.0),
#     ("Frogs", 99.999, 0.0),
#     ("Horses", 99.99, 0.0),
#     ("Human", 99.995, 0.0),
#     ("Jellyfish", 99.999, 0.0.0),
#     ("Rabbits", 99.99, 0.00.0),
#     ("Sandwiches", 99.999, 0.00.0),
#     ("Sea", 99.995, 0.0),
#     ("Statues", 99.995, 0.0),
#     ("Towers", 99.995, 0.0.0),
#     ("Trees", 99.99, 0.0),
#     ("Waterfalls", 99.99, 0.0.0)
# ]

data = [
    ("Architectures", 99.999, 0.0),
    ("Bears", 99.999, 0.0),
    ("Birds", 99.999, 0.0),
    ("Butterfly", 99.999, 0.0),
    ("Cats", 99.999, 0.0),
    ("Dogs", 99.999, 0.0),
    ("Fishes", 99.999, 0.0),
    ("Flame", 99.999, 0.0),
    ("Flowers", 99.999, 0.0),
    ("Frogs", 99.999, 0.0),
    ("Horses", 99.999, 0.0),
    ("Human", 99.999, 0.0),
    ("Jellyfish", 99.999, 0.0),
    ("Rabbits", 99.999, 0.0),
    ("Sandwiches", 99.999, 0.0),
    ("Sea", 99.999, 0.0),
    ("Statues", 99.999, 0.0),
    ("Towers", 99.999, 0.0),
    ("Trees", 99.999, 0.0),
    ("Waterfalls", 99.999, 0.0)
]

# Dummy metrics for demonstration - replace with your actual metrics if available
# For this example, we'll generate random values
import random
random.seed(42)  # For reproducibility

# Create the best_params dictionary
best_params = {}
for class_name, percentile, multiplier in data:
    # Generate dummy metrics for demonstration
    ua = random.uniform(0.7, 0.95)
    ira = random.uniform(0.7, 0.95)
    avg = (ua + ira) / 2
    
    best_params[class_name] = {
        "params": (percentile, multiplier),
        "metrics": {"UA": ua, "IRA": ira},
        "best_avg": avg
    }

# Output the results
print("\nBest parameters for each class (maximizing average of UA and IRA):")
print("-" * 80)

# Create dictionary for saving parameters
params_dict = {}
for class_name, data in best_params.items():
    print(f"\nClass: {class_name}")
    percentile, multiplier = data["params"]
    print(f"Parameters: percentile={percentile}, multiplier={multiplier}")
    print(f"UA:  {data['metrics']['UA']:.4f}")
    print(f"IRA: {data['metrics']['IRA']:.4f}")
    print(f"Average: {data['best_avg']:.4f}")
    params_dict[class_name] = {"percentile": percentile, "multiplier": multiplier}

# Save parameters
save_path = Path(base_path) / "class_params_uniform_99.999_0.0.pth"
torch.save(params_dict, save_path)
print(f"\nParameters saved to {save_path}")

# If you want to verify the saved data
loaded_params = torch.load(save_path)
print("\nVerifying saved parameters:")
for class_name, params in loaded_params.items():
    print(f"{class_name}: percentile={params['percentile']}, multiplier={params['multiplier']}")