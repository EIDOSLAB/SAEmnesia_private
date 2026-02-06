import os, sys
import fire
import numpy as np
import torch
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available, theme_available

def main(input_dir: str):
    avg_ua = np.zeros((20, 20))
    avg_ira = np.zeros((20, 20))  # Changed to 2D to track over time
    avg_cra = np.zeros((20, 20))  # Changed to 2D to track over time
    
    # Sequential objects to unlearn (matching the generation and evaluation scripts)
    sequential_objects_to_unlearn = {
        0 : ["Bears"],
        1 : ["Bears", "Cats"],
        2 : ["Bears", "Cats", "Flowers"],
        3 : ["Bears", "Cats", "Flowers", "Frogs"],
        4 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish"],
        5 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea"],
        6 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues"],
        7 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches"],
        8 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls"],
        9 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures"],
        10 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds"],
        11 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly"],
        12 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs"],
        13 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes"],
        14 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame"],
        15 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame", "Horses"],
        16 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame", "Horses", "Human"],
        17 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame", "Horses", "Human", "Rabbits"],
        18 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame", "Horses", "Human", "Rabbits", "Towers"],
        19 : ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls", "Architectures", "Birds", "Butterfly", "Dogs", "Fishes", "Flame", "Horses", "Human", "Rabbits", "Towers", "Trees"],
    }
    
    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    
    progress_bar = tqdm(sequential_objects_to_unlearn.keys(), desc="Processing tasks")
    
    for curr_task_idx in progress_bar:
        curr_objects = "_".join(sequential_objects_to_unlearn[curr_task_idx])
        progress_bar.set_description(f"Processing task {curr_task_idx}: {curr_objects}")
        
        # Load style and class accuracy data for current task
        data_style = torch.load(os.path.join(input_dir, f"{curr_objects}.pth"))
        data_class = torch.load(os.path.join(input_dir, f"{curr_objects}_cls.pth"))
        
        acc_data_style = data_style["acc"]
        acc_data_class = data_class["acc"]
        
        # Calculate unlearning accuracy (UA) for previous tasks
        for prev_task_idx in range(curr_task_idx + 1):
            prev_objects = sequential_objects_to_unlearn[prev_task_idx]
            
            # Calculate average unlearning accuracy for objects that should be unlearned
            for prev_object in prev_objects:
                avg_ua[prev_task_idx, curr_task_idx] += 1 - acc_data_class[prev_object]
            avg_ua[prev_task_idx, curr_task_idx] /= len(prev_objects)
            
            # Calculate In-domain Retention Accuracy (IRA)
            # For objects: accuracy on objects not unlearned
            other_objects = [obj for obj in class_available if obj not in prev_objects]
            curr_avg_ira = 0.0  # In-domain (Object) Retention Accuracy
            
            if len(other_objects) > 0:
                for other_object in other_objects:
                    curr_avg_ira += acc_data_class[other_object]
                curr_avg_ira /= len(other_objects)
            else:
                # When all objects are unlearned, set to NaN or 0
                curr_avg_ira = np.nan
            
            # Store IRA for this combination of prev_task and curr_task
            avg_ira[prev_task_idx, curr_task_idx] = curr_avg_ira
            
            # Calculate Cross-domain Retention Accuracy (CRA)
            # For styles: accuracy on all styles (since we're not unlearning styles)
            curr_avg_cra = 0.0  # Cross-domain (Style) Retention Accuracy
            for theme in theme_avail:
                curr_avg_cra += acc_data_style[theme]
            curr_avg_cra /= len(theme_avail)
            
            # Store CRA for this combination of prev_task and curr_task
            avg_cra[prev_task_idx, curr_task_idx] = curr_avg_cra
    
    print("UA table (Unlearning Accuracy for Objects):")
    print("Rows: which objects were unlearned, Cols: evaluation after which task")
    print(avg_ua)
    print("\n" + "="*80 + "\n")
    
    print("IRA table (In-domain Retention Accuracy - Objects):")
    print("Rows: which objects were unlearned, Cols: evaluation after which task")
    print(avg_ira)
    print("\n" + "="*80 + "\n")
    
    print("CRA table (Cross-domain Retention Accuracy - Styles):")
    print("Rows: which objects were unlearned, Cols: evaluation after which task")
    print(avg_cra)
    print("\n" + "="*80 + "\n")
    
    # Also print the diagonal (final performance after each unlearning step)
    print("Diagonal values (performance at each step):")
    print("Task | UA | IRA | CRA")
    print("-" * 40)
    for i in range(20):
        print(f"{i:4d} | {avg_ua[i, i]:.4f} | {avg_ira[i, i]:.4f} | {avg_cra[i, i]:.4f}")
    
    # Save to file for further analysis
    np.savez(
        os.path.join(input_dir, "metrics_analysis.npz"),
        ua=avg_ua,
        ira=avg_ira,
        cra=avg_cra,
        sequential_objects=sequential_objects_to_unlearn
    )
    print(f"\nMetrics saved to {os.path.join(input_dir, 'metrics_analysis.npz')}")

if __name__ == "__main__":
    fire.Fire(main)