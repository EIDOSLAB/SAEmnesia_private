import os, sys
import fire
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from UnlearnCanvas_resources.const import class_available, theme_available

def main(input_dir: str):
    avg_ua = np.zeros((9, 9))
    avg_ra = np.zeros(9)
    
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
            
            # Calculate retention accuracy (RA)
            # For objects: accuracy on objects not unlearned
            other_objects = [obj for obj in class_available if obj not in prev_objects]
            curr_avg_ora = 0.0  # Object Retention Accuracy
            for other_object in other_objects:
                curr_avg_ora += acc_data_class[other_object]
            curr_avg_ora /= len(other_objects)
            
            # For styles: accuracy on all styles (since we're not unlearning styles)
            curr_avg_sra = 0.0  # Style Retention Accuracy
            for theme in theme_avail:
                curr_avg_sra += acc_data_style[theme]
            curr_avg_sra /= len(theme_avail)
            
            # Overall retention accuracy is average of object and style retention
            avg_ra[prev_task_idx] = (curr_avg_ora + curr_avg_sra) / 2
    
    print("UA table (Unlearning Accuracy for Objects):")
    print(avg_ua)
    print("RA table (Retention Accuracy - Objects + Styles):")
    print(avg_ra)

if __name__ == "__main__":
    fire.Fire(main)