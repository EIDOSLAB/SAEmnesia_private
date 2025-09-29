import os
import fire
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available, theme_available

def main(input_dir: str, output_path: str = "accuracy_matrix.pdf", title: str = None):
    """
    Generate accuracy matrices showing the impact of unlearning each class.
    
    Args:
        input_dir: Directory containing the .pth files with accuracy data
        output_path: Path to save the visualization
        title: Custom title for the plot
    """
    
    # Initialize the accuracy matrix
    num_classes = len(class_available)
    accuracy_matrix = np.zeros((num_classes, num_classes))
    
    # Create mapping from class names to indices
    class_to_idx = {cls: idx for idx, cls in enumerate(class_available)}
    
    print("Generating accuracy matrix...")
    progress_bar = tqdm(class_available, desc="Processing classes")
    
    for unlearned_idx, unlearned_cls in enumerate(progress_bar):
        progress_bar.set_description(f"Processing unlearned class: {unlearned_cls}")
        
        try:
            # Load the class accuracy data for when this class was unlearned
            data_class = torch.load(os.path.join(input_dir, f"{unlearned_cls}_cls.pth"))
            acc_data_class = data_class["acc"]
            
            # Fill the row for this unlearned class
            for evaluated_cls in class_available:
                evaluated_idx = class_to_idx[evaluated_cls]
                
                if evaluated_cls in acc_data_class:
                    # CORRECTED: Keep as proportions (0-1) like SAeUron expects, 
                    # but multiply by 100 only for display in heatmap
                    accuracy_matrix[unlearned_idx, evaluated_idx] = acc_data_class[evaluated_cls] * 100
                    
        except FileNotFoundError:
            print(f"Warning: File {unlearned_cls}_cls.pth not found. Filling row with zeros.")
            continue
    
    # Create the full accuracy matrix plot
    plt.figure(figsize=(16, 14))
    
    sns.heatmap(accuracy_matrix, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                xticklabels=class_available,
                yticklabels=class_available,
                cbar_kws={'label': 'Classification Accuracy (%)'},
                square=True,
                linewidths=0.5,
                vmin=0,
                vmax=100,
                annot_kws={"size": 12})
    
    # Set title
    if title is None:
        plot_title = 'Impact of Unlearning on Class Accuracy\n(Rows: Unlearned Class, Columns: Evaluated Class)'
    else:
        plot_title = title
    
    plt.title(plot_title, fontsize=20, pad=20)
    plt.xlabel('Evaluated Class', fontsize=20)
    plt.ylabel('Unlearned Class', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Full matrix saved to {output_path}")
    
    # Create the IRA-only matrix plot (mask diagonal values)
    ira_matrix = accuracy_matrix.copy()
    
    # Create a mask to hide diagonal values (UA) since IRA only shows retention
    mask = np.eye(num_classes, dtype=bool)
    
    plt.figure(figsize=(16, 14))
    
    sns.heatmap(ira_matrix, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                xticklabels=class_available,
                yticklabels=class_available,
                cbar_kws={'label': 'Retention Accuracy (%)'},
                square=True,
                linewidths=0.5,
                vmin=0,
                vmax=100,
                mask=mask)  # Hide diagonal values
    
    # Set title for IRA plot
    if title is None:
        ira_title = 'In-domain Retention Accuracy (IRA)\n(Rows: Unlearned Class, Columns: Retained Class)'
    else:
        ira_title = f"{title} - IRA Only"
    
    plt.title(ira_title, fontsize=20, pad=20)
    plt.xlabel('Retained Class', fontsize=20)
    plt.ylabel('Unlearned Class', fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    
    plt.tight_layout()
    
    # Save IRA plot
    base_path, ext = os.path.splitext(output_path)
    ira_output_path = f"{base_path}_IRA{ext}"
    plt.savefig(ira_output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"IRA matrix saved to {ira_output_path}")
    
    # CORRECTED: Compute metrics using the same method as SAeUron's evaluation
    avg_ua = 0.0
    avg_ira = 0.0
    
    valid_classes = 0
    for i, cls in enumerate(class_available):
        # Use the actual proportion values (divide by 100 to get back to 0-1 range)
        accuracy_proportion = accuracy_matrix[i, i] / 100
        
        if accuracy_proportion >= 0:  # Count all classes with data
            # UA: 1 - accuracy of the unlearned class (higher is better unlearning)
            avg_ua += 1 - accuracy_proportion
            
            # IRA: average of other class accuracies
            other_accuracies_sum = 0.0
            other_classes_count = 0
            
            for j in range(len(class_available)):
                if j != i and accuracy_matrix[i, j] > 0:
                    other_accuracies_sum += accuracy_matrix[i, j] / 100  # Convert back to proportion
                    other_classes_count += 1
            
            if other_classes_count > 0:
                avg_ira += other_accuracies_sum / other_classes_count
            
            valid_classes += 1
    
    if valid_classes > 0:
        avg_ua /= valid_classes
        avg_ira /= valid_classes
        
        print(f"Average UA: {avg_ua * 100:.2f}%")
        print(f"Average IRA: {avg_ira * 100:.2f}%")
    
    # Debug: Print some sample values to understand the data format
    print(f"\nDebug - Sample diagonal values:")
    for i in range(min(5, len(class_available))):
        print(f"  {class_available[i]}: {accuracy_matrix[i, i]:.1f}%")
    
    print(f"\nDebug - Sample off-diagonal values for first class:")
    for j in range(min(5, len(class_available))):
        if j != 0:
            print(f"  {class_available[0]} -> {class_available[j]}: {accuracy_matrix[0, j]:.1f}%")

if __name__ == "__main__":
    fire.Fire(main)