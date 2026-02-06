import os
import fire
import torch
from tqdm import tqdm
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available, theme_available

def main(input_dir: str, output_file: str = None):
    avg_ua = 0.0
    avg_ira = 0.0
    avg_cra = 0.0
    
    # Store per-class metrics
    per_class_metrics = {
        "UA": {},
        "IRA": {},
        "CRA": {}
    }
    
    progress_bar = tqdm(class_available, desc="Processing classes")
    
    for cls in progress_bar:
        progress_bar.set_description(f"Processing class {cls}")
    
        # Load files
        style_file = os.path.join(input_dir, f"{cls}.pth")
        class_file = os.path.join(input_dir, f"{cls}_cls.pth")

        print(f"\nLoading files for class: {cls}")
        print(f"  Style file: {style_file}")
        print(f"  Class file: {class_file}")

        data_style = torch.load(style_file, weights_only=False)
        data_class = torch.load(class_file, weights_only=False)

        print(f"  Keys in acc_data_class: {list(data_class['acc'].keys())}")
        print(f"  Checking if '{cls}' is in acc_data_class: {cls in data_class['acc']}")

        acc_data_class = data_class["acc"]
        
        # UA: Unlearning Accuracy (1 - accuracy on the unlearned class)
        ua = 1 - acc_data_class[cls]
        per_class_metrics["UA"][cls] = ua
        avg_ua += ua
        
        # IRA: Innocent Retention Accuracy (average accuracy on other classes)
        curr_avg_ira = 0.0
        for cls_to_compare in class_available:
            if cls_to_compare != cls:
                curr_avg_ira += acc_data_class[cls_to_compare]
        ira = curr_avg_ira / (len(class_available) - 1)
        per_class_metrics["IRA"][cls] = ira
        avg_ira += ira
        
        # CRA: Concept Retention Accuracy (average accuracy on themes/styles)
        acc_data_style = data_style["acc"]
        curr_avg_cra = 0.0
        for theme in theme_available:
            if theme != "Seed_Images":
                curr_avg_cra += acc_data_style[theme]
        cra = curr_avg_cra / (len(theme_available) - 1)
        per_class_metrics["CRA"][cls] = cra
        avg_cra += curr_avg_cra / (len(theme_available) - 1)
    
    avg_ua /= len(class_available)
    avg_ira /= len(class_available)
    avg_cra /= len(class_available)
    
    # Build output string
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("PER-CLASS METRICS")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Class':<20} {'UA (%)':<15} {'IRA (%)':<15} {'CRA (%)':<15}")
    output_lines.append("-" * 80)
    
    for cls in class_available:
        ua_pct = per_class_metrics['UA'][cls] * 100
        ira_pct = per_class_metrics['IRA'][cls] * 100
        cra_pct = per_class_metrics['CRA'][cls] * 100
        output_lines.append(f"{cls:<20} {ua_pct:<15.2f} {ira_pct:<15.2f} {cra_pct:<15.2f}")
    
    output_lines.append("=" * 80)
    output_lines.append("AVERAGE METRICS")
    output_lines.append("=" * 80)
    output_lines.append(f"Average UA:  {avg_ua * 100:.2f}%")
    output_lines.append(f"Average IRA: {avg_ira * 100:.2f}%")
    output_lines.append(f"Average CRA: {avg_cra * 100:.2f}%")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("Metric Definitions:")
    output_lines.append("  UA  (Unlearning Accuracy):           How well the model forgot the target class (higher = better)")
    output_lines.append("  IRA (In-Domain Retention Accuracy):   How well the model retained other classes (higher = better)")
    output_lines.append("  CRA (Cross-Domain Retention Accuracy):    How well the model retained art styles/themes (higher = better)")
    
    # Print to console
    output_text = "\n".join(output_lines)
    print("\n" + output_text)
    
    # Save to text file if specified
    if output_file:
        # Force .txt extension
        if not output_file.endswith('.txt'):
            output_file = output_file.replace('.pth', '.txt') if output_file.endswith('.pth') else output_file + '.txt'
        
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"\nResults saved to {output_file}")
        
        # Also save the raw data as .pth for programmatic access
        pth_file = output_file.replace('.txt', '.pth')
        results = {
            "per_class": per_class_metrics,
            "averages": {
                "UA": avg_ua,
                "IRA": avg_ira,
                "CRA": avg_cra
            }
        }
        torch.save(results, pth_file)
        print(f"Raw data saved to {pth_file}")

if __name__ == "__main__":
    fire.Fire(main)