import os
import fire
import torch
from tqdm import tqdm
from UnlearnCanvas_resources.const import class_available


def main(input_dir: str):
    metrics = {}
    progress_bar = tqdm(class_available, desc="Processing classes")
    
    for cls in progress_bar:
        progress_bar.set_description(f"Processing class {cls}")
        data_class = torch.load(os.path.join(input_dir, f"{cls}_cls.pth"))
        acc_data_class = data_class["acc"]
        
        # Calculate UA for this class
        ua = 1 - acc_data_class[cls]
        
        # Calculate IRA for this class
        curr_avg_ira = 0.0
        for cls_to_compare in class_available:
            if cls_to_compare != cls:
                curr_avg_ira += acc_data_class[cls_to_compare]
        ira = curr_avg_ira / (len(class_available) - 1)
        
        # NEW: Calculate CCM (Classifier Confidence Metric)
        # We want high confidence in predictions (high max_confidence, low entropy, high top2_gap)
        # when the class is NOT the target class being unlearned
        
        # Get confidence metrics (these are already averaged per class in the previous script)
        max_conf_data = data_class["max_confidence"]
        entropy_data = data_class["entropy"]
        top2_gap_data = data_class["top2_gap"]
        
        # Calculate average confidence for non-target classes
        # (we want these to be confidently classified as something else)
        total_max_conf = 0.0
        total_entropy = 0.0
        total_top2_gap = 0.0
        count = 0
        
        for cls_to_compare in class_available:
            if cls_to_compare != cls:
                total_max_conf += max_conf_data[cls_to_compare]
                total_entropy += entropy_data[cls_to_compare]
                total_top2_gap += top2_gap_data[cls_to_compare]
                count += 1
        
        avg_max_conf = total_max_conf / count
        avg_entropy = total_entropy / count
        avg_top2_gap = total_top2_gap / count
        
        # CCM: Combine metrics
        # High confidence is good (weight positively)
        # Low entropy is good (weight negatively) - normalize entropy to 0-1 range
        # High top2 gap is good (weight positively)
        # Entropy for uniform distribution over N classes: log(N)
        max_entropy = torch.log(torch.tensor(len(class_available))).item()
        normalized_entropy = avg_entropy / max_entropy  # 0 to 1
        
        # CCM formula: combines the three aspects
        # We want: high confidence + low entropy + high decisiveness
        ccm = (avg_max_conf + (1 - normalized_entropy) + avg_top2_gap) / 3
        
        metrics[cls] = {
            "UA": ua * 100,  # Convert to percentage
            "IRA": ira * 100,  # Convert to percentage
            "CCM": ccm * 100,  # Convert to percentage (0-100 scale)
            # Store individual components for analysis
            "max_confidence": avg_max_conf * 100,
            "entropy": avg_entropy,
            "normalized_entropy": normalized_entropy * 100,
            "top2_gap": avg_top2_gap * 100,
        }
    
    # Calculate averages
    avg_ua = sum(m["UA"] for m in metrics.values()) / len(class_available)
    avg_ira = sum(m["IRA"] for m in metrics.values()) / len(class_available)
    avg_ccm = sum(m["CCM"] for m in metrics.values()) / len(class_available)
    avg_max_conf = sum(m["max_confidence"] for m in metrics.values()) / len(class_available)
    avg_entropy = sum(m["entropy"] for m in metrics.values()) / len(class_available)
    avg_norm_entropy = sum(m["normalized_entropy"] for m in metrics.values()) / len(class_available)
    avg_top2_gap = sum(m["top2_gap"] for m in metrics.values()) / len(class_available)
    
    # Add averages to metrics
    metrics["average"] = {
        "UA": avg_ua,
        "IRA": avg_ira,
        "CCM": avg_ccm,
        "max_confidence": avg_max_conf,
        "entropy": avg_entropy,
        "normalized_entropy": avg_norm_entropy,
        "top2_gap": avg_top2_gap,
    }
    
    # Save metrics to file
    output_path = os.path.join(input_dir, "class_metrics.pth")
    torch.save(metrics, output_path)
    
    print(f"{input_dir=}")
    print(f"Average UA:  {avg_ua:.2f}%")
    print(f"Average IRA: {avg_ira:.2f}%")
    print(f"Average CCM: {avg_ccm:.2f}%")
    print(f"  - Max Confidence: {avg_max_conf:.2f}%")
    print(f"  - Normalized Entropy: {avg_norm_entropy:.2f}%")
    print(f"  - Top-2 Gap: {avg_top2_gap:.2f}%")
    print(f"Detailed metrics saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)