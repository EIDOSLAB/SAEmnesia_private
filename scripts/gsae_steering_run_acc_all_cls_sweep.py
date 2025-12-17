import subprocess
import fire
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available


def run_scripts_sequentially(
    classes_to_unlearn,
    alpha,
    percentile,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_cls_sweep_fast.py "
        f"--input_dir '{input_dir_base}/percentile_{percentile}_alpha_{alpha}/' "
        f"--output_dir '{output_dir_base}/percentile_{percentile}_alpha_{alpha}/' "
        f"--class_ckpt '{class_ckpt}' "
        f"--cls '{{}}' --batch_size {batch_size} --seed [{seed}]"
    )
    
    for cls in classes_to_unlearn:
        command = base_command.format(cls)
        print(f"Running command: {command}")
        
        process = subprocess.run(command, shell=True)
        
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for cls '{cls}'"
            )
            break
        else:
            print(f"Successfully completed script for cls '{cls}'")


def main(
    alphas,
    percentiles,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
):
    """
    Run accuracy evaluation for decoder-based steering results.
    
    Args:
        alphas: List of alpha values (e.g., [-0.1, -0.2, -0.5, -1.0])
        percentiles: List of percentile values (e.g., [99.99, 99.995, 99.999])
        input_dir_base: Base directory containing generated images
        output_dir_base: Base directory for accuracy results
        class_ckpt: Path to classifier checkpoint
        batch_size: Batch size for evaluation
        seed: Random seed
    """
    for alpha in alphas:
        for percentile in percentiles:
            run_scripts_sequentially(
                class_available,
                alpha,
                percentile,
                input_dir_base,
                output_dir_base,
                class_ckpt,
                batch_size,
                seed,
            )
            
            process = subprocess.run(
                "PYTHONPATH=. python scripts/avg_accuracy_cls_sweep.py "
                f"'{output_dir_base}/percentile_{percentile}_alpha_{alpha}/'",
                shell=True,
            )
            
            if process.returncode != 0:
                print("Error: Failed to run average accuracy calculation")


if __name__ == "__main__":
    fire.Fire(main)