import subprocess
import fire
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import class_available


def run_scripts_sequentially(
    classes_to_unlearn,
    multiplier,
    percentile,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
    script_version="v2",  # NEW: Choose between original and v2 scripts
):
    """
    Run accuracy calculation for each class sequentially.
    
    Args:
        script_version: "v1" for original scripts, "v2" for CCM-enabled scripts
    """
    # Select which script to use
    if script_version == "v2":
        accuracy_script = "accuracy_unlearncanvas_cls_sweep_fast_v2.py"
        avg_script = "avg_accuracy_cls_sweep_v2.py"
    else:
        accuracy_script = "accuracy_unlearncanvas_cls_sweep_fast.py"
        avg_script = "avg_accuracy_cls_sweep.py"
    
    base_command = (
        f"PYTHONPATH=. python scripts/{accuracy_script} "
        f"--input_dir '{input_dir_base}/percentile_{percentile}_multiplier_{multiplier}/' "
        f"--output_dir '{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/' "
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
    
    # Run average calculation
    print(f"\nCalculating averages using {avg_script}...")
    avg_command = (
        f"PYTHONPATH=. python scripts/{avg_script} "
        f"'{output_dir_base}/percentile_{percentile}_multiplier_{multiplier}/'"
    )
    process = subprocess.run(avg_command, shell=True)
    
    if process.returncode != 0:
        print("Error: Failed to run average accuracy calculation")
    else:
        print("Successfully completed average calculation")


def main(
    multipliers,
    percentiles,
    input_dir_base,
    output_dir_base,
    class_ckpt,
    batch_size,
    seed,
    script_version="v2",  # NEW: Default to v2 (CCM-enabled)
):
    """
    Main function to run the complete accuracy sweep.
    
    Args:
        multipliers: List of multipliers to evaluate
        percentiles: List of percentiles to evaluate
        input_dir_base: Base directory containing generated images
        output_dir_base: Base directory for output metrics
        class_ckpt: Path to classifier checkpoint
        batch_size: Batch size for evaluation
        seed: Random seed(s) used for generation
        script_version: "v1" for original, "v2" for CCM-enabled (default: "v2")
    """
    print(f"Using script version: {script_version}")
    print(f"CCM metric: {'ENABLED' if script_version == 'v2' else 'DISABLED'}")
    print("="*80)
    
    for multiplier in multipliers:
        for percentile in percentiles:
            print(f"\nProcessing: percentile={percentile}, multiplier={multiplier}")
            print("-"*80)
            
            run_scripts_sequentially(
                class_available,
                multiplier,
                percentile,
                input_dir_base,
                output_dir_base,
                class_ckpt,
                batch_size,
                seed,
                script_version=script_version,
            )
    
    print("\n" + "="*80)
    print("All accuracy calculations complete!")
    print("="*80)


if __name__ == "__main__":
    fire.Fire(main)