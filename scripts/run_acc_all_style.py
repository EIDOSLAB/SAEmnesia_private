import subprocess
import fire
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from UnlearnCanvas_resources.const import theme_available

def run_scripts_sequentially(
    themes_to_unlearn, input_dir, output_dir, style_ckpt, class_ckpt, batch_size
):
    accuracy_script = os.path.join(SCRIPT_DIR, "accuracy_unlearncanvas_fast.py")
    
    # Ensure the script exists
    if not os.path.exists(accuracy_script):
        print(f"Error: Script not found at {accuracy_script}")
        print(f"Current directory: {os.getcwd()}")
        return False
    
    base_command = (
        f"PYTHONPATH=. python {accuracy_script} "
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        "--theme '{}' "  
        f"--batch_size {batch_size}"
    )
    
    for theme in themes_to_unlearn:
        command = base_command.format(theme)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for theme '{theme}'"
            )
            break
        else:
            print(f"Successfully completed script for theme '{theme}'")

def main(
    input_dir,
    output_dir,
    style_ckpt,
    class_ckpt,
    batch_size,
    avg_accuracy_input_dir,
):
    run_scripts_sequentially(
        [t for t in theme_available if t != "Seed_Images"],
        input_dir,
        output_dir,
        style_ckpt,
        class_ckpt,
        batch_size,
    )
    
    # Fix the path to the average accuracy script
    avg_script_path = os.path.join(SCRIPT_DIR, "avg_accuracy_style.py")
    
    # Ensure the script exists
    if not os.path.exists(avg_script_path):
        print(f"Error: Average accuracy script not found at {avg_script_path}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    process = subprocess.run(
        f"PYTHONPATH=. python {avg_script_path} '{avg_accuracy_input_dir}'",  # Using avg_accuracy_input_dir
        shell=True,
    )

if __name__ == "__main__":  # Fixed the syntax error
    fire.Fire(main)