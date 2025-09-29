import subprocess
import fire

def run_scripts_sequentially(
    objects_to_unlearn, input_dir, output_dir, style_ckpt, class_ckpt, batch_size
):
    base_command = (
        "PYTHONPATH=. python scripts/accuracy_unlearncanvas_cls_fast.py "  # Updated script name
        f"--input_dir '{input_dir}' "
        f"--output_dir '{output_dir}' "
        f"--style_ckpt '{style_ckpt}' "
        f"--class_ckpt '{class_ckpt}' "
        "--cls '{}' "  # Changed to --cls to match the actual parameter
        f"--batch_size {batch_size}"
    )
    
    for obj in objects_to_unlearn:
        command = base_command.format(obj)
        print(f"Running command: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            print(
                f"Error: Script failed with return code {process.returncode} for object group '{obj}'"
            )
            break
        else:
            print(f"Successfully completed script for object group '{obj}'")

def main(input_dir, output_dir, style_ckpt, class_ckpt, batch_size):
    # Sequential objects to unlearn (matching the generation script)
    sequential_objects_to_unlearn = [
        ["Bears"],
        ["Bears", "Cats"],
        ["Bears", "Cats", "Flowers"],
        ["Bears", "Cats", "Flowers", "Frogs"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches"],
        ["Bears", "Cats", "Flowers", "Frogs", "Jellyfish", "Sea", "Statues", "Sandwiches", "Waterfalls"],
    ]
    
    run_scripts_sequentially(
        ["_".join(s) for s in sequential_objects_to_unlearn],
        input_dir,
        output_dir,
        style_ckpt,
        class_ckpt,
        batch_size,
    )
    
    # Run averaging script for object sequential results
    subprocess.run(
        f"PYTHONPATH=. python scripts/avg_accuracy_object_sequential.py '{output_dir}'",
        shell=True,
    )

if __name__ == "__main__":
    fire.Fire(main)