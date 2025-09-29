#!/bin/bash
#SBATCH --job-name=fid_test
#SBATCH --output=sbatch_output/%j_fid_v4.out
#SBATCH --error=sbatch_output/%j_fid_v4.err
#SBATCH --time=1:00:00              # Increased time limit
#SBATCH --mem=80G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4           # Increased CPU cores per task
#SBATCH --account=IscrC_SAOU  # Replace with your actual account name

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi


echo "Computing FID scores."
python scripts/run_fid_all.py \
--p1 '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/ucanvas_dataset' \
--p2_base '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v4/hp_search' \
--output_path_base '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/fid/fine_tuned/from_scratch/v4/hp_search' \
echo "FID scores computed."

deactivate

echo "Job completed."
echo "End time: $(date)"