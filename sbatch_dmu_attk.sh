#!/bin/bash
#SBATCH --job-name=dmu_atk_c%a
#SBATCH --output=sbatch_output/%j_class_%a_attack_%A.out
#SBATCH --error=sbatch_output/%j_class_%a_attack_%A.err
#SBATCH --time=24:00:00
#SBATCH --mem=96G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --account=IscrC_INSAIT
#SBATCH --array=0-19  # 20 classes (indices 0-19)

# Get parameters
ATTACK_IDX=${1:-0}  # Attack index from command line, default 0
CLASS_IDX=$SLURM_ARRAY_TASK_ID  # Class index from array job

# Load environment
source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================="
echo "Job started at: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Attack Index: $ATTACK_IDX"
echo "Class Index: $CLASS_IDX"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "========================================="
nvidia-smi

cd Diffusion-MU-Attack
# --class_params_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/baseline/class_params.pth \
python run_atk_all_cls.py \
    --eval_seed 42 \
    --attack_idx $ATTACK_IDX \
    --class_params_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_188/class_params.pth \
    --class_idx $CLASS_IDX \
    --sampling_step_num 100

cd ..

# Deactivate the virtual environment
deactivate

echo "========================================="
echo "Job completed at: $(date)"
echo "Class index $CLASS_IDX finished"
echo "========================================="