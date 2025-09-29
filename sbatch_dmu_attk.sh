#!/bin/bash
#SBATCH --job-name=dmu_baseline_evaluation
#SBATCH --output=sbatch_output/%j_dmu_baseline_evaluation.out
#SBATCH --error=sbatch_output/%j_dmu_baseline_evaluation.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=384G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # Increased CPU cores per task
#SBATCH --account=IscrC_SAOU

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Running UnlearnDiffAtk."

# Bash loop over variable $i that goes from 0 to 141
cd Diffusion-MU-Attack

python run_atk_all_cls.py --attack_idx 0 --eval_seed 42 --class_params_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/class_params.pth --sampling_step_num 100

# for i in {0..141}
# do
#     echo "Running attack_idx: $i"
#     python run_atk_all_cls.py \
#         --eval_seed 42 \
#         --attack_idx $i \
#         --class_params_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/class_params_uniform_99.999_-1.0.pth \
#         --sampling_step_num 100 \
# done
# 

echo "UnlearnDiffAtk completed."

echo "Computing average accuracy difference."

cd ..

python scripts/avg_accuracy_cls_diffatk.py \
    --input_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diffatk_eval/baseline \
    --attk_idxs [0]

# echo "UnlearnDiffAtk completed."
#     --attk_idxs $(seq 0 141)

# --attack_idx <idx> \



# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"