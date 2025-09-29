#!/bin/bash
#SBATCH --job-name=fs_baseline_sequential_unlearning
#SBATCH --output=sbatch_output/%j_fs_baseline_sequential_unlearning_hp_parameters.out
#SBATCH --error=sbatch_output/%j_fs_baseline_sequential_unlearning_hp_parameters.err
#SBATCH --time=10:00:00              # Increased time limit
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
echo "Start time: $(date)"

## OBJECTS: unet.up_blocks.1.attentions.1
## STYLES: unet.up_blocks.1.attentions.2

# Run object unlearning with sequential approach using class-specific parameters
accelerate launch --num_processes 4 scripts/cls_sample_unlearning_sequential_distr.py \
    --seed 42 \
    --output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sequential_unlearning/baseline' \
    --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
    --hookpoint 'unet.up_blocks.1.attentions.1' \
    --cls_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/non_finetuned/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
    --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best' \
    --class_params_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/class_params.pth' \
    --steps 100

echo "Image generation completed at: $(date)"
echo "Starting evaluation..."

# Run object evaluation with sequential approach
# Note: The input_dir path will now be different since we removed percentile/multiplier from the path
python scripts/run_acc_all_cls_sequential.py \
    --input_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sequential_unlearning/baseline' \
    --output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sequential_unlearning/evaluation_results/baseline' \
    --style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
    --class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
    --batch_size 32

# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"