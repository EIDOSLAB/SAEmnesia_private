#!/bin/bash
#SBATCH --job-name=styles_sequential_unlearning
#SBATCH --output=sbatch_output/%j_styles_sequential_unlearning.out
#SBATCH --error=sbatch_output/%j_styles_sequential_unlearning.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=384G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # Increased CPU cores per task
#SBATCH --account=IscrC_INSAIT

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

# accelerate launch --num_processes 4 scripts/sample_unlearning_sequential_distr.py \
# --percentile 99.999 \
# --multiplier -1.0 \
# --seed 42 \
# --output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sequential_unlearning/styles' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.2' \
# --style_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.2/style_latents_dict_unet.up_blocks.1.attentions.2.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best' \
# --steps 100
# 
# echo "Image generation completed at: $(date)"
echo "Starting evaluation..."

# Run object evaluation with sequential approach
# Note: The input_dir path will now be different since we removed percentile/multiplier from the path
python scripts/run_acc_all_style_sequential.py \
    --input_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sequential_unlearning/styles/percentile_99.999_multiplier_-1.0' \
    --output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sequential_unlearning/evaluation_results/styles' \
    --style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
    --class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
    --batch_size 128

# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"