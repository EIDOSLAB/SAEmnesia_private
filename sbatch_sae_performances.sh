#!/bin/bash
#SBATCH --job-name=sae_analysis
#SBATCH --output=%j_sae_analysis.out
#SBATCH --error=%j_sae_analysis.err
#SBATCH --account=iscrc_magnify
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Name of the Python script
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/sae_performances.py"

# Path to SAE checkpoint directory
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/unet.up_blocks.1.attentions.2"

# Path to style activations dictionary file
STYLE_ACTIVATIONS_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.2/style_activations_dict_unet.up_blocks.1.attentions.2.pkl"

# Directory to save analysis outputs
OUTPUT_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_analysis/$(date +%Y%m%d_%H%M%S)"

# Directory for wandb logs (offline mode)
WANDB_DIR="${OUTPUT_DIR}/wandb_logs"

# Name for this analysis run
RUN_NAME="theme_latent_analysis_$(date +%Y%m%d_%H%M%S)"

# Make sure directories exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

# Install safetensors if needed
pip install safetensors --no-index --find-links=/leonardo/home/userexternal/ecassano/.pip_packages/ || echo "Warning: Could not install safetensors from local directory"

# Set PyTorch CUDA memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set WANDB_MODE to offline explicitly in the environment
export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR}

# Copy the script to the output directory for reference
cp ${SCRIPT_NAME} ${OUTPUT_DIR}/script_used.py

# Run the analysis script with memory optimization parameters
python ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --style_activations_path ${STYLE_ACTIVATIONS_PATH} \
    --device cuda \
    --output_dir ${OUTPUT_DIR} \
    --wandb_project "sae_theme_latent_analyzer" \
    --run_name ${RUN_NAME} \
    --seed 42 \
    --batch_size 4 \
    --analyze_timesteps \
    # --max_samples 200 \
    # --memory_efficient

echo "Analysis job completed at $(date)"
echo "Analysis outputs have been saved to: ${OUTPUT_DIR}"
echo "Wandb logs have been saved to: ${WANDB_DIR}"
echo "After getting internet access, sync the logs with: wandb sync ${WANDB_DIR}"