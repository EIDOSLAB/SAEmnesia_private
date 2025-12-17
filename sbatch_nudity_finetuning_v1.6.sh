#!/bin/bash
#SBATCH --job-name=nudity_sae
#SBATCH --output=sbatch_output/%j_nudity_training.out
#SBATCH --error=sbatch_output/%j_nudity_training.err
#SBATCH --account=IscrC_MAGNIFY
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Use a location with more disk space
LARGE_CACHE_BASE="/leonardo_work/IscrC_MAGNIFY/cassano/temp_cache"

# NCCL settings - OPTIMIZED
export NCCL_BLOCKING_WAIT=0
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker

# CUDA memory management - OPTIMIZED
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Temporary files
export TMPDIR="${LARGE_CACHE_BASE}/tmp"
export TMP="${LARGE_CACHE_BASE}/tmp"
export TEMP="${LARGE_CACHE_BASE}/tmp"

# HuggingFace cache directories
export HF_DATASETS_CACHE="${LARGE_CACHE_BASE}/hf_datasets"
export HF_DATASETS_DOWNLOADED_DATASETS_PATH="${LARGE_CACHE_BASE}/hf_datasets/downloads"
export HF_HOME="${LARGE_CACHE_BASE}/hf_home"
export TRANSFORMERS_CACHE="${LARGE_CACHE_BASE}/transformers"
export HF_HUB_CACHE="${LARGE_CACHE_BASE}/hf_hub"

# Torch caches
export TORCH_HOME="${LARGE_CACHE_BASE}/torch"
export TORCH_CACHE="${LARGE_CACHE_BASE}/torch_cache"

# Python caches
export PYTHONPYCACHEPREFIX="${LARGE_CACHE_BASE}/pycache"
export PYTHONDONTWRITEBYTECODE=1

# Weights & Biases
export WANDB_MODE="offline"
export WANDB_DIR="${LARGE_CACHE_BASE}/wandb"
export WANDB_CACHE_DIR="${LARGE_CACHE_BASE}/wandb_cache"

# Additional caches
export MPLCONFIGDIR="${LARGE_CACHE_BASE}/matplotlib"
export NUMBA_CACHE_DIR="${LARGE_CACHE_BASE}/numba"
export JUPYTER_RUNTIME_DIR="${LARGE_CACHE_BASE}/jupyter"
export ARROW_TMPDIR="${LARGE_CACHE_BASE}/arrow_tmp"
export CUDA_CACHE_PATH="${LARGE_CACHE_BASE}/cuda_cache"

# Datasets settings
export HF_DATASETS_OFFLINE=1

# OMP settings - REDUCED to match lower worker count
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Create all cache directories
mkdir -p $TMPDIR $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $WANDB_DIR 
mkdir -p $HF_HOME $TORCH_HOME $PYTHONPYCACHEPREFIX $MPLCONFIGDIR 
mkdir -p $NUMBA_CACHE_DIR $JUPYTER_RUNTIME_DIR $ARROW_TMPDIR $CUDA_CACHE_PATH
mkdir -p $HF_HUB_CACHE $WANDB_CACHE_DIR $TORCH_CACHE

# Set permissions
chmod -R 755 ${LARGE_CACHE_BASE}

# Clean up existing temporary files
echo "Cleaning up existing temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*.lock" -delete 2>/dev/null || true

# Script and paths
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/nudity_sae_finetuning_v1.6.py"
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity/best/unet.up_blocks.1.attentions.1"
ACTIVATIONS_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/activations"
SCORES_JSON_PATH="/leonardo_scratch/fast/IscrC_SAOU/nudity_scores.json"
SAVE_DIR="/leonardo_scratch/fast/IscrC_SAOU/sae_checkpoints/nudity/finetuned/v1.6"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p sbatch_output

# Verify required files (abbreviated)
echo "Verifying required files..."
if [ ! -f "${SCRIPT_NAME}" ]; then
    echo "ERROR: Script not found"
    exit 1
fi

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

# Run training with MEMORY-SAFE settings
echo "Running nudity detection SAE training..."

torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --scores_json_path ${SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 50 \
    --reconstruction_weight 1.0 \
    --cross_entropy_weight 3.0 \
    --sparsity_weight 0.01 \
    --batch_size 32 \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --validation_split 0.2 \
    --num_gpus 4 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --patience 5 \
    --pos_class_weight 1110.0 \
    --max_val_batches 20 \
    --resume


if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $?"
fi

# Clean up
echo "Cleaning up temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true

echo "Job completed at $(date)"