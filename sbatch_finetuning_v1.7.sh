#!/bin/bash
#SBATCH --job-name=v1.7_ft
#SBATCH --output=sbatch_output/%j_fine_tuning-v1.7.out
#SBATCH --error=sbatch_output/%j_fine_tuning_v1.7.err
#SBATCH --account=iscrc_SAOU
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Use a location with more disk space - typically /leonardo_work has more quota than /leonardo_scratch/fast
LARGE_CACHE_BASE="/leonardo_work/IscrC_MAGNIFY/cassano/temp_cache"

# Increase NCCL timeout and add debugging
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Add CUDA memory management
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Redirect ALL temporary files to the larger storage location
export TMPDIR="${LARGE_CACHE_BASE}/tmp"
export TMP="${LARGE_CACHE_BASE}/tmp"
export TEMP="${LARGE_CACHE_BASE}/tmp"

# HuggingFace cache directories
export HF_DATASETS_CACHE="${LARGE_CACHE_BASE}/hf_datasets"
export HF_DATASETS_DOWNLOADED_DATASETS_PATH="${LARGE_CACHE_BASE}/hf_datasets/downloads"
export HF_HOME="${LARGE_CACHE_BASE}/hf_home"
export TRANSFORMERS_CACHE="${LARGE_CACHE_BASE}/transformers"
export HF_HUB_CACHE="${LARGE_CACHE_BASE}/hf_hub"

# Torch and PyTorch caches
export TORCH_HOME="${LARGE_CACHE_BASE}/torch"
export TORCH_CACHE="${LARGE_CACHE_BASE}/torch_cache"

# Python caches
export PYTHONPYCACHEPREFIX="${LARGE_CACHE_BASE}/pycache"
export PYTHONDONTWRITEBYTECODE=1  # Disable .pyc file creation

# Weights & Biases
export WANDB_MODE="offline"
export WANDB_DIR="${LARGE_CACHE_BASE}/wandb"
export WANDB_CACHE_DIR="${LARGE_CACHE_BASE}/wandb_cache"

# Additional environment variables for popular libraries that create temp files
export MPLCONFIGDIR="${LARGE_CACHE_BASE}/matplotlib"
export NUMBA_CACHE_DIR="${LARGE_CACHE_BASE}/numba"
export JUPYTER_RUNTIME_DIR="${LARGE_CACHE_BASE}/jupyter"

# PyArrow (used by datasets library)
export ARROW_TMPDIR="${LARGE_CACHE_BASE}/arrow_tmp"

# CUDA cache (if using GPU compilation)
export CUDA_CACHE_PATH="${LARGE_CACHE_BASE}/cuda_cache"

# FIXED: Allow datasets to use the large cache directory for shuffle operations
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE_MAX_SIZE="100GB"  # Set reasonable cache limit

# Create all cache directories
mkdir -p $TMPDIR $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $WANDB_DIR 
mkdir -p $HF_HOME $TORCH_HOME $PYTHONPYCACHEPREFIX $MPLCONFIGDIR 
mkdir -p $NUMBA_CACHE_DIR $JUPYTER_RUNTIME_DIR $ARROW_TMPDIR $CUDA_CACHE_PATH
mkdir -p $HF_HUB_CACHE $WANDB_CACHE_DIR $TORCH_CACHE

# Set permissions
chmod -R 755 ${LARGE_CACHE_BASE}

# Clean up any existing temporary files first
echo "Cleaning up existing temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*.lock" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*partial*" -delete 2>/dev/null || true

# Check available disk space in both locations
echo "Checking disk space:"
echo "Source data location:"
df -h /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/
echo "Temporary files location:"
df -h /leonardo_work/IscrC_MAGNIFY/cassano/

# Name of the NEW dual concept Python script
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sae_finetuning_v1.7.py"

# Path to SAE checkpoint directory
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/unet.up_blocks.1.attentions.1"

# Directory containing concept activations WITH STYLE RECOVERY METADATA
# This should contain the recovered_object_to_style_index.json file in metadata/
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects"

# JSON file paths for SEPARATE object and style scores
OBJECT_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json"
STYLE_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/non_finetuned/scores.json"

# Directory to save models and logs
SAVE_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/dual_concept_optimized/v1.7/ce_weight_3.0_sparsity_0.01"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p sbatch_output

# Verify that the required files exist
echo "Verifying required files..."
if [ ! -f "${SCRIPT_NAME}" ]; then
    echo "ERROR: Script not found at ${SCRIPT_NAME}"
    exit 1
fi

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: SAE checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -d "${ACTIVATIONS_DIR}" ]; then
    echo "ERROR: Activations directory not found at ${ACTIVATIONS_DIR}"
    exit 1
fi

# Check for style recovery metadata (corrected path for hookpoint structure)
METADATA_PATH="${ACTIVATIONS_DIR}/unet.up_blocks.1.attentions.1/metadata/recovered_object_to_style_index.json"
if [ ! -f "${METADATA_PATH}" ]; then
    echo "ERROR: Style recovery metadata not found at ${METADATA_PATH}"
    echo "Please run style recovery first!"
    exit 1
fi

if [ ! -f "${OBJECT_SCORES_JSON_PATH}" ]; then
    echo "ERROR: Object scores JSON not found at ${OBJECT_SCORES_JSON_PATH}"
    exit 1
fi

if [ ! -f "${STYLE_SCORES_JSON_PATH}" ]; then
    echo "ERROR: Style scores JSON not found at ${STYLE_SCORES_JSON_PATH}"
    exit 1
fi

echo "✅ All required files found!"

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

# Run training with the NEW dual concept script
echo "Running dual object-style concept training..."
echo "Object scores: ${OBJECT_SCORES_JSON_PATH}"
echo "Style scores: ${STYLE_SCORES_JSON_PATH}"
echo "Activations with styles: ${ACTIVATIONS_DIR}"
echo "Cache directory: ${HF_DATASETS_CACHE}"

torchrun --nproc_per_node=4 ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --object_scores_json_path ${OBJECT_SCORES_JSON_PATH} \
    --style_scores_json_path ${STYLE_SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 150 \
    --reconstruction_weight 1.0 \
    --cross_entropy_weight 3.0 \
    --sparsity_weight 0.01 \
    --batch_size 128 \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --validation_split 0.2 \
    --mixed_batches \
    --num_gpus 4 \
    --gradient_accumulation_steps 1 \
    --mixed_precision \
    --patience 5 \
    --resume \
    --use_float16


# --resume \
# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $?"
fi

# Clean up temporary files after completion
echo "Cleaning up temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*.lock" -delete 2>/dev/null || true

# Optionally, remove the entire temp cache directory if you want to save space
# Uncomment the next line if you want to clean everything after the job
# rm -rf ${LARGE_CACHE_BASE}

echo "Job completed at $(date)"
echo "Results have been saved to: ${SAVE_DIR}"
echo "Final disk usage:"
df -h /leonardo_work/IscrC_MAGNIFY/cassano/
df -h /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/

# Print summary of what was trained
echo ""
echo "=== TRAINING SUMMARY ==="
echo "Script: ${SCRIPT_NAME}"
echo "SAE Checkpoint: ${CHECKPOINT_PATH}"
echo "Activations (with styles): ${ACTIVATIONS_DIR}"
echo "Object Scores: ${OBJECT_SCORES_JSON_PATH}"
echo "Style Scores: ${STYLE_SCORES_JSON_PATH}"
echo "Output Directory: ${SAVE_DIR}"
echo "Training Type: Dual Object-Style Concept Assignment"
echo "======================="