#!/bin/bash
#SBATCH --job-name=style_v1.6_ft
#SBATCH --output=sbatch_output/%j_style_fine_tuning-v1.6.out
#SBATCH --error=sbatch_output/%j_style_fine_tuning_v1.6.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=24:00:00
#SBATCH --mem=380G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Use a location with more disk space
LARGE_CACHE_BASE="/leonardo_work/IscrC_INSAIT/cassano/temp_cache"

# Increase NCCL timeout and add debugging
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3000  # 30 minutes
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

# CHANGED: Remove HF_DATASETS_OFFLINE - it breaks operations
# export HF_DATASETS_OFFLINE=1  # <-- COMMENTED OUT
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0  # <-- ADDED: Force streaming/memory-mapped mode

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
df -h /leonardo_work/IscrC_INSAIT/
echo "Temporary files location:"
df -h /leonardo_work/IscrC_INSAIT/

# Name of the STYLE-FOCUSED Python script
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sae_styles_finetuning_v1.6.py"

# Path to SAE checkpoint directory - UP.1.2 for style learning
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.2"

# Directory containing STYLE-FOCUSED concept activations WITH STYLE RECOVERY METADATA
# This should contain the recovered_object_to_style_index.json file in metadata/
ACTIVATIONS_DIR="/leonardo_work/IscrC_INSAIT/styles_finetuning_dataset"

# JSON file paths for SEPARATE object and style scores - for up.1.2 block
OBJECT_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/up_1_2/scores.json"
STYLE_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/up_1_2/scores.json"

# Directory to save models and logs - STYLE-FOCUSED
SAVE_DIR="/leonardo_work/IscrC_INSAIT/cassano/saeuron/sae_checkpoints/style_optimized/v1.6/ce_weight_2.0_style_sep_1.5_sparsity_0.01"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p sbatch_output

# Verify that the required files exist
echo "Verifying required files for STYLE-FOCUSED training..."
if [ ! -f "${SCRIPT_NAME}" ]; then
    echo "ERROR: Style-focused script not found at ${SCRIPT_NAME}"
    exit 1
fi

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: SAE checkpoint (up.1.2) not found at ${CHECKPOINT_PATH}"
    exit 1
fi

if [ ! -d "${ACTIVATIONS_DIR}" ]; then
    echo "ERROR: Style activations directory not found at ${ACTIVATIONS_DIR}"
    exit 1
fi

# Check for style recovery metadata (corrected path for hookpoint structure)
METADATA_PATH="${ACTIVATIONS_DIR}/unet.up_blocks.1.attentions.2/metadata/recovered_object_to_style_index.json"
if [ ! -f "${METADATA_PATH}" ]; then
    echo "ERROR: Style recovery metadata not found at ${METADATA_PATH}"
    echo "Please run style recovery first!"
    exit 1
fi

if [ ! -f "${OBJECT_SCORES_JSON_PATH}" ]; then
    echo "WARNING: Object scores JSON not found at ${OBJECT_SCORES_JSON_PATH}"
    echo "Will use random assignment for objects if --from_scratch is used"
fi

if [ ! -f "${STYLE_SCORES_JSON_PATH}" ]; then
    echo "WARNING: Style scores JSON not found at ${STYLE_SCORES_JSON_PATH}"
    echo "Will use random assignment for styles if --from_scratch is used"
fi

echo "✅ Required files verified for style-focused training!"

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

# Run STYLE-FOCUSED training
echo "Running STYLE-FOCUSED SAE training for up.1.2 block..."
echo "Object scores: ${OBJECT_SCORES_JSON_PATH}"
echo "Style scores: ${STYLE_SCORES_JSON_PATH}"
echo "Style-focused activations: ${ACTIVATIONS_DIR}"
echo "Cache directory: ${HF_DATASETS_CACHE}"
echo "Target block: up.1.2 (style-specific)"

torchrun --nproc_per_node=4 ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --object_scores_json_path ${OBJECT_SCORES_JSON_PATH} \
    --style_scores_json_path ${STYLE_SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 150 \
    --reconstruction_weight 1.0 \
    --cross_entropy_weight 2.0 \
    --sparsity_weight 0.01 \
    --batch_size 64 \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --validation_split 0.2 \
    --mixed_batches \
    --num_gpus 4 \
    --gradient_accumulation_steps 1 \
    --mixed_precision \
    --patience 8 \
    --resume \
    --use_float16

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Style-focused training completed successfully!"
else
    echo "❌ Style-focused training failed with exit code: $?"
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
df -h /leonardo_work/IscrC_INSAIT/

# Print summary of what was trained
echo ""
echo "=== STYLE-FOCUSED TRAINING SUMMARY ==="
echo "Script: ${SCRIPT_NAME} (STYLE-OPTIMIZED)"
echo "SAE Checkpoint: ${CHECKPOINT_PATH} (up.1.2 - STYLE BLOCK)"
echo "Activations (style-focused): ${ACTIVATIONS_DIR}"
echo "Object Scores: ${OBJECT_SCORES_JSON_PATH}"
echo "Style Scores: ${STYLE_SCORES_JSON_PATH}"
echo "Output Directory: ${SAVE_DIR}"
echo "Training Type: STYLE-PRIORITY Concept Assignment"
echo "Block Target: up.1.2 (Style-specific layer)"
echo "Key Features: Style separation loss, weighted CE loss, style priority"
echo "=========================================="