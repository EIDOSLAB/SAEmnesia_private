#!/bin/bash
#SBATCH --job-name=v4_dual_post_topk
#SBATCH --output=sbatch_output/%j_resume_v4_dual_post_topk.out
#SBATCH --error=sbatch_output/%j_resume_v4_dual_post_topk.err
#SBATCH --account=iscrc_magnify
#SBATCH --time=24:00:00
#SBATCH --mem=300G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

#--output=sbatch_output/%j_from_scratch_v4_dual_post_topk.out
#--error=sbatch_output/%j_from_scratch_v4_dual_post_topk.err

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

# Name of the Python script (v4 - dual concept version with POST-TOPK BCE)
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sae_finetuning_v4.py"

# Path to SAE checkpoint directory
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/unet.up_blocks.1.attentions.1"

# Directory containing concept activations with style recovery metadata
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects"

# JSON file paths for BOTH object and style scores
OBJECT_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json"
STYLE_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/non_finetuned/scores.json"

# Directory to save models and logs - Updated for v4 dual concept version with POST-TOPK BCE
SAVE_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/dual_concept_optimized/v4_post_topk/ce_weight_3.0_sparsity_0.01"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p sbatch_output

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

# Print configuration for verification
echo "=== DUAL CONCEPT TRAINING CONFIGURATION v4 (POST-TOPK) ==="
echo "Script: ${SCRIPT_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Activations: ${ACTIVATIONS_DIR}"
echo "Object Scores JSON: ${OBJECT_SCORES_JSON_PATH}"
echo "Style Scores JSON: ${STYLE_SCORES_JSON_PATH}"
echo "Save Directory: ${SAVE_DIR}"
echo "Loss Configuration:"
echo "  - Reconstruction Weight: 1.0"
echo "  - Binary Cross-Entropy Weight: 3.0 (applied AFTER topk)"
echo "  - BCE operates on post-topk activations only"
echo "  - Applied to BOTH object and style latents"
echo "  - Sparsity Weight: 0.01 (applied to topk activations)"
echo "Concept Assignment:"
echo "  - Objects get priority boost in latent assignment"
echo "  - Style recovery from metadata used"
echo "  - BCE gradients only on assigned latents in top-k"
echo "Resume Mode: ENABLED (will continue from latest checkpoint)"
echo "Wandb: OFFLINE mode, incremental logging on resume"
echo "Key Difference from v3: BCE loss computed AFTER topk selection"
echo "This ensures BCE only affects actually selected sparse activations"
echo "================================"

# Verify required files exist
echo "=== VERIFICATION ==="
if [ -f "${OBJECT_SCORES_JSON_PATH}" ]; then
    echo "✅ Object scores file found"
else
    echo "❌ Object scores file NOT found: ${OBJECT_SCORES_JSON_PATH}"
    exit 1
fi

if [ -f "${STYLE_SCORES_JSON_PATH}" ]; then
    echo "✅ Style scores file found"
else
    echo "❌ Style scores file NOT found: ${STYLE_SCORES_JSON_PATH}"
    exit 1
fi

# Check for style recovery metadata
METADATA_CHECK="${ACTIVATIONS_DIR}/unet.up_blocks.1.attentions.1/metadata/recovered_object_to_style_index.json"
if [ -f "${METADATA_CHECK}" ]; then
    echo "✅ Style recovery metadata found"
else
    echo "❌ Style recovery metadata NOT found: ${METADATA_CHECK}"
    echo "   Please run style recovery first!"
    exit 1
fi

echo "All required files verified!"
echo "================================"

# Run training with Dual Concept Binary Cross-Entropy loss (POST-TOPK version)
echo "Running DUAL CONCEPT training with Binary Cross-Entropy loss (POST-topk)..."
echo "Loss will be applied to BOTH object and style assigned latents"
echo "BCE operates only on activations that survive topk selection"
echo "Unassigned latents remain completely unaffected"
echo "Shuffle operations will use: ${HF_DATASETS_CACHE}"

torchrun --nproc_per_node=4 ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --scores_json_path ${OBJECT_SCORES_JSON_PATH} \
    --style_scores_json_path ${STYLE_SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 100 \
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
    --resume \
    --patience 5 \
    --use_float16

# Alternative configurations (commented out):

# For training from scratch with random concept assignments:
# --from_scratch 

# For different loss weights:
# --cross_entropy_weight 1.0   # Lower BCE weight
# --cross_entropy_weight 5.0   # Higher BCE weight

# For different batch sizes if memory issues:
# --batch_size 64              # Smaller batch size
# --batch_size 256             # Larger batch size (if memory allows)

echo ""
echo "=== DUAL CONCEPT TRAINING COMPLETED (v4 POST-TOPK) ==="
echo "Binary Cross-Entropy loss was applied AFTER topk selection"
echo "BCE operated only on sparse activations that survived topk"
echo "Loss was applied to BOTH object and style assigned latents"
echo "Unassigned latents were completely unaffected by gradients"
echo "Objects received priority in latent assignment"
echo "Style information recovered from metadata"
echo "Wandb logs are incremental and will continue previous run if resumed"

# Clean up temporary files after completion
echo "Cleaning up temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*.lock" -delete 2>/dev/null || true

# Optionally, remove the entire temp cache directory if you want to save space
# Uncomment the next line if you want to clean everything after the job
# rm -rf ${LARGE_CACHE_BASE}

echo "Job completed at $(date)"
echo "Results have been saved to: ${SAVE_DIR}"
echo "Model checkpoints include incremental saves with optimizer state"
echo "Final disk usage:"
df -h /leonardo_work/IscrC_MAGNIFY/cassano/
df -h /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/

echo ""
echo "=== NEXT STEPS ==="
echo "1. Check training logs in: ${SAVE_DIR}/wandb"
echo "2. Latest model checkpoint: ${SAVE_DIR}/latest/"
echo "3. Epoch-specific checkpoints: ${SAVE_DIR}/epoch_*/"
echo "4. To resume training, simply re-run this script with --resume flag"
echo "5. Wandb logs can be synced when online: wandb sync ${SAVE_DIR}/wandb"
echo "6. Analyze dual concept assignment success rates in logs"
echo "7. Compare object vs style latent activation patterns"
echo "8. Compare v4 (post-topk) vs v3 (pre-topk) BCE performance"

echo ""
echo "=== DUAL CONCEPT SUMMARY v4 (POST-TOPK) ==="
echo "This version trains on BOTH objects and styles simultaneously:"
echo "- Objects: Get priority in latent assignment (score + 1.0 boost)"
echo "- Styles: Assigned to remaining high-scoring latents"
echo "- BCE Loss: Applied AFTER topk selection to sparse activations"
echo "- Target latents: Only encouraged if they appear in top-k"
echo "- Unassigned latents: No gradients, free to learn other patterns"
echo "- Style Recovery: Automatic from metadata, no manual labeling needed"
echo "- Key benefit: More realistic training on actual sparse representation"
echo "- Sparsity loss: Also operates on post-topk activations"