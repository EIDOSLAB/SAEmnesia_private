#!/bin/bash
#SBATCH --job-name=benchmark_baseline
#SBATCH --output=benchmark_output/baseline_%j.out
#SBATCH --error=benchmark_output/baseline_%j.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "=== BENCHMARK MODE - BASELINE METHOD ==="
echo "Starting at $(date)"

# Set benchmark mode environment variables
export BENCHMARK_MODE=1
export BENCHMARK_STEPS=100

# Use a location with more disk space
LARGE_CACHE_BASE="/leonardo_work/IscrC_MAGNIFY/cassano/temp_cache"

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
export PYTHONDONTWRITEBYTECODE=1

# Weights & Biases
export WANDB_MODE="offline"
export WANDB_DIR="${LARGE_CACHE_BASE}/wandb"
export WANDB_CACHE_DIR="${LARGE_CACHE_BASE}/wandb_cache"

# Additional environment variables
export MPLCONFIGDIR="${LARGE_CACHE_BASE}/matplotlib"
export NUMBA_CACHE_DIR="${LARGE_CACHE_BASE}/numba"
export JUPYTER_RUNTIME_DIR="${LARGE_CACHE_BASE}/jupyter"
export ARROW_TMPDIR="${LARGE_CACHE_BASE}/arrow_tmp"
export CUDA_CACHE_PATH="${LARGE_CACHE_BASE}/cuda_cache"

# FIXED: Allow datasets to use the large cache directory for shuffle operations
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE_MAX_SIZE="100GB"

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

# Paths - YOUR ACTUAL PATHS
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/timed_train.py"
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects"
HOOKPOINT="unet.up_blocks.1.attentions.1"
SAVE_DIR="./benchmark_results/baseline"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p benchmark_output

# Verify files exist
echo "Verifying required files..."
if [ ! -f "${SCRIPT_NAME}" ]; then
    echo "ERROR: Script not found at ${SCRIPT_NAME}"
    exit 1
fi

if [ ! -d "${ACTIVATIONS_DIR}" ]; then
    echo "ERROR: Activations directory not found at ${ACTIVATIONS_DIR}"
    exit 1
fi

# Check if the hookpoint directory exists in activations
if [ ! -d "${ACTIVATIONS_DIR}/${HOOKPOINT}" ]; then
    echo "ERROR: Hookpoint directory not found at ${ACTIVATIONS_DIR}/${HOOKPOINT}"
    echo "Available directories in ${ACTIVATIONS_DIR}:"
    ls -d ${ACTIVATIONS_DIR}/unet.* 2>/dev/null | head -5
    exit 1
fi

echo "✅ All required files found!"

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

echo "Running benchmark for BASELINE method..."
echo "Will measure 100 steps after 10 warmup steps"
echo "Effective batch size: 4096"
echo "Hookpoint: ${HOOKPOINT}"
echo ""

# Run baseline training script with proper parameters
python ${SCRIPT_NAME} \
    --dataset_path ${ACTIVATIONS_DIR} \
    --hookpoints ${HOOKPOINT} \
    --effective_batch_size 4096 \
    --auxk_alpha 0.03125 \
    --expansion_factor 16 \
    --k 32 \
    --multi_topk False \
    --num_workers 4 \
    --wandb_log_frequency 4000 \
    --num_epochs 1 \
    --dead_feature_threshold 10000000 \
    --lr 4e-4 \
    --lr_scheduler linear \
    --lr_warmup_steps 0 \
    --batch_topk True \
    --log_to_wandb False \
    --save_every 0 \
    --run_name baseline_benchmark \
    --seed 42

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    echo "Results saved to: sae-ckpts/"
    find sae-ckpts -name "benchmark_results_*.json" -exec ls -lh {} \;
else
    echo "❌ Benchmark failed with exit code: $?"
fi

# Clean up temporary files after completion
echo "Cleaning up temporary files..."
find ${LARGE_CACHE_BASE} -name "*.tmp" -delete 2>/dev/null || true
find ${LARGE_CACHE_BASE} -name "*.lock" -delete 2>/dev/null || true

echo "Benchmark completed at $(date)"