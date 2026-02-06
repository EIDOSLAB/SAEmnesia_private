#!/bin/bash
#SBATCH --job-name=benchmark_new_method
#SBATCH --output=benchmark_output/new_method_%j.out
#SBATCH --error=benchmark_output/new_method_%j.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=02:00:00
#SBATCH --mem=100G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

echo "=== BENCHMARK MODE - NEW METHOD ==="
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

# Create all cache directories
mkdir -p $TMPDIR $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $WANDB_DIR 
mkdir -p $HF_HOME $TORCH_HOME $PYTHONPYCACHEPREFIX $MPLCONFIGDIR 
mkdir -p $NUMBA_CACHE_DIR $JUPYTER_RUNTIME_DIR $ARROW_TMPDIR $CUDA_CACHE_PATH
mkdir -p $HF_HUB_CACHE $WANDB_CACHE_DIR $TORCH_CACHE

# Set permissions
chmod -R 755 ${LARGE_CACHE_BASE}

# Paths
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/timed_sae_finetuning_v1.6.py"
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.1"
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects"
OBJECT_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json"
STYLE_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/non_finetuned/scores.json"
SAVE_DIR="./benchmark_results/new_method"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p benchmark_output

# Verify files exist
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

echo "✅ All required files found!"

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

echo "Running benchmark for NEW method..."
echo "Will measure 100 steps after 10 warmup steps"
echo "Batch size: 32 (for fair comparison with baseline)"
echo ""

# Run with SINGLE GPU (no torchrun for fair comparison)
python ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --object_scores_json_path ${OBJECT_SCORES_JSON_PATH} \
    --style_scores_json_path ${STYLE_SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 1 \
    --reconstruction_weight 1.0 \
    --cross_entropy_weight 1.0 \
    --sparsity_weight 0.01 \
    --batch_size 32 \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --validation_split 0.2 \
    --gradient_accumulation_steps 1 \
    --patience 5

# Check if benchmark completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Benchmark completed successfully!"
    echo "Results saved to: ${SAVE_DIR}"
    ls -lh ${SAVE_DIR}/benchmark_results_*.json
else
    echo "❌ Benchmark failed with exit code: $?"
fi

echo "Benchmark completed at $(date)"