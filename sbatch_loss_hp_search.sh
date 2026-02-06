#!/bin/bash
#SBATCH --job-name=hyperparam_search
#SBATCH --output=hyperparam_logs/%x_%A.out
#SBATCH --error=hyperparam_logs/%x_%A.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=24:00:00
#SBATCH --mem=380G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8

# Accept command line arguments
BETA=$1
LAMBDA=$2
GAMMA=$3

# Validate arguments
if [ -z "$BETA" ] || [ -z "$LAMBDA" ] || [ -z "$GAMMA" ]; then
    echo "Error: Missing arguments"
    echo "Usage: sbatch run_hyperparam_config.sbatch BETA LAMBDA GAMMA"
    exit 1
fi

CONFIG_NAME="beta${BETA}_lambda${LAMBDA}_gamma${GAMMA}"

echo "=== HYPERPARAMETER CONFIGURATION ==="
echo "Beta (cross_entropy_weight): ${BETA}"
echo "Lambda (sparsity_weight): ${LAMBDA}"
echo "Gamma (orthogonality_weight): ${GAMMA}"
echo "Config name: ${CONFIG_NAME}"
echo "Starting at $(date)"

export BENCHMARK_MODE=0

# NCCL Configuration for Leonardo
export NCCL_TIMEOUT=1800  # 30 minutes in seconds
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_TIMEOUT=22

# PyTorch distributed debugging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export OMP_NUM_THREADS=8

# Set cache directories
LARGE_CACHE_BASE="/leonardo_work/IscrC_MAGNIFY/cassano/temp_cache"

export TMPDIR="${LARGE_CACHE_BASE}/tmp"
export TMP="${LARGE_CACHE_BASE}/tmp"
export TEMP="${LARGE_CACHE_BASE}/tmp"
export HF_DATASETS_CACHE="${LARGE_CACHE_BASE}/hf_datasets"
export HF_DATASETS_DOWNLOADED_DATASETS_PATH="${LARGE_CACHE_BASE}/hf_datasets/downloads"
export HF_HOME="${LARGE_CACHE_BASE}/hf_home"
export TRANSFORMERS_CACHE="${LARGE_CACHE_BASE}/transformers"
export HF_HUB_CACHE="${LARGE_CACHE_BASE}/hf_hub"
export TORCH_HOME="${LARGE_CACHE_BASE}/torch"
export TORCH_CACHE="${LARGE_CACHE_BASE}/torch_cache"
export PYTHONPYCACHEPREFIX="${LARGE_CACHE_BASE}/pycache"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="offline"
export WANDB_DIR="${LARGE_CACHE_BASE}/wandb"
export WANDB_CACHE_DIR="${LARGE_CACHE_BASE}/wandb_cache"
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

chmod -R 755 ${LARGE_CACHE_BASE}

# Paths
SCRIPT_NAME="/leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/timed_sae_finetuning_v1.6.py"
CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.1"
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects"
OBJECT_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json"
STYLE_SCORES_JSON_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/non_finetuned/scores.json"
SAVE_DIR="./hyperparam_search_results/${CONFIG_NAME}"

# Make sure directories exist
mkdir -p ${SAVE_DIR}
mkdir -p hyperparam_logs

# Activate the environment
source ../../envs/saeuron_cassano/bin/activate

# Display GPU info
nvidia-smi

echo ""
echo "Running training..."
echo ""

# Run training with hyperparameters
# Note: removed --nproc_per_node since it's set by torchrun automatically from SLURM
torchrun --nproc_per_node=4 ${SCRIPT_NAME} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --activations_dir ${ACTIVATIONS_DIR} \
    --object_scores_json_path ${OBJECT_SCORES_JSON_PATH} \
    --style_scores_json_path ${STYLE_SCORES_JSON_PATH} \
    --device cuda \
    --learning_rate 5e-6 \
    --num_epochs 5 \
    --reconstruction_weight 1.0 \
    --cross_entropy_weight ${BETA} \
    --sparsity_weight ${LAMBDA} \
    --orthogonality_weight ${GAMMA} \
    --batch_size 32 \
    --save_dir ${SAVE_DIR} \
    --seed 42 \
    --validation_split 0.2 \
    --gradient_accumulation_steps 1 \
    --patience 10

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Results saved to: ${SAVE_DIR}"
else
    echo "❌ Training failed with exit code: $?"
fi

echo "Completed at $(date)"