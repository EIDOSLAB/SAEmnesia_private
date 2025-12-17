#!/bin/bash
#SBATCH --job-name=knn_pipeline
#SBATCH --output=sbatch_output/%j_knn_pipeline.out
#SBATCH --error=sbatch_output/%j_knn_pipeline.err
#SBATCH --account=IscrC_MAGNIFY
#SBATCH --time=00:02:00
#SBATCH --mem=32G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Create output directories
mkdir -p sbatch_output
mkdir -p /leonardo_scratch/fast/IscrC_SAOU/results/knn_evaluation

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# Set paths

CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best"
PIPE_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50"
HOOKPOINT="unet.up_blocks.1.attentions.1"
ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_SAOU/results/activations/saemnesia"
ACTIVATIONS_PATH="${ACTIVATIONS_DIR}/${HOOKPOINT}/"
RESULTS_DIR="/leonardo_scratch/fast/IscrC_SAOU/results/knn_evaluation/saemnesia"

# CHECKPOINT_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best"
# PIPE_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50"
# HOOKPOINT="unet.up_blocks.1.attentions.2"
# ACTIVATIONS_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/"
# ACTIVATIONS_DIR="/leonardo_scratch/fast/IscrC_SAOU/results/activations/baseline_styles"
# ACTIVATIONS_PATH="${ACTIVATIONS_DIR}/${HOOKPOINT}/"
# RESULTS_DIR="/leonardo_scratch/fast/IscrC_SAOU/results/knn_evaluation/baseline_styles"

# Create activations directory
mkdir -p "$ACTIVATIONS_DIR"

# echo "==========================================="
# echo "Step 1: Gathering Activations For Objects"
# echo "==========================================="
# echo "Checkpoint path: $CHECKPOINT_PATH"
# echo "Pipeline path: $PIPE_PATH"
# echo "Hookpoint: $HOOKPOINT"
# echo "Saving activations to: $ACTIVATIONS_PATH"
# echo ""
# 
# python scripts/knn_activations_gathering.py \
#     --checkpoint_path="$CHECKPOINT_PATH" \
#     --hookpoint="$HOOKPOINT" \
#     --pipe_path="$PIPE_PATH" \
#     --save_dir="$ACTIVATIONS_PATH" \
#     --steps=100 \
#     --seed=188
# 
# echo ""
# echo "Activations gathering completed!"
# echo ""


echo "==========================================="
echo "Step 2: Running k-NN Evaluation For Objects"
echo "==========================================="
echo "Activations path: $ACTIVATIONS_PATH"
echo "Hookpoint: $HOOKPOINT"
echo "Results directory: $RESULTS_DIR"
echo ""

python scripts/knn_evaluation.py \
    --activations_path="$ACTIVATIONS_PATH" \
    --hookpoint="$HOOKPOINT" \
    --num_timesteps=100 \
    --results_save_path="$RESULTS_DIR/knn_results.pkl" \
    --plot_save_path="$RESULTS_DIR/figure4_reproduction.pdf" \
    ----precomputed_results_path="/leonardo_scratch/fast/IscrC_SAOU/results/knn_evaluation/saemnesia/knn_results.pkl"

echo ""
echo "==========================================="
echo "PIPELINE COMPLETED!"
echo "==========================================="
echo "Activations saved to: $ACTIVATIONS_PATH"
echo "Results saved to: $RESULTS_DIR"
echo "  - knn_results.pkl"
echo "  - figure4_reproduction.png"
