#!/bin/bash
#SBATCH --job-name=v1_feature_overlap_analysis
#SBATCH --output=sbatch_output/%j_v1_feature_overlap_analysis.out
#SBATCH --error=sbatch_output/%j_v1_feature_overlap_analysis.err
#SBATCH --time=00:20:00              # Should be enough for the analysis
#SBATCH --mem=128G                   # Less memory needed than full evaluation
#SBATCH --partition=boost_usr_prod   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # Only need 1 GPU for this analysis
#SBATCH --cpus-per-task=4            

# Load any necessary GPU modules (system-specific)
# module load cuda

# Activate your environment
source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Running Feature Overlap Analysis..."

# Set your paths here - adjust these to match your setup
PIPE_CHECKPOINT="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50"  # Use your local SD model
HOOKPOINT="unet.up_blocks.1.attentions.1"
CLASS_LATENTS_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/from_scratch/v1/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl"
SAE_CHECKPOINT="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/object_concept_optimized/v1/ce_weight_3.0_sparsity_0.01/best"
CLASS_PARAMS_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/class_params_uniform_99.999_-1.0.pth"
OUTPUT_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/feature_overlap_results/v1"

# Create output directory
mkdir -p sbatch_output
mkdir -p $OUTPUT_DIR

# Run the feature overlap analysis
python scripts/compute_latents_overlap.py \
    --pipe_checkpoint "$PIPE_CHECKPOINT" \
    --hookpoint "$HOOKPOINT" \
    --class_latents_path "$CLASS_LATENTS_PATH" \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --class_params_path "$CLASS_PARAMS_PATH" \
    --seed 42 \
    --steps 100 \
    --guidance_scale 9.0 \
    --output_dir "$OUTPUT_DIR"

echo "Feature Overlap Analysis completed."
echo "Results saved to: $OUTPUT_DIR"

# List the generated files
echo "Generated files:"
ls -la "$OUTPUT_DIR"

# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"