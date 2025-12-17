#!/bin/bash
#SBATCH --job-name=fid_clip_eval
#SBATCH --output=sbatch_output/%j_fid_clip.out
#SBATCH --error=sbatch_output/%j_fid_clip.err
#SBATCH --time=2:00:00
#SBATCH --mem=380G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=IscrC_SAOU

# Create output directory
mkdir -p sbatch_output

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo ""

# Configuration
REFERENCE_DIR="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/real_images/matched_30k"
GENERATED_DIR="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/generated_with_sae/matched_val_unique"
OUTPUT_DIR="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/evaluation_results"
CAPTIONS_FILE="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/coco_30k_captions.json"

# Print configuration
echo "Configuration:"
echo "  Reference: $REFERENCE_DIR"
echo "  Generated: $GENERATED_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Captions: $CAPTIONS_FILE"
echo ""

# Run evaluation
echo "Starting FID and CLIP evaluation..."
python scripts/nudity_fid_clip_evaluation.py \
    --generated_dir "$GENERATED_DIR" \
    --reference_dir "$REFERENCE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 128 \
    --compute_fid \
    --compute_clip \
    --target_size 299 \
    --captions_file "$CAPTIONS_FILE" \
    --clip_model "ViT-B/32" \

# python scripts/clean_fid.py \
#     --generated_dir "$GENERATED_DIR" \
#     --reference_dir "$REFERENCE_DIR" \

echo ""
echo "Evaluation completed!"
echo "Job finished at: $(date)"

deactivate