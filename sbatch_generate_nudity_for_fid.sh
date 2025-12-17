#!/bin/bash
#SBATCH --job-name=coco_gen_sae
#SBATCH --output=sbatch_output/%j_coco_gen_sae.out
#SBATCH --error=sbatch_output/%j_coco_gen_sae.err
#SBATCH --account=IscrC_MAGNIFY
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Create output directory
mkdir -p sbatch_output

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# Print some info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi
echo ""

# Paths
SD_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4"
SAE_PATH="/leonardo_scratch/fast/IscrC_SAOU/sae_checkpoints/nudity/finetuned/v1.6/"
CAPTIONS_PATH="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/coco_30k_captions.json"
OUTPUT_DIR="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/generated_with_sae"
TARGET_HOOKPOINT="unet.up_blocks.1.attentions.1"

# Print configuration
echo "Configuration:"
echo "  SD Model: $SD_PATH"
echo "  SAE Path: $SAE_PATH"
echo "  Captions: $CAPTIONS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Hookpoint: $TARGET_HOOKPOINT"
echo ""

# Run generation
echo "Starting COCO image generation with SAE..."
python scripts/fid_nudity_images_generation.py \
    --captions_path "$CAPTIONS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sd_path "$SD_PATH" \
    --sae_path "$SAE_PATH" \
    --target_hookpoint "$TARGET_HOOKPOINT" \
    --checkpoint_type best \
    --num_steps 100 \
    --guidance_scale 9 \
    --seed 42 \
    --device cuda

echo ""
echo "Job finished at: $(date)"