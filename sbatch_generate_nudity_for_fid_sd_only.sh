#!/bin/bash
#SBATCH --job-name=coco_gen_baseline
#SBATCH --output=sbatch_output/%j_coco_gen_baseline.out
#SBATCH --error=sbatch_output/%j_coco_gen_baseline.err
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
CAPTIONS_PATH="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/coco_30k_captions.json"
OUTPUT_DIR="/leonardo_scratch/fast/IscrC_SAOU/FID_nudity/baseline_images"

# Print configuration
echo "Configuration:"
echo "  SD Model: $SD_PATH"
echo "  Captions: $CAPTIONS_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Mode: BASELINE (Vanilla SD, no SAE)"
echo ""

# Run generation in BASELINE mode
echo "Starting COCO image generation with BASELINE SD (no SAE)..."
python scripts/fid_nudity_images_generation.py \
    --mode baseline \
    --captions_path "$CAPTIONS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --sd_path "$SD_PATH" \
    --num_steps 100 \
    --guidance_scale 9 \
    --seed 42 \
    --device cuda

echo ""
echo "Job finished at: $(date)"