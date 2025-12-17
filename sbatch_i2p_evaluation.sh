#!/bin/bash
#SBATCH --job-name=i2p_eval
#SBATCH --output=sbatch_output/%j_i2p_eval.out
#SBATCH --error=sbatch_output/%j_i2p_eval.err
#SBATCH --account=IscrC_MAGNIFY
#SBATCH --time=00:05:00
#SBATCH --mem=32G
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

# Paths
SD_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4"
I2P_DATASET="/leonardo_scratch/fast/IscrC_SAOU/i2p_dataset"
# SAE_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity"
SAE_PATH="/leonardo_scratch/fast/IscrC_SAOU/sae_checkpoints/nudity/finetuned/v1.6/"
# NUDITY_LATENTS="/leonardo_scratch/fast/IscrC_SAOU/nudity_activations/nudity_latents_dict_unet.up_blocks.1.attentions.1.pkl"
NUDITY_LATENTS="/leonardo_scratch/fast/IscrC_SAOU/nudity_activations/finetuned/v1.6/nudity_latents_dict_unet.up_blocks.1.attentions.1.pkl"
OUTPUT_DIR="/leonardo_scratch/fast/IscrC_SAOU/i2p_evaluation_results"
TARGET_HOOKPOINT="unet.up_blocks.1.attentions.1"

# Evaluation mode - choose one:
# MODE="baseline"          # Baseline only (SD1.4, no SAE)
# MODE="with_sae"          # With SAE unlearning (percentile=99.0, multiplier=-10)
# MODE="test"              # Test on small subset (10 images)
# MODE="hyperparam_sweep"  # Sweep multipliers with percentile=99.999 (100 images each)
MODE="final_eval"        # Full evaluation with best multiplier (4703 images)

echo "========================================================================"
echo "Evaluation mode: $MODE"
echo "========================================================================"

if [ "$MODE" = "baseline" ]; then
    echo "Running BASELINE evaluation (SD1.4 without SAE)..."
    echo "Output: Armpits | Belly | Buttocks | Feet | Breasts(F) | Genitalia(F) | Breasts(M) | Genitalia(M) | Total"
    echo ""
    
    python scripts/i2p_eval.py \
        --method_name "SD1.4 Baseline" \
        --sd_path "$SD_PATH" \
        --i2p_dataset "$I2P_DATASET" \
        --output_dir "${OUTPUT_DIR}/baseline" \
        --baseline_only \
        --num_steps 100 \
        --guidance_scale 9 \
        --seed 42 \
        --device cuda
        
elif [ "$MODE" = "with_sae" ]; then
    echo "Running evaluation WITH SAE UNLEARNING..."
    echo "Parameters: percentile=99.0, multiplier=-10.0"
    echo "Output: Armpits | Belly | Buttocks | Feet | Breasts(F) | Genitalia(F) | Breasts(M) | Genitalia(M) | Total"
    echo ""
    
    python scripts/i2p_eval.py \
        --method_name "SAE Unlearning p99 m-10" \
        --sd_path "$SD_PATH" \
        --i2p_dataset "$I2P_DATASET" \
        --output_dir "${OUTPUT_DIR}/sae_p99_m-10" \
        --sae_path "$SAE_PATH" \
        --nudity_latents_path "$NUDITY_LATENTS" \
        --target_hookpoint "$TARGET_HOOKPOINT" \
        --checkpoint_type best \
        --percentile 99.0 \
        --multiplier -10.0 \
        --num_steps 100 \
        --guidance_scale 9 \
        --seed 42 \
        --device cuda

elif [ "$MODE" = "test" ]; then
    echo "Running TEST evaluation (10 images only)..."
    echo ""
    
    python scripts/i2p_eval.py \
        --method_name "SAE Test" \
        --sd_path "$SD_PATH" \
        --i2p_dataset "$I2P_DATASET" \
        --output_dir "${OUTPUT_DIR}/test" \
        --sae_path "$SAE_PATH" \
        --nudity_latents_path "$NUDITY_LATENTS" \
        --target_hookpoint "$TARGET_HOOKPOINT" \
        --checkpoint_type best \
        --percentile 99.0 \
        --multiplier -10.0 \
        --num_steps 100 \
        --guidance_scale 9 \
        --max_images 10 \
        --seed 42 \
        --device cuda

elif [ "$MODE" = "hyperparam_sweep" ]; then
    echo "========================================================================"
    echo "HYPERPARAMETER SWEEP: Finding best multiplier"
    echo "========================================================================"
    echo "Configuration:"
    echo "  - Percentile: 99.999 (using only top 1 SAE latent)"
    echo "  - Images per multiplier: 100"
    echo "  - Multipliers to test: -1 to -30 (all integers)"
    echo "  - Total evaluations: 30"
    echo "  - Dataset: SEXUAL CATEGORY ONLY (filtered)"
    echo ""
    
    # Create summary file
    SUMMARY_FILE="${OUTPUT_DIR}/hyperparam_sweep/summary.txt"
    mkdir -p "${OUTPUT_DIR}/hyperparam_sweep"
    
    echo "Multiplier Sweep Results (percentile=99.999, 150 images, sexual category only)" > "$SUMMARY_FILE"
    echo "================================================================================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Run evaluation for each multiplier from -1 to -30
    for mult in $(seq -10 -10 -150); do
        echo ""
        echo "--------------------------------------------------------------------"
        echo "Testing multiplier: $mult ($((-mult + 1))/30)"
        echo "--------------------------------------------------------------------"
        
        python scripts/i2p_eval.py \
            --method_name "SAE p99.999 m${mult}" \
            --sd_path "$SD_PATH" \
            --i2p_dataset "$I2P_DATASET" \
            --output_dir "${OUTPUT_DIR}/hyperparam_sweep/m${mult}" \
            --sae_path "$SAE_PATH" \
            --nudity_latents_path "$NUDITY_LATENTS" \
            --target_hookpoint "$TARGET_HOOKPOINT" \
            --checkpoint_type best \
            --percentile 99.99 \
            --multiplier $mult \
            --num_steps 50 \
            --guidance_scale 7.5 \
            --max_images 100 \
            --filter_sexual_only \
            --seed 42 \
            --device cuda
        
        # Extract total nudity count and append to summary
        RESULT_CSV="${OUTPUT_DIR}/hyperparam_sweep/m${mult}/results_sae_p99.999_m${mult}.csv"
        if [ -f "$RESULT_CSV" ]; then
            TOTAL=$(tail -n 1 "$RESULT_CSV" | cut -d',' -f10)
            echo "Multiplier $mult: Total nudity detections = $TOTAL" >> "$SUMMARY_FILE"
            echo "  Result: $TOTAL total detections"
        else
            echo "Multiplier $mult: ERROR - Results file not found" >> "$SUMMARY_FILE"
            echo "  ERROR: Results file not found"
        fi
    done
    
    echo ""
    echo "========================================================================"
    echo "HYPERPARAMETER SWEEP COMPLETED"
    echo "========================================================================"
    echo ""
    cat "$SUMMARY_FILE"
    echo ""
    echo "Summary saved to: $SUMMARY_FILE"
    echo ""
    echo "NEXT STEP: Review the summary above and set BEST_MULTIPLIER in the script,"
    echo "           then change MODE to 'final_eval' to run full evaluation."

elif [ "$MODE" = "final_eval" ]; then
    # TODO: Set this based on hyperparam sweep results
    BEST_MULTIPLIER=-60  # <-- UPDATE THIS AFTER REVIEWING SWEEP RESULTS
    
    echo "========================================================================"
    echo "FINAL EVALUATION: Full dataset with best multiplier"
    echo "========================================================================"
    echo "Configuration:"
    echo "  - Percentile: 99.999 (using only top 1 SAE latent)"
    echo "  - Multiplier: $BEST_MULTIPLIER (selected from sweep)"
    echo "  - Images: ALL 4703 prompts"
    echo ""
    
    python scripts/i2p_eval.py \
        --method_name "SAE Final p99.999 m${BEST_MULTIPLIER}" \
        --sd_path "$SD_PATH" \
        --i2p_dataset "$I2P_DATASET" \
        --output_dir "${OUTPUT_DIR}/final_eval/-60" \
        --sae_path "$SAE_PATH" \
        --nudity_latents_path "$NUDITY_LATENTS" \
        --target_hookpoint "$TARGET_HOOKPOINT" \
        --checkpoint_type best \
        --percentile 99.99 \
        --multiplier $BEST_MULTIPLIER \
        --num_steps 50 \
        --guidance_scale 7.5 \
        --filter_sexual_only \
        --seed 42 \
        --device cuda
    
    echo ""
    echo "========================================================================"
    echo "FINAL EVALUATION COMPLETED"
    echo "========================================================================"
    echo "Results saved to: ${OUTPUT_DIR}/final_eval/"
fi

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Check the CSV files for the final table output."