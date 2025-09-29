#!/bin/bash
#SBATCH --job-name=fs_hp_search
#SBATCH --output=sbatch_output/%j_fs_hp_search.out
#SBATCH --error=sbatch_output/%j_fs_hp_search.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=384G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # Increased CPU cores per task
#SBATCH --account=IscrC_SAOU

# Parse command line arguments
usage() {
    echo "Usage: sbatch $0 <VERSION_PATH> <CLASS_PARAMS_PATH>"
    echo ""
    echo "Arguments:"
    echo "  VERSION_PATH       Either:"
    echo "                     - 'baseline' for baseline model"
    echo "                     - '/<concept_type>/<version>' for fine-tuned model"
    echo "                     Examples: '/dual_concept_optimized/v1.5'"
    echo "                              '/object_concept_optimized/v1.7'"
    echo "                              '/from_scratch/dual_concept_optimized/v1.5'"
    echo "  CLASS_PARAMS_PATH  Full path to the class_params.pth file"
    echo "                     Expected format: class_params_uniform_XX.XXX_-XX.X.pth"
    echo ""
    echo "Examples:"
    echo "sbatch $0 baseline /path/to/class_params_uniform_99.999_-20.0.pth"
    echo "sbatch $0 /dual_concept_optimized/v1.5 /path/to/class_params_uniform_99.999_-20.0.pth"
    echo "sbatch $0 /object_concept_optimized/v1.7 /path/to/class_params_uniform_99.999_-20.0.pth"
    echo "sbatch $0 /from_scratch/dual_concept_optimized/v1.5 /path/to/class_params_uniform_99.999_-20.0.pth"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Error: Incorrect number of arguments provided."
    usage
fi

# Assign arguments to variables
VERSION_PATH="$1"
CLASS_PARAMS_PATH="$2"

# Parse VERSION_PATH to determine if baseline or fine-tuned
if [ "$VERSION_PATH" = "baseline" ]; then
    IS_BASELINE=true
    MODEL_TYPE="baseline"
    VERSION="baseline"
    CONCEPT_TYPE=""
    SAE_CHECKPOINT_SUBPATH=""
    FEATURES_ACTIVATIONS_SUBPATH="non_finetuned/unet.up_blocks.1.attentions.1"
    OUTPUT_SUBPATH="baseline"
elif [[ "$VERSION_PATH" =~ ^/from_scratch/([^/]+)/([^/]+)$ ]]; then
    # Handle /from_scratch/<concept_type>/<version> format
    IS_BASELINE=false
    MODEL_TYPE="from_scratch"
    CONCEPT_TYPE="${BASH_REMATCH[1]}"
    VERSION="${BASH_REMATCH[2]}"
    
    # Validate concept type
    if [[ "$CONCEPT_TYPE" != "object_concept_optimized" && "$CONCEPT_TYPE" != "dual_concept_optimized" ]]; then
        echo "Error: CONCEPT_TYPE must be either 'object_concept_optimized' or 'dual_concept_optimized'"
        echo "Found: $CONCEPT_TYPE"
        usage
    fi
    
    SAE_CHECKPOINT_SUBPATH="from_scratch/$CONCEPT_TYPE/$VERSION/ce_weight_3.0_sparsity_0.01/best"
    FEATURES_ACTIVATIONS_SUBPATH="finetuned/from_scratch/$VERSION/unet.up_blocks.1.attentions.1"
    OUTPUT_SUBPATH="fine_tuned/from_scratch/$VERSION"
elif [[ "$VERSION_PATH" =~ ^/([^/]+)/([^/]+)$ ]]; then
    # Handle /<concept_type>/<version> format (without from_scratch prefix)
    IS_BASELINE=false
    MODEL_TYPE="finetuned"
    CONCEPT_TYPE="${BASH_REMATCH[1]}"
    VERSION="${BASH_REMATCH[2]}"
    
    # Validate concept type
    if [[ "$CONCEPT_TYPE" != "object_concept_optimized" && "$CONCEPT_TYPE" != "dual_concept_optimized" ]]; then
        echo "Error: CONCEPT_TYPE must be either 'object_concept_optimized' or 'dual_concept_optimized'"
        echo "Found: $CONCEPT_TYPE"
        usage
    fi
    
    SAE_CHECKPOINT_SUBPATH="$CONCEPT_TYPE/$VERSION/ce_weight_3.0_sparsity_0.01/best"
    FEATURES_ACTIVATIONS_SUBPATH="finetuned/$VERSION/unet.up_blocks.1.attentions.1"
    OUTPUT_SUBPATH="fine_tuned/$CONCEPT_TYPE/$VERSION"
else
    echo "Error: Invalid VERSION_PATH format: $VERSION_PATH"
    echo "Must be either:"
    echo "  - 'baseline'"
    echo "  - '/<concept_type>/<version>'"
    echo "  - '/from_scratch/<concept_type>/<version>'"
    usage
fi

# Extract multiplier from class_params filename
# Expected format: class_params_uniform_99.999_-20.0.pth
CLASS_PARAMS_FILENAME=$(basename "$CLASS_PARAMS_PATH")
if [[ $CLASS_PARAMS_FILENAME =~ class_params_uniform_[0-9]+\.[0-9]+_(-?[0-9]+\.[0-9]+)\.pth ]]; then
    MULTIPLIER="${BASH_REMATCH[1]}"
    echo "Extracted multiplier: $MULTIPLIER"
else
    echo "Error: Could not extract multiplier from filename: $CLASS_PARAMS_FILENAME"
    echo "Expected format: class_params_uniform_XX.XXX_-XX.X.pth"
    exit 1
fi

# Update job name and output files with version and multiplier info
if [ "$IS_BASELINE" = true ]; then
    JOB_SUFFIX="baseline_mult${MULTIPLIER}_hp_search"
    SBATCH_OUTPUT_DIR="sbatch_output/baseline/mult_${MULTIPLIER}"
else
    # Include both concept type abbreviation and version for clarity
    if [ "$CONCEPT_TYPE" = "dual_concept_optimized" ]; then
        CONCEPT_ABBREV="dual"
    else
        CONCEPT_ABBREV="obj"
    fi
    
    if [ "$MODEL_TYPE" = "from_scratch" ]; then
        JOB_SUFFIX="fs_${CONCEPT_ABBREV}_${VERSION}_mult${MULTIPLIER}_hp_search"
        SBATCH_OUTPUT_DIR="sbatch_output/from_scratch/${CONCEPT_ABBREV}_${VERSION}/mult_${MULTIPLIER}"
    else
        JOB_SUFFIX="${CONCEPT_ABBREV}_${VERSION}_mult${MULTIPLIER}_hp_search"
        SBATCH_OUTPUT_DIR="sbatch_output/${CONCEPT_ABBREV}_${VERSION}/mult_${MULTIPLIER}"
    fi
fi

scontrol update job=$SLURM_JOB_ID name="fs_${JOB_SUFFIX}"

# Create version and multiplier-specific output directory if it doesn't exist
mkdir -p "$SBATCH_OUTPUT_DIR"

# Redirect output and error to version and multiplier-specific files
exec > >(tee "${SBATCH_OUTPUT_DIR}/${SLURM_JOB_ID}_fs_${JOB_SUFFIX}.out")
exec 2> >(tee "${SBATCH_OUTPUT_DIR}/${SLURM_JOB_ID}_fs_${JOB_SUFFIX}.err" >&2)

# Validate that class_params.pth file exists
if [ ! -f "$CLASS_PARAMS_PATH" ]; then
    echo "Error: class_params.pth file not found at: $CLASS_PARAMS_PATH"
    exit 1
fi

# Base paths
BASE_PATH="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron"

# Derived paths based on parameters (now including multiplier and baseline/finetuned)
if [ "$IS_BASELINE" = true ]; then
    # For baseline, the Python script will append the hookpoint name, so we just provide the base path
    SAE_CHECKPOINT_PATH="$BASE_PATH/sae_checkpoints"
else
    # For finetuned models, use the full subpath
    SAE_CHECKPOINT_PATH="$BASE_PATH/sae_checkpoints/$SAE_CHECKPOINT_SUBPATH"
fi

FEATURES_ACTIVATIONS_PATH="$BASE_PATH/features_activations/$FEATURES_ACTIVATIONS_SUBPATH"
OUTPUT_DIR="$BASE_PATH/sweep_outputs/objects/$OUTPUT_SUBPATH/mult_${MULTIPLIER}/hp_search"

if [ "$IS_BASELINE" = true ]; then
    EVALUATIONS_DIR="$BASE_PATH/evaluations/objects/baseline/mult_${MULTIPLIER}/hp_search"
elif [ "$MODEL_TYPE" = "from_scratch" ]; then
    EVALUATIONS_DIR="$BASE_PATH/evaluations/objects/fine_tuned/from_scratch/$VERSION/mult_${MULTIPLIER}/hp_search"
else
    EVALUATIONS_DIR="$BASE_PATH/evaluations/objects/fine_tuned/$CONCEPT_TYPE/$VERSION/mult_${MULTIPLIER}/hp_search"
fi

echo "==================== Job Configuration ===================="
echo "Model Type: $MODEL_TYPE"
echo "Version: $VERSION"
echo "Concept Type: $CONCEPT_TYPE"
echo "Multiplier: $MULTIPLIER"
echo "Class Params Path: $CLASS_PARAMS_PATH"
echo "SAE Checkpoint: $SAE_CHECKPOINT_PATH"
echo "Features Activations: $FEATURES_ACTIVATIONS_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Evaluations Directory: $EVALUATIONS_DIR"
echo "=========================================================="

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

## OBJECTS: unet.up_blocks.1.attentions.1
## STYLES: unet.up_blocks.1.attentions.2

# Validate that required directories/files exist
if [ ! -d "$SAE_CHECKPOINT_PATH" ]; then
    echo "Error: SAE checkpoint directory not found at: $SAE_CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$FEATURES_ACTIVATIONS_PATH" ]; then
    echo "Error: Features activations directory not found at: $FEATURES_ACTIVATIONS_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$SBATCH_OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$EVALUATIONS_DIR"

# Step 1

# python /leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/load_from_hub.py \
# --name bcywinski/SAeUron \
# --hookpoint unet.up_blocks.1.attentions.1 \
# --save_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/


# Step 2

# echo "Running FULL PRODUCTION step 2."
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_unlearn_canvas.py \
# --mode metadata \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects \
# --batch_size 128 \
# --class_start 0 \
# --class_end 20

# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_unlearn_canvas.py \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_MAGNIFY/tmp \
# --batch_size 128

# echo "Completed FULL PRODUCTION step 2."

# accelerate launch --num-processes=2 /leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/collect_activations_unlearn_canvas.py \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/cached_activations \
# --batch_size 128 

# Step 4 - Object

# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_gather_sae_acts_ca_prompts_cls.py \
# --checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/non_finetuned/unet.up_blocks.1.attentions.1"

# Step 4 - Object (with finetuned model)

# echo "Running step 4."
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_gather_sae_acts_ca_prompts_cls.py \
# --checkpoint_path "$SAE_CHECKPOINT_PATH" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "$FEATURES_ACTIVATIONS_PATH"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for objects."
# 
# python scripts/save_scores.py \
# --model_checkpoint "$SAE_CHECKPOINT_PATH/unet.up_blocks.1.attentions.1" \
# --latents_path "$FEATURES_ACTIVATIONS_PATH/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl" \
# --concept_type "objects" \
# --num_timesteps 100 \
# --output_json "$BASE_PATH/scores/objects/finetuned/from_scratch/$VERSION/scores.json" \
# --plot_scores  \
# --plot_output_dir "$BASE_PATH/scores/objects/finetuned/from_scratch/$VERSION"
# 
# echo "Scores saved for objects."
# 
# echo "Running step 4 for styles."
# 
# python scripts/efficient_gather_sae_acts_ca_prompts.py \
# --checkpoint_path "$SAE_CHECKPOINT_PATH" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "$FEATURES_ACTIVATIONS_PATH"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for styles."
# 
# python scripts/save_scores.py \
# --model_checkpoint "$SAE_CHECKPOINT_PATH/unet.up_blocks.1.attentions.1" \
# --latents_path "$FEATURES_ACTIVATIONS_PATH/style_latents_dict_unet.up_blocks.1.attentions.1.pkl" \
# --concept_type "styles" \
# --num_timesteps 100 \
# --output_json "$BASE_PATH/scores/styles/finetuned/from_scratch/$VERSION/scores.json" \
# --plot_scores  \
# --plot_output_dir "$BASE_PATH/scores/styles/finetuned/from_scratch/$VERSION"
# 
# echo "Scores saved for styles."


# Step 5.0 - Hyperparameter Sweep for Object Unlearning 
# original_multipliers = [-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
# original_percentiles = [99.99, 99.995, 99.999]

# echo "Running step 5.0, phase 1 - Hyperparameter Sweep for Object Unlearning"
# 
# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_sweep_cls_distr.py \
# --percentiles [99.999] \
# --multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] \
# --seed 42 \
# --output_dir "$OUTPUT_DIR" \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --class_latents_path "$FEATURES_ACTIVATIONS_PATH/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl" \
# --sae_checkpoint "$SAE_CHECKPOINT_PATH" \
# --steps 100
# 
# echo "Phase 1 of step 5.0 completed."
# 
# echo "Running step 5.0, phase 2 - Hyperparameter Sweep for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/run_acc_all_cls_sweep.py \
# --percentiles [99.999] \
# --multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] \
# --input_dir_base "$OUTPUT_DIR" \
# --output_dir_base "$OUTPUT_DIR" \
# --class_ckpt /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth \
# --batch_size 256 \
# --seed 42
# 
# echo "Phase 2 of step 5.0 completed."
# 
# 
# echo "Running step 5.0, phase 3 - Hyperparameter Sweep for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/find_best_params_cls_sweep.py [99.999] [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] "$OUTPUT_DIR"
# 
# echo "Phase 3 of step 5.0 completed."


# Adaptive Multipliers for Object Unlearning

# echo "Running step 5.1 - Adaptive Multipliers for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/adaptive_class_params.py \
# --model_path "$SAE_CHECKPOINT_PATH/unet.up_blocks.1.attentions.1" \
# --data_path /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects \
# --output_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/adaptive_multipliers \
# 
# echo "Step 5.1 completed."

# Step 5.1 - Object Unlearning

# echo "Running step 5 - Object Unlearning"
# 
# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sample_unlearning_cls_distr.py  \
# --class_params_path "$CLASS_PARAMS_PATH" \
# --seed 42 \
# --output_dir "$OUTPUT_DIR" \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --class_latents_path "$FEATURES_ACTIVATIONS_PATH/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl" \
# --sae_checkpoint "$SAE_CHECKPOINT_PATH" \
# --steps 100
# 
# echo "Step 5 completed."

# 
# Step 5.2 - Style Unlearning

# echo "Running step 5."
# 
# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/sample_unlearning_distr.py \
# --percentile 99.999 \
# --multiplier -1.0 \
# --seed 42 \
# --output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/styles/' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.2' \
# --style_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.2/style_latents_dict_unet.up_blocks.1.attentions.2.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/' \
# --steps 100

# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/sample_unlearning_distr.py \
# --percentile 99.999 \
# --multiplier -1.0 \
# --seed 42 \
# --output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/styles/fine_tuned' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.2' \
# --style_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/styles/fine_tuned/style_latents_dict_unet.up_blocks.1.attentions.2.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/style_fine_tuned/style_latent_finetuning_20250418_141014/latest' \
# --steps 100
# 
# echo "Step 5 completed."

# Benchmark - Style

# echo "Running evaluations."
# 
# python scripts/run_acc_all_style.py \
# --input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/styles/percentile_99.999_multiplier_-1.0" \
# --output_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/evaluations/styles/" \
# --style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
# --class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
# --batch_size 128 \
# --avg_accuracy_input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/evaluations/styles/"

# echo "Evaluations completed."

# Benchmark - Objects

echo "Running evaluations."

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/run_acc_all_cls.py \
--input_dir "$OUTPUT_DIR" \
--output_dir "$OUTPUT_DIR" \
--style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
--class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
--batch_size 128

echo "Evaluations completed."

# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"