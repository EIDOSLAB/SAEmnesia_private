#!/bin/bash
#SBATCH --job-name=v3_seed_0_evaluation
#SBATCH --output=sbatch_output/%j_v3_seed_0.out
#SBATCH --error=sbatch_output/%j_v3_seed_0.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=384G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8           # Increased CPU cores per task
#SBATCH --account=IscrC_SAOU

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Pre-create all directory structures for all multipliers
echo "Pre-creating directory structures..."
BASE_DIR='/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_0'
for multiplier in -1.0 -5.0 -10.0 -15.0 -20.0 -25.0 -30; do
    mkdir -p "${BASE_DIR}/percentile_99.999_multiplier_${multiplier}"/{Architectures,Bears,Birds,Butterfly,Cats,Dogs,Fishes,Flame,Flowers,Frogs,Horses,Human,Jellyfish,Rabbits,Sandwiches,Sea,Statues,Towers,Trees,Waterfalls}
done
echo "Directory structures created."

## OBJECTS: unet.up_blocks.1.attentions.1
## STYLES: unet.up_blocks.1.attentions.2

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
# --checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.1"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for objects."
# 
# python scripts/save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/dual_concept_optimized/v3/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.1.attentions.1 \
# --latents_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v3/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl \
# --concept_type "objects" \
# --num_timesteps 100 \
# --output_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/scores.json \
# --plot_scores  \
# --plot_output_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/v3
# 
# echo "Scores saved for objects."

# echo "Running step 4 for styles."
# 
# python scripts/efficient_gather_sae_acts_ca_prompts.py \
# --checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/from_scratch/object_concept_optimized/v1/ce_weight_3.0_sparsity_0.01/best" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/from_scratch/v1/unet.up_blocks.1.attentions.1"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for styles."
# 
# python scripts/save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/from_scratch/object_concept_optimized/v1/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.1.attentions.1 \
# --latents_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/from_scratch/v1/unet.up_blocks.1.attentions.1/style_latents_dict_unet.up_blocks.1.attentions.1.pkl \
# --concept_type "styles" \
# --num_timesteps 100 \
# --output_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/finetuned/from_scratch/v1/scores.json \
# --plot_scores  \
# --plot_output_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/finetuned/from_scratch/v1
# 
# echo "Scores saved for styles."


# Step 5.0 - Hyperparameter Sweep for Object Unlearning 
# original_multipliers = [-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
# original_percentiles = [99.99, 99.995, 99.999]

echo "Running step 5.0, phase 1 - Hyperparameter Sweep for Object Unlearning"

accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_sweep_cls_distr.py \
--percentiles [99.999] \
--multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30]> --seed 0 \
--output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0' \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--hookpoint 'unet.up_blocks.1.attentions.1' \
--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/from_scratch/v3/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/from_scratch/dual_concept_optimized/v3/ce_weight_3.0_sparsity_0.01/best' \
--steps 100

echo "Phase 1 of step 5.0 completed."

echo "Running step 5.0, phase 2 - Hyperparameter Sweep for Object Unlearning"

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/run_acc_all_cls_sweep.py \
--percentiles [99.999] \
--multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30]> \
--input_dir_base /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0 \
--output_dir_base /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0 \
--class_ckpt /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth \
--batch_size 256 \
--seed 0

echo "Phase 2 of step 5.0 completed."


echo "Running step 5.0, phase 3 - Hyperparameter Sweep for Object Unlearning"

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/find_best_params_cls_sweep.py [99.999] [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0"

echo "Phase 3 of step 5.0 completed."


# Adaptive Multipliers for Object Unlearning

# echo "Running step 5.1 - Adaptive Multipliers for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/adaptive_class_params.py \
# --model_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/object_concept_optimized/epoch_13/unet.up_blocks.1.attentions.1 \
# --data_path /leonardo_scratch/fast/IscrC_MAGNIFY/cassano/finetuning_activations/objects \
# --output_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/adaptive_multipliers \
# 
# echo "Step 5.1 completed."

# Step 5.1 - Object Unlearning

echo "Running step 5"

accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sample_unlearning_cls_distr.py  \
--class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0/class_params.pth \
--seed 0 \
--output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0' \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--hookpoint 'unet.up_blocks.1.attentions.1' \
--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/from_scratch/v3/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/from_scratch/dual_concept_optimized/v3/ce_weight_3.0_sparsity_0.01/best' \
--steps 100

echo "Step 5 completed."

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
--input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0" \
--output_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/from_scratch/v3/hp_search/seed_0" \
--style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
--class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
--batch_size 128 \
--avg_accuracy_input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/evaluations/objects/fine_tuned/from_scratch/v3/hp_search/seed_0"

echo "Evaluations completed."


# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"