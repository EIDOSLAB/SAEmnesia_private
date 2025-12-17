#!/bin/bash
#SBATCH --job-name=dual_object_metadata
#SBATCH --output=sbatch_output/%j_dual_object_metadata.out
#SBATCH --error=sbatch_output/%j_dual_object_metadata.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=380G                   # Increased memory
#SBATCH --partition=boost_usr_prod   # Ensure this is your highest-resource partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # Increased CPU cores per task
#SBATCH --account=IscrC_SAOU

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

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
# --mode dual_object_metadata \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_SAOU/cassano/finetuning_activations/dual_objects \
# --batch_size 128 \
# --class_start 0 \
# --class_end 20
# echo "Completed FULL PRODUCTION step 2."

# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/styles_metadata_recovery.py
# 
# echo "Running SDXL-Turbo activation collection"
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_unlearn_canvas.py \
# --mode metadata \
# --hook_names unet.up_blocks.0.attentions.1 \
# --model_name /leonardo_work/IscrC_SAOU/sdxl-turbo \
# --new_cached_activations_path /leonardo_scratch/fast/IscrC_SAOU/cassano/finetuning_activations/sdxl_objects \
# --batch_size 64 \
# --class_start 0 \
# --class_end 20 \
# --organization_type object
 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_unlearn_canvas.py \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_MAGNIFY/tmp \
# --batch_size 128
# 

# accelerate launch --num-processes=2 /leonardo/home/userexternal/ecassano/projects/SAeUron/scripts/collect_activations_unlearn_canvas.py \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/cached_activations \
# --batch_size 128 

# echo "Generating dataset for styles finetuning."
# accelerate launch --num-processes=2 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_unlearn_canvas.py \
# --mode finetuning \
# --hook_names unet.up_blocks.1.attentions.2 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
# --new_cached_activations_path /leonardo_work/IscrC_SAOU/styles_finetuning_dataset \
# --batch_size 128 \
# --class_start 0 \
# --class_end 50
# 
# 

# echo "Dataset collected."

# echo "Generating dataset for nudity finetuning."
# accelerate launch --num-processes=2 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/collect_activations_nudity.py \
# --hook_names unet.up_blocks.1.attentions.1 \
# --model_name /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4 \
# --new_cached_activations_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/activations \
# --prompts_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/coco_30k_captions.txt \
# --batch_size 128 \
# --num_inference_steps 50 \
# --cache_every_n_timesteps 1 \
# --column caption \
# --output_or_diff output \
# --seed 42
# 
# echo "Dataset collected."

# Step 4 - Nudity 

# echo "Running step 4 for Nudity."
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/nudity_efficient_gather_sae_acts_ca_prompts_cls.py \
# --checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4" \
# --save_dir "/leonardo_scratch/fast/IscrC_SAOU/nudity_activations" \
# --nudity_prompts_file "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/nudity_prompts.txt" \
# --non_nudity_prompts_file "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/non_nudity_prompts.txt" \
# --batch_size 128
# echo "Step 4 for nudity completed."
# echo "Saving scores for nudity."
# python scripts/nudity_save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity/unet.up_blocks.1.attentions.1 \
# --latents_path /leonardo_scratch/fast/IscrC_SAOU/nudity_activations/nudity_latents_dict_unet.up_blocks.1.attentions.1.pkl \
# --num_timesteps 50 \
# --output_json /leonardo_scratch/fast/IscrC_SAOU/nudity_scores.json \
# --plot_scores \
# --plot_output_dir /leonardo_scratch/fast/IscrC_SAOU/nudity_plots
# echo "Scores saved for nudity."

# Step 4 - Object (with finetuned model)

# echo "Running step 4."
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_gather_sae_acts_ca_prompts_cls.py \
# --checkpoint_path "/leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/sdxl-turbo/v1.6/ce_weight_3.0_sparsity_0.01/best/" \
# --hookpoint "unet.up_blocks.0.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/sdxl-turbo/unet.up_blocks.0.attentions.1"
# 
# echo "Step 4 completed."

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
# --checkpoint_path "/leonardo_scratch/large/userexternal/ecassano/sae_checkpoints/dual_concept_optimized/g_sae/ce_weight_1.0/best" \
# --hookpoint "unet.up_blocks.1.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_scratch/large/userexternal/ecassano/saeuron/features_activations/finetuned/gsae/unet.up_blocks.1.attentions.1"
# 
# echo "Step 4 completed."

# 
# 
# echo "Saving scores for objects."
# 
# python scripts/save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/sdxl-turbo/v1.6/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.0.attentions.1 \
# --latents_path /leonardo_work/IscrC_SAOU/cassano/saeuron/features_activations/finetuned/sdxl-turbo/unet.up_blocks.0.attentions.1/cls_latents_dict_unet.up_blocks.0.attentions.1.pkl \
# --concept_type "objects" \
# --num_timesteps 100 \
# --output_json /leonardo_work/IscrC_SAOU/cassano/saeuron/scores/objects/finetuned/sdxl-turbo/v1.6/scores.json \
# --plot_scores  \
# --plot_output_dir /leonardo_work/IscrC_SAOU/cassano/saeuron/scores/objects/finetuned/sdxl-turbo/v1.6
# 
# echo "Scores saved for objects."

# echo "Styles Finetuning - Running step 4."
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_gather_sae_acts_ca_prompts_cls.py \
# --checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best" \
# --hookpoint "unet.up_blocks.1.attentions.2" \
# --pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
# --save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.2"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for objects."
# 
# python scripts/save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.2 \
# --latents_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.2/cls_latents_dict_unet.up_blocks.1.attentions.2.pkl \
# --concept_type "objects" \
# --num_timesteps 100 \
# --output_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/up_1_2/scores.json \
# --plot_scores  \
# --plot_output_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/up_1_2
# 
# echo "Scores saved for objects."
 
 
 
 
# echo "Running step 4 for styles."
# 
# python scripts/efficient_gather_sae_acts_ca_prompts.py \
# --checkpoint_path "/leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/sdxl-turbo/v1.6/ce_weight_3.0_sparsity_0.01/best/" \
# --hookpoint "unet.up_blocks.0.attentions.1" \
# --pipe_path "/leonardo_work/IscrC_SAOU/sdxl-turbo" \
# --save_dir "/leonardo_work/IscrC_SAOU/cassano/saeuron/features_activations/finetuned/sdxl-turbo/unet.up_blocks.0.attentions.1"
# 
# echo "Step 4 completed."
# 
# echo "Saving scores for styles."
# 
# python scripts/save_scores.py \
# --model_checkpoint /leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/sdxl-turbo/v1.6/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.0.attentions.1 \
# --latents_path /leonardo_work/IscrC_SAOU/cassano/saeuron/features_activations/finetuned/sdxl-turbo/unet.up_blocks.0.attentions.1/cls_latents_dict_unet.up_blocks.0.attentions.1.pkl \
# --concept_type "styles" \
# --num_timesteps 100 \
# --output_json /leonardo_work/IscrC_SAOU/cassano/saeuron/scores/styles/finetuned/sdxl-turbo/v1.6/scores.json \
# --plot_scores  \
# --plot_output_dir /leonardo_work/IscrC_SAOU/cassano/saeuron/scores/styles/finetuned/sdxl-turbo/v1.6
# 
# echo "Scores saved for styles."


# Step 5.0 - Hyperparameter Sweep for Object Unlearning 
# original_multipliers = [-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
# original_percentiles = [99.99, 99.995, 99.999]


#echo "Running step 5.0, phase 1 - Hyperparameter Sweep for Object Unlearning"
#
#accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/gsae_steering_efficient_sweep_cls_distr.py \
#--alphas [-1.5,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.2,-0.1]> --seed 188 \
#--percentiles [99.999] \
#--output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/gsae_steering/hp_search/seed_188' \
#--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
#--hookpoint 'unet.up_blocks.1.attentions.1' \
#--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
#--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best' \
#--steps 100
#
#echo "Phase 1 of step 5.0 completed."
#
## 
#echo "Running step 5.0, phase 2 - Hyperparameter Sweep for Object Unlearning"
#
#python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/gsae_steering_run_acc_all_cls_sweep.py \
#--percentiles [99.999] \
#--alphas [-1.1,-1.2,-1.3,-1.5,-1.0,-0.8,-0.7,-0.6,-0.5]> \
#--input_dir_base /leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/gsae_steering/hp_search/seed_188 \
#--output_dir_base /leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/gsae_steering/hp_search/seed_188 \
#--class_ckpt /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth \
#--batch_size 256 \
#--seed 188

# # --script_version v2
# 
# echo "Phase 2 of step 5.0 completed."
# 
# 
# echo "Running step 5.0, phase 3 - Hyperparameter Sweep for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/gsae_steering_find_best_params_cls_sweep.py [99.999] [-0.1,-0.2,-0.3,-0.4,-0.5] "/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/gsae_steering/hp_search/seed_188" 
# # 
# echo "Phase 3 of step 5.0 completed."

# echo "Running step 5.0, phase 3 - Hyperparameter Sweep for Object Unlearning"
# 
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/find_best_params_cls_sweep_v2.py [99.999] [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] "/leonardo_work/IscrC_SAOU/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/improved_m_selection/hp_search/seed_188" \
# --selection_strategy weighted_average \
# --ua_weight 1.0 \
# --ira_weight 1.0 \
# --ccm_weight 1.0
# 
# echo "Phase 3 of step 5.0 completed."
# Step 5.1 - Object Unlearning

# echo "Running step 5"

# --class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_188/class_params.pth \

# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sample_unlearning_cls_distr.py  \
# --seed 188 \
# --output_dir '/leonardo_work/IscrC_SAOU/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/sdxl-turbo/hp_search/seed_188' \
# --class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/baseline/class_params_-1.0.pth \
# --pipe_checkpoint '/leonardo_work/IscrC_SAOU/sdxl-turbo' \
# --hookpoint 'unet.up_blocks.0.attentions.1' \
# --class_latents_path '/leonardo_work/IscrC_SAOU/cassano/saeuron/features_activations/finetuned/sdxl-turbo/unet.up_blocks.0.attentions.1/cls_latents_dict_unet.up_blocks.0.attentions.1.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/sdxl-turbo/v1.6/ce_weight_3.0_sparsity_0.01/best' \
# --steps 100 

# G-SAE
# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/gsae_steering.py  \
# --output_dir '/leonardo_work/IscrC_SAOU/cassano/saeuron/sweep_outputs/objects/fine_tuned/gsae' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --sae_checkpoint '/leonardo_work/IscrC_SAOU/cassano/saeuron/sae_checkpoints/dual_concept_optimized/g_sae/ce_weight_1.0/best' \
# --class_latents_path '/leonardo_work/IscrC_SAOU/cassano/saeuron/features_activations/finetuned/gsae/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
# --alpha=-1.0 \
# --steps=100 \
# --gamma=1.0
#--start_timestep 25

# Inferenza G-SAE, checkpoint SAEmnesia

# python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/gsae_steering.py  \
# --output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/SAEmnesia_gsae_steering' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best' \
# --class_latents_path '/leonardo_scratch/large/userexternal/ecassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
# --class_params_path  /leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/gsae_steering/hp_search/seed_188/class_params.pth \
# --steps=100 \
# --gamma=1.0


# Inferenza come G-SAE, conditional steering. Uso il sae solo se il concetto è presente.

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/conditional_gsae_steering.py  \
--output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/SAEmnesia_conditional_gsae_steering' \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--hookpoint 'unet.up_blocks.1.attentions.1' \
--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best' \
--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
--class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/baseline/class_params_-1.5.pth \
--steps=100 \
--gamma=1.0

# accelerate launch scripts/sample_unlearning_cls_distr.py \
#   --seed 188 \
#   --output_dir /leonardo_work/IscrC_SAOU/cassano/saeuron/sweep_outputs/objects/no_sae/seed_188 \
#   --pipe_checkpoint /leonardo_work/IscrC_SAOU/sdxl-turbo \
#   --steps 50 \
#   --use_sdxl=True \
#   --use_sae=False

# --hookpoint 'unet.up_blocks.0.attentions.1' \


# Unlearning classico con Beta al posto della media nell'inferenza

# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/beta_sample_unlearning_cls_distr.py  \
# --seed 188 \
# --output_dir '/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/beta/hp_search/seed_188/0.9' \
# --class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/baseline/class_params_-.9.pth \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best' \
# --steps 100 
# 
# echo "Step 5 completed."

# echo "Running step 5 with replacements."
# 
# accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/with_replacement_sample_unlearning_cls_distr.py  \
# --class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_188/class_params.pth \
# --seed 42 \
# --output_dir '/leonardo_work/IscrC_SAOU/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/with_replacement/hp_search/seed_42' \
# --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
# --hookpoint 'unet.up_blocks.1.attentions.1' \
# --class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
# --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best' \
# --steps 100 \
# --use_replacement_map True
# 
# echo "Step 5 with replacements completed."

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
--input_dir "/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/SAEmnesia_conditional_gsae_steering" \
--output_dir "/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/SAEmnesia_conditional_gsae_steering" \
--style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
--class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
--batch_size 128 \
--avg_accuracy_input_dir "/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/SAEmnesia_conditional_gsae_steering"

echo "Evaluations completed."


# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"