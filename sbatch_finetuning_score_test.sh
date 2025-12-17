#!/bin/bash
#SBATCH --job-name=score_finetuning_test
#SBATCH --output=sbatch_output/%j_score_finetuning_test.out
#SBATCH --error=sbatch_output/%j_score_finetuning_test.err
#SBATCH --time=24:00:00              # Increased time limit
#SBATCH --mem=380G                   # Increased memory
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

# Step 4 - Object (with finetuned model)

echo "Running step 4."

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_gather_sae_acts_ca_prompts_cls.py \
--checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/score_loss_optimized/batchtopk/best" \
--hookpoint "unet.up_blocks.1.attentions.1" \
--pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
--save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1"

echo "Step 4 completed."


echo "Saving scores for objects."

python scripts/save_scores.py \
--model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/score_loss_optimized/batchtopk/best/unet.up_blocks.1.attentions.1 \
--latents_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl \
--concept_type "objects" \
--num_timesteps 100 \
--output_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/batchtopk/scores/scores.json \
--plot_scores  \
--plot_output_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/batchtopk/scores

echo "Scores saved for objects."
 
echo "Running step 4 for styles."

python scripts/efficient_gather_sae_acts_ca_prompts.py \
--checkpoint_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/score_loss_optimized/batchtopk/best" \
--hookpoint "unet.up_blocks.1.attentions.1" \
--pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50" \
--save_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1"

echo "Step 4 completed."

echo "Saving scores for styles."

python scripts/save_scores.py \
--model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/score_loss_optimized/batchtopk/best/unet.up_blocks.1.attentions.1 \
--latents_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1/style_latents_dict_unet.up_blocks.1.attentions.1.pkl \
--concept_type "styles" \
--num_timesteps 100 \
--output_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/finetuned/batchtopk/scores/scores.json \
--plot_scores  \
--plot_output_dir /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/styles/finetuned/batchtopk/scores

echo "Scores saved for styles."

# # Step 5.0 - Hyperparameter Sweep for Object Unlearning 
# # original_multipliers = [-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0]
# # original_percentiles = [99.99, 99.995, 99.999]

echo "Running step 5.0, phase 1 - Hyperparameter Sweep for Object Unlearning"

accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/efficient_sweep_cls_distr.py \
--percentiles [99.999] \
--multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30]> --seed 42 \
--output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42' \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--hookpoint 'unet.up_blocks.1.attentions.1' \
--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/scores_loss_optimized/batchtopk/best' \
--steps 100

echo "Phase 1 of step 5.0 completed."

echo "Running step 5.0, phase 2 - Hyperparameter Sweep for Object Unlearning"

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/run_acc_all_cls_sweep.py \
--percentiles [99.999] \
--multipliers [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30]> \
--input_dir_base /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42 \
--output_dir_base /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42 \
--class_ckpt /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth \
--batch_size 256 \
--seed 42

echo "Phase 2 of step 5.0 completed."


echo "Running step 5.0, phase 3 - Hyperparameter Sweep for Object Unlearning"

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/find_best_params_cls_sweep.py [99.999] [-1.0,-5.0,-10.0,-15.0,-20.0,-25.0,-30.0] "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42"

echo "Phase 3 of step 5.0 completed."

# Step 5.1 - Object Unlearning

echo "Running step 5"

accelerate launch --num_processes 4 /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/sample_unlearning_cls_distr.py  \
--class_params_path  /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42/class_params.pth \
--seed 42 \
--output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42' \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--hookpoint 'unet.up_blocks.1.attentions.1' \
--class_latents_path '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/batchtopk/scores/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl' \
--sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/score_loss_optimized/batchtopk/best' \
--steps 100

echo "Step 5 completed."

# Benchmark - Objects

echo "Running evaluations."

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/run_acc_all_cls.py \
--input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42" \
--output_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/batchtopk/scores/hp_search/seed_42" \
--style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
--class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
--batch_size 128 \
--avg_accuracy_input_dir "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/evaluations/objects/fine_tuned/batchtopk/scores/hp_search/seed_42"

echo "Evaluations completed."


# Deactivate the virtual environment when done
deactivate

echo "Job completed."
echo "End time: $(date)"