#!/bin/bash
#SBATCH --job-name=visualize_features
#SBATCH --output=sbatch_output/%j_visualize_features.out
#SBATCH --error=sbatch_output/%j_visualize_features.err
#SBATCH --account=IscrC_MAGNIFY
#SBATCH --time=00:05:00
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Define the object to unlearn
OBJECT="Architectures"

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# python scripts/latents_image_visualization.py \
#     --sae_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.1 \
#     --pipe_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
#     --hookpoint="unet.up_blocks.1.attentions.1" \
#     --mode=top_features_grid \
#     --scores_json=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json \
#     --concept_name="${OBJECT}" \
#     --prompt="an image of a ${OBJECT,,}" \
#     --top_k=5 \
#     --timesteps_to_show="47,30,10,1" \
#     --num_inference_steps=50 \
#     --output_dir=/leonardo_scratch/fast/IscrC_SAOU/visualizations/baseline/${OBJECT}/ \
#     --class_latents_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl \
#     --class_params_path=<PATH_TO_YOUR_CLASS_PARAMS_PT>

python scripts/latents_image_visualization.py \
    --sae_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.1.attentions.1 \
    --pipe_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
    --hookpoint="unet.up_blocks.1.attentions.1" \
    --mode=top_features_grid \
    --scores_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/v1.6/scores.json \
    --concept_name="${OBJECT}" \
    --prompt="an image of a ${OBJECT,,}" \
    --top_k=5 \
    --timesteps_to_show="47,30,10,1" \
    --num_inference_steps=50 \
    --output_dir=/leonardo_scratch/fast/IscrC_SAOU/visualizations/saemnesia/${OBJECT}/ \
    --class_latents_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl \
    --class_params_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_42/class_params.pth

# for timestep in {0..49..5}; do
#     echo "Processing timestep: $timestep"
#     
#     # Baseline visualization
#     python scripts/latents_image_visualization.py \
#         --sae_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best/unet.up_blocks.1.attentions.1 \
#         --pipe_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
#         --prompt "an image of a tree" \
#         --mode top_features \
#         --scores_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/non_finetuned/scores.json \
#         --concept_name Trees \
#         --timestep $timestep \
#         --output_dir /leonardo_scratch/fast/IscrC_SAOU/visualizations/baseline/$timestep \
#         --hookpoint unet.up_blocks.1.attentions.1
#     
#     # SAEmnesia visualization
#     python scripts/latents_image_visualization.py \
#         --sae_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/wrong/dual_concept_optimized/v1.6/ce_weight_3.0_sparsity_0.01/best/unet.up_blocks.1.attentions.1 \
#         --pipe_path /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50 \
#         --prompt "an image of a tree" \
#         --mode top_features \
#         --scores_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/v1.6/scores.json \
#         --concept_name Trees \
#         --timestep $timestep \
#         --output_dir /leonardo_scratch/fast/IscrC_SAOU/visualizations/saemnesia/$timestep \
#         --hookpoint unet.up_blocks.1.attentions.1
#     
#     echo "Completed timestep: $timestep"
# done

echo "All visualizations complete!"