#!/bin/bash
#SBATCH --job-name=visualize_features
#SBATCH --output=sbatch_output/%j_visualize_features.out
#SBATCH --error=sbatch_output/%j_visualize_features.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=02:05:00
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Define available classes and themes
class_available=(
    "Architectures"
    "Bears"
    "Birds"
    "Butterfly"
    "Cats"
    "Dogs"
    "Fishes"
    "Flame"
    "Flowers"
    "Frogs"
    "Horses"
    "Human"
    "Jellyfish"
    "Rabbits"
    "Sandwiches"
    "Sea"
    "Statues"
    "Towers"
    "Trees"
    "Waterfalls"
)

theme_available=(
    "Abstractionism"
    "Artist_Sketch"
    "Blossom_Season"
    "Bricks"
    "Byzantine"
    "Cartoon"
    "Cold_Warm"
    "Color_Fantasy"
    "Comic_Etch"
    "Crayon"
    "Cubism"
    "Dadaism"
    "Dapple"
    "Defoliation"
    "Early_Autumn"
    "Expressionism"
    "Fauvism"
    "French"
    "Glowing_Sunset"
    "Gorgeous_Love"
    "Greenfield"
    "Impressionism"
    "Ink_Art"
    "Joy"
    "Liquid_Dreams"
    "Magic_Cube"
    "Meta_Physics"
    "Meteor_Shower"
    "Monet"
    "Mosaic"
    "Neon_Lines"
    "On_Fire"
    "Pastel"
    "Pencil_Drawing"
    "Picasso"
    "Pop_Art"
    "Red_Blue_Ink"
    "Rust"
    "Seed_Images"
    "Sketch"
    "Sponge_Dabbed"
    "Structuralism"
    "Superstring"
    "Surrealism"
    "Ukiyoe"
    "Van_Gogh"
    "Vibrant_Flow"
    "Warm_Love"
    "Warm_Smear"
    "Watercolor"
    "Winter"
)

# Activate environment
source ../../envs/saeuron_cassano/bin/activate

# Loop through all combinations of objects and styles
for OBJECT in "${class_available[@]}"; do
    for STYLE in "${theme_available[@]}"; do
        echo "Processing ${OBJECT} with ${STYLE} style..."
        
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
            --mode=top_latent_ablation \
            --scores_json /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/scores/objects/finetuned/v1.6/scores.json \
            --concept_name="${OBJECT}" \
            --prompt="an image of a ${OBJECT,,} in ${STYLE} style" \
            --n_latents=2 \
            --timesteps_to_show="47,30,10,1" \
            --num_inference_steps=50 \
            --output_dir=/leonardo_scratch/fast/IscrC_SAOU/visualizations/saemnesia/${OBJECT}/${STYLE}/ \
            --class_latents_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/finetuned/v1.6/unet.up_blocks.1.attentions.1/cls_latents_dict_unet.up_blocks.1.attentions.1.pkl \
            --class_params_path=/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/hp_search/seed_42/class_params.pth \
            --alpha=0.5
    done
done

echo "All visualizations complete!"