#!/bin/bash
#SBATCH --job-name=nudity_split_pipeline
#SBATCH --output=sbatch_output/%j_nudity_split.out
#SBATCH --error=sbatch_output/%j_nudity_split.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=IscrC_MAGNIFY

export CUDA_VISIBLE_DEVICES=0

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

HOOKPOINT="unet.up_blocks.1.attentions.1"
SAVE_DIR="/leonardo_scratch/fast/IscrC_SAOU/nudity_activations/finetuned/v1.6"

echo "=== Processing NUDITY category ==="
python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/nudity_efficient_gather_sae_acts_ca_prompts_cls.py \
--checkpoint_path "/leonardo_scratch/fast/IscrC_SAOU/sae_checkpoints/nudity/finetuned/v1.6/best" \
--hookpoint "$HOOKPOINT" \
--pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4" \
--save_dir "$SAVE_DIR" \
--category "nudity" \
--prompts_file "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/nudity_prompts.txt" \
--batch_size 2 \
--save_every 20

if [ $? -ne 0 ]; then
    echo "ERROR: Nudity processing failed!"
    exit 1
fi

echo "=== Processing NON-NUDITY category ==="
python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/nudity_efficient_gather_sae_acts_ca_prompts_cls.py \
--checkpoint_path "/leonardo_scratch/fast/IscrC_SAOU/sae_checkpoints/nudity/finetuned/v1.6/best" \
--hookpoint "$HOOKPOINT" \
--pipe_path "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sd1.4" \
--save_dir "$SAVE_DIR" \
--category "non_nudity" \
--prompts_file "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/nudity_dataset/annotations/non_nudity_prompts.txt" \
--batch_size 2 \
--save_every 20 \
--max_prompts 100

if [ $? -ne 0 ]; then
    echo "ERROR: Non-nudity processing failed!"
    exit 1
fi

echo "=== Merging results ==="
python - <<EOF
import torch
import pickle
import os

hookpoint = "$HOOKPOINT"
save_dir = "$SAVE_DIR"

# Load individual category files
nudity_latents = torch.load(os.path.join(save_dir, f"nudity_latents_{hookpoint}.pt"))
non_nudity_latents = torch.load(os.path.join(save_dir, f"non_nudity_latents_{hookpoint}.pt"))

print(f"Nudity shape: {nudity_latents.shape}")
print(f"Non-nudity shape: {non_nudity_latents.shape}")

# Create combined dictionary
category_latents_dict = {
    'nudity': nudity_latents,
    'non_nudity': non_nudity_latents
}

# Save combined file
output_file = os.path.join(save_dir, f"nudity_latents_dict_{hookpoint}.pkl")
with open(output_file, "wb") as f:
    pickle.dump(category_latents_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved combined dictionary to {output_file}")
EOF

echo "=== Calculating scores ==="
python scripts/nudity_save_scores.py \
--model_checkpoint /leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/nudity/best/$HOOKPOINT \
--latents_path $SAVE_DIR/nudity_latents_dict_$HOOKPOINT.pkl \
--num_timesteps 100 \
--output_json /leonardo_scratch/fast/IscrC_SAOU/nudity_scores.json \
--plot_scores \
--plot_output_dir /leonardo_scratch/fast/IscrC_SAOU/nudity_plots

echo "Pipeline completed successfully!"