#!/bin/bash

source ~/envs/saeuron_cassano/bin/activate

# Interactive srun command to run SD generation without SAE
srun --job-name=sd_generation_no_sae \
     --time=00:10:00 \
     --mem=16G \
     --partition=boost_usr_prod \
     --nodes=1 \
     --ntasks-per-node=1 \
     --gres=gpu:1 \
     --cpus-per-task=4 \
     --account=IscrC_SAOU \
     --pty bash -c "
# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo 'CUDA devices: \$CUDA_VISIBLE_DEVICES'
nvidia-smi

echo 'Running SD generation without SAE...'

python scripts/sd_generation_without_sae.py \
--pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
--output_dir '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/images_without_sae/results/custom_images/' \
--seed 888 \
--steps 50 \
--guidance_scale 7.5

echo 'Generation completed!'

# Deactivate the virtual environment when done
deactivate

echo 'Job completed.'
echo 'End time: \$(date)'
"