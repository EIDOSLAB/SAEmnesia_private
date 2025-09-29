#!/bin/bash
#SBATCH --job-name=sample_unlearning_distr
#SBATCH --output=sbatch_output/%j_gather_sae.out
#SBATCH --error=sbatch_output/%j_gather_sae.err
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --partition=boost_usr_prod    # Change to a GPU partition on your system
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2                  # Request 1 GPU
#SBATCH --cpus-per-task=8

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python /leonardo/home/userexternal/ecassano/projects/SAeUron_finetuning/scripts/objects_activations_checker.py \
--index-file "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/features_activations/unet.up_blocks.1.attentions.1/activations_index_unet.up_blocks.1.attentions.1.txt" \

deactivate

echo "Job completed."
echo "End time: $(date)"