#!/bin/bash
#SBATCH --job-name=sae_benchmark
#SBATCH --output=sbatch_output/%j_sae_benchmark.out
#SBATCH --error=sbatch_output/%j_sae_benchmark.err
#SBATCH --time=02:00:00              # 2 hours should be enough for benchmarking
#SBATCH --mem=100G                   # Less memory needed than training
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                 # Only need 1 GPU for benchmarking
#SBATCH --cpus-per-task=8
#SBATCH --account=IscrC_INSAIT

# Load any necessary GPU modules (system-specific)
# module load cuda

source ../../envs/saeuron_cassano/bin/activate

# Set PyTorch memory configuration
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Create output directory for results
mkdir -p benchmark_results

# Run the benchmark
python scripts/inference_performances.py \
    --pipe_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/diff_models_checkpoints/style50' \
    --hookpoint 'unet.up_blocks.1.attentions.1' \
    --sae_checkpoint '/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/sae_checkpoints/best' \
    --seed 188 \
    --steps 100 \
    --guidance_scale 9.0 \
    --num_warmup_runs 3 \
    --num_benchmark_runs 10 \
    --batch_size 1 \
    --start_timestep 0

echo "Benchmark completed!"