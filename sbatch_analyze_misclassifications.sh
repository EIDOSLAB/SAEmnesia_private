#!/bin/bash
#SBATCH --job-name=analyze_misclassifications
#SBATCH --output=sbatch_output/%j_analyze_misclassifications.out
#SBATCH --error=sbatch_output/%j_analyze_misclassifications.err
#SBATCH --account=IscrC_INSAIT
#SBATCH --time=02:05:00
#SBATCH --mem=128G
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


# Activate environment
source ../../envs/saeuron_cassano/bin/activate

echo "Beginning misclassification analysis."

python scripts/analyze_misclassifications.py \
  --unlearn_class "Trees" \
  --images_dir "/leonardo_scratch/large/userexternal/ecassano/saeuron/sweep_outputs/objects/fine_tuned/v1.6/noise_injection/seed_188/replace_with_neighbor/Trees" \
  --class_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50_cls.pth" \
  --style_ckpt "/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/classifier_checkpoints/cls_model/style50.pth" \
  --seed 188

echo "Misclassification analysis completed."