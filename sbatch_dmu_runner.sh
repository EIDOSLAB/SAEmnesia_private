#!/bin/bash
# Submit each attack_idx as a separate independent job
echo "Submitting individual jobs for attack_idx 0-141..."

for i in {1..1}; do
    sbatch sbatch_dmu_attk.sh $i
    echo "Submitted job for attack_idx $i"
done

echo "All 142 jobs submitted!"