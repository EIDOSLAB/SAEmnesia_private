#!/bin/bash

# Batch experiment runner for SAeUron hyperparameter study
# This script runs all the experiments from your history in an organized way

# Set base paths
SCRIPT_PATH="sbatch_hp_study.sh"
CLASS_PARAMS_BASE="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron"

# Define multipliers to test (from most negative to positive)
MULTIPLIERS=(-30.0 -25.0 -20.0 -15.0 -10.0 -5.0 -1.0 0.0)

# Define model versions to test
DUAL_CONCEPT_VERSIONS=("v4" "v3" "v1.5" "v1.6" "v1.7")
OBJECT_CONCEPT_VERSIONS=("v2" "v1")

echo "Starting batch experiment submission..."
echo "Script: $SCRIPT_PATH"
echo "Base path: $CLASS_PARAMS_BASE"
echo ""

# Function to submit job and show progress
submit_job() {
    local version_path="$1"
    local multiplier="$2"
    local class_params_file="${CLASS_PARAMS_BASE}/class_params_uniform_99.999_${multiplier}.pth"
    
    echo "Submitting: $version_path with multiplier $multiplier"
    sbatch "$SCRIPT_PATH" "$version_path" "$class_params_file"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
}

# Run dual_concept_optimized experiments
echo "=== DUAL CONCEPT OPTIMIZED EXPERIMENTS ==="
for version in "${DUAL_CONCEPT_VERSIONS[@]}"; do
    echo "Running experiments for /dual_concept_optimized/$version"
    for multiplier in "${MULTIPLIERS[@]}"; do
        submit_job "/dual_concept_optimized/$version" "$multiplier"
    done
    echo ""
done

# Run object_concept_optimized experiments
echo "=== OBJECT CONCEPT OPTIMIZED EXPERIMENTS ==="
for version in "${OBJECT_CONCEPT_VERSIONS[@]}"; do
    echo "Running experiments for /object_concept_optimized/$version"
    for multiplier in "${MULTIPLIERS[@]}"; do
        submit_job "/object_concept_optimized/$version" "$multiplier"
    done
    echo ""
done

echo "All experiments submitted!"
echo ""
echo "Summary:"
echo "- Dual concept versions: ${DUAL_CONCEPT_VERSIONS[*]}"
echo "- Object concept versions: ${OBJECT_CONCEPT_VERSIONS[*]}"
echo "- Multipliers tested: ${MULTIPLIERS[*]}"
echo "- Total jobs submitted: $((${#DUAL_CONCEPT_VERSIONS[@]} * ${#MULTIPLIERS[@]} + ${#OBJECT_CONCEPT_VERSIONS[@]} * ${#MULTIPLIERS[@]}))"
echo ""
echo "Use 'squeue -u \$USER' to check job status"