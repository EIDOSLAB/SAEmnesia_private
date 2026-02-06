#!/bin/bash

# Script to submit all hyperparameter combinations as parallel SLURM jobs

echo "========================================="
echo "HYPERPARAMETER SEARCH - PARALLEL SUBMISSION"
echo "========================================="
echo ""

# Create log directory
mkdir -p hyperparam_logs

# Define hyperparameter grid
BETAS=(1.0 3.0 5.0)
LAMBDAS=(0.005 0.01 0.02)
GAMMAS=(0.05 0.1 0.2)

# Count total combinations
TOTAL=$((${#BETAS[@]} * ${#LAMBDAS[@]} * ${#GAMMAS[@]}))
echo "Total combinations: ${TOTAL}"
echo ""

# Array to store job IDs
declare -a JOB_IDS

# Counter
COUNT=0

# Submit each combination
for BETA in "${BETAS[@]}"; do
    for LAMBDA in "${LAMBDAS[@]}"; do
        for GAMMA in "${GAMMAS[@]}"; do
            COUNT=$((COUNT + 1))
            
            # Create job name
            JOB_NAME="hp_b${BETA}_l${LAMBDA}_g${GAMMA}"
            
            echo "[$COUNT/$TOTAL] Submitting: Beta=${BETA}, Lambda=${LAMBDA}, Gamma=${GAMMA}"
            
            # Submit job
            JOB_ID=$(sbatch --parsable \
                --job-name="${JOB_NAME}" \
                sbatch_loss_hp_search.sh ${BETA} ${LAMBDA} ${GAMMA})
            
            JOB_IDS+=("${JOB_ID}")
            echo "  → Job ID: ${JOB_ID}"
            echo ""
            
            # Optional: small delay to avoid overwhelming scheduler
            sleep 0.5
        done
    done
done

echo "========================================="
echo "SUBMISSION COMPLETE"
echo "========================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
echo ""
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor specific job:"
echo "  squeue -j JOB_ID"
echo ""
echo "Check job output:"
echo "  tail -f hyperparam_logs/hp_b*_JOB_ID.out"
echo ""
echo "Results will be saved to: ./hyperparam_search_results/"
echo "========================================="