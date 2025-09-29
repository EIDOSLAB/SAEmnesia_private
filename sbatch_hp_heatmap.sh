#!/bin/bash
#SBATCH --job-name=saeuron_heatmap
#SBATCH --output=sbatch_output/%j_saeuron_heatmap.out
#SBATCH --error=sbatch_output/%j_saeuron_heatmap.err
#SBATCH --time=00:05:00              # Short time - just generating heatmaps
#SBATCH --mem=4G                     # Light memory usage for matplotlib
#SBATCH --partition=boost_usr_prod   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2            # Light CPU usage
#SBATCH --account=iscrc_SAOU
# No GPU needed for matplotlib heatmaps

# Activate your environment
source ../../envs/saeuron_cassano/bin/activate

echo "Start time: $(date)"
echo "Running SAeUron-style heatmap generation..."

# Create output directory for sbatch logs
mkdir -p sbatch_output

# Define output directory for heatmaps
OUTPUT_DIR="/leonardo_work/IscrC_MAGNIFY/cassano/saeuron/evaluations/objects/fine_tuned/from_scratch/v1.6/heatmaps"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the SAeUron heatmap generation script
python scripts/hp_heatmap.py --output_dir "$OUTPUT_DIR"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "✅ SAeUron heatmap generation completed successfully."
    
    # List the generated files
    echo ""
    echo "📋 Generated files in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"/*.pdf 2>/dev/null || echo "  No PDF files found"
    echo ""
    echo "Total PDF files generated: $(ls -1 "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l)"
else
    echo "❌ Error: SAeUron heatmap generation failed."
    exit 1
fi

# Deactivate the virtual environment when done
deactivate

echo "🎉 Job completed successfully!"
echo "📁 Output directory: $OUTPUT_DIR"
echo "End time: $(date)"