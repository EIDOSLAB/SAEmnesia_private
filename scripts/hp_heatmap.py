import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate multipliers analysis plots')
parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Print current working directory and check if we can write files
print(f"Output directory: {args.output_dir}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents before saving: {os.listdir(args.output_dir)}")

# Try to create a simple test file to check write permissions
try:
    test_file = os.path.join(args.output_dir, "test_write.txt")
    with open(test_file, "w") as f:
        f.write("test")
    print("✓ Write permission confirmed")
    os.remove(test_file)
except Exception as e:
    print(f"✗ Write permission issue: {e}")

# --- Data ---
multipliers = [-30, -25, -20, -15, -10, -5, -1, 0.0]

# Baseline data (reversed order)
baseline_UA = [81.55, 76.50, 75.65, 65.65, 61.20, 51.75, 31.7, 5.55]
baseline_IRA = [74.58, 77.89, 78.93, 80.73, 82.99, 87.86, 97.36, 97.77]
baseline_CRA = [69.82, 71.33, 73.06, 75.30, 78.78, 84.59, 93.94, 98.95]

## MODELS FROM SCRATCH ##

# V1.5 data (reversed order)
# v15_UA = [94.60, 94.75, 94.55, 92.45, 94.60, 94.70, 59.1, 2.85]
# v15_IRA = [53.42, 56.58, 61.16, 66.31, 74.52, 87.80, 97.29, 97.65]
# v15_CRA = [49.64, 52.49, 56.32, 60.97, 68.52, 83.29, 96.78, 99.12]
# 
# V2 data (reversed order)
v2_UA = [94.45, 94.45, 92.40, 94.70, 90.40, 83.55, 40.70, 1.70]
v2_IRA = [21.34, 21.34, 23.01, 24.82, 28.89, 42.44, 93.41, 98.10]
v2_CRA = [17.73, 17.73, 19.00, 20.87, 23.92, 31.29, 81.85, 99.07]
# 
# # V1.6 data (reversed order)
v16_UA = [91.85, 94.25, 91.15, 96.40, 94.25, 88.65, 73.35, 3.50]
v16_IRA = [56.58, 60.15, 64.29, 68.98, 76.36, 87.05, 96.32, 97.45]
v16_CRA = [52.34, 55.39, 59.21, 63.83, 72.13, 84.78, 95.90, 98.98]
# 
# V3 data (reversed order)
v3_UA = [90.70, 90.55, 87.80, 91.40, 91.35, 91.60, 60.60, 2.75]
v3_IRA = [55.86, 58.54, 62.12, 67.58, 78.40, 90.90, 97.77, 98.21]
v3_CRA = [49.71, 52.18, 55.70, 60.16, 69.31, 84.68, 96.59, 98.97]
# 
# # V1.7 data (reversed order)
# # v17_UA = [46.05, 41.85, 32.35, 23.95, 20.05, 8.80, 2.65, 1.90]
# # v17_IRA = [96.58, 97.34, 97.53, 97.46, 97.75, 98.11, 98.20, 98.08]
# # v17_CRA = [94.20, 94.85, 95.70, 96.69, 97.33, 97.96, 98.46, 98.47]
# 
# # V4 data (reversed order)
v4_UA = [74.50, 71.70, 69.00, 59.95, 46.60, 43.80, 20.10, 1.80]
v4_IRA = [88.39, 90.69, 93.19, 95.48, 97.33, 98.13, 98.50, 98.54]
v4_CRA = [82.65, 86.00, 89.33, 92.37, 95.07, 96.90, 98.22, 98.80]

## MODELS FINETUNED ##

# # V1.5 data
# v15_UA = [95.69, 95.59, 97.16, 97.94, 96.47, 92.94, 66.37, 7.45]
# v15_IRA = [64.36, 66.78, 70.08, 73.68, 78.68, 87.59, 95.69, 96.70]
# v15_CRA = [59.21, 61.47, 64.51, 68.04, 73.98, 84.40, 97.37, 99.33]
# 
# V2 data
# v2_UA = [97.16, 97.84, 98.53, 98.73, 98.33, 96.47, 69.80, 4.80]
# v2_IRA = [48.76, 49.80, 51.41, 53.80, 58.80, 73.65, 96.48, 97.37]
# v2_CRA = [44.38, 45.68, 47.07, 48.96, 52.60, 62.47, 90.34, 98.92]
# 
# # V1.6 data
# v16_UA = [95.59, 94.71, 94.71, 95.10, 95.29, 91.47, 67.84, 8.04]
# v16_IRA = [69.70, 72.23, 74.75, 78.26, 82.52, 89.78, 96.31, 96.64]
# v16_CRA = [64.61, 67.44, 70.83, 74.63, 79.70, 86.51, 97.79, 99.36]
# 
# # V3 data
# v3_UA = [95.69, 96.18, 96.76, 97.55, 96.76, 92.55, 67.16, 7.75]
# v3_IRA = [64.51, 66.79, 69.95, 73.55, 78.60, 87.56, 95.73, 96.82]
# v3_CRA = [59.29, 61.47, 64.60, 68.26, 73.91, 84.34, 97.41, 99.34]
# # 
# # # V1.7 data
# # v17_UA = [71.67, 68.24, 65.59, 60.69, 52.06, 43.14, 27.06, 3.63]
# # v17_IRA = [89.41, 90.72, 92.44, 94.41, 96.49, 97.51, 97.38, 97.37]
# # v17_CRA = [85.84, 87.12, 88.91, 90.68, 92.88, 97.04, 97.97, 98.96]
# # 
# # # V4 data
# v4_UA = [69.90, 66.76, 65.98, 62.55, 53.63, 38.73, 24.22, 4.90]
# v4_IRA = [95.62, 96.09, 96.50, 96.96, 97.16, 97.40, 97.44, 97.53]
# v4_CRA = [91.11, 92.46, 93.81, 95.07, 96.58, 98.12, 98.77, 99.10]
# 
# # V1 data
# v1_UA = [95.00, 94.90, 95.20, 95.49, 97.16, 94.31, 94.12, 4.22]
# v1_IRA = [5.78, 6.26, 6.46, 6.43, 7.06, 13.47, 85.24, 97.44]
# v1_CRA = [1.45, 1.57, 1.70, 1.86, 2.45, 5.76, 57.71, 98.41]

# Organize data by metric
metrics = [
    # "UA (Unlearning Accuracy)",
    # "IRA (In-domain Retain Accuracy)", 
    # "CRA (Cross-domain Retain Accuracy)"
    "UA",
    "IRA", 
    "CRA"
]

# ua_data = [baseline_UA, v16_UA]
# ira_data = [baseline_IRA, v16_IRA]
# cra_data = [baseline_CRA,  v16_CRA]
# 
# # Calculate mean across all three metrics for each model
# baseline_mean = np.mean([baseline_UA, baseline_IRA, baseline_CRA], axis=0)
# v16_mean = np.mean([v16_UA, v16_IRA, v16_CRA], axis=0)
# 
# mean_data = [baseline_mean, v16_mean]
# 
# # All data for plotting
# all_data = [ua_data, ira_data, cra_data, mean_data]
# all_metrics = metrics + ["Mean (UA, IRA, CRA)"]

# Model names and colors (reordered)
# model_names = ["SAeUron", "S-OS-OC-CA-FS", "S-OO-CA-FS", "S-OS-CA-FS", "S-OS-TK-CA-FS"]
# model_names = ["SAeUron", "SAEmnesia"]
# colors = [  "#ff7f0e","#2ca02c","#1f77b4", "#9467bd", "#999999","#ff7f0e", "#d62728", "#8c564b"]

# Reorder data to match the new model order
# ua_data_reordered = [baseline_UA, v16_UA]
# ira_data_reordered = [baseline_IRA, v16_IRA]
# cra_data_reordered = [baseline_CRA, v16_CRA]
# mean_data_reordered = [baseline_mean, v16_mean]

ua_data = [baseline_UA, v2_UA, v16_UA, v3_UA, v4_UA]
ira_data = [baseline_IRA, v2_IRA, v16_IRA, v3_IRA, v4_IRA]
cra_data = [baseline_CRA, v2_CRA, v16_CRA, v3_CRA, v4_CRA]

# Calculate mean across all three metrics for each model
baseline_mean = np.mean([baseline_UA, baseline_IRA, baseline_CRA], axis=0)
v2_mean = np.mean([v2_UA, v2_IRA, v2_CRA], axis=0)
v16_mean = np.mean([v16_UA, v16_IRA, v16_CRA], axis=0)
v3_mean = np.mean([v3_UA, v3_IRA, v3_CRA], axis=0)
v4_mean = np.mean([v4_UA, v4_IRA, v4_CRA], axis=0)

mean_data = [baseline_mean, v2_mean, v16_mean, v3_mean, v4_mean]

# All data for plotting
all_data = [ua_data, ira_data, cra_data, mean_data]
all_metrics = metrics + ["Mean (UA, IRA, CRA)"]

# Model names and colors (reordered)
model_names = ["SAeUron", "S-OS-OC-CA-FS", "S-OO-CA-FS", "S-OS-CA-FS", "S-OS-TK-CA-FS"]
# model_names = ["SAeUron", "S-OS-OC-CA-FT", "S-OO-CA-FT", "S-OS-CA-FT", "S-OS-TK-CA-FT"]
colors = [ "#ff7f0e", "#2ca02c", "#999999", "#1f77b4", "#9467bd", "#d62728", "#8c564b"]

# Reorder data to match the new model order
ua_data_reordered = [baseline_UA, v16_UA, v2_UA, v3_UA, v4_UA]
ira_data_reordered = [baseline_IRA, v16_IRA, v2_IRA, v3_IRA, v4_IRA]
cra_data_reordered = [baseline_CRA, v16_CRA, v2_CRA, v3_CRA, v4_CRA]
mean_data_reordered = [baseline_mean, v16_mean, v2_mean, v3_mean, v4_mean]

# All data for plotting (reordered)
all_data = [ua_data_reordered, ira_data_reordered, cra_data_reordered, mean_data_reordered]

# --- Plot settings ---
plt.figure(figsize=(16, 4))  # Wide layout for 4 panels

for i in range(4):
    plt.subplot(1, 4, i+1)
    
    # Plot each model
    for j, (model_data, color, name) in enumerate(zip(all_data[i], colors, model_names)):
        plt.plot(range(len(multipliers)), model_data,
                marker='o', color=color,
                label=name if i==0 else "", markersize=6, linewidth=2)
    
    plt.title(all_metrics[i], fontsize=22)
    plt.xlabel("Multiplier", fontsize=24)
#     if i == 0:
#         plt.ylabel("Score (%)", fontsize=24)
    
    if i == 0:  # First plot
        plt.ylim(0, 100)
        plt.yticks(fontsize=22)
    else:
        plt.ylim(0, 100)
        ax = plt.gca()
        ax.tick_params(axis='y', which='both', left=True, labelleft=False)
        
    # Add grid to all plots (outside the if/else)
    plt.grid(axis='y', alpha=0.3)
    
    plt.grid(alpha=0.3)
    plt.xticks(range(len(multipliers)), [str(m) for m in multipliers], fontsize=18)

# Title and legend
# plt.suptitle("Model Performance Across Multipliers", fontsize=15, y=1.05)
plt.figlegend(model_names, loc="lower center", ncol=len(model_names), 
              frameon=False, fontsize=25, bbox_to_anchor=(0.5, -0.15))

plt.tight_layout()

# Check if matplotlib backend supports file saving
print(f"Matplotlib backend: {plt.get_backend()}")

# Save with more explicit error handling and correct output directory
output_pdf = os.path.join(args.output_dir, "multipliers_analysis_all_models.pdf")
output_png = os.path.join(args.output_dir, "multipliers_analysis_all_models.png")

try:
    plt.savefig(output_pdf, bbox_inches="tight")
    print("✓ PDF saved successfully")
except Exception as e:
    print(f"✗ Error saving PDF: {e}")

try:
    plt.savefig(output_png, dpi=400, bbox_inches="tight")
    print("✓ PNG saved successfully")
except Exception as e:
    print(f"✗ Error saving PNG: {e}")

# Check if files were actually created
if os.path.exists(output_pdf):
    print(f"✓ PDF file exists, size: {os.path.getsize(output_pdf)} bytes")
else:
    print("✗ PDF file not found after saving")

if os.path.exists(output_png):
    print(f"✓ PNG file exists, size: {os.path.getsize(output_png)} bytes")
else:
    print("✗ PNG file not found after saving")

print(f"Directory contents after saving: {os.listdir(args.output_dir)}")
print(f"Plots should be saved in: {args.output_dir}")
plt.show()

# Calculate (UA+IRA)/2 for each model at each multiplier
print("\n" + "="*80)
print("BEST MODEL BY (UA+IRA)/2 AT EACH MULTIPLIER")
print("="*80)

# Calculate combined UA+IRA scores for each model at each multiplier
combined_scores = []
for i, model_name in enumerate(model_names):
    model_combined = []
    for j in range(len(multipliers)):
        ua_score = ua_data_reordered[i][j]
        ira_score = ira_data_reordered[i][j]
        combined = (ua_score + ira_score) / 2
        model_combined.append(combined)
    combined_scores.append(model_combined)

# Print header
print(f"{'Multiplier':<12} {'Best Model':<12} {'(UA+IRA)/2':<12} {'UA':<8} {'IRA':<8}")
print("-" * 80)

# Find best model for each multiplier
for j, multiplier in enumerate(multipliers):
    # Get scores for all models at this multiplier
    multiplier_scores = [(combined_scores[i][j], i) for i in range(len(model_names))]
    # Find the best score and corresponding model index
    best_score, best_model_idx = max(multiplier_scores)
    
    # Get individual UA and IRA scores for the best model
    best_ua = ua_data_reordered[best_model_idx][j]
    best_ira = ira_data_reordered[best_model_idx][j]
    
    print(f"{multiplier:<12} {model_names[best_model_idx]:<12} {best_score:<12.2f} "
          f"{best_ua:<8.2f} {best_ira:<8.2f}")

print("\n" + "="*80)
print("DETAILED SCORES FOR ALL MODELS AT EACH MULTIPLIER")
print("="*80)

# Print detailed table showing (UA+IRA)/2 for all models at each multiplier
print(f"{'Multiplier':<12}", end="")
for model_name in model_names:
    print(f"{model_name:<10}", end="")
print()
print("-" * (12 + 10 * len(model_names)))

for j, multiplier in enumerate(multipliers):
    print(f"{multiplier:<12}", end="")
    for i in range(len(model_names)):
        combined = combined_scores[i][j]
        print(f"{combined:<10.2f}", end="")
    print()

print("\nNote: Values represent (UA + IRA) / 2 for each model at each multiplier")
print("="*80)