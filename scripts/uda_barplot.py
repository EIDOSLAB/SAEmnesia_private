import matplotlib.pyplot as plt
import numpy as np

# Set up matplotlib to use LaTeX font
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['font.size'] = 26          # Base font size
plt.rcParams['axes.labelsize'] = 26     # Axis labels
plt.rcParams['axes.titlesize'] = 26     # Title
plt.rcParams['xtick.labelsize'] = 26    # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 26    # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 26    # Legend

# Data
# methods = ['ESD', 'FMN', 'UCE', 'CA', 'SalUn', 'SEOT', 'SPM', 'EDiff', 'SHS', 'SAeUron', 'OC-FT', 'FS']
# before_unlearnDiffAtk = [98, 88, 98, 60, 85, 55, 60, 92, 95, 26.19, 73.30, 66.10]
# after_unlearnDiffAtk = [55, 50, 32, 28, 44, 12, 30, 45, 34, 2.22, 28.30, 26.40]

# methods = ['ESD', 'FMN', 'UCE', 'CA', 'SalUn', 'SEOT', 'SPM', 'EDiff', 'SHS', 'SAeUron', 'SAEmnesia']
# before_unlearnDiffAtk = [98, 88, 98, 60, 85, 55, 60, 92, 95, 26.19, 73.30]
# after_unlearnDiffAtk = [55, 50, 32, 28, 44, 12, 30, 45, 34, 2.22, 28.30]

methods = ['SAeUron', 'SAEmnesia']
before_unlearnDiffAtk = [26.19, 73.30]
after_unlearnDiffAtk = [2.22, 28.30]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set positions for bars
x = np.arange(len(methods))
width = 0.35

# Create color arrays with vivid colors for GOFT and TBFS
before_colors = ['#CD853F'] * len(methods)  # Default brown for all
after_colors = ['#4682B4'] * len(methods)   # Default blue for all

# Make GOFT and TBFS more vivid while keeping the color scheme
before_colors[1] = '#8B4513'  # Darker, more vivid brown for GOFT
# before_colors[11] = '#8B4513'  # Darker, more vivid brown for GOFT
after_colors[1] = '#2563EB'   # Brighter blue for TBFS
# after_colors[11] = '#2563EB'   # Brighter blue for TBFS

# Create bars with black borders
bars1 = ax.bar(x - width/2, before_unlearnDiffAtk, width, 
               label='Before UnlearnDiffAtk', color=before_colors, alpha=0.9,
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2, after_unlearnDiffAtk, width,
               label='After UnlearnDiffAtk', color=after_colors, alpha=0.9,
               edgecolor='black', linewidth=0.8)

# Customize the plot
ax.set_ylabel('Unlearning Accuracy (\%)')
ax.set_xlabel('')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30)

# Place legend above the plot in one line
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

# Set y-axis limits to make plot height match 100 line
ax.set_ylim(0, 100)

ax.set_xlim(-0.5, len(methods) - 0.5)

# Add grid for better readability
ax.grid(True, alpha=0.3, axis='y')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot as PDF
plt.savefig('unlearning_accuracy_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Plot saved as 'unlearning_accuracy_comparison.pdf'")