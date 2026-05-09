#!/usr/bin/env python3
"""Regenerate residual plots from existing CSV."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Load data
csv_path = Path("final_results/physics_residuals/test_residuals.csv")
df_results = pd.read_csv(csv_path)

output_dir = Path("final_results/physics_residuals/")

print(f"Loaded {len(df_results)} samples from {csv_path}")

# Plot 1: Just the histogram (cleaner)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Compute statistics
pinn_median = df_results["pinn_residual_median"].median()
baseline_median = df_results["baseline_residual_median"].median()
improvement_pct = (baseline_median - pinn_median) / baseline_median * 100

# Histogram comparison
bins = np.linspace(
    min(df_results["pinn_residual_median"].min(), df_results["baseline_residual_median"].min()),
    max(df_results["pinn_residual_median"].max(), df_results["baseline_residual_median"].max()),
    50
)

ax.hist(df_results["baseline_residual_median"], bins=bins, alpha=0.6, 
        label=f'Baseline (median={baseline_median:.2e})', color='#e74c3c', edgecolor='black')
ax.hist(df_results["pinn_residual_median"], bins=bins, alpha=0.6,
        label=f'PINN (median={pinn_median:.2e})', color='#2ecc71', edgecolor='black')
ax.axvline(baseline_median, color='#c0392b', linestyle='--', linewidth=2.5)
ax.axvline(pinn_median, color='#27ae60', linestyle='--', linewidth=2.5)

ax.set_xlabel('Physics Residual Magnitude', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title('Test-Set Induction Equation Residual Distribution (N=5,716 samples)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True, fancybox=True,
          framealpha=0.95, edgecolor='black', facecolor='white')
ax.grid(True, alpha=0.3, axis='y')

# Add improvement text box
ax.text(0.98, 0.65, f'{improvement_pct:.1f}%\nreduction',
        transform=ax.transAxes, ha='right', va='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))

plt.tight_layout()
plot_path = output_dir / "residual_comparison.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {plot_path}")
plt.close()

# Plot 2: Grouped bar chart by flare status (using seaborn)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Prepare data in long format for seaborn
plot_data = []
for _, row in df_results.iterrows():
    flare_status = "Flare Samples" if row["has_flare_24h"] else "Non-Flare Samples"
    plot_data.append({
        "Sample Type": flare_status,
        "Model": "PINN (Physics)",
        "Residual": row["pinn_residual_median"]
    })
    plot_data.append({
        "Sample Type": flare_status,
        "Model": "Baseline (No Physics)",
        "Residual": row["baseline_residual_median"]
    })

df_plot = pd.DataFrame(plot_data)

# Create barplot with seaborn
sns.barplot(data=df_plot, x="Sample Type", y="Residual", hue="Model", 
            estimator=np.median, errorbar=("pi", 50), capsize=0.1, ax=ax,
            palette={"PINN (Physics)": "#2ecc71", "Baseline (No Physics)": "#e74c3c"},
            alpha=0.8, edgecolor='black', linewidth=1.5)

# Get counts for labels
flare_count = df_results["has_flare_24h"].sum()
noflare_count = (~df_results["has_flare_24h"]).sum()

ax.set_xlabel('Sample Type', fontsize=13)
ax.set_ylabel('Median Physics Residual', fontsize=13)
ax.set_title('Physics Residual by Flare Status (24h horizon)', fontsize=14, fontweight='bold', pad=20)
ax.set_xticklabels([f'Flare Samples\n(N={flare_count})', f'Non-Flare Samples\n(N={noflare_count})'], fontsize=11)
ax.legend(fontsize=11, loc='upper left', frameon=True, shadow=True, fancybox=True, 
          framealpha=0.95, edgecolor='black', facecolor='white')

# Remove the scientific notation offset label (the 1e-6)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
ax.yaxis.get_offset_text().set_visible(False)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.2e', fontsize=9, fontweight='bold', padding=3)

plt.tight_layout()
plot_path_flare = output_dir / "residual_by_flare.png"
plt.savefig(plot_path_flare, dpi=300, bbox_inches="tight")
print(f"Saved flare comparison to: {plot_path_flare}")
plt.close()

print("\nDone! Updated plots saved.")

