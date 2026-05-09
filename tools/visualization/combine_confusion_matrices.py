#!/usr/bin/env python3
"""
Combine individual confusion matrix PNGs into a single 3×3 grid figure.

Layout:
- Rows: Flare-PINN, Strong-Form, Benchmark
- Columns: 6h, 12h, 24h horizons
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def main():
    parser = argparse.ArgumentParser(description="Combine confusion matrices into 3×3 grid")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with individual CM PNGs")
    parser.add_argument("--output", type=str, default="final_results/confusion_matrices_combined.png", 
                       help="Output combined figure path")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # Define the grid layout: (model_name, file_suffix)
    models = [
        ("Flare-PINN (Weak-Form)", "PINN"),
        ("Strong-Form", "Strong"),
        ("Benchmark (No Physics)", "Base"),
    ]
    
    horizons = ["6h", "12h", "24h"]
    
    # Create figure with 3×3 grid
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.15,
                  left=0.05, right=0.98, top=0.96, bottom=0.04)
    
    print("Loading confusion matrices...")
    
    for row_idx, (model_name, suffix) in enumerate(models):
        for col_idx, horizon in enumerate(horizons):
            # Construct filename
            filename = f"{horizon}{suffix}.png"
            filepath = input_dir / filename
            
            if not filepath.exists():
                print(f"  Warning: {filename} not found, skipping...")
                continue
            
            # Load image
            img = mpimg.imread(filepath)
            
            # Create subplot
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(img)
            ax.axis('off')
            
            # Add model label on leftmost column
            if col_idx == 0:
                ax.text(-0.15, 0.5, model_name, transform=ax.transAxes,
                       fontsize=20, fontweight='bold', va='center', rotation=90,
                       ha='center')
            
            # Add horizon label on bottom row
            if row_idx == 2:
                ax.text(0.5, -0.08, f'{horizon} Horizon', transform=ax.transAxes,
                       fontsize=20, fontweight='bold', ha='center', va='top')
            
            print(f"  Added: {filename}")
    
    # Overall title
    fig.suptitle('Confusion Matrices: Model Comparison Across Forecast Horizons',
                fontsize=24, fontweight='bold', y=0.99)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n Saved combined confusion matrices to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()

