import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_loss_landscape(npz_path, save_path=None, levels=100, cmap="YlOrRd"):
    """
    Plot the loss landscape from a saved .npz file.

    Args:
        npz_path (str): Path to the .npz file containing 'alphas', 'betas', and 'losses'.
        save_path (str, optional): If given, save the plot instead of showing it.
        levels (int): Number of contour levels.
        cmap (str): Matplotlib colormap.
    """
    # Load loss landscape
    data = np.load(npz_path)
    alphas = data['alphas']
    betas = data['betas']
    losses = data['losses']

    # Create the plot
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(alphas, betas, losses, levels=levels, cmap=cmap)
    
    lines = plt.contour(alphas, betas, losses, levels=levels, colors='black', linewidths=0.5)
    plt.clabel(lines, inline=True, fontsize=8, fmt="%.2f")  # Optional: label the lines with loss values

    cbar = plt.colorbar(cp)
    cbar.set_label('Loss', fontsize=12)

    plt.xlabel('Direction 1 (alpha)', fontsize=14)
    plt.ylabel('Direction 2 (beta)', fontsize=14)
    plt.title('Loss Landscape', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot loss landscape from .npz file.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the saved loss landscape (.npz file).")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the plot.")
    args = parser.parse_args()

    plot_loss_landscape(args.npz_path, save_path=args.save_path)

if __name__ == "__main__":
    main()
