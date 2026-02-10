#!/usr/bin/env python3
"""
Visualize patches from preprocessed Cryo-EM data.

Usage:
    python visualize_patches.py --data data/cryo_data_streaming/
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_patches(binary_path, patch_size=1024, num_patches=None):
    """Read patches from binary file (memory-efficient streaming read)."""
    # Get file size to determine total number of patches
    file_size = os.path.getsize(binary_path)
    patch_bytes = patch_size * patch_size * 4  # 4 bytes per float32
    total_patches = file_size // patch_bytes

    if num_patches is None:
        num_patches = total_patches
    else:
        num_patches = min(num_patches, total_patches)

    # Read only the requested number of patches (memory-efficient!)
    num_elements = num_patches * patch_size * patch_size
    data = np.fromfile(binary_path, dtype=np.float32, count=num_elements)

    # Reshape
    patches = data.reshape(num_patches, patch_size, patch_size)

    return patches


def compute_psnr(clean, noisy):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((clean - noisy) ** 2)
    if mse == 0:
        return float("inf")
    max_val = 1.0  # Normalized to [0, 1]
    psnr = 10 * np.log10(max_val**2 / mse)
    return psnr


def main():
    parser = argparse.ArgumentParser(description="Visualize Cryo-EM patches")
    parser.add_argument(
        "--data", type=str, required=True, help="Data directory with binary files"
    )
    parser.add_argument(
        "--patch_size", type=int, default=1024, help="Patch size (default: 1024)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of samples to visualize (default: 6)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for figure (default: show plot)",
    )

    args = parser.parse_args()

    # Check files exist
    input_path = os.path.join(args.data, "train_input.bin")
    target_path = os.path.join(args.data, "train_target.bin")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        print(f"Run preprocessing first:")
        print(
            f"  python tools/preprocess_empiar.py --input data/empiar_10025_subset/ --output {args.data}"
        )
        exit(1)

    if not os.path.exists(target_path):
        print(f"Error: {target_path} not found")
        exit(1)

    print(f"Reading patches from {args.data}...")

    # Read patches
    noisy_patches = read_patches(input_path, args.patch_size, args.num_samples)
    clean_patches = read_patches(target_path, args.patch_size, args.num_samples)

    print(f"Loaded {len(noisy_patches)} patches")
    print(f"Shape: {noisy_patches.shape}")

    # Compute statistics
    print("\nDataset statistics:")
    print(
        f"  Clean - Min: {clean_patches.min():.4f}, Max: {clean_patches.max():.4f}, Mean: {clean_patches.mean():.4f}, Std: {clean_patches.std():.4f}"
    )
    print(
        f"  Noisy - Min: {noisy_patches.min():.4f}, Max: {noisy_patches.max():.4f}, Mean: {noisy_patches.mean():.4f}, Std: {noisy_patches.std():.4f}"
    )

    # Compute PSNR for each sample
    psnrs = []
    for i in range(len(clean_patches)):
        psnr = compute_psnr(clean_patches[i], noisy_patches[i])
        psnrs.append(psnr)

    print(f"\nPSNR (noisy vs clean):")
    print(f"  Mean: {np.mean(psnrs):.2f} dB")
    print(f"  Std:  {np.std(psnrs):.2f} dB")
    print(f"  Range: {np.min(psnrs):.2f} - {np.max(psnrs):.2f} dB")

    # Visualize
    num_samples = min(args.num_samples, len(clean_patches))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Noisy
        axes[i, 0].imshow(noisy_patches[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"Noisy (Sample {i})\nPSNR: {psnrs[i]:.2f} dB")
        axes[i, 0].axis("off")

        # Clean
        axes[i, 1].imshow(clean_patches[i], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Clean (Target)")
        axes[i, 1].axis("off")

        # Difference
        diff = np.abs(clean_patches[i] - noisy_patches[i])
        axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[i, 2].set_title(f"|Difference|\nMax: {diff.max():.4f}")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
