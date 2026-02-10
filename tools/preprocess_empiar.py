#!/usr/bin/env python3
"""
Preprocess EMPIAR-10025 frame-averaged micrographs for Cryo-EM denoising.

This script:
1. Reads MRC files (7420×7676 images)
2. Normalizes using 95th percentile
3. Extracts 1024×1024 patches
4. Adds synthetic noise (Gaussian + Poisson)
5. Converts to binary format for training

Usage:
    python preprocess_empiar.py --input data/empiar_10025_subset/ --output data/cryo_data_streaming/
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import mrcfile
except ImportError:
    print("Error: mrcfile not installed")
    print("Install with: pip install mrcfile")
    exit(1)


def read_mrc(mrc_path):
    """Read MRC file and return data as numpy array."""
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data.copy()
        # Convert to float32
        data = data.astype(np.float32)
    return data


def normalize_percentile(img, percentile=95):
    """
    Normalize image using percentile of positive values.

    Standard normalization in cryo-EM:
    - Compute 95th percentile of positive values
    - Divide all values by this percentile
    - Clip to [0, 1]
    """
    positive_values = img[img > 0]

    if len(positive_values) == 0:
        print("Warning: No positive values in image, using abs values")
        positive_values = np.abs(img)

    if len(positive_values) == 0:
        print("Warning: All zeros, skipping normalization")
        return img

    p_val = np.percentile(positive_values, percentile)

    if p_val == 0:
        print(f"Warning: {percentile}th percentile is zero, using max instead")
        p_val = np.max(positive_values)

    normalized = img / p_val
    normalized = np.clip(normalized, 0, 1)

    return normalized.astype(np.float32)


def extract_patches(img, patch_size=1024, stride=512):
    """
    Extract patches from large micrograph.

    Args:
        img: Input image (H, W)
        patch_size: Size of square patches
        stride: Step size between patches

    Returns:
        patches: Array of shape (num_patches, patch_size, patch_size)
    """
    h, w = img.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]
            patches.append(patch)

    return np.array(patches, dtype=np.float32)


def add_synthetic_noise(clean_patches, noise_type="poisson_gaussian", sigma=0.05):
    """
    Add synthetic noise to clean patches.

    Args:
        clean_patches: Clean image patches
        noise_type: 'gaussian', 'poisson', or 'poisson_gaussian'
        sigma: Noise level for Gaussian component

    Returns:
        noisy_patches: Patches with added noise
    """
    noisy = clean_patches.copy()

    if noise_type == "gaussian":
        # Simple Gaussian noise
        noise = np.random.normal(0, sigma, noisy.shape).astype(np.float32)
        noisy = noisy + noise

    elif noise_type == "poisson":
        # Poisson noise (shot noise from electron counting)
        # Scale up, apply Poisson, scale back
        scale = 100.0
        noisy = np.random.poisson(noisy * scale).astype(np.float32) / scale

    elif noise_type == "poisson_gaussian":
        # Combined: Poisson (shot noise) + Gaussian (detector noise)
        scale = 100.0
        noisy = np.random.poisson(noisy * scale).astype(np.float32) / scale
        noise = np.random.normal(0, sigma, noisy.shape).astype(np.float32)
        noisy = noisy + noise

    # Clip to valid range
    noisy = np.clip(noisy, 0, 1).astype(np.float32)

    return noisy


def write_binary(patches, output_path):
    """
    Write patches to binary file (sample-major format).

    Format:
        - Shape: (num_patches, height, width)
        - Type: float32
        - Order: Sample-major (patch 0 fully, then patch 1, ...)
    """
    num_patches, height, width = patches.shape

    with open(output_path, "wb") as f:
        for i in range(num_patches):
            patches[i].astype(np.float32).tofile(f)

    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Wrote {num_patches} patches to {output_path}")
    print(f"File size: {file_size_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EMPIAR-10025 for Cryo-EM denoising"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory with MRC files"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for binary files"
    )
    parser.add_argument(
        "--patch_size", type=int, default=1024, help="Patch size (default: 1024)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride for patch extraction (default: 512, 50%% overlap)",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="poisson_gaussian",
        choices=["gaussian", "poisson", "poisson_gaussian"],
        help="Type of synthetic noise (default: poisson_gaussian)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.05,
        help="Gaussian noise sigma (default: 0.05)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data for testing (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Find all MRC files
    mrc_files = sorted(glob.glob(os.path.join(args.input, "*.mrc")))

    if len(mrc_files) == 0:
        print(f"Error: No MRC files found in {args.input}")
        print("Please download the frame-averaged micrographs first:")
        print("  Visit: https://www.ebi.ac.uk/empiar/EMPIAR-10025/")
        print("  Download: Frame-averaged micrographs (41 GB)")
        exit(1)

    print(f"Found {len(mrc_files)} MRC files")
    print(f"Patch size: {args.patch_size}×{args.patch_size}")
    print(
        f"Stride: {args.stride} ({'50%' if args.stride == args.patch_size // 2 else '100%'} overlap)"
    )
    print(f"Noise type: {args.noise_type}, sigma: {args.noise_level}")
    print()

    # Split into train/test
    num_test = max(1, int(len(mrc_files) * args.test_split))
    num_train = len(mrc_files) - num_test

    # Shuffle and split
    indices = np.random.permutation(len(mrc_files))
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    train_files = [mrc_files[i] for i in train_indices]
    test_files = [mrc_files[i] for i in test_indices]

    print(f"Train: {num_train} images")
    print(f"Test: {num_test} images")
    print()

    # Process training data
    print("=" * 70)
    print("Processing training data...")
    print("=" * 70)

    train_clean_patches = []
    train_noisy_patches = []

    for mrc_path in tqdm(train_files, desc="Training images"):
        # Read and normalize
        img = read_mrc(mrc_path)
        img = normalize_percentile(img)

        # Extract patches
        patches = extract_patches(img, args.patch_size, args.stride)

        # Add to clean list
        train_clean_patches.append(patches)

        # Create noisy versions
        noisy = add_synthetic_noise(patches, args.noise_type, args.noise_level)
        train_noisy_patches.append(noisy)

    # Concatenate all patches
    train_clean_patches = np.concatenate(train_clean_patches, axis=0)
    train_noisy_patches = np.concatenate(train_noisy_patches, axis=0)

    print(f"\nTotal training patches: {len(train_clean_patches)}")
    print(f"Shape: {train_clean_patches.shape}")

    # Write training data
    print("\nWriting training data...")
    write_binary(train_noisy_patches, os.path.join(args.output, "train_input.bin"))
    write_binary(train_clean_patches, os.path.join(args.output, "train_target.bin"))

    # Process test data
    print()
    print("=" * 70)
    print("Processing test data...")
    print("=" * 70)

    test_clean_patches = []
    test_noisy_patches = []

    for mrc_path in tqdm(test_files, desc="Test images"):
        # Read and normalize
        img = read_mrc(mrc_path)
        img = normalize_percentile(img)

        # Extract patches
        patches = extract_patches(img, args.patch_size, args.stride)

        # Add to clean list
        test_clean_patches.append(patches)

        # Create noisy versions (different noise realization than training)
        noisy = add_synthetic_noise(patches, args.noise_type, args.noise_level)
        test_noisy_patches.append(noisy)

    # Concatenate all patches
    test_clean_patches = np.concatenate(test_clean_patches, axis=0)
    test_noisy_patches = np.concatenate(test_noisy_patches, axis=0)

    print(f"\nTotal test patches: {len(test_clean_patches)}")
    print(f"Shape: {test_clean_patches.shape}")

    # Write test data
    print("\nWriting test data...")
    write_binary(test_noisy_patches, os.path.join(args.output, "test_input.bin"))
    write_binary(test_clean_patches, os.path.join(args.output, "test_target.bin"))

    # Summary
    print()
    print("=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {args.output}")
    print(f"\nFiles created:")
    for filename in [
        "train_input.bin",
        "train_target.bin",
        "test_input.bin",
        "test_target.bin",
    ]:
        filepath = os.path.join(args.output, filename)
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"  {filename:20s} {size_gb:8.2f} GB")

    print(f"\nDataset statistics:")
    print(f"  Training patches: {len(train_clean_patches):,}")
    print(f"  Test patches:     {len(test_clean_patches):,}")
    print(f"  Patch size:       {args.patch_size}×{args.patch_size}")
    print(
        f"  Total size:       {(os.path.getsize(os.path.join(args.output, 'train_input.bin')) + os.path.getsize(os.path.join(args.output, 'train_target.bin')) + os.path.getsize(os.path.join(args.output, 'test_input.bin')) + os.path.getsize(os.path.join(args.output, 'test_target.bin'))) / (1024**3):.2f} GB"
    )

    print(f"\nNoise parameters:")
    print(f"  Type:  {args.noise_type}")
    print(f"  Sigma: {args.noise_level}")

    print(f"\nNext steps:")
    print(f"  1. Visualize samples: python tools/visualize_patches.py")
    print(f"  2. Train model: ./cryo_train_unet --stream --epochs 15 --save")
    print()


if __name__ == "__main__":
    main()
