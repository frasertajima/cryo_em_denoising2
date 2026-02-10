#!/usr/bin/env python3
"""
Memory-efficient streaming preprocessor for EMPIAR-10025 Cryo-EM denoising.

This version writes patches directly to disk as they're generated, avoiding
the memory explosion from accumulating all patches in RAM.

Key improvements over preprocess_empiar.py:
1. Streams patches directly to binary files (no accumulation)
2. Processes one image at a time
3. Memory usage: ~100MB (constant, independent of dataset size)
4. Shows progress with patch counts

Usage:
    python preprocess_empiar_streaming.py --input data/empiar_10025_subset/ --output data/cryo_data_streaming/
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


def extract_and_write_patches(
    img,
    clean_file,
    noisy_file,
    patch_size=1024,
    stride=512,
    noise_type="poisson_gaussian",
    sigma=0.05,
):
    """
    Extract patches from image and write directly to binary files.

    This streaming approach avoids memory accumulation by writing patches
    immediately after generation.

    Args:
        img: Input image (H, W)
        clean_file: Open file handle for clean patches
        noisy_file: Open file handle for noisy patches
        patch_size: Size of square patches
        stride: Step size between patches
        noise_type: Type of synthetic noise
        sigma: Noise level

    Returns:
        num_patches: Number of patches written
    """
    h, w = img.shape
    num_patches = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = img[y : y + patch_size, x : x + patch_size].astype(np.float32)

            # Write clean patch
            patch.tofile(clean_file)

            # Create and write noisy patch
            noisy = add_synthetic_noise(patch, noise_type, sigma)
            noisy.tofile(noisy_file)

            num_patches += 1

    return num_patches


def add_synthetic_noise(clean_patch, noise_type="poisson_gaussian", sigma=0.05):
    """
    Add synthetic noise to a single clean patch.

    Args:
        clean_patch: Clean image patch
        noise_type: 'gaussian', 'poisson', or 'poisson_gaussian'
        sigma: Noise level for Gaussian component

    Returns:
        noisy_patch: Patch with added noise
    """
    noisy = clean_patch.copy()

    if noise_type == "gaussian":
        # Simple Gaussian noise
        noise = np.random.normal(0, sigma, noisy.shape).astype(np.float32)
        noisy = noisy + noise

    elif noise_type == "poisson":
        # Poisson noise (shot noise from electron counting)
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


def process_dataset(
    mrc_files, output_dir, prefix, patch_size, stride, noise_type, sigma
):
    """
    Process a list of MRC files with streaming writes.

    Args:
        mrc_files: List of MRC file paths
        output_dir: Output directory
        prefix: Filename prefix ('train' or 'test')
        patch_size: Patch size
        stride: Stride for extraction
        noise_type: Type of noise
        sigma: Noise level

    Returns:
        total_patches: Total number of patches written
    """
    clean_path = os.path.join(output_dir, f"{prefix}_target.bin")
    noisy_path = os.path.join(output_dir, f"{prefix}_input.bin")

    total_patches = 0

    # Open files in binary append mode
    with open(clean_path, "wb") as clean_file, open(noisy_path, "wb") as noisy_file:
        for mrc_path in tqdm(mrc_files, desc=f"{prefix.capitalize()} images"):
            # Read and normalize image
            img = read_mrc(mrc_path)
            img = normalize_percentile(img)

            # Extract patches and write directly to disk
            num_patches = extract_and_write_patches(
                img, clean_file, noisy_file, patch_size, stride, noise_type, sigma
            )

            total_patches += num_patches

    # Report file sizes
    clean_size_gb = os.path.getsize(clean_path) / (1024**3)
    noisy_size_gb = os.path.getsize(noisy_path) / (1024**3)

    print(f"\n{prefix.capitalize()} set complete:")
    print(f"  Patches: {total_patches:,}")
    print(f"  Clean file: {clean_size_gb:.2f} GB")
    print(f"  Noisy file: {noisy_size_gb:.2f} GB")

    return total_patches


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient preprocessor for EMPIAR-10025 Cryo-EM denoising"
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

    # Process training data with streaming writes
    print("=" * 70)
    print("Processing training data (streaming mode)...")
    print("=" * 70)

    train_patches = process_dataset(
        train_files,
        args.output,
        "train",
        args.patch_size,
        args.stride,
        args.noise_type,
        args.noise_level,
    )

    # Process test data with streaming writes
    print()
    print("=" * 70)
    print("Processing test data (streaming mode)...")
    print("=" * 70)

    test_patches = process_dataset(
        test_files,
        args.output,
        "test",
        args.patch_size,
        args.stride,
        args.noise_type,
        args.noise_level,
    )

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
    print(f"  Training patches: {train_patches:,}")
    print(f"  Test patches:     {test_patches:,}")
    print(f"  Patch size:       {args.patch_size}×{args.patch_size}")

    total_size_gb = sum(
        os.path.getsize(os.path.join(args.output, f)) / (1024**3)
        for f in [
            "train_input.bin",
            "train_target.bin",
            "test_input.bin",
            "test_target.bin",
        ]
        if os.path.exists(os.path.join(args.output, f))
    )
    print(f"  Total size:       {total_size_gb:.2f} GB")

    print(f"\nNoise parameters:")
    print(f"  Type:  {args.noise_type}")
    print(f"  Sigma: {args.noise_level}")

    print(f"\nMemory efficiency:")
    print(f"  ✓ Streaming writes (constant ~100MB RAM)")
    print(f"  ✓ No patch accumulation")
    print(f"  ✓ Single-image processing")

    print(f"\nNext steps:")
    print(f"  1. Visualize samples: python tools/visualize_patches.py")
    print(f"  2. Train model: ./cryo_train_unet --stream --epochs 15 --save")
    print()


if __name__ == "__main__":
    main()
