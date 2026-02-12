#!/usr/bin/env python3
"""
Preprocess Real Cryo-EM Micrographs for Denoising Training

This script handles real cryo-EM data (averaged micrographs) and creates
training pairs using a practical self-supervised approach:

Approach: "Neighbor Denoising"
- For each patch, use a slightly offset version as the target
- Real noise is spatially uncorrelated, so offset patches have independent noise
- This is a form of Noise2Noise without needing movie frames

Alternative: Add controlled noise to already-noisy images
- Input: real_image + extra_noise
- Target: real_image
- Teaches network to remove additive noise component

Usage:
    python preprocess_real_cryo.py --input data/empiar_10025_subset/ \
                                    --output data/cryo_real_streaming/ \
                                    --mode neighbor
"""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import mrcfile
except ImportError:
    print("Error: mrcfile not installed. Run: pip install mrcfile")
    exit(1)


def read_mrc(mrc_path):
    """Read MRC file and return data as numpy array."""
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data.copy().astype(np.float32)
    return data


def normalize_image(img):
    """Normalize image to [0, 1] using percentile clipping."""
    p1, p99 = np.percentile(img, [1, 99])
    img_clipped = np.clip(img, p1, p99)
    img_norm = (img_clipped - p1) / (p99 - p1 + 1e-8)
    return img_norm.astype(np.float32)


def extract_patches_neighbor(img, patch_size=1024, stride=512, offset=2):
    """
    Extract patch pairs using neighbor offset approach.

    For each patch location, extract two patches with a small offset.
    Since real cryo-EM noise is spatially uncorrelated, offset patches
    have independent noise realizations of the same underlying signal.

    Args:
        img: 2D image array
        patch_size: Size of patches to extract
        stride: Stride between patch centers
        offset: Pixel offset between input and target patches

    Returns:
        input_patches: List of input patches
        target_patches: List of target patches (offset by 'offset' pixels)
    """
    h, w = img.shape
    input_patches = []
    target_patches = []

    # Need extra margin for offset
    margin = offset

    for y in range(margin, h - patch_size - margin, stride):
        for x in range(margin, w - patch_size - margin, stride):
            # Input patch
            input_patch = img[y : y + patch_size, x : x + patch_size]

            # Target patch with offset (diagonal offset for independence)
            target_patch = img[
                y + offset : y + patch_size + offset,
                x + offset : x + patch_size + offset,
            ]

            # Only use if both patches are valid
            if input_patch.shape == (patch_size, patch_size) and target_patch.shape == (
                patch_size,
                patch_size,
            ):
                input_patches.append(input_patch)
                target_patches.append(target_patch)

    return input_patches, target_patches


def extract_patches_additive_noise(img, patch_size=1024, stride=512, noise_sigma=0.15):
    """
    Extract patches using additive noise approach.

    Add extra Gaussian noise to already-noisy image.
    Input: real_noisy + extra_noise
    Target: real_noisy

    This teaches the network to remove the additive component.

    Args:
        img: 2D image array (already noisy from cryo-EM)
        patch_size: Size of patches
        stride: Stride between patches
        noise_sigma: Std of additional noise to add

    Returns:
        input_patches: Patches with extra noise added
        target_patches: Original noisy patches (target)
    """
    h, w = img.shape
    input_patches = []
    target_patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y : y + patch_size, x : x + patch_size]

            if patch.shape == (patch_size, patch_size):
                # Target is the real noisy image
                target_patches.append(patch.copy())

                # Input is real noisy + extra noise
                extra_noise = (
                    np.random.randn(*patch.shape).astype(np.float32) * noise_sigma
                )
                noisy_patch = patch + extra_noise
                noisy_patch = np.clip(noisy_patch, 0, 1)
                input_patches.append(noisy_patch)

    return input_patches, target_patches


def main():
    parser = argparse.ArgumentParser(description="Preprocess real cryo-EM data")
    parser.add_argument("--input", required=True, help="Input directory with MRC files")
    parser.add_argument(
        "--output", required=True, help="Output directory for streaming data"
    )
    parser.add_argument(
        "--mode",
        choices=["neighbor", "additive"],
        default="neighbor",
        help="Training pair creation mode",
    )
    parser.add_argument("--patch_size", type=int, default=1024, help="Patch size")
    parser.add_argument(
        "--stride", type=int, default=512, help="Stride between patches"
    )
    parser.add_argument(
        "--offset", type=int, default=2, help="Offset for neighbor mode"
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.15,
        help="Extra noise sigma for additive mode",
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Test set fraction"
    )
    parser.add_argument(
        "--max_files", type=int, default=None, help="Max files to process"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find MRC files
    mrc_files = sorted(list(input_dir.glob("*.mrc")))
    if args.max_files:
        mrc_files = mrc_files[: args.max_files]

    print(f"=" * 60)
    print(f"  Real Cryo-EM Preprocessing")
    print(f"=" * 60)
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Mode: {args.mode}")
    print(f"  MRC files found: {len(mrc_files)}")
    print(f"  Patch size: {args.patch_size}")
    print(f"  Stride: {args.stride}")
    if args.mode == "neighbor":
        print(f"  Offset: {args.offset}")
    else:
        print(f"  Extra noise sigma: {args.noise_sigma}")
    print(f"=" * 60)
    print()

    # Collect all patches
    all_input_patches = []
    all_target_patches = []

    for mrc_file in tqdm(mrc_files, desc="Processing micrographs"):
        try:
            img = read_mrc(mrc_file)

            # Handle 3D stacks (take first frame or average)
            if len(img.shape) == 3:
                img = np.mean(img, axis=0)

            # Normalize
            img = normalize_image(img)

            # Extract patches based on mode
            if args.mode == "neighbor":
                inp, tgt = extract_patches_neighbor(
                    img, args.patch_size, args.stride, args.offset
                )
            else:
                inp, tgt = extract_patches_additive_noise(
                    img, args.patch_size, args.stride, args.noise_sigma
                )

            all_input_patches.extend(inp)
            all_target_patches.extend(tgt)

        except Exception as e:
            print(f"  Warning: Failed to process {mrc_file.name}: {e}")
            continue

    print(f"\nTotal patches extracted: {len(all_input_patches)}")

    if len(all_input_patches) == 0:
        print("Error: No patches extracted!")
        return

    # Convert to arrays
    input_patches = np.array(all_input_patches, dtype=np.float32)
    target_patches = np.array(all_target_patches, dtype=np.float32)

    # Shuffle
    indices = np.random.permutation(len(input_patches))
    input_patches = input_patches[indices]
    target_patches = target_patches[indices]

    # Split train/test
    n_test = int(len(input_patches) * args.test_split)
    n_train = len(input_patches) - n_test

    train_input = input_patches[:n_train]
    train_target = target_patches[:n_train]
    test_input = input_patches[n_train:]
    test_target = target_patches[n_train:]

    print(f"Train patches: {n_train}")
    print(f"Test patches: {n_test}")

    # Save to binary files
    print("\nSaving to streaming binary format...")

    # Flatten patches for streaming format
    train_input_flat = train_input.reshape(n_train, -1)
    train_target_flat = train_target.reshape(n_train, -1)
    test_input_flat = test_input.reshape(n_test, -1)
    test_target_flat = test_target.reshape(n_test, -1)

    # Write files (sample-major order for streaming)
    train_input_flat.tofile(output_dir / "train_input.bin")
    train_target_flat.tofile(output_dir / "train_target.bin")
    test_input_flat.tofile(output_dir / "test_input.bin")
    test_target_flat.tofile(output_dir / "test_target.bin")

    # Calculate file sizes
    train_size = train_input_flat.nbytes / (1024**3)
    test_size = test_input_flat.nbytes / (1024**3)

    print(f"\nFiles saved:")
    print(f"  train_input.bin:  {train_size:.2f} GB ({n_train} patches)")
    print(f"  train_target.bin: {train_size:.2f} GB")
    print(f"  test_input.bin:   {test_size:.2f} GB ({n_test} patches)")
    print(f"  test_target.bin:  {test_size:.2f} GB")
    print(f"  Total: {2 * (train_size + test_size):.2f} GB")

    # Save metadata
    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Patch size: {args.patch_size}\n")
        f.write(f"Train patches: {n_train}\n")
        f.write(f"Test patches: {n_test}\n")
        f.write(f"Source files: {len(mrc_files)}\n")
        if args.mode == "neighbor":
            f.write(f"Offset: {args.offset}\n")
        else:
            f.write(f"Noise sigma: {args.noise_sigma}\n")

    print(f"\n" + "=" * 60)
    print("  Preprocessing Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  cd v33_deep_residual")
    print(f"  ./cryo_train_deep --stream --data_dir {output_dir}/ --epochs 5 --save")


if __name__ == "__main__":
    main()
