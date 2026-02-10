#!/usr/bin/env python3
"""
Quick diagnostic: Compare validation loss (from training data) vs test loss.

This will help us understand if the brightness offset is due to:
1. Different data distributions (train vs test)
2. A bug in the model/training

We'll compute MSE loss on:
- Last 10% of training data (what the model used for validation)
- First 10% of test data
"""

import struct

import numpy as np

# Parameters
PATCH_SIZE = 1024
PIXELS_PER_PATCH = PATCH_SIZE * PATCH_SIZE

# File paths
TRAIN_INPUT = "../data/cryo_data_streaming/train_input.bin"
TRAIN_TARGET = "../data/cryo_data_streaming/train_target.bin"
TEST_INPUT = "../data/cryo_data_streaming/test_input.bin"
TEST_TARGET = "../data/cryo_data_streaming/test_target.bin"

print("=" * 70)
print("VALIDATION DATA vs TEST DATA COMPARISON")
print("=" * 70)
print()

# Load training data dimensions
import os

train_size = os.path.getsize(TRAIN_INPUT)
total_train_patches = train_size // (PIXELS_PER_PATCH * 4)  # 4 bytes per float32

print(f"Total training patches: {total_train_patches:,}")

# Calculate validation split (last 10% of training data)
val_split = 0.1
num_val_patches = int(total_train_patches * val_split)
train_patches = total_train_patches - num_val_patches

print(f"Training patches:       {train_patches:,}")
print(f"Validation patches:     {num_val_patches:,} (last 10% of train data)")
print()

# Load validation data (last 10% of training file)
print("Loading validation data from training file...")
val_start_byte = train_patches * PIXELS_PER_PATCH * 4

with open(TRAIN_INPUT, "rb") as f:
    f.seek(val_start_byte)
    val_noisy = np.fromfile(
        f, dtype=np.float32, count=num_val_patches * PIXELS_PER_PATCH
    )

with open(TRAIN_TARGET, "rb") as f:
    f.seek(val_start_byte)
    val_clean = np.fromfile(
        f, dtype=np.float32, count=num_val_patches * PIXELS_PER_PATCH
    )

val_noisy = val_noisy.reshape(num_val_patches, PATCH_SIZE, PATCH_SIZE)
val_clean = val_clean.reshape(num_val_patches, PATCH_SIZE, PATCH_SIZE)

print(f"✓ Loaded {len(val_noisy):,} validation patches")

# Load test data (same number of patches for fair comparison)
print("Loading test data...")
test_noisy = np.fromfile(
    TEST_INPUT, dtype=np.float32, count=num_val_patches * PIXELS_PER_PATCH
)
test_clean = np.fromfile(
    TEST_TARGET, dtype=np.float32, count=num_val_patches * PIXELS_PER_PATCH
)

test_noisy = test_noisy.reshape(num_val_patches, PATCH_SIZE, PATCH_SIZE)
test_clean = test_clean.reshape(num_val_patches, PATCH_SIZE, PATCH_SIZE)

print(f"✓ Loaded {len(test_noisy):,} test patches")
print()

# Compute statistics
print("=" * 70)
print("DATA STATISTICS COMPARISON")
print("=" * 70)
print()

print("VALIDATION DATA (from training file):")
print(
    f"  Noisy  - Mean: {val_noisy.mean():.6f}, Std: {val_noisy.std():.6f}, Min: {val_noisy.min():.6f}, Max: {val_noisy.max():.6f}"
)
print(
    f"  Clean  - Mean: {val_clean.mean():.6f}, Std: {val_clean.std():.6f}, Min: {val_clean.min():.6f}, Max: {val_clean.max():.6f}"
)
print()

print("TEST DATA (from test file):")
print(
    f"  Noisy  - Mean: {test_noisy.mean():.6f}, Std: {test_noisy.std():.6f}, Min: {test_noisy.min():.6f}, Max: {test_noisy.max():.6f}"
)
print(
    f"  Clean  - Mean: {test_clean.mean():.6f}, Std: {test_clean.std():.6f}, Min: {test_clean.min():.6f}, Max: {test_clean.max():.6f}"
)
print()

# Compute brightness difference
val_brightness = val_clean.mean()
test_brightness = test_clean.mean()
brightness_diff = test_brightness - val_brightness

print(f"BRIGHTNESS DIFFERENCE:")
print(f"  Validation clean mean: {val_brightness:.6f}")
print(f"  Test clean mean:       {test_brightness:.6f}")
print(f"  Difference:            {brightness_diff:+.6f}")
print()

# If the model learned validation data at brightness 0.XX but test is 0.YY,
# the predictions will be systematically offset!

# Compute what the baseline MSE would be if we just predicted the validation mean
print("=" * 70)
print("BASELINE MSE (predicting mean of validation data)")
print("=" * 70)
print()

# Predict validation mean for test data
baseline_prediction = np.full_like(test_clean, val_brightness)
baseline_mse = np.mean((baseline_prediction - test_clean) ** 2)
baseline_rmse = np.sqrt(baseline_mse)

print(f"If we predict {val_brightness:.4f} (validation mean) for all test pixels:")
print(f"  MSE:  {baseline_mse:.6f}")
print(f"  RMSE: {baseline_rmse:.6f}")
print()

# This is the "brightness offset penalty" - the model will pay this cost
# just for learning the wrong brightness level!

# Compare to the actual model test MSE you reported
model_test_mse = 0.054791
print(f"Your model's actual test MSE: {model_test_mse:.6f}")
print()

if baseline_mse > 0.01:
    print("⚠ WARNING: Large baseline MSE due to brightness mismatch!")
    print("   The model learned brightness from validation data, but test")
    print("   data has a different brightness distribution.")
    print()
    print(f"   Brightness offset squared: {brightness_diff**2:.6f}")
    print(f"   This alone contributes:    {brightness_diff**2:.6f} to MSE")
    print()
else:
    print("✓ Brightness levels are similar between validation and test")

print("=" * 70)
print("CONCLUSION")
print("=" * 70)
print()

if abs(brightness_diff) > 0.05:
    print("The test data has significantly different brightness than validation.")
    print("This explains why:")
    print(
        "  - Training val loss: 0.007 (on data with brightness ~{:.2f})".format(
            val_brightness
        )
    )
    print(
        "  - Test MSE: 0.055 (on data with brightness ~{:.2f})".format(test_brightness)
    )
    print()
    print("Solution: Use global normalization across train+test instead of")
    print("          per-image normalization.")
else:
    print("Brightness levels are similar - the issue may be elsewhere.")

print()
