#!/usr/bin/env python3
"""
Test Fortran-trained model on TRAINING data to check if it matches validation loss
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Load model
checkpoint_dir = "../v28f_e_final_training/saved_models/cryo_cnn/epoch_0001/"
model = SimpleCNN()

w1 = (
    np.fromfile(checkpoint_dir + "conv1_weights.bin", dtype=np.float32)
    .reshape(3, 3, 1, 16)
    .transpose(3, 2, 0, 1)
)
b1 = np.fromfile(checkpoint_dir + "conv1_bias.bin", dtype=np.float32)
model.conv1.weight.data = torch.from_numpy(w1)
model.conv1.bias.data = torch.from_numpy(b1)

w2 = (
    np.fromfile(checkpoint_dir + "conv2_weights.bin", dtype=np.float32)
    .reshape(3, 3, 16, 16)
    .transpose(3, 2, 0, 1)
)
b2 = np.fromfile(checkpoint_dir + "conv2_bias.bin", dtype=np.float32)
model.conv2.weight.data = torch.from_numpy(w2)
model.conv2.bias.data = torch.from_numpy(b2)

w3 = (
    np.fromfile(checkpoint_dir + "conv3_weights.bin", dtype=np.float32)
    .reshape(3, 3, 16, 1)
    .transpose(3, 2, 0, 1)
)
b3 = np.fromfile(checkpoint_dir + "conv3_bias.bin", dtype=np.float32)
model.conv3.weight.data = torch.from_numpy(w3)
model.conv3.bias.data = torch.from_numpy(b3)

model.eval()

print("=" * 70)
print("Testing Fortran model on TRAINING data")
print("=" * 70)

# Test on 10 random patches from TRAINING set
train_file_input = "../data/cryo_data_streaming/train_input.bin"
train_file_target = "../data/cryo_data_streaming/train_target.bin"

import os

train_size = os.path.getsize(train_file_input)
patch_size = 1024 * 1024
train_patches = train_size // (patch_size * 4)

print(f"Training set: {train_patches:,} patches")
print()

# Sample last 10% (validation set used during training)
val_start = int(train_patches * 0.9)
val_patches = train_patches - val_start

print(
    f"Validation split used during training: patches {val_start:,} to {train_patches:,}"
)
print(f"Testing on first 20 patches from validation split...")
print()

mse_values = []
corr_values = []

with torch.no_grad():
    for i in range(min(20, val_patches)):
        idx = val_start + i

        # Load patch
        with open(train_file_input, "rb") as f:
            f.seek(idx * patch_size * 4)
            noisy = np.fromfile(f, dtype=np.float32, count=patch_size).reshape(
                1, 1, 1024, 1024
            )

        with open(train_file_target, "rb") as f:
            f.seek(idx * patch_size * 4)
            clean = np.fromfile(f, dtype=np.float32, count=patch_size).reshape(
                1, 1, 1024, 1024
            )

        # Inference
        x = torch.from_numpy(noisy)
        y = model(x).numpy()

        # Metrics
        mse = np.mean((y - clean) ** 2)
        corr, _ = stats.pearsonr(y.flatten(), clean.flatten())

        mse_values.append(mse)
        corr_values.append(corr)

        if i < 5:
            print(f"Patch {idx:5d}: MSE={mse:.6f}, Corr={corr:.6f}")

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Mean MSE on validation patches:  {np.mean(mse_values):.6f}")
print(f"Mean Correlation:                {np.mean(corr_values):.6f}")
print()
print(f"Fortran reported validation loss: 0.006967")
print(f"PyTorch evaluation on val data:   {np.mean(mse_values):.6f}")
print()
if abs(np.mean(mse_values) - 0.006967) < 0.001:
    print("✓ MATCH! Model works on training/val data")
    print("  Problem: Severe overfitting to training distribution")
else:
    print("✗ MISMATCH! Something is fundamentally broken")
    print("  Problem: Loss calculation or model saving is wrong")
