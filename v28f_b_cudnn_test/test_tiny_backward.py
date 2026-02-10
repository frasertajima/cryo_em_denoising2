#!/usr/bin/env python3
"""
Tiny backward pass test to verify gradients with simple, verifiable values

Create a minimal 4×4 input, 2×2 kernel to check gradients match exactly
"""

import numpy as np
import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create tiny network: 1→2 channels, 2×2 kernel, 4×4 input
model = nn.Conv2d(1, 2, kernel_size=2, padding=0, stride=1, bias=True)

# Simple input
input_tensor = torch.randn(1, 1, 4, 4, requires_grad=False)
target = torch.randn(1, 2, 3, 3, requires_grad=False)

print("=" * 60)
print("Tiny Network Backward Pass Test")
print("=" * 60)
print()
print("Network: Conv2d(in=1, out=2, kernel=2×2)")
print("Input shape:", tuple(input_tensor.shape))
print("Output shape:", tuple(target.shape))
print()

# Forward pass
output = model(input_tensor)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

print(f"Forward loss: {loss.item():.8f}")
print()

print("Forward output (first row, both channels):")
print(f"  Channel 0: {output[0, 0, 0, :].detach().numpy()}")
print(f"  Channel 1: {output[0, 1, 0, :].detach().numpy()}")
print()

# Backward pass
model.zero_grad()
loss.backward()

# Check gradients
print("Weight gradient stats:")
print(f"  Shape: {model.weight.grad.shape}")
print(f"  Min: {model.weight.grad.min().item():.8f}")
print(f"  Max: {model.weight.grad.max().item():.8f}")
print(f"  Mean: {model.weight.grad.mean().item():.8f}")
print(f"  Std: {model.weight.grad.std().item():.8f}")
print()

print("Weight gradient (first kernel):")
print(model.weight.grad[0, 0].numpy())
print()

print("Bias gradient:")
print(model.bias.grad.numpy())
print()

# Export for Fortran
output_dir = "tiny_test"
import os

os.makedirs(output_dir, exist_ok=True)

# Export weights
weight = model.weight.detach().cpu().numpy().astype(np.float32)
bias = model.bias.detach().cpu().numpy().astype(np.float32)

# Transform weights: flip + transpose + F-order (matching export_for_fortran.py)
weight_flipped = np.flip(weight, axis=(2, 3)).copy()
weight_transposed = weight_flipped.transpose(3, 2, 1, 0)  # (Out,In,H,W) -> (H,W,In,Out)
weight_transposed.reshape(-1, order="F").tofile(f"{output_dir}/weight.bin")

bias.tofile(f"{output_dir}/bias.bin")

# Export input and target with proper layout
input_np = input_tensor.numpy()
target_np = target.numpy()

input_transposed = input_np.transpose(1, 2, 3, 0)
input_transposed.reshape(-1, order="F").tofile(f"{output_dir}/input.bin")

target_transposed = target_np.transpose(1, 2, 3, 0)
target_transposed.reshape(-1, order="F").tofile(f"{output_dir}/target.bin")

# Export gradients
grad_weight = model.weight.grad.numpy().astype(np.float32)
grad_bias = model.bias.grad.numpy().astype(np.float32)

# Transform weight gradient same as weights
grad_weight_flipped = np.flip(grad_weight, axis=(2, 3)).copy()
grad_weight_transposed = grad_weight_flipped.transpose(3, 2, 1, 0)
grad_weight_transposed.reshape(-1, order="F").tofile(f"{output_dir}/grad_weight.bin")

grad_bias.tofile(f"{output_dir}/grad_bias.bin")

print(f"Exported to {output_dir}/")
print(f"  weight.bin: {weight.shape} -> {weight_transposed.shape}")
print(f"  grad_weight.bin: {grad_weight.shape} -> {grad_weight_transposed.shape}")
print()

# Print actual gradient values for manual verification
print("Detailed gradient check:")
print("=" * 60)
print("PyTorch weight gradient (Out=2, In=1, H=2, W=2):")
for out_c in range(2):
    print(f"\n  Kernel {out_c} (output channel {out_c}):")
    print(f"    {grad_weight[out_c, 0, :, :]}")

print()
print("Expected Fortran transformed (H=2, W=2, In=1, Out=2):")
print("  After flip and transpose:")
for i in range(2):
    for j in range(2):
        print(f"    [{i},{j},0,:] = {grad_weight_transposed[i, j, 0, :]}")
