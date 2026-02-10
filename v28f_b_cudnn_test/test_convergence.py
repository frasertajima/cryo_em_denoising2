#!/usr/bin/env python3
"""
Test 1: Gradient Descent Step Comparison

Verify that despite gradient differences, one optimization step produces
similar weight updates in PyTorch and Fortran.

This is the definitive test: if gradients are correct (just with numerical
error), the updated weights should match closely.
"""

import os

import numpy as np
import torch
import torch.nn as nn

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Convergence Test: One Optimization Step Comparison")
print("=" * 70)
print()

# Create simple 2-layer network for manageable comparison
model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1, bias=True),
    nn.ReLU(),
    nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=True),
)

# Small input for faster computation
batch_size = 2
image_size = 128
input_tensor = torch.randn(batch_size, 1, image_size, image_size)
target = torch.randn(batch_size, 1, image_size, image_size)

print(f"Network: Conv(1→16, 3×3) → ReLU → Conv(16→1, 3×3)")
print(f"Input shape: {tuple(input_tensor.shape)}")
print(f"Target shape: {tuple(target.shape)}")
print()

# Save initial weights
initial_weights = {}
for name, param in model.named_parameters():
    initial_weights[name] = param.detach().clone()

print("Initial weights saved")
print()

# Forward pass
output = model(input_tensor)
loss_fn = nn.MSELoss()
loss = loss_fn(output, target)

print(f"Forward loss: {loss.item():.8f}")
print()

# Backward pass
model.zero_grad()
loss.backward()

# Check gradients exist
print("Gradient statistics:")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(
            f"  {name}: max={param.grad.abs().max().item():.6f}, "
            f"mean={param.grad.abs().mean().item():.6f}"
        )
print()

# Manual gradient descent step (like what Fortran will do)
learning_rate = 0.001
print(f"Learning rate: {learning_rate}")
print()

updated_weights_manual = {}
weight_changes = {}

for name, param in model.named_parameters():
    if param.grad is not None:
        # Manual update: w = w - lr * grad
        updated = param.detach() - learning_rate * param.grad.detach()
        updated_weights_manual[name] = updated

        # Calculate change magnitude
        change = (updated - initial_weights[name]).abs()
        weight_changes[name] = {
            "max": change.max().item(),
            "mean": change.mean().item(),
        }

print("Weight updates after one step:")
for name, change in weight_changes.items():
    print(f"  {name}: max_change={change['max']:.8f}, mean_change={change['mean']:.8f}")
print()

# Export for Fortran
output_dir = "convergence_test"
os.makedirs(output_dir, exist_ok=True)

# Export initial weights
print("Exporting to Fortran format...")
for name, param in model.named_parameters():
    safe_name = name.replace(".", "_")
    param_np = param.detach().cpu().numpy().astype(np.float32)

    if "weight" in name and len(param_np.shape) == 4:
        # Conv weight: (Out, In, H, W) -> flip + transpose to (H, W, In, Out)
        weight_flipped = np.flip(param_np, axis=(2, 3)).copy()
        weight_transposed = weight_flipped.transpose(3, 2, 1, 0)
        weight_transposed.reshape(-1, order="F").tofile(
            f"{output_dir}/{safe_name}_initial.bin"
        )
        print(f"  {name}: {param_np.shape} -> {weight_transposed.shape} (Fortran)")
    else:
        # Bias: direct export
        param_np.tofile(f"{output_dir}/{safe_name}_initial.bin")
        print(f"  {name}: {param_np.shape}")

# Export input and target
input_np = input_tensor.numpy()
target_np = target.numpy()

input_transposed = input_np.transpose(1, 2, 3, 0)  # (N,C,H,W) -> (C,H,W,N)
target_transposed = target_np.transpose(1, 2, 3, 0)

input_transposed.reshape(-1, order="F").tofile(f"{output_dir}/input.bin")
target_transposed.reshape(-1, order="F").tofile(f"{output_dir}/target.bin")

print(f"  input: {input_np.shape} -> {input_transposed.shape} (Fortran)")
print(f"  target: {target_np.shape} -> {target_transposed.shape} (Fortran)")
print()

# Export expected updated weights for comparison
print("Exporting expected updated weights...")
for name, updated in updated_weights_manual.items():
    safe_name = name.replace(".", "_")
    updated_np = updated.cpu().numpy().astype(np.float32)

    if "weight" in name and len(updated_np.shape) == 4:
        # Conv weight: (Out, In, H, W) -> flip + transpose to (H, W, In, Out)
        weight_flipped = np.flip(updated_np, axis=(2, 3)).copy()
        weight_transposed = weight_flipped.transpose(3, 2, 1, 0)
        weight_transposed.reshape(-1, order="F").tofile(
            f"{output_dir}/{safe_name}_expected.bin"
        )
    else:
        # Bias: direct export
        updated_np.tofile(f"{output_dir}/{safe_name}_expected.bin")

print(f"  Exported {len(updated_weights_manual)} updated weight tensors")
print()

# Create metadata file
with open(f"{output_dir}/metadata.txt", "w") as f:
    f.write(f"batch_size={batch_size}\n")
    f.write(f"image_size={image_size}\n")
    f.write(f"learning_rate={learning_rate}\n")
    f.write(f"loss={loss.item()}\n")

print(f"Exported to {output_dir}/")
print()

print("=" * 70)
print("Next step: Run Fortran test_convergence to verify updates match")
print("=" * 70)
