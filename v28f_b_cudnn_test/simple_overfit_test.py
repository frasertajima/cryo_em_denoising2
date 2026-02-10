#!/usr/bin/env python3
"""
Simple Overfit Test - PyTorch Reference

Matches the Fortran simple_overfit_test.cuf exactly:
- 1→1 conv layer, 3×3 kernel
- Constant input = 0.5
- Target = 0.5 (identity function)
- Plain SGD optimizer
- Should easily converge to near-zero loss

This tests the absolute simplest case: learn to copy a constant.
"""

import torch
import torch.nn as nn
import numpy as np

# Configuration (match Fortran test)
IMAGE_SIZE = 64
BATCH_SIZE = 1
NUM_STEPS = 500
LEARNING_RATE = 0.001

print("=" * 70)
print("Simple Overfit Test - PyTorch Reference")
print("=" * 70)
print()
print("Task: Learn to copy input to output (identity function)")
print("Network: Single conv layer 1→1, 3×3")
print("Data: One 64×64 constant image")
print()

# Set seed for reproducibility (match Fortran random seed if possible)
torch.manual_seed(42)
np.random.seed(42)

# Create network: single conv layer, no ReLU
conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=True)

# Create constant input and target
input_tensor = torch.full((BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), 0.5)
target = torch.full((BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE), 0.5)

print(f"Input shape: {tuple(input_tensor.shape)}")
print(f"Input mean: {input_tensor.mean().item():.6f}")
print()

# Initial forward pass
with torch.no_grad():
    output = conv(input_tensor)
    initial_loss = ((output - target) ** 2).mean()
    print(f"Initial loss (should be high due to random weights):")
    print(f"  Loss: {initial_loss.item():.6f}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print()

# Manual SGD (not using optimizer to match Fortran exactly)
print("Training...")
print(f"{'Step':>6} {'Loss':>15}")
print("-" * 70)

for step in range(1, NUM_STEPS + 1):
    # Forward pass
    output = conv(input_tensor)
    
    # Compute loss (MSE with reduction='mean')
    loss = ((output - target) ** 2).mean()
    
    # Backward pass
    conv.zero_grad()
    loss.backward()
    
    # Print diagnostics for first 3 steps
    if step <= 3:
        print()
        print(f"Step {step} diagnostics:")
        print(f"  Input mean: {input_tensor.mean().item():.6f} (should be 0.5)")
        print(f"  Output mean: {output.mean().item():.6e} (target: 0.5)")
        print(f"  Loss: {loss.item():.6e}")
        
        # Gradient statistics
        grad_output = 2.0 * (output - target) / (IMAGE_SIZE * IMAGE_SIZE * BATCH_SIZE)
        print(f"  grad_output mean: {grad_output.mean().item():.6e}")
        print(f"  Weight grad mean: {conv.weight.grad.mean().item():.6e}")
        print(f"  Weight mean (before update): {conv.weight.data.mean().item():.6e}")
        print(f"  Bias grad: {conv.bias.grad[0].item():.6e}")
        print(f"  Bias (before update): {conv.bias.data[0].item():.6e}")
    
    # Manual SGD update (match Fortran exactly)
    with torch.no_grad():
        conv.weight.data -= LEARNING_RATE * conv.weight.grad
        conv.bias.data -= LEARNING_RATE * conv.bias.grad
        
        if step <= 3:
            print(f"  Weight mean (after update): {conv.weight.data.mean().item():.6e}")
            print(f"  Bias (after update): {conv.bias.data[0].item():.6e}")
    
    # Print progress
    if step % 50 == 0 or step == 1:
        print(f"{step:6d} {loss.item():15.6e}")
    
    # Early stop if converged
    if loss.item() < 1e-6:
        print()
        print(f"Converged at step {step}!")
        break

print("-" * 70)
print()

# Final evaluation
with torch.no_grad():
    output = conv(input_tensor)
    final_loss = ((output - target) ** 2).mean()

print("Final results:")
print(f"  Loss: {final_loss.item():.4e}")
print(f"  Target mean: {target.mean().item():.6f}")
print(f"  Output mean: {output.mean().item():.6f}")
print()

if final_loss < 0.01:
    print("✓ SUCCESS: Learned to copy constant")
else:
    print("✗ FAIL: Could not learn simple constant")
    print(f"  Final loss: {final_loss.item():.4e}")
