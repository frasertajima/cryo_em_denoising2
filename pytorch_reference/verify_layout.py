#!/usr/bin/env python3
"""
Verify tensor layout conversion with a simple test case
"""

import numpy as np

# Create a simple 2x2x2x2 test tensor with known values
# PyTorch format: (Out, In, H, W)
test_weight = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)

print("Original PyTorch weight (Out, In, H, W):")
print(f"Shape: {test_weight.shape}")
print(test_weight)
print()

# Apply our conversion: flip, transpose, asfortranarray
weight_flipped = np.flip(test_weight, axis=(2, 3)).copy()
print("After flip on axes (2,3) - H and W:")
print(weight_flipped)
print()

weight_transposed = weight_flipped.transpose(2, 3, 1, 0)
print(f"After transpose (2,3,1,0) - shape {weight_transposed.shape} (H, W, In, Out):")
print(weight_transposed)
print()

weight_fortran = np.asfortranarray(weight_transposed)
print("After asfortranarray (memory layout changed):")
print(f"Is Fortran order: {weight_fortran.flags.f_contiguous}")
print(f"Is C order: {weight_fortran.flags.c_contiguous}")
print()

# Now simulate reading in Fortran
# Fortran reads in column-major order
print("If we write this to a file and read in Fortran with shape (2,2,2,2)...")
print("Fortran will interpret it as (H,W,In,Out) in column-major order")
print()

# Test the reverse transform to verify
print("=== Testing Reverse Transform ===")
# Start from Fortran layout
fortran_data = weight_fortran

# Reverse: reshape with F-order, transpose, flip
recovered = fortran_data.reshape((2, 2, 2, 2), order="F")
print(f"After reshape with F-order: shape {recovered.shape}")
print(recovered)
print()

recovered = recovered.transpose(3, 2, 1, 0)  # (H,W,In,Out) -> (Out,In,H,W)
print(f"After transpose (3,2,1,0): shape {recovered.shape}")
print(recovered)
print()

recovered = np.flip(recovered, axis=(2, 3)).copy()
print("After flip on axes (2,3):")
print(recovered)
print()

print("Original vs Recovered match:", np.allclose(test_weight, recovered))
