#!/usr/bin/env python3
"""
Write in exact Fortran memory order
"""

import numpy as np

# Create test tensor
test = np.zeros((2, 1, 4, 4), dtype=np.float32)
test[0, 0, :, :] = np.arange(16).reshape(4, 4)
test[1, 0, :, :] = np.arange(16, 32).reshape(4, 4)

print("Original PyTorch tensor (N=2, C=1, H=4, W=4):")
print("Batch 0:")
print(test[0, 0])
print("\nBatch 1:")
print(test[1, 0])
print()

# To write for Fortran (C,H,W,N) column-major:
# Fortran memory order: C varies fastest, then H, then W, then N
# This is exactly what F-order reshape does!

# Transpose to (C,H,W,N)
test_transposed = test.transpose(1, 2, 3, 0)  # (N,C,H,W) -> (C,H,W,N)

# Now reshape to 1D using F-order (column-major) - this gives Fortran memory order
flat_fortran = test_transposed.reshape(-1, order="F")

print(f"After transpose to {test_transposed.shape}")
print("Flattened with F-order (first 20 values):")
print(flat_fortran[:20])
print()

# Write this flat array
flat_fortran.tofile("test_layout3.bin")
print("Written to test_layout3.bin")
