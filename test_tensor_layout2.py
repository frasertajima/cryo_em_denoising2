#!/usr/bin/env python3
"""
Try different layout transformation
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

# Method: Transpose then flatten with C-order (not F-order!)
# Fortran reads in column-major, which means if we write in row-major
# and Fortran interprets as column-major, dimensions get swapped
test_transposed = test.transpose(1, 2, 3, 0)  # (N,C,H,W) -> (C,H,W,N)

print(f"After transpose: shape = {test_transposed.shape}")
print()

# Write with C-order (row-major) - this is the default for tofile()
test_transposed.tofile("test_layout2.bin")
print("Written to test_layout2.bin with C-order")
print()

print("First 20 values in file:")
print(test_transposed.flatten()[:20])
