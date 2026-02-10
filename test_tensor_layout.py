#!/usr/bin/env python3
"""
Create a small test tensor with known values to verify layout
"""

import numpy as np

# Create a small test: 2 batches, 1 channel, 4x4 images
# Fill with easily identifiable values
test = np.zeros((2, 1, 4, 4), dtype=np.float32)

# Batch 0, channel 0
test[0, 0, :, :] = np.arange(16).reshape(4, 4)

# Batch 1, channel 0
test[1, 0, :, :] = np.arange(16, 32).reshape(4, 4)

print("Original PyTorch tensor (N=2, C=1, H=4, W=4):")
print("Batch 0:")
print(test[0, 0])
print("\nBatch 1:")
print(test[1, 0])
print()

# Export with our transformation
test_transposed = test.transpose(1, 2, 3, 0)  # (N,C,H,W) -> (C,H,W,N)
test_fortran = np.asfortranarray(test_transposed)

print(f"After transpose to (C,H,W,N): shape = {test_fortran.shape}")
print("Memory layout (first 20 values):")
flat = test_fortran.flatten(order="F")
print(flat[:20])
print()

# Write to file
test_fortran.tofile("test_layout.bin")
print("Written to test_layout.bin")
print()

# What Fortran should see when it reads with shape (1,4,4,2):
print("Fortran should read this and see:")
print("When accessing array(1, :, :, 1) - batch 1:")
print(test_fortran[0, :, :, 0])
print()
print("When accessing array(1, :, :, 2) - batch 2:")
print(test_fortran[0, :, :, 1])
