#!/usr/bin/env python3
"""
Verify the EXACT inverse transform
"""

import numpy as np

# Create test tensor
test_weight = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
print("Original PyTorch weight (Out, In, H, W):")
print(test_weight)
print()

# Forward transform (Fortran -> PyTorch) from model_loader.py:
# 1. reshape with F-order
# 2. transpose(3,2,1,0)
# 3. flip(axis=(2,3))

# Inverse transform (PyTorch -> Fortran):
# 1. flip(axis=(2,3)) - undo the flip
# 2. transpose(3,2,1,0) - undo the transpose
# 3. flatten with F-order

print("=== Applying Inverse Transform ===")
step1 = np.flip(test_weight, axis=(2, 3)).copy()
print("Step 1 - Flip axes (2,3):")
print(step1)
print()

step2 = step1.transpose(3, 2, 1, 0)
print(f"Step 2 - Transpose (3,2,1,0): shape {step2.shape}")
print(step2)
print()

# Flatten with F-order (asfortranarray then tofile does this)
step3 = np.asfortranarray(step2)
print("Step 3 - asfortranarray (changes memory layout):")
print(f"Is F-contiguous: {step3.flags.f_contiguous}")
print()

# Now simulate Fortran reading: read binary as 1D, then reshape with F-order
flat_data = step3.flatten(order="F")  # This is what tofile() writes
print("Flat data (what gets written to file):")
print(flat_data[:8], "...")
print()

# Fortran reads this and reshapes to (2,2,2,2) with F-order
fortran_view = flat_data.reshape((2, 2, 2, 2), order="F")
print("Fortran view after reshape (2,2,2,2) with F-order:")
print(fortran_view)
print()

# Now apply forward transform to see if we get back original
print("=== Applying Forward Transform to Fortran data ===")
recovered = fortran_view.reshape((2, 2, 2, 2), order="F")
recovered = recovered.transpose(3, 2, 1, 0)
recovered = np.flip(recovered, axis=(2, 3)).copy()

print("Recovered:")
print(recovered)
print()

print("Match:", np.allclose(test_weight, recovered))
