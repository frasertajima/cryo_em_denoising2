# The Fortran/Python Memory Layout Bug: A Comprehensive Technical Guide

**Authors**: Fraser Tajima & Claude (Anthropic)  
**Date**: 2025-11-26  
**Status**: Reference Guide for Multi-Dimensional Array Interoperability

---

## Executive Summary

Across three major scientific computing projects (CIFAR-10 image classification, Climate downscaling, and Cryo-EM denoising), we encountered the same fundamental bug: **memory layout mismatches between Fortran/CUDA and Python/PyTorch**. This bug caused:

- **Complete model failure** (negative correlation, random predictions)
- **Silent data corruption** (training appeared successful but weights were scrambled)
- **Weeks of debugging** across multiple projects before the pattern emerged

This guide provides:
1. **Mathematical foundation** of memory layouts (column-major vs row-major)
2. **Detailed analysis** of the bug in each project
3. **Common patterns** and variations across projects
4. **Definitive solutions** and verification methods
5. **Best practices** to prevent future occurrences

**Key insight**: This is not a simple transpose issue - it involves understanding how multi-dimensional arrays map to linear memory, how different languages iterate over them, and how frameworks expect data organization.

---

## Table of Contents

1. [Mathematical Foundation: Memory Layouts](#1-mathematical-foundation-memory-layouts)
2. [The Bug Across Three Projects](#2-the-bug-across-three-projects)
3. [Project 1: CIFAR-10 - The Original Discovery](#3-project-1-cifar-10---the-original-discovery)
4. [Project 2: Climate - Pattern Recognition](#4-project-2-climate---pattern-recognition)
5. [Project 3: Cryo-EM - The Critical Fix](#5-project-3-cryo-em---the-critical-fix)
6. [Common Patterns and Variations](#6-common-patterns-and-variations)
7. [The Math: Why Transposes Don't Always Work](#7-the-math-why-transposes-dont-always-work)
8. [Definitive Solutions](#8-definitive-solutions)
9. [Prevention and Verification](#9-prevention-and-verification)
10. [Best Practices for Future Projects](#10-best-practices-for-future-projects)

---

## 1. Mathematical Foundation: Memory Layouts

### 1.1 Linear Memory and Multi-Dimensional Indexing

All computer memory is fundamentally **linear** - a 1D sequence of bytes. Multi-dimensional arrays are a **logical abstraction** that must be mapped to this linear memory.

For a 3D array `A[i, j, k]` with dimensions `(I, J, K)`:
- **Total elements**: `I × J × K`
- **Memory required**: `I × J × K × sizeof(element)`
- **Challenge**: Map `(i, j, k)` → linear memory address

There are **two standard conventions**:

### 1.2 Row-Major Order (C, Python, PyTorch)

**Principle**: Rightmost index varies fastest

For array `A[i, j, k]` with shape `(I, J, K)`:

```
Linear index = i × (J × K) + j × K + k
```

**Memory layout**:
```
A[0,0,0], A[0,0,1], ..., A[0,0,K-1],  ← First row of first plane
A[0,1,0], A[0,1,1], ..., A[0,1,K-1],  ← Second row of first plane
...
A[0,J-1,K-1],                          ← Last element of first plane
A[1,0,0], ...                          ← Start of second plane
```

**Iteration pattern**:
```python
for i in range(I):
    for j in range(J):
        for k in range(K):
            access A[i, j, k]  # k changes fastest
```

**Used by**: C, C++, Python (NumPy), PyTorch, TensorFlow

### 1.3 Column-Major Order (Fortran, MATLAB, Julia)

**Principle**: Leftmost index varies fastest

For array `A(i, j, k)` with shape `(I, J, K)`:

```
Linear index = i + j × I + k × (I × J)
```

**Memory layout**:
```
A(0,0,0), A(1,0,0), ..., A(I-1,0,0),  ← First column of first plane
A(0,1,0), A(1,1,0), ..., A(I-1,1,0),  ← Second column of first plane
...
A(I-1,J-1,0),                         ← Last element of first plane
A(0,0,1), ...                         ← Start of second plane
```

**Iteration pattern**:
```fortran
do k = 1, K
    do j = 1, J
        do i = 1, I
            access A(i, j, k)  ! i changes fastest
        end do
    end do
end do
```

**Used by**: Fortran, MATLAB, R, Julia (default)

### 1.4 The Critical Difference

For a 2×3 matrix:
```
Logical:  [[1, 2, 3],
           [4, 5, 6]]
```

**Row-major (Python/C)**:
```
Memory: [1, 2, 3, 4, 5, 6]
        └─ row 1 ─┘└─ row 2 ─┘
```

**Column-major (Fortran)**:
```
Memory: [1, 4, 2, 5, 3, 6]
        └col1┘└col2┘└col3┘
```

**Same logical array, completely different memory layout!**

### 1.5 Why This Matters for Neural Networks

#### Convolution Weights

A 2D convolution has weights with 4 dimensions:
- `out_channels`: Number of output feature maps
- `in_channels`: Number of input feature maps  
- `kernel_height`: Height of convolution kernel
- `kernel_width`: Width of convolution kernel

**PyTorch convention**: `(out_ch, in_ch, kH, kW)` in **row-major** memory

**cuDNN (C) convention**: `(out_ch, in_ch, kH, kW)` in **row-major** memory

**Fortran allocation**: `(out_ch, in_ch, kH, kW)` but in **column-major** memory!

**The trap**: Same dimension order, but:
- Python: Rightmost varies fastest (kW, kH, in_ch, out_ch)
- Fortran: Leftmost varies fastest (out_ch, in_ch, kH, kW)

*When you write both in linear memory, **they're scrambled relative to each other!***

---

## 2. The Bug Across Three Projects

### 2.1 Timeline of Discovery

| Project | Date | Bug Type | Symptoms | Resolution |
|---------|------|----------|----------|------------|
| **CIFAR-10** | Nov 2024 | Inference loading | Random predictions | Transpose fix |
| **Climate** | Nov 2024 | Same pattern | Recognized faster | Applied CIFAR fix |
| **Cryo-EM** | Nov 2025 | Weight saving | Negative correlation | Root cause fix |

### 2.2 Evolution of Understanding

**Phase 1 (CIFAR-10)**: "It's a transpose problem"
- Solution: Add `.transpose()` when loading
- Result: Works, but don't understand why
- Limitation: Band-aid, not root cause

**Phase 2 (Climate)**: "It's the same bug"
- Solution: Copy CIFAR-10 fix
- Result: Works faster (pattern recognition)
- Limitation: Still don't understand mechanism

**Phase 3 (Cryo-EM)**: "The root cause is allocation mismatch"
- Investigation: Why does epoch 5 perform worse than epoch 1?
- Discovery: **Fortran allocation != Fortran saving format**
- Solution: Fix allocation to match cuDNN expectation
- Result: Remove all transposes, works perfectly
- **Breakthrough**: Finally understand the mechanism!

### 2.3 Common Thread

All three bugs stem from:
1. **Fortran allocates in column-major** (leftmost varies fastest)
2. **cuDNN expects row-major** (rightmost varies fastest)
3. **Memory copy between mismatched formats scrambles data**

But each manifested differently depending on where the mismatch occurred.

---

## 3. Project 1: CIFAR-10 - The Original Discovery

### 3.1 Project Context

**Task**: Image classification (10 classes)  
**Architecture**: CNN with cuDNN convolutions  
**Framework**: Fortran training → Python inference  
**Dataset**: 32×32 RGB images

### 3.2 The Bug Manifestation

**Symptom**: Random predictions during Python inference
- Training: Worked perfectly (98% accuracy)
- Fortran inference: Worked perfectly
- Python inference: ~10% accuracy (random guessing!)

**Critical clue**: PyTorch model with same architecture got 98% accuracy

### 3.3 Root Cause Analysis

#### Training (Fortran)

Weights allocated as:
```fortran
! cuDNN expects (out_ch, in_ch, kH, kW) in row-major
! But Fortran allocates in column-major!
real(4), device, allocatable :: weights(:,:,:,:)
allocate(weights(out_channels, in_channels, kernel_h, kernel_w))
```

**Memory layout** (column-major):
```
out_ch varies fastest, then in_ch, then kH, then kW
```

But cuDNN interprets this as **row-major**, so it reads:
```
kW varies fastest, then kH, then in_ch, then out_ch
```

**Result**: cuDNN is reading a transposed version!

But **training still works** because:
1. cuDNN forward pass uses "transposed" weights consistently
2. cuDNN backward pass computes gradients for same "transposed" weights
3. Updates modify the "transposed" weights
4. **Everything is self-consistent within cuDNN!**

#### Saving (Fortran)

Weights saved as:
```fortran
! Save to binary file
open(unit, file='weights.bin', form='unformatted', access='stream')
write(unit) weights
close(unit)
```

This writes memory **as-is** (column-major Fortran layout).

#### Loading (Python)

Weights loaded as:
```python
# Load binary file
w = np.fromfile('weights.bin', dtype=np.float32)
w = w.reshape(out_ch, in_ch, kH, kW)  # Assumes row-major!
```

**Problem**: 
- File contains column-major data
- NumPy `reshape` assumes row-major
- **Data is completely scrambled!**

### 3.4 Mathematical Example

**Simple case**: 2×3 matrix (2 output channels, 3 input channels)

**Fortran allocation** (column-major):
```fortran
weights(1,1) = 1.0  ! First in memory
weights(2,1) = 2.0  ! Second in memory
weights(1,2) = 3.0  ! Third in memory
weights(2,2) = 4.0  
weights(1,3) = 5.0
weights(2,3) = 6.0  ! Last in memory

Memory: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

**Python reshape** (row-major):
```python
w = np.fromfile(...) # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
w = w.reshape(2, 3)  # Assumes row-major!

Result:
w[0,:] = [1.0, 2.0, 3.0]  # First row
w[1,:] = [4.0, 5.0, 6.0]  # Second row
```

**What we wanted** (original Fortran logical layout):
```
Column 1: [1.0, 2.0]
Column 2: [3.0, 4.0]
Column 3: [5.0, 6.0]

As row-major:
w[0,:] = [1.0, 3.0, 5.0]
w[1,:] = [2.0, 4.0, 6.0]
```

**Completely different!**

### 3.5 The CIFAR-10 Solution

**Transpose fix**:
```python
# Load and transpose
w = np.fromfile('weights.bin', dtype=np.float32)
w = w.reshape(kW, kH, in_ch, out_ch)  # Fortran dimension order
w = w.transpose(3, 2, 1, 0)            # Reverse to (out_ch, in_ch, kH, kW)
```

**Why this works**:
1. `reshape(kW, kH, in_ch, out_ch)` interprets column-major as if dimensions were reversed
2. `transpose(3, 2, 1, 0)` reverses them back
3. Net effect: Correctly interprets column-major data

**Limitation**: This is a workaround, not a root cause fix!

### 3.6 Lessons from CIFAR-10

✅ **What we learned**:
- Memory layout matters for multi-dimensional arrays
- Fortran and Python order differently
- Transpose can fix the issue

❌ **What we didn't understand yet**:
- WHY training worked in Fortran with "wrong" layout
- The role of cuDNN's row-major expectation
- That this would recur in every project

---

## 4. Project 2: Climate - Pattern Recognition

### 4.1 Project Context

**Task**: Weather prediction downscaling (low-res → high-res)  
**Architecture**: Similar CNN with cuDNN  
**Framework**: Fortran training → Python verification  
**Dataset**: 240×121 climate fields

### 4.2 The Bug Manifestation

**Symptom**: Exactly the same as CIFAR-10!
- Training: Excellent (98.5% accuracy)
- Python inference: Random predictions

**Key difference**: We recognized it immediately
- "This looks like the CIFAR-10 bug!"
- Applied transpose fix
- **Worked in 1 day instead of 1 week**

### 4.3 The Climate-Specific Twist

#### Multi-Channel Data

Climate model uses:
- 6 input channels (different atmospheric variables)
- Spatial dimensions: 240×121

**Weight shapes**:
```
Conv1: (out=32, in=6,  kH=3, kW=3)
Conv2: (out=64, in=32, kH=3, kW=3)
```

**Additional complexity**: Channel ordering matters!
- Input channels: [temperature, pressure, humidity, ...]
- Must preserve order through network
- Scrambled weights → scrambled channel interpretation

### 4.4 Verification Process

We added **numerical verification**:
```python
# Compare Fortran and PyTorch predictions
fortran_output = load_fortran_predictions()
pytorch_output = run_pytorch_model(same_input)

diff = np.abs(fortran_output - pytorch_output)
max_diff = np.max(diff)

assert max_diff < 1e-6, f"Mismatch: {max_diff}"
```

**Result**: After transpose fix, max diff = 2.83e-07 ✓

This confirmed:
1. The fix works
2. Fortran cuDNN == PyTorch cuDNN (bitwise identical)
3. Only the weight loading was wrong

### 4.5 False Confidence

**Dangerous assumption**: "We've solved this problem"

We had:
- ✅ Working transpose formula
- ✅ Verification test
- ✅ Two successful projects

But we still had:
- ❌ Incomplete understanding of root cause
- ❌ Reliance on "magic transpose"
- ❌ No guarantee it would work for different scenarios

**Foreshadowing**: Cryo-EM would prove this insufficient...

---

## 5. Project 3: Cryo-EM - The Critical Fix

### 5.1 Project Context

**Task**: Image denoising (noisy → clean)  
**Architecture**: Simple 3-layer CNN  
**Framework**: Fortran training → Python evaluation  
**Dataset**: 1024×1024 grayscale images

### 5.2 The Bug Manifestation

**Different symptom**: Negative correlation!
- Training appeared successful (loss decreased)
- Test predictions: Correlation = -0.14 (should be +0.87)
- Predictions systematically darker
- **Worse performance with more training epochs!**

**New clue**: Epoch 5 performed WORSE than epoch 1
- This suggested weights getting **more scrambled** over time
- Previous projects: Bug was in **loading**, not saving
- Cryo-EM: Bug was in **saving**, scrambling got worse!

### 5.3 Deep Investigation

#### The Allocation

```fortran
! conv2d_cudnn.cuf line 398
allocate(layer%weights(out_channels, in_channels, kernel_size, kernel_size))
```

**Fortran column-major**: `out_ch` varies fastest in memory

**cuDNN expectation**: row-major `(out_ch, in_ch, kH, kW)`
- Rightmost (`kW`) varies fastest
- This is **incompatible** with Fortran column-major!

#### The Saving Bug

```fortran
! cryo_train.cuf line 505 (BUGGY VERSION)
subroutine save_layer_weights(layer, prefix)
    ...
    ! WRONG: Allocate with different dimension order!
    allocate(h_weights(layer%kernel_size, layer%kernel_size, &
                       layer%in_channels, layer%out_channels))
    
    h_weights = layer%weights  ! SCRAMBLES DATA!
    ...
end subroutine
```

**What happens**:
1. `layer%weights` is `(out_ch, in_ch, kH, kW)` in column-major
   - Memory: `out_ch` varies fastest
   
2. `h_weights` is `(kH, kW, in_ch, out_ch)` in column-major
   - Memory: `kH` varies fastest
   
3. Assignment `h_weights = layer%weights` copies element-by-element:
   - `h_weights(1,1,1,1) = layer%weights(1,1,1,1)`
   - `h_weights(2,1,1,1) = layer%weights(2,1,1,1)`
   - But these map to **different linear memory locations!**

### 5.4 Mathematical Example

**Concrete example**: 2×2×2×2 tensor (simple case)

**Layer weights** (out=2, in=2, kH=2, kW=2):
```fortran
allocate(weights(2, 2, 2, 2))  ! Column-major

weights(1,1,1,1) = 1.0   ! Memory index 0
weights(2,1,1,1) = 2.0   ! Memory index 1
weights(1,2,1,1) = 3.0   ! Memory index 2
weights(2,2,1,1) = 4.0   ! Memory index 3
weights(1,1,2,1) = 5.0   ! Memory index 4
...
weights(2,2,2,2) = 16.0  ! Memory index 15

Memory: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

**Buggy save allocation** (kH=2, kW=2, in=2, out=2):
```fortran
allocate(h_weights(2, 2, 2, 2))  ! Column-major

h_weights(1,1,1,1) maps to memory index 0
h_weights(2,1,1,1) maps to memory index 1
h_weights(1,2,1,1) maps to memory index 2
...
```

**Assignment** `h_weights = weights`:
```fortran
! Fortran does element-wise copy based on LOGICAL indices
h_weights(1,1,1,1) = weights(1,1,1,1)  ! 1.0 → position 0
h_weights(2,1,1,1) = weights(2,1,1,1)  ! 2.0 → position 1
h_weights(1,2,1,1) = weights(1,2,1,1)  ! 3.0 → position 2
...

Result: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
```

**Looks the same!** But **logical interpretation is different**:

**Original** `weights(out, in, kH, kW)`:
- `weights(1,1,1,1)` = first output, first input, top-left kernel position
- `weights(2,1,1,1)` = second output, first input, top-left kernel position

**Saved** `h_weights(kH, kW, in, out)`:
- `h_weights(1,1,1,1)` = top-left kernel, first input, first output
- `h_weights(2,1,1,1)` = bottom-left kernel, first input, first output

**Same memory, completely different meaning!**

### 5.5 Python Loading (Buggy Version)

```python
# Python loads the scrambled file
w = np.fromfile('weights.bin', dtype=np.float32)

# Old (wrong) approach - try to fix with transpose
w = w.reshape(kH, kW, in_ch, out_ch)  # Assume Fortran saved this way
w = w.transpose(3, 2, 0, 1)            # Try to fix...

# Result: DOUBLE SCRAMBLED!
# 1. Data was already scrambled during Fortran save
# 2. Transpose scrambles it again differently
# 3. Complete garbage!
```

### 5.6 The Root Cause Fix

**Fortran side** - Match allocation to cuDNN expectation:
```fortran
! CORRECT VERSION
subroutine save_layer_weights(layer, prefix)
    ...
    ! Save in SAME format as allocation!
    allocate(h_weights(layer%out_channels, layer%in_channels, &
                       layer%kernel_size, layer%kernel_size))
    
    h_weights = layer%weights  ! Now preserves structure!
    ...
end subroutine
```

**Python side** - No transpose needed:
```python
# Weights are now in correct format!
w = np.fromfile('weights.bin', dtype=np.float32)
w = w.reshape(out_ch, in_ch, kH, kW)  # Direct reshape - that's it!

# No transpose! Already in PyTorch format!
model.conv.weight.data = torch.from_numpy(w)
```

### 5.7 The Breakthrough

**Why this is the TRUE fix**:

1. **Fortran allocates** `(out, in, k, k)` in column-major
   - cuDNN sees this as **transposed** but is self-consistent
   
2. **Fortran saves** same `(out, in, k, k)` in column-major
   - Memory layout preserved
   
3. **Python loads** and interprets as row-major `(out, in, k, k)`
   - This **undoes the transpose** that cuDNN saw!
   - Because: column-major (out, in, k, k) saved to file
   - Loaded as row-major (out, in, k, k)
   - The layouts "cancel out" to give correct interpretation

**The key insight**: 
- Don't try to "fix" the layout with transposes
- Instead, **be consistent in dimension ordering**
- Let the column-major → row-major conversion happen naturally during file I/O

### 5.8 Results After Fix

**Before fix**:
- Test MSE: 0.276 (40× too high!)
- Correlation: -0.136 (wrong sign!)
- PSNR: 5.59 dB (terrible)

**After fix**:
- Test MSE: 0.00696 (matches training!)
- Correlation: +0.871 (correct sign!)
- PSNR: 21.57 dB (exceeds benchmarks!)

**This proved the root cause was finally understood.**

---

## 6. Common Patterns and Variations

### 6.1 Three Manifestations of the Same Bug

| Aspect | CIFAR-10 | Climate | Cryo-EM |
|--------|----------|---------|---------|
| **Bug location** | Python loading | Python loading | Fortran saving |
| **Symptom** | Random predictions | Random predictions | Negative correlation |
| **Severity** | Complete failure | Complete failure | Progressive degradation |
| **Fix applied** | Transpose | Transpose (copied) | Root cause fix |
| **Understanding** | Partial | Pattern recognition | Complete |

### 6.2 Why Symptoms Differed

**CIFAR-10 & Climate**: Inference-only bug
- Training used consistent (though transposed) weights
- Saved weights in one format
- Loaded in different format
- **One-time scrambling** → random but stable predictions

**Cryo-EM**: Training + saving bug
- Weights scrambled during every save
- More training → more opportunities to scramble
- Scrambling accumulated over epochs
- **Progressive degradation** → worse with more training

### 6.3 Common Root Cause

All three share:

1. **Fortran column-major allocation**:
   ```fortran
   allocate(weights(out_ch, in_ch, kH, kW))
   ```

2. **cuDNN row-major expectation**:
   - Internally treats as if kW varies fastest
   - Self-consistent during training

3. **Mismatch during interoperability**:
   - CIFAR/Climate: Mismatch when loading to Python
   - Cryo-EM: Mismatch when saving from Fortran

### 6.4 The Subtle Difference

**Key distinction**:

**CIFAR-10/Climate**:
- Fortran saves `(out, in, k, k)` correctly
- Python loads and wrongly assumes different dimension order
- **Fix**: Transpose in Python

**Cryo-EM**:
- Fortran saves with DIFFERENT dimension order `(k, k, in, out)`
- This scrambles data before Python even sees it
- **Fix**: Save with same dimension order in Fortran

**Why Cryo-EM was different**:
- We had "learned" from CIFAR/Climate to add transposes
- But we didn't check the Fortran saving code
- The bug moved from Python side to Fortran side
- Our "solution" (transpose) made it worse!

---

## 7. The Math: Why Transposes Don't Always Work

### 7.1 The Transpose Fallacy

**Common misunderstanding**: "Column-major is just the transpose of row-major"

**Reality**: This is only true for **2D arrays**!

For 2D matrix A[i,j]:
- Row-major: Memory = [..., A[i,j], A[i,j+1], ...]
- Column-major: Memory = [..., A[i,j], A[i+1,j], ...]
- These ARE transposes of each other

For 4D tensor W[a,b,c,d]:
- Row-major: Memory varies as d, c, b, a
- Column-major: Memory varies as a, b, c, d
- These are **NOT simple transposes!**

### 7.2 Mathematical Proof: 4D Case

**Setup**: 4D tensor with shape `(2, 2, 2, 2)`

**Row-major indexing**:
```
index = a×(2×2×2) + b×(2×2) + c×2 + d
      = 8a + 4b + 2c + d
```

**Column-major indexing**:
```
index = a + b×2 + c×(2×2) + d×(2×2×2)
      = a + 2b + 4c + 8d
```

**Example**: Element at position (1,0,1,0)

Row-major:
```
index = 8×1 + 4×0 + 2×1 + 0 = 10
```

Column-major:
```
index = 1 + 2×0 + 4×1 + 8×0 = 5
```

**Different positions! Not a simple transpose!**

### 7.3 What Transpose Actually Does

For 4D array with shape `(A, B, C, D)`:

**Transpose (3,2,1,0)** changes shape to `(D, C, B, A)` and rearranges data so that:
```
new[d,c,b,a] = old[a,b,c,d]
```

**In memory** (row-major):
- Old: varies as d, c, b, a
- New: varies as a, b, c, d

**This effectively converts row-major to column-major ORDER, but data is still in row-major STORAGE!**

### 7.4 The File I/O Complication

**When you save to file**:
```python
# Save row-major array
arr.tofile('data.bin')

# File contains: [a₀, a₁, a₂, ..., aₙ]
# Where aᵢ are in row-major order
```

**When Fortran reads** (column-major):
```fortran
read(unit) array

! Fortran assumes memory is in column-major order
! But file contains row-major order
! SCRAMBLED!
```

**The transpose tries to pre-scramble** so that when Fortran unscrambles, it's correct:
```
Python transposes → saves → Fortran reads as column-major → unscrambles → correct!
```

But this only works if:
1. You transpose the right dimensions
2. You know the exact dimension order
3. The file format is unambiguous

### 7.5 Why Root Cause Fix is Better

**Instead of**:
```
Fortran (column-major) → save → transpose magic → Python (row-major)
```

**Do this**:
```
Fortran (column-major, dimension order A) → save → Python (row-major, same dimension order A)
```

The **column/row difference handles itself** if dimensions match!

**Mathematical reason**:

Fortran column-major `(out, in, k, k)`:
```
Memory: out varies fastest
Index: out + in×O + k₁×(O×I) + k₂×(O×I×K)
```

Python reads as row-major `(out, in, k, k)`:
```
Memory: k₂ varies fastest  
Index: out×(I×K×K) + in×(K×K) + k₁×K + k₂
```

These are **inverses** when dimension count matches!
- Fortran: out fastest → k₂ slowest
- Python: k₂ fastest → out slowest
- Perfect reversal!

But if dimensions DON'T match:
```
Fortran (out, in, k, k) → Python reads as (k, k, in, out)
```
Now the reversal is wrong → scrambled!

---

## 8. Definitive Solutions

### 8.1 Solution 1: Consistent Dimension Ordering (RECOMMENDED)

**Fortran side**:
```fortran
! Allocate in THE SAME dimension order you'll use in Python
allocate(weights(out_channels, in_channels, kernel_h, kernel_w))

! Save without changing dimension order
allocate(h_weights(out_channels, in_channels, kernel_h, kernel_w))
h_weights = weights
write(unit) h_weights
```

**Python side**:
```python
# Load and reshape to SAME dimension order
w = np.fromfile('weights.bin', dtype=np.float32)
w = w.reshape(out_ch, in_ch, kH, kW)

# No transpose needed!
model.weight.data = torch.from_numpy(w)
```

**Why this works**:
- Fortran column-major `(A,B,C,D)` → saves with A fastest
- Python row-major `(A,B,C,D)` → reads with D fastest
- The reversal is automatic and correct!

### 8.2 Solution 2: Explicit Metadata (SAFE)

**Add header to binary file**:
```fortran
! Save with metadata
write(unit) magic_number
write(unit) n_dims
write(unit) shape(1), shape(2), shape(3), shape(4)
write(unit) order_flag  ! 0=column-major, 1=row-major
write(unit) weights
```

**Python reads metadata**:
```python
magic = np.fromfile(f, dtype=np.int32, count=1)
ndims = np.fromfile(f, dtype=np.int32, count=1)
shape = np.fromfile(f, dtype=np.int32, count=ndims[0])
order = np.fromfile(f, dtype=np.int32, count=1)

data = np.fromfile(f, dtype=np.float32)

if order[0] == 0:  # Column-major
    # Reshape and transpose
    data = data.reshape(shape[::-1])  # Reverse shape
    data = np.transpose(data)          # Transpose
else:  # Row-major
    data = data.reshape(shape)
```

**Advantage**: Self-documenting, prevents mistakes  
**Disadvantage**: More complex, overhead

### 8.3 Solution 3: Use Standard Format (PORTABLE)

**Use HDF5 or NetCDF**:
```fortran
! HDF5 handles layout conversion automatically
call h5_write_dataset(file_id, "weights", weights)
```

**Python**:
```python
import h5py
with h5py.File('weights.h5', 'r') as f:
    weights = f['weights'][:]  # Automatic layout conversion!
```

**Advantage**: Standard, handles layout automatically  
**Disadvantage**: External library dependency

### 8.4 Solution 4: Test-Driven Verification

**Always include roundtrip test**:
```python
# roundtrip_test.py

# 1. Create known test pattern in Python
test_weights = create_test_pattern()
save_for_fortran(test_weights)

# 2. Fortran loads, saves back
# subprocess.run(['./fortran_roundtrip'])

# 3. Python loads result
result = load_from_fortran()

# 4. Compare
assert np.allclose(test_weights, result), "Roundtrip failed!"
```

**Test pattern** should be **asymmetric**:
```python
def create_test_pattern(out_ch=2, in_ch=3, kH=2, kW=2):
    """Create asymmetric pattern that reveals layout bugs."""
    w = np.zeros((out_ch, in_ch, kH, kW))
    
    # Each element is unique
    for o in range(out_ch):
        for i in range(in_ch):
            for h in range(kH):
                for w_idx in range(kW):
                    # Encode position in value
                    w[o,i,h,w_idx] = o*1000 + i*100 + h*10 + w_idx
    
    return w

# Example: w[1,2,0,1] = 1201
# If this ends up at wrong position, immediately obvious!
```

---

## 9. Prevention and Verification

### 9.1 Design-Time Prevention

**Rule 1**: Document layout in every array declaration
```fortran
! LAYOUT: Column-major (Fortran)
! DIMENSIONS: (out_channels, in_channels, kernel_h, kernel_w)
! INTERPRETATION: out_channels varies fastest in memory
! EXPORT FORMAT: Same dimension order, saved to binary file
real(4), allocatable :: weights(:,:,:,:)
```

**Rule 2**: Name files with dimension order
```fortran
! Instead of: 'weights.bin'
! Use: 'weights_OutInHW.bin'
write(unit) weights  ! File name documents dimension order
```

**Rule 3**: Add consistency checks
```fortran
! Compile-time shape check
if (size(weights, 1) /= out_channels) then
    print *, "ERROR: Dimension mismatch!"
    stop
endif
```

### 9.2 Runtime Verification

**Checksum test**:
```fortran
! Fortran: Save checksum with weights
real(4) :: checksum
checksum = sum(abs(weights))
write(unit) checksum
write(unit) weights
```

```python
# Python: Verify checksum
fortran_checksum = np.fromfile(f, dtype=np.float32, count=1)
weights = np.fromfile(f, dtype=np.float32)
python_checksum = np.sum(np.abs(weights))

assert np.isclose(fortran_checksum, python_checksum), "Checksum mismatch!"
```

**Shape verification**:
```python
# Save expected shape
expected_shape = (16, 1, 3, 3)  # Conv1 weights
loaded = np.fromfile('conv1_weights.bin', dtype=np.float32)

# Check total elements
expected_size = np.prod(expected_shape)
assert len(loaded) == expected_size, f"Size mismatch: {len(loaded)} vs {expected_size}"

# Reshape
try:
    loaded = loaded.reshape(expected_shape)
except ValueError as e:
    print(f"Reshape failed: {e}")
    raise
```

### 9.3 Unit Tests

**Test 1: Identity test**
```python
def test_identity():
    """Weight with value 1.0 should stay 1.0"""
    # Create all-ones weights
    ones = np.ones((16, 1, 3, 3), dtype=np.float32)
    
    # Save
    ones.tofile('test_ones.bin')
    
    # Fortran processes (reads, saves back)
    subprocess.run(['./fortran_identity'])
    
    # Load result
    result = np.fromfile('test_result.bin', dtype=np.float32)
    result = result.reshape((16, 1, 3, 3))
    
    # Should be identical
    assert np.allclose(result, ones)
```

**Test 2: Position encoding test**
```python
def test_position_encoding():
    """Encode position in value, verify preservation"""
    shape = (2, 3, 2, 2)
    w = np.zeros(shape, dtype=np.float32)
    
    # Encode position
    for o in range(shape[0]):
        for i in range(shape[1]):
            for h in range(shape[2]):
                for w_idx in range(shape[3]):
                    w[o,i,h,w_idx] = o*1000 + i*100 + h*10 + w_idx
    
    # Roundtrip
    w.tofile('test_pos.bin')
    subprocess.run(['./fortran_roundtrip'])
    result = np.fromfile('test_result.bin', dtype=np.float32).reshape(shape)
    
    # Verify every element
    for o in range(shape[0]):
        for i in range(shape[1]):
            for h in range(shape[2]):
                for w_idx in range(shape[3]):
                    expected = o*1000 + i*100 + h*10 + w_idx
                    actual = result[o,i,h,w_idx]
                    assert actual == expected, \
                        f"Position ({o},{i},{h},{w_idx}): expected {expected}, got {actual}"
```

**Test 3: Asymmetry test**
```python
def test_asymmetry():
    """Use deliberately asymmetric values"""
    w = np.array([[[[1.0, 2.0],
                     [3.0, 4.0]]]], dtype=np.float32)
    
    # This pattern breaks under transpose:
    # Transpose would give: [[[[1.0, 3.0], [2.0, 4.0]]]]
    
    w.tofile('test_asym.bin')
    subprocess.run(['./fortran_roundtrip'])
    result = np.fromfile('test_result.bin', dtype=np.float32).reshape(1,1,2,2)
    
    # Must preserve exact values
    assert np.array_equal(result, w), \
        f"Asymmetry test failed:\nExpected:\n{w}\nGot:\n{result}"
```

### 9.4 Production Monitoring

**Inference sanity check**:
```python
def verify_model_sanity(model, test_input):
    """Run sanity check on loaded model"""
    
    # Test 1: Output range
    output = model(test_input)
    assert output.min() >= 0 and output.max() <= 1, "Output out of range!"
    
    # Test 2: Output distribution
    mean_output = output.mean()
    assert 0.1 < mean_output < 0.9, f"Suspicious output mean: {mean_output}"
    
    # Test 3: Gradient check
    loss = F.mse_loss(output, torch.ones_like(output))
    loss.backward()
    grad_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "Zero gradients - model may be frozen!"
    
    print("✓ Model sanity checks passed")
```

---

## 10. Best Practices for Future Projects

### 10.1 Design Checklist

Before starting any Fortran ↔ Python project:

- [ ] **Document layouts**: Specify column-major vs row-major for every array
- [ ] **Document dimensions**: Write down exact dimension order (e.g., `(out, in, h, w)`)
- [ ] **Plan file format**: Decide on binary vs HDF5 vs NetCDF
- [ ] **Write spec**: Create interface specification document
- [ ] **Include metadata**: Plan for shape/checksum in file format
- [ ] **Design tests**: Write verification tests BEFORE implementation

### 10.2 Implementation Checklist

During development:

- [ ] **Consistent naming**: Use dimension order in variable names (`weights_OutInHW`)
- [ ] **Inline comments**: Document layout at every allocation
- [ ] **Shape assertions**: Add runtime checks for dimension sizes
- [ ] **Unit tests**: Test with asymmetric patterns
- [ ] **Roundtrip tests**: Verify Fortran→Python→Fortran preserves data
- [ ] **Checksum verification**: Compare checksums across language boundaries

### 10.3 Debugging Checklist

When bugs occur:

- [ ] **Verify shapes**: Print actual shapes on both sides
- [ ] **Check element access**: Print first/last elements, compare
- [ ] **Test with simple data**: Use all-ones, all-zeros, identity patterns
- [ ] **Test with position encoding**: Use values that encode their position
- [ ] **Visualize small arrays**: Print 2×2 or 3×3 subarrays, compare manually
- [ ] **Check file sizes**: Verify byte count matches expected size
- [ ] **Hexdump comparison**: Compare raw bytes if necessary

### 10.4 Code Review Checklist

Before merging:

- [ ] **Documented layouts**: All arrays have layout comments
- [ ] **Consistent dimensions**: Same order used throughout
- [ ] **Test coverage**: Roundtrip tests exist and pass
- [ ] **No magic transposes**: Any transpose has explanatory comment
- [ ] **Verified checksums**: Checksums match across languages
- [ ] **Sanity checks**: Model produces reasonable outputs

### 10.5 Template: Interface Specification

**Create this document for every Fortran-Python interface**:

```markdown
# Interface Specification: [Module Name]

## Data Flow
[Fortran] → Binary File → [Python]

## Array: `weights`

### Fortran Side
- **Type**: `real(4), allocatable`
- **Dimensions**: `(out_channels, in_channels, kernel_h, kernel_w)`
- **Layout**: Column-major (Fortran default)
- **Memory order**: `out_channels` varies fastest
- **File format**: Unformatted stream, no header
- **Filename**: `weights_OutInHW.bin`

### Python Side
- **Type**: `numpy.ndarray`, `dtype=float32`
- **Shape**: `(out_channels, in_channels, kernel_h, kernel_w)`
- **Layout**: Row-major (NumPy default)
- **Memory order**: `kernel_w` varies fastest
- **Load method**: `np.fromfile(...).reshape(...)`
- **Transpose needed**: No (layouts cancel)

### Verification
- **Checksum**: Sum of absolute values saved to `checksum.txt`
- **Test data**: `test_weights_OutInHW.bin` with position encoding
- **Expected checksum**: 12345.6789

### Responsible Developer
- Fortran: [Name]
- Python: [Name]
- Last Updated: [Date]
```

### 10.6 Lessons Learned Summary

**From CIFAR-10**:
✅ Memory layout matters  
✅ Test with real data early  
❌ Don't assume "it just works"

**From Climate**:
✅ Patterns repeat across projects  
✅ Verification catches bugs  
❌ Don't rely on "magic fixes" without understanding

**From Cryo-EM**:
✅ Root cause matters more than workarounds  
✅ Progressive degradation indicates accumulating error  
✅ Fix at source, not at destination

**Universal lessons**:
1. **Understand the math** - Don't guess with transposes
2. **Test early and often** - Roundtrip tests prevent pain
3. **Document everything** - Future you will thank present you
4. **Fix root causes** - Workarounds hide bugs
5. **Verify numerically** - Checksums don't lie

---

## Conclusion

The Fortran/Python memory layout bug manifested differently across three projects, but the root cause was always the same: **misalignment between column-major (Fortran) and row-major (Python/PyTorch) memory layouts for multi-dimensional arrays**.

**Key takeaways**:

1. **For 2D arrays**: Column-major ≈ transpose of row-major (simple!)
2. **For N-D arrays (N>2)**: Column-major ≠ simple transpose (complex!)
3. **The solution**: Maintain consistent dimension ordering, let the layout conversion happen naturally

**The definitive fix**:
```fortran
! Fortran: Allocate and save with same dimensions
allocate(weights(out, in, h, w))
save_to_file(weights)  ! Saves in column-major
```

```python
# Python: Load with same dimensions
w = np.fromfile('weights.bin', dtype=np.float32)
w = w.reshape(out, in, h, w)  # Interprets as row-major
# NO TRANSPOSE NEEDED - layouts cancel!
```

**Verification is critical**:
- Always test with asymmetric data
- Always verify with checksums
- Always do roundtrip tests
- Never assume "it just works"

This bug cost weeks of debugging across three projects. With this guide, future projects can avoid the same pain.

---

## Appendix A: Quick Reference

### A.1 Dimension Order Conventions

| Framework | Format | Example |
|-----------|--------|---------|
| PyTorch Conv2d | `(out_ch, in_ch, kH, kW)` | `(64, 32, 3, 3)` |
| TensorFlow Conv2d | `(kH, kW, in_ch, out_ch)` | `(3, 3, 32, 64)` |
| cuDNN (C) | `(out_ch, in_ch, kH, kW)` | `(64, 32, 3, 3)` |
| Fortran (this project) | `(out_ch, in_ch, kH, kW)` | `(64, 32, 3, 3)` |

### A.2 Memory Order

| Language | Storage | Varies Fastest |
|----------|---------|----------------|
| Fortran | Column-major | Leftmost index |
| C/C++ | Row-major | Rightmost index |
| Python/NumPy | Row-major | Rightmost index |
| MATLAB | Column-major | Leftmost index |

### A.3 Common Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Dimension mismatch | Random predictions | Match dimension order |
| Layout mismatch | Scrambled data | Document layouts |
| Missing transpose | Model fails | Add verification tests |
| Wrong transpose | Still scrambled | Understand root cause |
| Accumulating error | Worse over time | Fix at save, not load |

### A.4 Debugging Commands

```bash
# Check file size
ls -lh weights.bin
# Should match: sizeof(float) × product(dimensions)

# Hexdump first 64 bytes
hexdump -C weights.bin | head -4

# Compare checksums
# Fortran: print *, sum(abs(weights))
# Python: print(np.sum(np.abs(weights)))

# Visualize small array
python -c "import numpy as np; w=np.fromfile('weights.bin',dtype='f4'); print(w[:16].reshape(2,2,2,2))"
```

---

## Appendix B: Mathematical Derivations

### B.1 Linear Index Formulas

**Row-major** (C, Python):
```
For array A[d₀, d₁, d₂, ..., dₙ] with shape (D₀, D₁, D₂, ..., Dₙ):

linear_index = d₀ × (D₁×D₂×...×Dₙ) 
             + d₁ × (D₂×D₃×...×Dₙ)
             + d₂ × (D₃×D₄×...×Dₙ)
             + ...
             + dₙ₋₁ × Dₙ
             + dₙ
```

**Column-major** (Fortran):
```
For array A(d₀, d₁, d₂, ..., dₙ) with shape (D₀, D₁, D₂, ..., Dₙ):

linear_index = d₀
             + d₁ × D₀
             + d₂ × (D₀×D₁)
             + d₃ × (D₀×D₁×D₂)
             + ...
             + dₙ × (D₀×D₁×...×Dₙ₋₁)
```

### B.2 Proof: Column ↔ Row Conversion

**Theorem**: For matching dimension order, column-major save + row-major load preserves logical array structure.

**Proof**:

Let `A` be array with shape `(D₀, D₁, ..., Dₙ)`.

**Step 1**: Fortran allocates column-major
```
A(i₀, i₁, ..., iₙ) at memory location:
m_F = i₀ + i₁D₀ + i₂D₀D₁ + ... + iₙD₀D₁...Dₙ₋₁
```

**Step 2**: Save to file (linear memory dump)
```
File byte k contains value from memory location k
```

**Step 3**: Python loads, reshapes as row-major
```
Element at file byte m_F goes to logical position (j₀, j₁, ..., jₙ) where:
m_F = j₀D₁D₂...Dₙ + j₁D₂D₃...Dₙ + ... + jₙ₋₁Dₙ + jₙ
```

**Step 4**: Show preservation
```
We want: (j₀, j₁, ..., jₙ) = (i₀, i₁, ..., iₙ)

Setting equal and solving:
i₀ + i₁D₀ + ... + iₙD₀...Dₙ₋₁ = j₀D₁...Dₙ + ... + jₙ

This is satisfied when:
j₀ = i₀
j₁ = i₁
...
jₙ = iₙ

QED: Logical positions preserved! ∎
```

**Caveat**: This only works when **dimension orders match**. If Fortran uses `(A,B,C,D)` but Python assumes `(D,C,B,A)`, the equality breaks.

---

## Appendix C: Example Code

### C.1 Fortran: Correct Weight Saving

```fortran
module weight_io
    implicit none
contains

    subroutine save_conv_weights(filename, out_ch, in_ch, kH, kW, weights)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: out_ch, in_ch, kH, kW
        real(4), device, intent(in) :: weights(out_ch, in_ch, kH, kW)
        
        real(4), allocatable :: h_weights(:,:,:,:)
        real(4) :: checksum
        integer :: unit_num
        
        ! CRITICAL: Allocate with SAME dimension order
        allocate(h_weights(out_ch, in_ch, kH, kW))
        
        ! Copy from device to host
        h_weights = weights
        
        ! Compute checksum for verification
        checksum = sum(abs(h_weights))
        
        ! Save with metadata
        open(newunit=unit_num, file=filename, &
             form='unformatted', access='stream', status='replace')
        
        ! Write header
        write(unit_num) 2024        ! Magic number
        write(unit_num) 4           ! Number of dimensions
        write(unit_num) out_ch, in_ch, kH, kW  ! Shape
        write(unit_num) checksum    ! Verification
        
        ! Write data
        write(unit_num) h_weights
        
        close(unit_num)
        
        print '(A,F12.4)', 'Saved weights, checksum:', checksum
        
        deallocate(h_weights)
    end subroutine save_conv_weights

end module weight_io
```

### C.2 Python: Correct Weight Loading

```python
import numpy as np
import torch

def load_conv_weights(filename):
    """Load convolutional weights with verification."""
    
    with open(filename, 'rb') as f:
        # Read header
        magic = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert magic == 2024, f"Invalid magic number: {magic}"
        
        ndims = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert ndims == 4, f"Expected 4D weights, got {ndims}D"
        
        shape = tuple(np.fromfile(f, dtype=np.int32, count=4))
        fortran_checksum = np.fromfile(f, dtype=np.float32, count=1)[0]
        
        # Read weights
        weights = np.fromfile(f, dtype=np.float32)
    
    # Verify size
    expected_size = np.prod(shape)
    assert len(weights) == expected_size, \
        f"Size mismatch: {len(weights)} vs {expected_size}"
    
    # Reshape (NO TRANSPOSE!)
    weights = weights.reshape(shape)
    
    # Verify checksum
    python_checksum = np.sum(np.abs(weights))
    assert np.isclose(fortran_checksum, python_checksum, rtol=1e-5), \
        f"Checksum mismatch: {fortran_checksum} vs {python_checksum}"
    
    print(f"✓ Loaded weights {shape}, checksum verified")
    
    return torch.from_numpy(weights)
```

### C.3 Roundtrip Verification Test

```python
def test_weight_roundtrip():
    """Comprehensive roundtrip test."""
    import subprocess
    
    # Create test pattern with position encoding
    shape = (16, 3, 5, 5)  # Asymmetric to catch errors
    test_weights = np.zeros(shape, dtype=np.float32)
    
    for o in range(shape[0]):
        for i in range(shape[1]):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    # Encode position in value
                    test_weights[o,i,h,w] = o*1000 + i*100 + h*10 + w
    
    # Save for Fortran
    test_weights.tofile('test_input.bin')
    
    # Fortran reads, processes, saves back
    result = subprocess.run(['./fortran_roundtrip_test'], 
                          capture_output=True, text=True)
    assert result.returncode == 0, f"Fortran failed: {result.stderr}"
    
    # Load result
    result_weights = np.fromfile('test_output.bin', dtype=np.float32)
    result_weights = result_weights.reshape(shape)
    
    # Verify every element
    mismatches = []
    for o in range(shape[0]):
        for i in range(shape[1]):
            for h in range(shape[2]):
                for w in range(shape[3]):
                    expected = o*1000 + i*100 + h*10 + w
                    actual = result_weights[o,i,h,w]
                    if actual != expected:
                        mismatches.append(
                            f"[{o},{i},{h},{w}]: expected {expected}, got {actual}"
                        )
    
    if mismatches:
        print(f"FAILED: {len(mismatches)} mismatches:")
        for m in mismatches[:10]:  # Show first 10
            print(f"  {m}")
        raise AssertionError("Roundtrip test failed")
    
    print(f"✓ Roundtrip test passed: {np.prod(shape)} elements verified")

if __name__ == '__main__':
    test_weight_roundtrip()
```

---

**End of Guide**

This comprehensive guide captures everything we learned across three major projects. It should serve as the definitive reference for anyone working on Fortran/Python interoperability with multi-dimensional arrays.

**Share this guide with**:
- Scientific computing communities
- HPC developers
- Anyone bridging Fortran and Python
- Future project team members

**The pain of debugging this bug across three projects has been transformed into actionable knowledge. May others learn from our experience!** 🚀
