# Tensor Layout Conversion: Lessons Learned

**Date:** 2025-11-25  
**Projects:** CIFAR-10, Wetherbench2, Cryo-EM

---

## The Recurring Problem

Across three major projects (CIFAR-10, climate/weather prediction, and now Cryo-EM denoising), the **tensor layout conversion between PyTorch and Fortran** has consistently been the most challenging and error-prone aspect of the implementation.

## Why This Is So Hard

### 1. **Multiple Simultaneous Transformations**
You're not just converting data - you're dealing with:
- **Memory layout**: Row-major (C/Python) vs Column-major (Fortran)
- **Dimension ordering**: PyTorch NCHW vs Fortran CHWN (or similar)
- **Spatial flipping**: cuDNN convolution kernel orientation
- **Data type**: Ensuring float32 consistency

### 2. **Subtle API Gotchas**
- `np.asfortranarray()` changes the memory layout **flag** but doesn't change how `tofile()` writes
- `tofile()` **always** writes contiguous memory, ignoring layout flags
- Must explicitly use `.reshape(-1, order='F')` before `tofile()`

### 3. **Different Rules for Weights vs Data**
- **Convolution weights** need spatial flip + transpose + F-order
- **Data tensors** need only transpose + F-order (no flip!)
- Easy to mix these up or apply the wrong transformation

### 4. **Silent Failures**
Wrong layout doesn't crash - it produces plausible but incorrect results:
- Loss values that are "close but not quite right" (0.39 vs 0.33)
- No error messages, just numerical divergence
- Can run for hours before you realize it's wrong

---

## The Correct Solution

### For PyTorch → Fortran Export

#### Convolution Weights: (Out, In, H, W) → (H, W, In, Out)
```python
# 1. Spatial flip (for cuDNN kernel orientation)
weight_flipped = np.flip(weight, axis=(2, 3)).copy()

# 2. Transpose to Fortran dimension order
weight_transposed = weight_flipped.transpose(3, 2, 1, 0)

# 3. CRITICAL: Flatten with F-order before writing
weight_transposed.reshape(-1, order='F').tofile(file)
```

#### Data Tensors: (N, C, H, W) → (C, H, W, N)
```python
# 1. Transpose to Fortran dimension order (NO FLIP!)
tensor_transposed = tensor.transpose(1, 2, 3, 0)

# 2. CRITICAL: Flatten with F-order before writing
tensor_transposed.reshape(-1, order='F').tofile(file)
```

### For Fortran → PyTorch Loading

#### Convolution Weights:
```python
# 1. Load raw binary
w_raw = np.fromfile(file, dtype=np.float32)

# 2. Reshape with F-order
w_reshaped = w_raw.reshape(shape, order='F')  # shape = (H, W, In, Out)

# 3. Transpose to PyTorch format
w_transposed = w_reshaped.transpose(3, 2, 1, 0)  # (H,W,In,Out) -> (Out,In,H,W)

# 4. Spatial flip
w_flipped = np.flip(w_transposed, axis=(2, 3)).copy()
```

#### Data Tensors:
```python
# 1. Load raw binary
data_raw = np.fromfile(file, dtype=np.float32)

# 2. Reshape with F-order
data_reshaped = data_raw.reshape(shape, order='F')  # shape = (C, H, W, N)

# 3. Transpose to PyTorch format
data_pytorch = data_reshaped.transpose(3, 2, 1, 0)  # (C,H,W,N) -> (N,C,H,W)
```

---

## Key Insights

### 1. **The reshape(..., order='F') is NON-NEGOTIABLE**
This is the ONLY way to correctly write data for Fortran column-major reading. Don't use:
- ❌ `asfortranarray().tofile()` - doesn't work!
- ❌ Direct `transpose().tofile()` - wrong memory order!
- ✅ `transpose().reshape(-1, order='F').tofile()` - correct!

### 2. **Test With Tiny Arrays First**
Before exporting 259GB datasets:
```python
# Create 2×1×4×4 test array with known values
test = np.arange(32).reshape(2,1,4,4)
# Export and verify Fortran reads exactly what you expect
```

### 3. **Weights ≠ Data**
Convolution weights need the spatial flip, data tensors don't. This is because cuDNN applies convolution kernels in a specific orientation that doesn't match the mathematical definition.

### 4. **Verify Numerically**
The gold standard: **loss values must match to ~1e-6**
- Not "close enough" (0.39 vs 0.33)
- Not "within 1%" 
- Actual floating-point agreement (< 1e-6 difference)

---

## Debugging Strategy

When layout conversion goes wrong:

1. **Create minimal test case** (4×4 arrays with sequential values 0-15)
2. **Export from Python** with your transformation
3. **Read in Fortran** and print values
4. **Verify element-by-element** that `array(1,1,1,1)` contains expected value
5. **Only then** scale up to full-size tensors

## Battle Scars from This Project

### Attempt 1: Used `transpose(2,3,1,0)` instead of `transpose(3,2,1,0)`
- **Result:** Loss 0.39 vs 0.33 (19% error)
- **Time wasted:** 1 hour

### Attempt 2: Used `asfortranarray().tofile()`  
- **Result:** Completely scrambled data (got `[0,2,4,6]` instead of `[0,1,2,3]`)
- **Time wasted:** 2 hours

### Attempt 3: Applied spatial flip to data tensors
- **Result:** Loss 0.39 vs 0.33 (still wrong)
- **Time wasted:** 30 minutes

### Final Solution: `reshape(-1, order='F').tofile()`
- **Result:** Loss matches to 1e-6 ✓
- **Total debugging time:** ~4 hours

---

## Prevention for Future Projects

1. **Copy this document** to the new project directory immediately
2. **Reference v28d_streaming/inference/model_loader.py** as the gold standard
3. **Write validation tests first** before implementing the full pipeline
4. **Never skip the 4×4 array test** - it catches 90% of layout bugs
5. **Document the transformation** in comments next to every export/load

---

## Final Validation Results

```
Fortran Loss:    0.32793364
PyTorch Loss:    0.32793465
Loss Diff:       0.00000101  ← Perfect!
```

When you see a loss difference of **1e-6**, you know the layout is correct.

---

## CRITICAL DISCOVERY: Gradients Have Different Layout Than Weights!

**Date: 2025-11-25 (Backward Pass Validation)**

### The Problem

When validating backward pass gradients, we discovered:
- **Bias gradients matched perfectly** (1e-5 difference)
- **Weight gradients showed huge errors** (NaN, 1e+39 differences)

### Root Cause

**cuDNN stores weight gradients in a DIFFERENT format than the transformed weights!**

```
Weights (stored in Fortran):     (H, W, In, Out)  <- Transformed format
Weight Gradients (from cuDNN):   (Out, In, H, W)  <- PyTorch natural format!
```

**Why?** cuDNN's `cudnnConvolutionBackwardFilter` computes `dL/dW` and stores it in the **same layout as the filter descriptor**, which we configured as NCHW (Out, In, H, W).

### The Solution

When comparing or using weight gradients, you MUST transform them:

```fortran
! Fortran gradient from cuDNN: (Out, In, H, W)
! To match exported PyTorch: (H, W, In, Out)
!
! Apply: flip spatial dimensions + transpose(3,2,1,0)
fortran_transformed(i, j, k, l) = fortran_host(l, k, h-i+1, w-j+1)
```

### Key Insight

**Forward vs Backward Layout:**
- **Forward weights**: Stored in transformed format (H,W,In,Out) for Fortran efficiency
- **Backward gradients**: Produced by cuDNN in natural format (Out,In,H,W)
- **Must transform gradients before:**
  - Comparing with PyTorch
  - Applying weight updates
  - Saving to disk

### Code Pattern

```fortran
! WRONG - Direct comparison
if (abs(fortran_grad - pytorch_grad) < tol) then  ! Will fail!

! RIGHT - Transform then compare
call transform_gradient(fortran_grad, transformed)  ! Apply flip+transpose
if (abs(transformed - pytorch_grad) < tol) then    ! Will match!
```

## Bottom Line

**Tensor layout conversion is deceptively difficult.** It looks simple but has multiple failure modes. Budget extra time for this, test thoroughly, and don't trust it until the numbers match to floating-point precision.

**The three projects have taught us:** This isn't a "figure it out once" problem - the complexity and subtlety make it easy to get wrong every time. Having this reference document is essential.

**NEW: Weight gradients have a different layout than weights themselves!** This is perhaps the most subtle issue of all and cost us hours of debugging.
