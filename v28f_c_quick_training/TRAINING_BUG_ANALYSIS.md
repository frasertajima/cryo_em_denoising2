# Training Divergence Bug - Complete Analysis

## Executive Summary

Multi-step training completely diverges despite perfect one-step convergence test. The bug appears to be in the forward pass producing incorrect outputs that contradict basic mathematics.

## The Paradox

With a simple 1→1 conv, constant input=0.5:

**Step 1:**
- Before FW: weight_mean=-0.00330, bias=0.000
- Output: -0.0232

**Step 2:**  
- Before FW: weight_mean=-0.00278, bias=0.00105
- Output: -0.0314

**Mathematical Impossibility:**
```
Step 1: output = W₁ × 0.5 + B₁ = -0.00330 × 0.5 + 0 = -0.00165 (approx)
Step 2: output = W₂ × 0.5 + B₂ = -0.00278 × 0.5 + 0.00105 = -0.00139 + 0.00105 = -0.00034

Expected: Step 2 output > Step 1 output (less negative)
Actual: Step 2 output < Step 1 output (more negative): -0.0314 < -0.0232
```

Weights increased, bias increased, input constant → output MUST increase.
But output decreased!

## Verified Facts

1. ✓ Input is constant at 0.5
2. ✓ Weights update correctly (-0.00330 → -0.00278)
3. ✓ Bias updates correctly (0 → 0.00105)
4. ✓ Updated weights persist to next iteration
5. ✓ Forward pass reads correct weight values
6. ✓ Forward pass uses `layer%weights` not `layer%grad_weights`
7. ✓ grad_output not modified by backward
8. ✓ Device synchronization present
9. ✓ One-step convergence test works perfectly
10. ✗ Forward pass produces mathematically impossible outputs

## Hypotheses

### H1: cuDNN State Corruption
- Workspace memory being reused incorrectly?
- cuDNN descriptors getting corrupted?
- Need to recreate descriptors between iterations?

### H2: Weight Layout Mismatch  
- Fortran `(out,in,k,k)` allocation is column-major
- cuDNN expects row-major NCHW `(out,in,k,k)`
- Should be `(k,k,in,out)` in Fortran?
- BUT: Convergence test worked with same layout!

### H3: Numerical Issue
- Weights are order 10⁻³, gradients are order 10⁻¹
- Gradient×LR  ≈ 10⁻⁴, comparable to weight magnitude
- Maybe precision loss in device→host→device transfers?

## Next Steps

1. **Test with weight layout fix**: Allocate as `(k,k,in,out)` instead of `(out,in,k,k)`
2. **Minimal reproduction**: Single forward pass, update ONE weight manually, forward again
3. **cuDNN descriptor investigation**: Check if descriptors need refresh
4. **Direct device kernel**: Bypass cuDNN entirely for simple test case

## Files

- Training test: `simple_overfit_test.cuf` 
- Conv module: `common/conv2d_cudnn.cuf`
- Analysis output: `training_output.log`
- This document: `TRAINING_BUG_ANALYSIS.md`
