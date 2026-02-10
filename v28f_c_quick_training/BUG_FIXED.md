# BUG FIXED: Fortran Variable Initialization Issue

## The Bug

In `common/conv2d_cudnn.cuf`, the forward and backward passes had:

```fortran
real(4), target :: alpha = 1.0, beta = 0.0
```

In Fortran, when you initialize a local variable like this, it has **implicit SAVE attribute**. This means the initialization only happens on the FIRST call, and the variable retains its value between calls!

## What Went Wrong

### Forward Pass (`conv2d_forward`)
1. First call: `beta = 0.0` (initialized)
2. During first call: `beta = 1.0` (modified for bias addition)
3. **Second call: `beta = 1.0` (NOT re-initialized!)**
4. Result: Convolution output accumulated with previous output!

```fortran
! First call: Correct
output = 1.0 * (weights * input) + 0.0 * old_output  ✓

! Second call: WRONG - beta still 1.0!
output = 1.0 * (weights * input) + 1.0 * old_output  ✗
```

## The Evidence

**Test Case**: weight=0, bias=0.5, input=0.5
- Expected output: 0.5
- Actual output (buggy): 0.94 (almost double!)
- Actual output (fixed): 0.5 ✓

The extra 0.44 was the accumulated previous output!

## The Fix

Separate declaration from initialization:

```fortran
! Before (BUGGY):
real(4), target :: alpha = 1.0, beta = 0.0

! After (CORRECT):
real(4), target :: alpha, beta

! Then explicitly initialize on EVERY call:
alpha = 1.0
beta = 0.0
```

## Impact

This bug caused:
- ✗ Training divergence (loss exploding to 189)
- ✗ Output moving in wrong direction despite correct gradient updates
- ✗ Forward pass producing mathematically impossible results

After fix:
- ✓ Training converges (loss decreases from 0.27 to 0.004)
- ✓ Output moves in correct direction
- ✓ Forward pass produces correct results

## Files Modified

- `v28f_cryo_em/v28f_c_quick_training/common/conv2d_cudnn.cuf`
  - `conv2d_forward`: Lines ~495-500
  - `conv2d_backward`: Lines ~554-562

## Lesson Learned

**Fortran implicit SAVE semantics for initialized local variables!**

When you write:
```fortran
real :: x = 1.0
```

It's equivalent to:
```fortran
real, save :: x = 1.0
```

Always explicitly re-initialize variables if you need fresh values on each call!
