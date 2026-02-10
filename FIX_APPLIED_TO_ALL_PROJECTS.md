# Critical Bug Fix Applied to All Projects

## Bug Summary

**Fortran implicit SAVE semantics** caused `beta` variable to retain value between function calls, leading to output accumulation instead of replacement.

## Projects Updated

✓ **v28f_cryo_em** - Cryo-EM denoising (where bug was found and fixed)
  - `common/conv2d_cudnn.cuf`
  - `v28f_a_simple_cnn/common/conv2d_cudnn.cuf`
  - `v28f_b_cudnn_test/common/conv2d_cudnn.cuf`
  - `v28f_c_quick_training/common/conv2d_cudnn.cuf`

✓ **v28e_climate_cnn** - Climate U-Net (72GB dataset)
  - `common/conv2d_cudnn.cuf`

## Impact

**Before Fix:**
- Training diverged (loss exploding to 100+)
- Output accumulated previous results
- Mathematically impossible behavior

**After Fix:**
- Training converges normally
- Loss decreases as expected
- Correct forward/backward pass behavior

## The Fix

Changed in both `conv2d_forward` and `conv2d_backward`:

```fortran
! Before (BUGGY):
real(4), target :: alpha = 1.0, beta = 0.0

! After (CORRECT):
real(4), target :: alpha, beta
...
alpha = 1.0
beta = 0.0
```

## Verification

Tested with:
- Simple overfit test (constant image) ✓
- 1x1 convolution test ✓
- Single-step gradient test ✓
- Multi-step training ✓

All tests now pass correctly.

## Date Applied

November 25, 2025

## Files Modified

- v28f_cryo_em/common/conv2d_cudnn.cuf
- v28f_cryo_em/v28f_a_simple_cnn/common/conv2d_cudnn.cuf  
- v28f_cryo_em/v28f_b_cudnn_test/common/conv2d_cudnn.cuf
- v28f_cryo_em/v28f_c_quick_training/common/conv2d_cudnn.cuf
- v28e_climate_cnn/common/conv2d_cudnn.cuf

## Note

Any other projects using `conv2d_cudnn.cuf` will need this fix applied!
