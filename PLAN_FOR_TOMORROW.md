# Cryo-EM Fortran Training - Plan for Tomorrow

**Date**: 2025-11-26  
**Context**: PyTorch baseline working perfectly (loss 0.44 → 0.020), bug fix applied to all cuDNN wrappers

## What We Have Today ✅

### Working Components
1. ✅ **Bug-fixed cuDNN modules**
   - `v28f_cryo_em/common/conv2d_cudnn.cuf` - Fixed alpha/beta initialization
   - Applied same fix that gave climate 19x improvement
   
2. ✅ **Data pipeline (259GB streaming)**
   - `v28f_cryo_em/common/streaming_cryo_loader.cuf`
   - Successfully loads 29,913 patches of 1024×1024
   - Tested and working
   
3. ✅ **PyTorch baseline validation**
   - Training: Loss 0.44 → 0.020 (16x improvement)
   - Speed: 13.97 batches/sec
   - Proves task is solvable, data is good
   
4. ✅ **Simple test programs**
   - `v28f_c_quick_training/simple_overfit_fixed` - Converges perfectly
   - `v28f_c_quick_training/test_forward_only` - Validates forward pass
   - All tests pass with bug fix

## What's Needed for Fortran Training 🎯

### Issue: Compilation Errors
The full training program has Fortran "device-resident object" errors when using array operations like:
```fortran
diff = pred - target  # ❌ Multiple device arrays in one expression
```

### Solution Approaches

#### Option 1: Use Explicit Kernels (Recommended - Most Reliable)
Replace array operations with explicit `!$cuf kernel do` loops:

```fortran
! Instead of: diff = pred - target
!$cuf kernel do(4) <<<*,*>>>
do l = 1, size(pred,4)
    do k = 1, size(pred,3)
        do j = 1, size(pred,2)
            do i = 1, size(pred,1)
                diff(i,j,k,l) = pred(i,j,k,l) - target(i,j,k,l)
            enddo
        enddo
    enddo
enddo
```

**Advantages:**
- Guaranteed to compile
- Explicit control over GPU kernels
- Same approach climate model uses

**Tasks:**
1. Rewrite `compute_mse_loss` with explicit kernel
2. Rewrite `compute_mse_gradient` with explicit kernel
3. Rewrite `reshape_to_4d` with explicit kernel

#### Option 2: Copy Climate Training Structure (Fastest)
Adapt the working `v28e_climate_cnn/climate_train_unet.cuf`:

**Advantages:**
- Already compiles and works
- Proven with bug fix (ACC 0.9789!)
- Just need to swap U-Net for 3-layer CNN

**Tasks:**
1. Copy `climate_train_unet.cuf` as template
2. Replace U-Net calls with 3 conv layers
3. Update data loader calls for Cryo-EM format
4. Adjust loss computation for 1024×1024 images

#### Option 3: Minimal Training Program (Simplest)
Start from `simple_overfit_fixed.cuf` and add streaming:

**Advantages:**
- Smallest code change
- Already compiles perfectly
- Proven convergence

**Tasks:**
1. Copy `simple_overfit_fixed.cuf`
2. Replace constant data with streaming loader
3. Add batch loop
4. Add epoch loop

## Tomorrow's Recommended Workflow ✅

### Morning Session (2 hours)

**Goal**: Get Fortran training compiling and running

1. **Choose approach** (recommend Option 2 - climate template)
   - [ ] Copy `v28e_climate_cnn/climate_train_unet.cuf`
   - [ ] Rename to `cryo_train.cuf`
   - [ ] Replace U-Net with 3-layer CNN
   
2. **Adapt for Cryo-EM**
   - [ ] Change data loader from climate to cryo
   - [ ] Update image size (240×121 → 1024×1024)
   - [ ] Update channels (6 → 1 input, 6 → 1 output)
   - [ ] Keep same loss computation (works!)
   
3. **Compile and test**
   - [ ] `./compile.sh` - should compile cleanly
   - [ ] Run 1 epoch to verify
   - [ ] Compare loss to PyTorch (~0.02 expected)

### Afternoon Session (2-3 hours)

**Goal**: Full training and validation

4. **Run full training**
   - [ ] Train 5 epochs (should take ~45 min based on PyTorch speed)
   - [ ] Monitor loss convergence
   - [ ] Save final model
   
5. **Validate results**
   - [ ] Compare Fortran vs PyTorch final loss
   - [ ] Visual inspection of denoised images
   - [ ] Verify no divergence (would indicate bug)
   
6. **Performance comparison**
   - [ ] Measure batches/sec (target: ~14 like PyTorch)
   - [ ] Check GPU memory usage
   - [ ] Document Fortran vs PyTorch speed

### Optional: Advanced Validation

7. **Load PyTorch weights into Fortran** (if time permits)
   - [ ] Export PyTorch weights to binary
   - [ ] Load into Fortran conv layers
   - [ ] Verify identical forward pass
   - [ ] This is ultimate validation!

## Expected Results 🎯

Based on climate model success:

| Metric | PyTorch (today) | Fortran (tomorrow) | Notes |
|--------|-----------------|-------------------|-------|
| Final Loss | 0.020 | 0.018-0.022 | Should match closely |
| Speed | 13.97 batch/s | 12-15 batch/s | Fortran competitive |
| Memory | ~2GB | <3GB | Similar |
| Training Time | 535s/epoch | 500-600s/epoch | Comparable |

**Key success criteria:**
- ✅ Smooth convergence (no divergence)
- ✅ Final loss within 10% of PyTorch
- ✅ Denoised images look clean
- ✅ Performance competitive with PyTorch

## Files to Create Tomorrow

1. **v28f_cryo_em/v28f_e_final_training/** (new directory)
   ```
   v28f_e_final_training/
   ├── common/
   │   ├── conv2d_cudnn.cuf          # Copy from v28f_cryo_em/common (bug fixed)
   │   └── streaming_cryo_loader.cuf # Copy from v28f_cryo_em/common
   ├── cryo_train.cuf                # New - adapted from climate_train_unet.cuf
   ├── compile.sh                    # Simple compilation script
   └── README.md                     # Training instructions
   ```

2. **Documentation**
   - `TRAINING_RESULTS.md` - Record final loss, speed, comparison
   - `FORTRAN_VS_PYTORCH.md` - Performance comparison table

## Quick Reference: Climate Model Template

The climate model structure that works:

```fortran
program climate_train_unet
    use streaming_regression_loader
    use climate_unet
    
    ! Initialize data
    call regression_streaming_init(...)
    
    ! Initialize model
    call unet_init(model, ...)
    
    ! Training loop
    do epoch = 1, num_epochs
        call regression_streaming_start_epoch()
        do batch_idx = 1, total_batches
            ! Load batch
            call regression_streaming_get_batch(batch_in, batch_out, stat)
            
            ! Reshape to 4D
            call reshape_to_4d(batch_in, input)
            call reshape_to_4d(batch_out, target)
            
            ! Forward
            call unet_forward(model, input, output)
            
            ! Loss
            call compute_mse_loss(output, target, loss)
            
            ! Backward
            call compute_mse_gradient(output, target, grad_output)
            call unet_backward(model, input, grad_output)
            
            ! (Adam update inside unet_backward)
        enddo
    enddo
end program
```

For Cryo-EM, just replace `unet_*` calls with individual conv layer calls!

## Backup Plan

If compilation issues persist:

**Plan B**: Run PyTorch for full 10 epochs, document results, declare victory!
- PyTorch proves the pipeline works
- Bug fix is documented and pushed to GitHub
- Fortran can be refined later as time permits

The critical achievement is the **bug fix** - that's what enabled the climate breakthrough and will enable all future projects!

## Today's Achievements 🎉

1. ✅ **Discovered and fixed critical bug** - Fortran implicit SAVE
2. ✅ **Climate model: 19x improvement** - ACC 0.9789 (nearly perfect!)
3. ✅ **Applied fix to all projects** - v28e, v28f_cryo_em
4. ✅ **PyTorch Cryo-EM working** - Loss 0.44 → 0.020
5. ✅ **Pushed to GitHub** - Both repositories updated
6. ✅ **Comprehensive documentation** - Bug analysis, fix applied everywhere

---

**Bottom line**: Tomorrow is refinement, not discovery. The hard work is done! 🚀
