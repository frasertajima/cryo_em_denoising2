# v28f Cryo-EM Project Status

**Date**: 2025-11-24  
**Phase**: Data Preparation (In Progress)  
**Next Session**: Training Implementation

---

## Today's Accomplishments

### ✅ Planning Complete
- [x] Comprehensive design document with research
- [x] Project checklist (10 phases)
- [x] Final decisions documented
- [x] Dataset selected: EMPIAR-10025 (41 GB frame-averaged)
- [x] Patch size: 1024×1024
- [x] Training: Synthetic noise (Poisson + Gaussian)
- [x] Infrastructure: Both managed memory AND streaming support

### ✅ Tools Created
- [x] `tools/download_empiar_10025.sh` - Download helper
- [x] `tools/preprocess_empiar.py` - Full preprocessing pipeline
- [x] `tools/visualize_patches.py` - Data visualization

### 🔄 In Progress
- [ ] Preprocessing EMPIAR-10025 (running now, ~1 hour)
  - Found 196 MRC files (7420×7676 each)
  - Extracting 1024×1024 patches (stride=512, 50% overlap)
  - Adding synthetic noise (Poisson + Gaussian, σ=0.05)
  - Train: 177 images → ~34,692 patches
  - Test: 19 images → ~3,724 patches
  - Output: 4 binary files (~150 GB total)

---

## Dataset Specifications

**Source**: EMPIAR-10025 T20S Proteasome  
**Data type**: Frame-averaged micrographs (motion-corrected)  
**Format**: MRC files  
**Size**: 41 GB (196 images)  
**Image dimensions**: 7420 × 7676 pixels  
**Pixel size**: ~0.66 Å/pixel

**After preprocessing**:
- Training patches: ~34,692 (1024×1024)
- Test patches: ~3,724 (1024×1024)
- Binary format: ~150 GB total (clean + noisy)

---

## Expected Output Files

```
data/cryo_data_streaming/
├── train_input.bin   (~141 GB - noisy patches)
├── train_target.bin  (~141 GB - clean patches)
├── test_input.bin    (~15 GB - noisy patches)
└── test_target.bin   (~15 GB - clean patches)

Total: ~312 GB (requires streaming!)
```

**Note**: This exceeds 50 GB RAM, so streaming is necessary. This is perfect - it proves our infrastructure works!

---

## Tomorrow's Plan: Training Implementation

### Session 2 Checklist

1. **Copy U-Net infrastructure** from v28e_climate_cnn
   - conv2d_cudnn.cuf
   - pooling_cudnn.cuf
   - unet_blocks.cuf
   - training_export.cuf, unet_export.cuf

2. **Create new modules**:
   - `common/cryo_unet.cuf` (1 channel in/out)
   - `common/streaming_image_loader.cuf` (1 channel loader)
   - `data/cryo_config.cuf` (dataset configuration)

3. **Write training program**:
   - `cryo_train_unet.cuf` (main training loop)
   - Support `--stream` flag
   - MSE loss, Adam optimizer
   - Checkpoint saving

4. **Build system**:
   - `compile.sh` (adapted from climate)
   - Compile all modules
   - Run tests

5. **Initial training**:
   - 5-10 epochs (proof of concept)
   - Monitor loss, GPU memory
   - Export samples for visualization

**Expected time**: 2-3 hours

---

## Performance Expectations

Based on climate project (240×121 → 1024×1024 is 18.5x more pixels):

| Metric | Climate | Cryo-EM (Estimate) |
|--------|---------|-------------------|
| Patch size | 256×128 | 1024×1024 |
| Pixels per sample | 32,768 | 1,048,576 (32x more) |
| Batch size | 8 | 2-4 (adjust for memory) |
| GPU memory | ~4 GB | ~6-7 GB (larger activations) |
| Throughput | 93 samples/sec | ~20-30 samples/sec (estimated) |
| Training time (15 epochs) | ~2 hours | ~4-6 hours (estimated) |

**Note**: These are rough estimates. Actual performance depends on:
- Batch size we can fit
- SSD read speed for streaming
- cuDNN performance on 1024×1024 convolutions

---

## Success Criteria Reminder

### Minimum (MVP)
- ✓ Compiles without errors
- ✓ Trains on preprocessed data
- ✓ Loss decreases
- ✓ Visual inspection shows denoising works
- ✓ PSNR > 18 dB (better than noisy input)

### Target (Good)
- ✓ PSNR 20-21 dB (match Topaz-Denoise)
- ✓ SSIM > 0.8
- ✓ PyTorch verification passes
- ✓ Streaming works on 150 GB dataset

### Stretch (Excellent)
- ✓ PSNR > 21 dB
- ✓ SSIM > 0.85
- ✓ 2-3x faster than PyTorch (with optimization)

---

## Current Preprocessing Progress

**Status**: Running (started at current time)  
**Estimated completion**: ~1 hour  
**Processing**: 177 training images @ 20 sec/image

Once complete, you'll have binary files ready for training!

---

**Next Action**: Wait for preprocessing to complete, then visualize data to verify quality.
