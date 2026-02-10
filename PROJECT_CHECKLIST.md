# v28f Cryo-EM Project Checklist

**Methodology**: Replicate the successful v28e_climate_cnn workflow

This checklist mirrors exactly what worked for the climate project. Each phase builds on the previous one, with clear validation steps.

---

## Phase 1: Planning & Design ✓ (CURRENT SESSION)

**Status**: IN PROGRESS

- [x] Research Cryo-EM datasets (EMPIAR, CryoCRAB, etc.)
- [x] Research denoising benchmarks (Topaz-Denoise, PSNR/SSIM metrics)
- [x] Choose task scope (2D denoising, not 3D reconstruction)
- [x] Design data format (MRC → streaming binary)
- [x] Plan U-Net architecture (1 channel vs 6 channels)
- [x] Write comprehensive design document
- [ ] **USER REVIEW & DECISIONS**
  - Dataset choice (EMPIAR subset? Size?)
  - Patch size (512×512 vs 1024×1024)
  - Training approach (synthetic first vs Noise2Noise)

**Climate comparison**: ✓ Same - we planned thoroughly before coding

---

## Phase 2: Data Preparation

**Goal**: Create streaming binary data files for training

### 2.1 Small Dataset (Prototype)
- [ ] Download small EMPIAR dataset or create synthetic data (~1-5 GB)
- [ ] Write Python MRC reader
  - [ ] Parse MRC header (dimensions, pixel size)
  - [ ] Read pixel data (handle different modes)
  - [ ] Test on sample files
- [ ] Implement preprocessing
  - [ ] 95th percentile normalization
  - [ ] Patch extraction (512×512 or 1024×1024)
  - [ ] Handle edge cases
- [ ] Create synthetic noisy pairs
  - [ ] Add Gaussian noise (σ = 0.1, 0.2)
  - [ ] Visualize noisy vs clean
- [ ] Convert to streaming binary format
  - [ ] Sample-major ordering
  - [ ] Float32 precision
  - [ ] Verify file sizes
- [ ] **VALIDATION**: Python visualization of patches

**Climate comparison**: Similar - we converted NetCDF → binary

### 2.2 Tools Creation
- [ ] Write `tools/mrc_to_streaming.py`
- [ ] Write `tools/create_synthetic_data.py`
- [ ] Write `tools/visualize_patches.py`
- [ ] Document usage in `data/README_DATA.md`

**Climate comparison**: We created conversion tools for WeatherBench2 data

---

## Phase 3: Minimal Viable Implementation

**Goal**: Get basic training loop compiling and running

### 3.1 Copy Infrastructure from v28e_climate_cnn
- [ ] Copy `common/conv2d_cudnn.cuf`
- [ ] Copy `common/pooling_cudnn.cuf`
- [ ] Copy `common/unet_blocks.cuf`
- [ ] Copy `common/cmdline_args.cuf`
- [ ] Copy `common/training_export.cuf`
- [ ] Copy `common/unet_export.cuf`

**Climate comparison**: ✓ Exact same modules reused

### 3.2 Create New Modules
- [ ] Write `common/streaming_image_loader.cuf`
  - Adapt from `streaming_regression_loader.cuf`
  - Single channel instead of multi-channel
  - Load patches from binary files
- [ ] Write `common/cryo_unet.cuf`
  - Adapt from `climate_unet.cuf`
  - Input: 1 channel (grayscale)
  - Output: 1 channel (denoised)
  - Padding: 512×512 → 512×512 (no padding if power of 2)
- [ ] Write `data/cryo_config.cuf`
  - Dataset paths
  - Dimensions (1 channel, 512×512 or 1024×1024)
  - Train/test split

**Climate comparison**: Same pattern - adapt config and model for new dataset

### 3.3 Training Program
- [ ] Write `cryo_train_unet.cuf`
  - Adapt from `climate_train_unet.cuf`
  - MSE loss (same as climate)
  - Adam optimizer (same as climate)
  - Command-line args (--stream, --epochs, --lr, --save)
  - Checkpoint saving
  - Sample export for verification

**Climate comparison**: Nearly identical structure

### 3.4 Build System
- [ ] Write `compile.sh`
  - Adapt from v28e_climate_cnn/compile.sh
  - Update module names (cryo_unet vs climate_unet)
  - Update paths
- [ ] **TEST**: Run `./compile.sh`
- [ ] **VALIDATION**: All modules compile without errors

**Climate comparison**: ✓ Same build process

---

## Phase 4: Unit Testing

**Goal**: Verify each component works in isolation

- [ ] Write `tests/test_conv2d.cuf`
  - Copy from v28e, adapt for 1 channel
  - Test forward/backward pass
  - Verify gradients
- [ ] Write `tests/test_pooling.cuf`
  - Copy from v28e (no changes needed)
  - Test maxpool, upsample
- [ ] Write `tests/test_unet_blocks.cuf`
  - Copy from v28e, adapt for 1 channel
  - Test encoder/decoder
  - Test skip connections
- [ ] Write `tests/test_cryo_unet.cuf`
  - Adapt from test_climate_unet.cuf
  - Test full forward pass (1 channel in/out)
  - Test gradient flow
  - Verify output shape
- [ ] Update `compile.sh` to build tests
- [ ] **RUN ALL TESTS**: `./test_*`
- [ ] **VALIDATION**: All tests pass

**Climate comparison**: ✓ Same testing methodology, 20 tests total

---

## Phase 5: Small-Scale Training (Proof of Concept)

**Goal**: Prove denoising works on small synthetic dataset

- [ ] Create tiny dataset (1000 patches, ~8 MB)
  - Clean patches
  - Noisy patches (Gaussian σ=0.2)
- [ ] Train for 10 epochs
  - Batch size 8
  - Learning rate 0.0001
  - Monitor loss decrease
- [ ] Export weights and sample outputs
  - `saved_models/cryo_unet/debug_weights/`
  - `sample_0000/` (input, output, target)
- [ ] **VALIDATION**: 
  - Loss decreases monotonically
  - Visual inspection: output is less noisy than input
  - PSNR improves vs input

**Climate comparison**: Same - small test before scaling

---

## Phase 6: PyTorch Verification

**Goal**: Verify our implementation matches PyTorch exactly

### 6.1 PyTorch Model
- [ ] Write `inference/cryo_unet.py`
  - Adapt from climate_unet.py
  - 1 channel input/output
  - Same architecture (32→64→128→256)
- [ ] Write weight loader
  - Load Fortran binary exports
  - Map to PyTorch parameters

**Climate comparison**: Same verification approach

### 6.2 Verification Script
- [ ] Write `inference/verify_fortran_pytorch.py`
  - Load Fortran weights
  - Load sample input/output
  - Run PyTorch forward pass
  - Compare outputs
- [ ] **RUN**: `python verify_fortran_pytorch.py`
- [ ] **VALIDATION**: Max diff < 1e-6

**Climate comparison**: ✓ Exact same validation (achieved 2.83e-07)

### 6.3 Training Step Benchmark
- [ ] Write `tests/test_training_step.cuf`
  - Single batch forward + backward + Adam
  - Time each phase
- [ ] Write `inference/verify_training_step.py`
  - Equivalent PyTorch training step
  - Time comparison
- [ ] **VALIDATION**: Fortran faster than PyTorch

**Climate comparison**: We achieved 1.3x speedup, expect similar

---

## Phase 7: Scale to Real Data (Streaming Proof)

**Goal**: Train on ~80 GB dataset using 8GB GPU

### 7.1 Dataset Preparation
- [ ] Download EMPIAR dataset subset (~80 GB)
  - Or find appropriate EMPIAR dataset in this size range
- [ ] Preprocess micrographs
  - Extract patches
  - Normalize (95th percentile)
- [ ] Create training pairs
  - **Option A**: Synthetic noise (Gaussian/Poisson)
  - **Option B**: Noise2Noise (odd/even frames)
- [ ] Convert to streaming format
  - `data/cryo_data_streaming/train_input.bin`
  - `data/cryo_data_streaming/train_target.bin`
  - `data/cryo_data_streaming/test_input.bin`
  - `data/cryo_data_streaming/test_target.bin`
- [ ] **VALIDATION**: Verify file sizes (~80 GB total)

**Climate comparison**: Same - 72 GB streaming dataset

### 7.2 Full Training
- [ ] Train with streaming enabled
  - `./cryo_train_unet --stream --epochs 15 --save`
  - Monitor GPU memory (~4 GB expected)
  - Monitor throughput (samples/sec)
- [ ] Checkpoint best model (validation loss)
- [ ] Export samples for analysis
- [ ] **VALIDATION**: 
  - Training completes without crashes
  - GPU memory < 8 GB
  - Loss decreases
  - Best model saved

**Climate comparison**: ✓ Proved streaming works, expect same

---

## Phase 8: Evaluation & Benchmarking

**Goal**: Measure PSNR/SSIM, compare to baselines

### 8.1 Metrics Computation
- [ ] Write `inference/compute_metrics.py`
  - Load test set
  - Run inference (Fortran or PyTorch)
  - Compute PSNR, SSIM
  - Per-image and average
- [ ] **RUN**: Compute metrics on test set
- [ ] **VALIDATION**: PSNR > 18 dB (better than no denoising)

**Climate comparison**: We measured MSE/ACC, this uses PSNR/SSIM

### 8.2 Baseline Comparisons
- [ ] No denoising (raw input): PSNR baseline
- [ ] Gaussian blur: Simple baseline
- [ ] PyTorch U-Net: Verify equivalent performance
- [ ] (Optional) Topaz-Denoise: If time permits

**Climate comparison**: We compared to persistence forecast

### 8.3 Visual Analysis
- [ ] Side-by-side comparisons (noisy, denoised, target)
- [ ] Difference maps
- [ ] Frequency analysis (power spectra)
- [ ] **VALIDATION**: No obvious artifacts

**Climate comparison**: Similar visual validation

### 8.4 Jupyter Notebooks
- [ ] `notebooks/cryo_unet_analysis.ipynb`
  - Weight visualization
  - Activation maps
  - Output comparisons
- [ ] `notebooks/cryo_unet_evaluation.ipynb`
  - PSNR/SSIM plots
  - Per-image metrics
  - Histogram analysis
- [ ] **VALIDATION**: Results look reasonable

**Climate comparison**: ✓ Same notebook structure

---

## Phase 9: Documentation & Release

**Goal**: Complete documentation for public release

### 9.1 README
- [ ] Write comprehensive `README.md`
  - Project overview
  - Quick start guide
  - Training results (PSNR/SSIM)
  - Architecture description
  - Command-line options
  - PyTorch verification results
  - Repository structure
  - Citation
- [ ] Write `data/README_DATA.md`
  - Dataset download instructions
  - MRC format explanation
  - Preprocessing pipeline
  - Streaming format specification

**Climate comparison**: ✓ Same documentation structure

### 9.2 Repository Cleanup
- [ ] Add `.gitignore`
  - Exclude binaries, compiled files
  - Exclude data files
  - Exclude saved models
- [ ] Add `LICENSE` (MIT)
- [ ] Create `push_to_github.sh`
- [ ] Update `PUSH_TO_GITHUB.md`

**Climate comparison**: ✓ Same release process

### 9.3 Testing & Release
- [ ] Test fresh clone and compilation
- [ ] Push to GitHub (CryoEM-Denoise repository)
- [ ] Update repository settings
  - Description, topics
  - README displays correctly
- [ ] (Optional) Blog post
- [ ] **VALIDATION**: Public repository accessible

**Climate comparison**: Same workflow

---

## Phase 10: Optimization (ONLY IF PHASES 1-9 SUCCEED)

**Goal**: Improve performance beyond baseline

**DO NOT START THIS PHASE UNTIL PHASES 1-9 ARE COMPLETE**

### 10.1 Profiling
- [ ] Run `nsys profile ./cryo_train_unet --stream --max_batches 50`
- [ ] Analyze bottlenecks
- [ ] Identify optimization targets

### 10.2 Performance Improvements
- [ ] Increase batch size (if memory allows)
- [ ] Enable Tensor Cores (CUDNN_TENSOR_OP_MATH)
- [ ] Mixed precision (FP16) if stable
- [ ] Multi-stream pipeline

### 10.3 Architecture Experiments
- [ ] Different channel counts (64→128→256→512)
- [ ] Different patch sizes
- [ ] Data augmentation
- [ ] Learning rate scheduling

**Climate comparison**: Future work - we documented but didn't optimize

---

## Success Criteria Summary

### Minimum (MVP)
- ✓ Compiles without errors
- ✓ All 20+ tests pass
- ✓ Trains on synthetic data (loss decreases)
- ✓ PyTorch verification (max diff < 1e-6)
- ✓ Visual denoising works
- ✓ PSNR > 18 dB

### Target (Good)
- ✓ All MVP criteria
- ✓ Trains on 80 GB real data with streaming
- ✓ PSNR 20-21 dB (competitive with Topaz)
- ✓ SSIM > 0.8
- ✓ Complete documentation
- ✓ Public GitHub release

### Stretch (Excellent)
- ✓ All Target criteria
- ✓ PSNR > 21 dB
- ✓ SSIM > 0.85
- ✓ Faster than PyTorch
- ✓ Blog post with results

---

## What Worked for Climate (Apply Here)

1. **Plan thoroughly first** - Design doc before coding ✓
2. **Copy working code** - Reuse v28e modules ✓
3. **Test incrementally** - Unit tests, small data, then scale ✓
4. **Verify with PyTorch** - Catch bugs early ✓
5. **Document as we go** - Don't leave it to the end ✓
6. **Don't optimize prematurely** - Prove it works first ✓
7. **Use todo list** - Track progress, stay organized ✓

---

## Current Status

**Phase 1**: IN PROGRESS (awaiting user review)

**Next Steps**:
1. User reviews DESIGN_DOCUMENT.md
2. User answers open questions (dataset, patch size, etc.)
3. Begin Phase 2 (data preparation)

**Estimated Sessions to Completion**: 6-9 sessions (based on climate experience)
