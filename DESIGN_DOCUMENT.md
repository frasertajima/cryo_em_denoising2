# v28f Cryo-EM Denoising - Design Document

**Status**: Planning Phase  
**Created**: 2025-11-24  
**Goal**: Prove U-Net denoising works on large Cryo-EM datasets using streaming from 8GB GPU

---

## Executive Summary

Apply the successful WeatherBench2 methodology (v28e_climate_cnn) to Cryo-EM image denoising. This project will demonstrate that the same streaming + cuDNN infrastructure can handle different scientific domains with minimal adaptation.

**Key Success Criteria:**
1. Train on ~80GB Cryo-EM dataset using 8GB GPU with streaming
2. Achieve competitive PSNR/SSIM metrics vs published baselines
3. Verify against PyTorch implementation
4. Complete documentation and reproducible pipeline

---

## 1. Project Scope

### What We're Building

**Task**: 2D micrograph denoising (noisy → clean)
- **Input**: Noisy cryo-EM micrographs (raw or motion-corrected)
- **Output**: Denoised micrographs
- **Architecture**: U-Net (same as climate project)
- **Training**: Noise2Noise self-supervised or supervised with paired data

### What We're NOT Building (Out of Scope)

- 3D reconstruction from 2D projections (more complex, save for later)
- Particle picking (different task)
- Missing wedge reconstruction (tomography-specific)
- CTF correction (preprocessing step)

**Rationale for 2D denoising:**
- Clear input/output pairs (easier to validate)
- Direct visual evaluation (humans can see quality)
- Established benchmarks (Topaz-Denoise, etc.)
- Simpler than 3D reconstruction for proving concept

---

## 2. Dataset Selection

### Primary Dataset: EMPIAR-10028

**Problem**: EMPIAR-10028 is 1.17 TB (too large for initial testing)

**Alternative Approach:**

1. **Start with smaller EMPIAR dataset** for prototyping (~10-20 GB)
   - Verify pipeline works
   - Test data loading and preprocessing
   
2. **Scale to medium dataset** for streaming proof (~80 GB)
   - Use subset of EMPIAR-10028 (first 100-200 micrographs)
   - Or find different EMPIAR dataset in 80GB range
   
3. **Future: Full EMPIAR-10028** if streaming works well (1.17 TB)

### Benchmark Datasets (Research)

Based on 2024-2025 research:

| Dataset | Size | Purpose | Metrics |
|---------|------|---------|---------|
| **CryoCRAB** | 116.8 TB | Foundation models, denoising | PSNR, SSIM |
| **denoisecryodata** | 650 maps | 3D map denoising | PSNR, cross-correlation |
| **DRACO dataset** | 270K images | General denoising | PSNR, SSIM, visual quality |
| **CryoBench** | Various | Heterogeneity testing | PSNR, reconstruction quality |

**Recommendation**: Start with publicly available EMPIAR subset, compare against **Topaz-Denoise** benchmarks (PSNR ~20-21 dB, SSIM ~0.82-0.87 depending on SNR).

---

## 3. Data Format & Preprocessing

### MRC File Format Specification

Cryo-EM data uses **MRC/CCP4 format** (MRC2014 standard):

**Header Structure** (1024 bytes):
- Dimensions: NX, NY, NZ (columns, rows, sections)
- Mode: Data type (0=8-bit, 1=16-bit, 2=32-bit float, 6=16-bit unsigned)
- Pixel size: Cell dimensions (Angstroms)
- Statistics: min, max, mean, RMS
- Extended header for metadata

**Data Block**:
- Raw pixel values (typically 32-bit float for cryo-EM)
- Column-major ordering (Fortran-style)

### Preprocessing Pipeline

```
Raw MRC files
    ↓
1. Read MRC header (parse pixel size, dimensions)
    ↓
2. Extract pixel data (typically 32-bit float)
    ↓
3. Normalization:
   - Compute 95th percentile of positive values
   - Divide all values by percentile
   - Clip: values < 0 → 0, values > 1 → 1
    ↓
4. Patch extraction (if images too large):
   - Extract 512×512 or 1024×1024 patches
   - Overlap for edge handling
    ↓
5. Convert to streaming binary format:
   - Sample-major ordering
   - Float32 (4 bytes per pixel)
   - No header (raw binary)
    ↓
Streaming data files: train_input.bin, train_target.bin
```

### Streaming Binary Format

Same format as climate project:

```
Layout: Sample-major ordering
Type: float32 (4 bytes per value)
Endianness: Native (little-endian on x86-64)

Structure:
  sample_0:
    channel_0: [height × width] values (e.g., 512×512)
  sample_1:
    channel_0: [height × width] values
  ...

File size: num_samples × channels × height × width × 4 bytes
```

**Key differences from climate:**
- **1 channel** (grayscale) vs 6 channels (climate variables)
- **Larger spatial dimensions**: 512×512 or 1024×1024 vs 240×121
- **Patch-based**: Extract patches from large micrographs

---

## 4. Training Strategy

### Noise2Noise vs Supervised

**Option 1: Noise2Noise (Self-Supervised)** - RECOMMENDED
- **Input**: Noisy micrograph from odd frames
- **Target**: Noisy micrograph from even frames
- **Advantage**: No clean ground truth needed
- **How**: Split movie frames into odd/even, average separately
- **Basis**: Topaz-Denoise uses this approach

**Option 2: Supervised (Synthetic Noise)**
- **Input**: Clean micrograph + synthetic noise
- **Target**: Clean micrograph
- **Advantage**: Simpler to understand, easier to validate
- **How**: Add Gaussian/Poisson noise to clean images
- **Basis**: Traditional approach, easier for initial testing

**Recommendation**: Start with **Option 2 (synthetic noise)** for initial testing to prove infrastructure works, then move to **Option 1 (Noise2Noise)** for realistic evaluation.

### Training Configuration

```
Architecture: U-Net (same as climate)
  - Encoder: 1 → 32 → 64 → 128 channels
  - Bottleneck: 128 → 256 channels
  - Decoder: 256 → 128 → 64 → 32 → 1 channels
  - Skip connections at each level
  
Loss: MSE (mean squared error)
Optimizer: Adam (lr=0.0001, same as climate)
Batch size: 4-8 (depending on patch size and GPU memory)

Input size: 512×512 or 1024×1024 patches
Padding: Power of 2 (e.g., 512→512, 1024→1024)
```

### Data Augmentation (Future)

- Random crops
- Random flips (horizontal/vertical)
- Random rotations (90°, 180°, 270°)
- Brightness/contrast adjustments

Start without augmentation, add later if needed.

---

## 5. Evaluation Metrics

### Primary Metrics

| Metric | Formula | Target (vs Topaz) | Interpretation |
|--------|---------|-------------------|----------------|
| **PSNR** | 10 × log₁₀(MAX²/MSE) | 20-21 dB | Higher = better, >20 dB is good |
| **SSIM** | Structural similarity | 0.82-0.87 | 1.0 = perfect, >0.8 is good |
| **MSE** | Mean squared error | Low | Training loss |

### Secondary Metrics

- **Visual inspection**: Side-by-side comparison
- **Frequency domain**: Power spectra (check for artifacts)
- **Particle picking**: Does denoising improve picking accuracy?

### Baseline Comparisons

1. **No denoising** (raw noisy images) - lower bound
2. **Gaussian blur** - simple baseline
3. **Topaz-Denoise** - state-of-the-art (if time permits)
4. **PyTorch U-Net** - verify our implementation matches

---

## 6. Architecture Design

### U-Net Adaptation from Climate

**Reusable from v28e_climate_cnn:**
- ✓ `common/conv2d_cudnn.cuf` - Conv2D + ReLU + Adam
- ✓ `common/pooling_cudnn.cuf` - MaxPool + Upsample
- ✓ `common/unet_blocks.cuf` - Encoder/decoder blocks
- ✓ `common/cmdline_args.cuf` - Argument parsing
- ✓ `common/training_export.cuf` - Export utilities
- ✓ `common/unet_export.cuf` - Weight export

**Modifications needed:**
- Create new `common/cryo_unet.cuf`:
  - Input channels: 6 → 1 (grayscale)
  - Output channels: 6 → 1 (grayscale)
  - Padding: 256×128 → 512×512 or 1024×1024
- Create new `data/cryo_config.cuf`:
  - Dataset paths, dimensions
  - Patch size configuration
- Create new `common/streaming_image_loader.cuf`:
  - Simplified version of regression loader
  - Single-channel images instead of multi-channel regression

### Memory Estimates

For 512×512 patches, batch size 8:

```
Encoder activations:
  Level 1: 8 × 32 × 512 × 512 = 67M floats = 268 MB
  Level 2: 8 × 64 × 256 × 256 = 33M floats = 134 MB
  Level 3: 8 × 128 × 128 × 128 = 16M floats = 67 MB
  Bottleneck: 8 × 256 × 64 × 64 = 8M floats = 33 MB

Total activations: ~500 MB
Gradients: ~500 MB
Weights: ~2M params × 4 bytes = 8 MB

Total GPU memory: ~1.5-2 GB (fits comfortably in 8GB)
```

Could potentially use **batch size 16** for 512×512 patches!

---

## 7. Implementation Checklist (Climate Methodology)

This is the proven workflow from v28e_climate_cnn:

### Phase 1: Planning & Design ✓ (CURRENT)
- [x] Research datasets and benchmarks
- [x] Define task scope (2D denoising)
- [x] Choose metrics (PSNR, SSIM)
- [x] Design data format
- [x] Architecture planning
- [ ] **USER REVIEW** ← We are here

### Phase 2: Data Preparation
- [ ] Download small EMPIAR dataset (10-20 GB)
- [ ] Write MRC reader (Python)
- [ ] Implement preprocessing (normalization, patching)
- [ ] Convert to streaming binary format
- [ ] Verify data with Python (visualize patches)
- [ ] Create synthetic noisy pairs (initial testing)

### Phase 3: Minimal Viable Implementation
- [ ] Copy U-Net modules from v28e_climate_cnn
- [ ] Create `cryo_unet.cuf` (1 channel in/out)
- [ ] Create `cryo_config.cuf` (dataset config)
- [ ] Create `streaming_image_loader.cuf` (simplified loader)
- [ ] Write `cryo_train_unet.cuf` (main training)
- [ ] Write `compile.sh`
- [ ] **TEST COMPILATION**

### Phase 4: Unit Testing
- [ ] Test conv2d with 1 channel (existing test adapted)
- [ ] Test pooling (reuse existing)
- [ ] Test U-Net forward pass (1-channel input)
- [ ] Test data loader (read streaming file)
- [ ] **ALL TESTS PASS**

### Phase 5: Small-Scale Training
- [ ] Train on small synthetic dataset (1000 patches)
- [ ] Verify loss decreases
- [ ] Export weights and sample output
- [ ] Visual inspection of results
- [ ] **PROOF THAT DENOISING WORKS**

### Phase 6: PyTorch Verification
- [ ] Create PyTorch U-Net (1 channel)
- [ ] Load Fortran weights into PyTorch
- [ ] Compare forward pass outputs
- [ ] Compare training step timing
- [ ] **MAX DIFF < 1e-6**

### Phase 7: Scale to Real Data
- [ ] Download EMPIAR dataset subset (~80 GB)
- [ ] Preprocess real micrographs
- [ ] Create Noise2Noise pairs (odd/even frames)
- [ ] Convert to streaming format
- [ ] Train with streaming enabled
- [ ] **STREAMING PROOF ON 8GB GPU**

### Phase 8: Evaluation & Benchmarking
- [ ] Compute PSNR, SSIM on test set
- [ ] Compare vs Gaussian blur baseline
- [ ] Visual comparison (before/after)
- [ ] Frequency analysis (power spectra)
- [ ] Document results
- [ ] **COMPETITIVE WITH PUBLISHED BASELINES**

### Phase 9: Documentation & Release
- [ ] Write comprehensive README
- [ ] Data download/preparation guide
- [ ] Jupyter notebooks (visualization, evaluation)
- [ ] GitHub repository (CryoEM-Denoise)
- [ ] Blog post
- [ ] **PUBLIC RELEASE**

### Phase 10: Optimization (ONLY if Phases 1-9 succeed)
- [ ] Profile with nsys
- [ ] Optimize bottlenecks
- [ ] Experiment with batch sizes
- [ ] Mixed precision (FP16)
- [ ] **PERFORMANCE TUNING**

---

## 8. Technical Challenges & Solutions

### Challenge 1: Large Image Sizes

**Problem**: Cryo-EM micrographs are 4096×4096 (EMPIAR-10028), much larger than climate's 240×121.

**Solutions:**
1. **Patch extraction**: Extract 512×512 or 1024×1024 patches
2. **Overlapping patches**: Smooth reconstruction at edges
3. **Streaming**: Load patches on-demand, not full images

### Challenge 2: MRC File Format

**Problem**: MRC files have complex headers, different from raw binary.

**Solutions:**
1. **Use existing libraries**: mrcfile (Python), IMOD (C)
2. **Preprocess once**: MRC → streaming binary (offline)
3. **Training uses binary**: No runtime MRC parsing

### Challenge 3: Noise2Noise Pairing

**Problem**: Need to create odd/even frame pairs from movie stacks.

**Solutions:**
1. **Start with synthetic noise**: Add Gaussian noise to clean images
2. **Later: Process EMPIAR movies**: Split frames, align, average
3. **Use preprocessed data**: If available, download pre-split pairs

### Challenge 4: Different Data Distribution

**Problem**: Cryo-EM has different intensity distributions than climate data.

**Solutions:**
1. **Percentile normalization**: 95th percentile → 1.0 (standard in field)
2. **Per-image normalization**: Normalize each micrograph independently
3. **Monitor histograms**: Visualize distributions during training

### Challenge 5: Evaluation Requires Clean References

**Problem**: Noise2Noise doesn't have ground truth, hard to measure PSNR.

**Solutions:**
1. **Synthetic data**: Known clean + noise → measure PSNR
2. **Proxy metrics**: Compare to Gaussian blur, visual inspection
3. **Downstream tasks**: Particle picking accuracy improvement

---

## 9. Dataset Candidates (80 GB Target)

After research, here are practical options:

### Option A: EMPIAR Subset
- Take first 100-200 micrographs from EMPIAR-10028
- ~100 × 4096² × 4 bytes × frames ≈ 80 GB (rough estimate)
- Advantage: Well-studied dataset
- Disadvantage: Need to download and subset 1.17 TB

### Option B: Smaller Complete EMPIAR Dataset
Search for EMPIAR datasets in 50-100 GB range:
- EMPIAR-10025: ~30 GB
- EMPIAR-10081: ~45 GB
- EMPIAR-10288: ~70 GB
- Others to investigate

### Option C: Synthetic Dataset
- Generate synthetic micrographs with controlled noise
- Exact size control (make exactly 80 GB)
- Advantage: Known ground truth, easy PSNR evaluation
- Disadvantage: Not "real" data, less impressive demo

**Recommendation**: Start with **Option B** (find 50-80 GB EMPIAR dataset), use **Option C** for initial testing (synthetic), then scale to **Option A** if streaming works well.

---

## 10. Comparison: Climate vs Cryo-EM

| Aspect | Climate (v28e) | Cryo-EM (v28f) |
|--------|---------------|----------------|
| **Task** | Regression (6h forecast) | Denoising (noisy → clean) |
| **Input channels** | 6 (weather variables) | 1 (grayscale) |
| **Output channels** | 6 (same variables) | 1 (grayscale) |
| **Spatial size** | 240×121 (small) | 512×512 or 1024×1024 (large) |
| **Dataset size** | 72 GB | 80 GB (target) |
| **File format** | Custom binary | MRC → binary |
| **Normalization** | Per-channel stats | 95th percentile |
| **Loss** | MSE | MSE |
| **Metrics** | MSE, ACC | MSE, PSNR, SSIM |
| **Batch size** | 8 | 4-8 (depends on patch size) |
| **Training time** | ~2 hours (15 epochs) | TBD (similar expected) |
| **Baseline** | Persistence forecast | Gaussian blur |
| **Validation** | PyTorch verification | PyTorch + Topaz comparison |

**Key insight**: Most infrastructure is reusable! Main differences are data preprocessing and metrics.

---

## 11. Success Metrics

### Minimum Viable Product (MVP)
- ✓ Compiles without errors
- ✓ Trains on synthetic data (loss decreases)
- ✓ PyTorch verification passes (max diff < 1e-6)
- ✓ Visual inspection shows denoising works
- ✓ PSNR > 18 dB on test set (better than no denoising)

### Good Result
- ✓ All MVP criteria
- ✓ Trains on real EMPIAR data with streaming
- ✓ PSNR 20-21 dB (comparable to Topaz-Denoise)
- ✓ SSIM > 0.8
- ✓ Frequency analysis shows no artifacts
- ✓ Complete documentation and reproducibility

### Excellent Result
- ✓ All Good Result criteria
- ✓ PSNR > 21 dB (better than Topaz)
- ✓ SSIM > 0.85
- ✓ Faster training than PyTorch
- ✓ Scales to 1.17 TB EMPIAR-10028
- ✓ Demonstrates optimization opportunities

---

## 12. Timeline Estimate (No Specific Dates)

Following the climate project pace:

**Phase 1-2** (Data prep): ~1-2 sessions
- Download dataset, write MRC reader, create streaming files

**Phase 3-4** (Implementation + tests): ~1-2 sessions  
- Adapt U-Net, write training loop, unit tests

**Phase 5-6** (Initial training + verification): ~1 session
- Small-scale training, PyTorch comparison

**Phase 7-8** (Real data + evaluation): ~2-3 sessions
- Streaming training, metrics, benchmarking

**Phase 9** (Documentation): ~1 session
- README, notebooks, release

**Total**: 6-9 sessions (similar to climate project)

**Optimization** (Phase 10): Open-ended, only if earlier phases succeed

---

## 13. Repository Structure

```
v28f_cryo_em/
├── common/
│   ├── cmdline_args.cuf               # Copied from v28e
│   ├── streaming_image_loader.cuf     # New: simplified 1-channel loader
│   ├── conv2d_cudnn.cuf               # Copied from v28e
│   ├── pooling_cudnn.cuf              # Copied from v28e
│   ├── unet_blocks.cuf                # Copied from v28e
│   ├── cryo_unet.cuf                  # New: 1-channel U-Net
│   ├── training_export.cuf            # Copied from v28e
│   └── unet_export.cuf                # Copied from v28e
├── data/
│   ├── cryo_config.cuf                # New: dataset configuration
│   ├── cryo_data_streaming/           # User-provided streaming data
│   └── README_DATA.md                 # Data setup instructions
├── tools/
│   ├── mrc_to_streaming.py            # Convert MRC to binary
│   ├── create_synthetic_data.py       # Generate synthetic noisy pairs
│   └── visualize_patches.py           # Inspect data
├── inference/
│   ├── cryo_unet.py                   # PyTorch equivalent
│   ├── verify_fortran_pytorch.py      # Verification script
│   └── compute_metrics.py             # PSNR, SSIM calculation
├── notebooks/
│   ├── cryo_unet_analysis.ipynb       # Weight/output visualization
│   └── cryo_unet_evaluation.ipynb     # PSNR/SSIM evaluation
├── tests/
│   ├── test_conv2d.cuf                # Copied from v28e
│   ├── test_pooling.cuf               # Copied from v28e
│   ├── test_unet_blocks.cuf           # Copied from v28e
│   └── test_cryo_unet.cuf             # New: 1-channel U-Net test
├── saved_models/                      # Training checkpoints
├── cryo_train_unet.cuf                # Main training program
├── compile.sh                         # Build script
├── README.md                          # User-facing documentation
└── DESIGN_DOCUMENT.md                 # This file
```

---

## 14. Open Questions for User Review

1. **Dataset size**: Is 80 GB the right target, or should we start smaller (~20 GB) for faster iteration?

2. **Patch size**: 512×512 or 1024×1024? Larger = more context, smaller = bigger batches.

3. **Training approach**: Start with synthetic noise (easier) or jump to Noise2Noise (more realistic)?

4. **Baseline comparison**: Just PyTorch, or also try to compare against Topaz-Denoise?

5. **3D visualization**: Should we save 3D reconstruction for a future project, or try it if 2D works well?

6. **Optimization priority**: Prove it works first, or optimize early for speed comparisons?

7. **Dataset choice**: Subset of EMPIAR-10028, or find different 80 GB dataset?

---

## 15. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MRC parsing complexity | Medium | Low | Use Python libraries, preprocess offline |
| Large patch sizes exceed GPU memory | Low | Medium | Start with 512×512, reduce batch size |
| Real data quality issues | Medium | Medium | Start with synthetic, validate incrementally |
| PSNR doesn't reach baseline | Low | High | Follow Topaz architecture closely, tune hyperparams |
| Streaming doesn't work at 80 GB | Very Low | High | Already proved with 72 GB climate, reuse same code |
| User loses interest mid-project | Low | High | Keep sessions focused, show progress early |

**Overall risk**: LOW - We've already proved the infrastructure with climate. This is mostly data adaptation.

---

## 16. Next Steps

**Immediate** (this session):
1. User reviews this design document
2. User provides feedback on open questions
3. Agree on dataset choice and patch size

**Next session**:
1. Download/create initial dataset
2. Write MRC reader and preprocessing tools
3. Create synthetic data for testing
4. Verify data pipeline with visualization

**Future sessions**: Follow Phase 3-9 checklist above

---

## References

- **Topaz-Denoise**: Bepler et al. (2020), Nature Communications
- **MRC2014 format**: Cheng et al. (2015), J Struct Biol
- **CryoCRAB**: Large-scale cryo-EM dataset (2025)
- **EMPIAR**: Electron Microscopy Public Image Archive
- **Noise2Noise**: Lehtinen et al. (2018), ICML

---

**Document Status**: Ready for user review  
**Next Action**: User feedback and dataset decision
