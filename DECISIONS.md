# v28f Cryo-EM Project - Final Decisions

**Status**: Planning Complete - Ready to Start Implementation  
**Date**: 2025-11-24  
**Next Phase**: Data Preparation

---

## User Decisions (Finalized)

### 1. Dataset: EMPIAR-10025 (30 GB)
**Decision**: Use EMPIAR-10025 at 30 GB instead of EMPIAR-10028 (1.17 TB)

**Rationale**:
- ✓ More than sufficient to prove streaming works
- ✓ Faster download (no overnight wait)
- ✓ Allows testing both managed memory AND streaming for performance comparison
- ✓ Simpler workflow - one dataset instead of test + real dataset
- ✓ Still large enough to demonstrate the approach

**Action Items**:
- [ ] Research EMPIAR-10025 specifications
- [ ] Download EMPIAR-10025 dataset
- [ ] Document dataset characteristics

---

### 2. Patch Size: 1024×1024
**Decision**: Use 1024×1024 patches

**Rationale**:
- More context for denoising (larger receptive field)
- Still fits in GPU memory with batch size 4-8
- Standard size in cryo-EM community

**Implications**:
- Batch size: 4-8 (vs 8-16 for 512×512)
- GPU memory: ~2-3 GB for activations
- U-Net padding: Already power of 2, no padding needed

---

### 3. Training Approach: Noise2Noise
**Decision**: Start with Noise2Noise (odd/even frame pairs)

**Rationale**:
- More realistic evaluation (real noise model)
- Self-supervised (no ground truth needed)
- Standard approach in cryo-EM community
- Can compare directly to Topaz-Denoise

**Fallback**: If Noise2Noise proves too difficult, synthetic noise is backup

**Action Items**:
- [ ] Research how to split EMPIAR-10025 into odd/even frames
- [ ] Implement frame splitting in preprocessing
- [ ] Verify noise characteristics are similar between pairs

---

### 4. Benchmark: Topaz-Denoise (Primary)
**Decision**: Topaz-Denoise is the primary benchmark to beat

**Rationale**:
- State-of-the-art in cryo-EM denoising
- Published PSNR/SSIM metrics to compare against
- Using same Noise2Noise approach
- Industry standard tool

**Secondary**: PyTorch verification (for correctness checking)

**Target Metrics**:
- PSNR: 20-21 dB (match Topaz)
- SSIM: 0.82-0.87 (match Topaz)
- Speed: Faster than PyTorch (target: 1.5-2x)

---

### 5. Memory Strategy: Test Both Approaches
**Decision**: Test whether 1024×1024 patches fit in RAM, use streaming if needed

**Approach**:
1. Calculate total memory required for 30 GB dataset with 1024×1024 patches
2. If fits in RAM: Use managed memory for simplicity
3. If exceeds RAM: Use streaming (already implemented)
4. **Performance comparison**: Benchmark both approaches!

**Rationale**:
- Managed memory might be faster (no disk I/O)
- Streaming proves scalability
- Comparison shows trade-offs
- Educational value for both approaches

**Action Items**:
- [ ] Calculate number of 1024×1024 patches in EMPIAR-10025
- [ ] Estimate RAM requirements
- [ ] Implement both loaders (managed memory + streaming)
- [ ] Benchmark performance difference

---

### 6. Data Format: Python Preprocessing
**Decision**: Handle MRC format in Python, convert to binary for training

**Rationale**:
- ✓ Python has excellent MRC libraries (mrcfile)
- ✓ Fortran focuses on training (not file parsing)
- ✓ Preprocessing is one-time cost (offline)
- ✓ Training uses simple binary format (fast)

**Workflow**:
```
MRC files (EMPIAR download)
    ↓ [Python preprocessing - offline]
Streaming binary files (train_input.bin, train_target.bin)
    ↓ [Fortran training - GPU accelerated]
Trained model
```

---

### 7. Optimization Focus: Major Goal
**Decision**: Spend significant time on optimization AFTER baseline works

**Rationale**:
- Climate achieved only 1.3x speedup (Adam was 129x faster, but small portion)
- Opportunity to apply advanced techniques:
  - ✓ Tensor Core engine (from v28d)
  - ✓ Persistent memory pool (from managed memory experiments)
  - ✓ N-body optimizations (from earlier work)
  - ✓ Mixed precision (FP16/FP32)
  - ✓ Multi-stream pipelining
- Cryo-EM benefits more from speed (large datasets, many experiments)
- Learnings apply back to climate project

**Optimization Roadmap** (after baseline works):
1. Profile with nsys (identify bottlenecks)
2. Enable Tensor Cores (CUDNN_TENSOR_OP_MATH)
3. Implement persistent memory pool (reduce allocations)
4. Apply n-body kernel optimizations (if applicable to convolutions)
5. Mixed precision training (FP16 accumulation)
6. Multi-stream data loading + training pipeline
7. **Target**: 2-3x speedup over PyTorch (vs 1.3x in climate)

---

## Project Timeline (Revised)

### Today (Session 1): Data Preparation
- [ ] Research EMPIAR-10025 specifications
- [ ] Download EMPIAR-10025 (30 GB)
- [ ] Write MRC reader (Python)
- [ ] Implement Noise2Noise preprocessing (odd/even frames)
- [ ] Extract 1024×1024 patches
- [ ] Convert to binary streaming format
- [ ] Visualize samples (verify quality)

**Goal**: Have `train_input.bin` and `train_target.bin` ready

---

### Tomorrow (Session 2): Training Implementation
- [ ] Copy U-Net infrastructure from v28e_climate_cnn
- [ ] Adapt for 1 channel (cryo_unet.cuf)
- [ ] Implement data loader (managed memory OR streaming)
- [ ] Write training program (cryo_train_unet.cuf)
- [ ] Compile and test
- [ ] Train for 5-10 epochs (proof of concept)
- [ ] Visual inspection (does denoising work?)

**Goal**: First denoised images

---

### Session 3: Verification & Baseline
- [ ] PyTorch U-Net implementation
- [ ] Verification (max diff < 1e-6)
- [ ] Full training (15-30 epochs)
- [ ] Compute PSNR/SSIM metrics
- [ ] Compare to Topaz-Denoise benchmarks

**Goal**: Competitive baseline results

---

### Sessions 4-6: Optimization & Benchmarking
- [ ] Profile with nsys
- [ ] Implement tensor core acceleration
- [ ] Test managed memory vs streaming performance
- [ ] Apply persistent memory pool
- [ ] Mixed precision experiments
- [ ] Multi-stream pipeline
- [ ] Final PyTorch comparison (measure speedup)

**Goal**: 2-3x speedup over PyTorch

---

### Session 7: Documentation & Release
- [ ] Comprehensive README
- [ ] Jupyter notebooks
- [ ] Data preparation guide
- [ ] Push to GitHub
- [ ] Blog post (optional)

---

## Key Advantages of This Approach

### 1. Reuses Proven Infrastructure
- ✓ U-Net architecture (same as climate)
- ✓ cuDNN kernels (already optimized)
- ✓ Streaming loader (already implemented)
- ✓ PyTorch verification (same methodology)

### 2. Right-Sized Dataset
- ✓ 30 GB is perfect (not too small, not too large)
- ✓ Tests both managed memory and streaming
- ✓ Fast iteration during development
- ✓ Still impressive for demo

### 3. Clear Benchmarks
- ✓ Topaz-Denoise is well-established
- ✓ Published metrics to target
- ✓ Same Noise2Noise approach
- ✓ Direct apples-to-apples comparison

### 4. Optimization Opportunities
- ✓ Larger images = more compute (benefits from optimization)
- ✓ Can apply advanced techniques from previous work
- ✓ Potential for significant speedup (2-3x target)
- ✓ Learnings transfer to other projects

### 5. Educational Value
- ✓ Demonstrates methodology works across domains
- ✓ Shows trade-offs (managed memory vs streaming)
- ✓ Explores optimization techniques
- ✓ Proves consumer hardware viability

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 1024×1024 patches too large for GPU | Reduce batch size to 2-4 |
| Noise2Noise frame splitting complex | Use synthetic noise as fallback |
| EMPIAR-10025 download issues | Have backup dataset ready |
| Can't match Topaz PSNR | Focus on PyTorch parity first |
| Optimization doesn't yield speedup | Document baseline performance anyway |

---

## Success Criteria (Revised)

### Minimum (MVP)
- ✓ Compiles and runs
- ✓ Trains on EMPIAR-10025 (30 GB)
- ✓ PyTorch verification passes
- ✓ Visual denoising works
- ✓ PSNR > 18 dB

### Target (Good)
- ✓ All MVP criteria
- ✓ PSNR 20-21 dB (match Topaz)
- ✓ SSIM > 0.8
- ✓ Faster than PyTorch (any speedup)
- ✓ Complete documentation

### Stretch (Excellent)
- ✓ All Target criteria
- ✓ PSNR > 21 dB (beat Topaz)
- ✓ SSIM > 0.85
- ✓ 2-3x faster than PyTorch
- ✓ Demonstrates multiple optimization techniques
- ✓ Managed memory vs streaming comparison

---

## Next Actions (This Session)

1. **Research EMPIAR-10025**
   - Dataset specifications
   - Number of micrographs
   - Image dimensions
   - Frame structure (for Noise2Noise)

2. **Download EMPIAR-10025**
   - Find download link
   - Verify checksum
   - Extract files

3. **Write Preprocessing Tools**
   - `tools/mrc_reader.py` - Parse MRC files
   - `tools/noise2noise_splitter.py` - Split odd/even frames
   - `tools/patch_extractor.py` - Extract 1024×1024 patches
   - `tools/create_streaming_binary.py` - Convert to binary format

4. **Verify Data Quality**
   - `tools/visualize_patches.py` - Inspect patches
   - Check noise characteristics
   - Verify normalization

---

**Status**: ✓ Planning Complete  
**Ready to Start**: Data Preparation Phase  
**Estimated Time Today**: 2-3 hours (download + preprocessing)
