# EMPIAR-10025 Dataset Specifications

**Dataset**: T20S Proteasome at 2.8 Å Resolution  
**Source**: https://www.ebi.ac.uk/empiar/EMPIAR-10025/  
**Date**: 2025-11-24

---

## Selected Data: Frame-Averaged Micrographs

**Choice**: 41 GB frame-averaged micrographs  
**Rationale**:
- ✓ Already motion-corrected (preprocessing done)
- ✓ Perfect size (fits in 50 GB RAM)
- ✓ Single images (simpler than movies)
- ✓ Fast iteration for development

---

## Image Specifications

**Dimensions**: 7420 × 7676 pixels  
**Format**: MRC (32-bit float expected)  
**Pixel size**: ~0.66 Å/pixel  
**Number of micrographs**: ~196 images (estimated from full dataset)  
**Total size**: 41 GB

---

## Patch Extraction Plan

### From 7420×7676 → 1024×1024 patches

**Stride options**:

1. **Non-overlapping** (stride = 1024):
   - Patches per image: 7 × 7 = 49 patches
   - Total patches: 196 × 49 = 9,604 patches
   - Coverage: Wastes edges

2. **50% overlap** (stride = 512):
   - Patches per image: 14 × 14 = 196 patches  
   - Total patches: 196 × 196 = 38,416 patches
   - Coverage: Better edge handling

3. **Optimal** (stride = 820, ~80% coverage):
   - Patches per image: 9 × 9 = 81 patches
   - Total patches: 196 × 81 = 15,876 patches
   - Coverage: Good balance

**Recommendation**: Use stride = 512 (50% overlap) for ~38K patches

---

## Memory Estimates

### Patch Dataset Size

**With 50% overlap (38,416 patches)**:
```
Clean patches: 38,416 × 1024 × 1024 × 4 bytes = 157 GB
Noisy patches: 38,416 × 1024 × 1024 × 4 bytes = 157 GB
Total: 314 GB
```

**Problem**: Too large for 50 GB RAM!

**Solution**: Use non-overlapping (stride = 1024) or store compressed

**With non-overlapping (9,604 patches)**:
```
Clean patches: 9,604 × 1024 × 1024 × 4 bytes = 39 GB
Noisy patches: 9,604 × 1024 × 1024 × 4 bytes = 39 GB
Total: 78 GB
```

**Still too large!**

### Revised Strategy

**Option A: Smaller patches (512×512)**:
```
Patches per image: 14 × 14 = 196 (non-overlapping)
Total patches: 196 × 196 = 38,416 patches
Size: 38,416 × 512 × 512 × 4 bytes = 39 GB × 2 = 78 GB
```
Still too large for RAM.

**Option B: Use subset of images**:
```
Use 100 images (instead of 196)
Patches: 100 × 49 = 4,900 patches (1024×1024)
Size: 4,900 × 1024 × 1024 × 4 bytes = 20 GB × 2 = 40 GB
```
**This fits in RAM!**

**Option C: Streaming (our infrastructure already supports this)**:
```
Use all 196 images, all patches (~38K)
Total: 157 GB per dataset (clean + noisy)
Load batches on-demand from SSD
```

---

## Final Decision

**Use all data (196 images) with streaming**:
- Proves streaming works (157 GB > 50 GB RAM)
- More training data = better results
- Demonstrates scalability

**Managed memory fallback**:
- If issues arise, use 100-image subset (fits in RAM)
- Good for debugging and fast iteration

---

## Training Strategy: Synthetic Noise

Since we're using frame-averaged micrographs (already denoised), we'll add synthetic noise:

### Noise Model

**Gaussian Noise** (simple baseline):
```python
noisy = clean + np.random.normal(0, sigma, clean.shape)
sigma = 0.1  # Adjust based on data range
```

**Poisson Noise** (more realistic for cryo-EM):
```python
# Cryo-EM follows Poisson statistics (electron counting)
noisy = np.random.poisson(clean * scale) / scale
```

**Combined** (most realistic):
```python
# Poisson (shot noise) + Gaussian (detector noise)
noisy = np.random.poisson(clean * scale) / scale
noisy = noisy + np.random.normal(0, sigma, noisy.shape)
```

### Training Pairs

- **Input**: Noisy image (clean + synthetic noise)
- **Target**: Clean image (original frame-averaged)
- **Loss**: MSE between denoised output and clean target

This is supervised learning with known ground truth (easier than Noise2Noise).

---

## Preprocessing Pipeline

```
Frame-averaged MRC files (41 GB)
    ↓
1. Read MRC files (196 images, 7420×7676)
    ↓
2. Normalize (95th percentile → 1.0)
    ↓
3. Extract 1024×1024 patches (stride = 512)
    ↓
4. Create synthetic noisy versions
    ↓
5. Convert to binary format
    ↓
Binary files:
  - train_input.bin  (noisy patches, ~157 GB)
  - train_target.bin (clean patches, ~157 GB)
  - test_input.bin   (noisy patches, ~17 GB)
  - test_target.bin  (clean patches, ~17 GB)
    ↓
Training with streaming (load batches from SSD)
```

---

## Data Split

**Train/Test Split**: 90% / 10%

- **Training**: 176 images → ~34,574 patches
- **Testing**: 20 images → ~3,842 patches

---

## Next Steps

1. **Download** 41 GB frame-averaged micrographs
2. **Write** preprocessing tools:
   - MRC reader
   - Patch extractor
   - Synthetic noise generator
   - Binary converter
3. **Process** all data → binary files
4. **Visualize** samples to verify quality
5. **Train** with streaming loader

---

**Status**: Ready to download and preprocess  
**Estimated preprocessing time**: 1-2 hours (depending on disk speed)
