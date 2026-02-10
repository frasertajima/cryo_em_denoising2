# v28f Cryo-EM Final Implementation Plan

**Status**: Ready to Start  
**Date**: 2025-11-24  
**Dataset**: EMPIAR-10025 subset (8 GB, 20 movies) + streaming infrastructure for full 2 TB

---

## Key Design Decision: Both Managed Memory AND Streaming

### Our Approach
Build **BOTH** data loaders from the start:
1. **Managed memory loader** - for 8 GB subset (fits in 50 GB RAM)
2. **Streaming loader** - for full 2 TB dataset (or anyone with less RAM)

### Why This Is Important
- ✓ Users with 8 GB subset can use fast managed memory
- ✓ Users with full 2 TB dataset can use streaming
- ✓ We can benchmark performance difference (educational value)
- ✓ Proves infrastructure scales from 8 GB → 2 TB
- ✓ Documentation shows both approaches

### Command-Line Interface
```bash
# Managed memory (default for small datasets)
./cryo_train_unet --epochs 15 --save

# Streaming (for large datasets)
./cryo_train_unet --stream --epochs 15 --save

# Force streaming even on small dataset (for testing)
./cryo_train_unet --stream --force --epochs 15
```

---

## Implementation Strategy

### Phase 1: Data Preparation (Today)

**Download**:
- [ ] EMPIAR-10025 subset (8 GB, 20 movies)
- [ ] Location: `data/empiar_10025_subset/`

**Preprocessing Tools** (`tools/` directory):
- [ ] `download_empiar_10025.sh` - Download script with instructions
- [ ] `mrc_reader.py` - MRC file parser
- [ ] `noise2noise_splitter.py` - Split odd/even frames
- [ ] `patch_extractor.py` - Extract 1024×1024 patches
- [ ] `create_binary_format.py` - Convert to streaming binary
- [ ] `visualize_patches.py` - Verify data quality

**Output Files** (both formats):
```
data/cryo_data_managed/           # Managed memory format
  ├── train_input.bin              # All patches in single file
  ├── train_target.bin
  ├── test_input.bin
  └── test_target.bin

data/cryo_data_streaming/         # Streaming format (same as climate)
  ├── train_input.bin              # Sample-major binary
  ├── train_target.bin
  ├── test_input.bin
  └── test_target.bin
```

**Note**: Files are identical! Just different usage patterns.

---

### Phase 2: Training Implementation (Tomorrow)

**Data Loaders** (`common/` directory):

1. **Managed Memory Loader** (`managed_memory_loader.cuf`):
   - Load entire dataset into GPU managed memory at startup
   - Fast random access
   - Requires: Dataset size < GPU + RAM capacity
   - Based on: v28_managed_memory experiments

2. **Streaming Loader** (`streaming_image_loader.cuf`):
   - Load batches on-demand from disk
   - Sequential access from binary files
   - Requires: Fast SSD (NVMe recommended)
   - Based on: climate `streaming_regression_loader.cuf`

**Configuration** (`data/cryo_config.cuf`):
```fortran
module cryo_config
  ! Dataset configuration
  character(len=256) :: data_dir = "data/cryo_data_streaming"
  character(len=256) :: managed_dir = "data/cryo_data_managed"
  
  ! Dimensions
  integer :: num_train = 15000     ! Estimated patches from 20 movies
  integer :: num_test = 1500       ! 10% holdout
  integer :: channels = 1          ! Grayscale
  integer :: height = 1024
  integer :: width = 1024
  
  ! Memory strategy
  logical :: use_streaming = .false.  ! Default: managed memory
  logical :: force_streaming = .false. ! Override for testing
end module
```

**Training Program** (`cryo_train_unet.cuf`):
```fortran
! Parse command-line args
if (use_stream) then
  ! Initialize streaming loader
  call init_streaming_loader(...)
else
  ! Initialize managed memory loader
  call init_managed_loader(...)
end if

! Training loop (same interface for both!)
do epoch = 1, num_epochs
  do batch_idx = 1, num_batches
    ! Get batch (abstracted - works for both loaders)
    call get_batch(batch_input, batch_target, batch_idx)
    
    ! Forward, backward, optimize (same for both)
    call forward(...)
    call backward(...)
    call adam_update(...)
  end do
end do
```

---

## Data Preparation Workflow

### Step 1: Download EMPIAR-10025 Subset

**CryoSPARC Tutorial Subset**:
- 20 movies from full dataset
- ~8 GB total
- Well-documented, widely used

**Download Command** (to be scripted):
```bash
# Option 1: Direct download from tutorial
wget https://cryosparc.com/empiar_10025_subset.tar

# Option 2: Aspera (faster for large files)
ascp -QT -l 200m -P33001 -i ~/.aspera/cli/etc/asperaweb_id_dsa.openssh \
  emp_ext3@hx-fasp-1.ebi.ac.uk:/10025/data/empiar_10025_subset.tar ./

# Option 3: Globus (if available)
# Instructions: https://www.ebi.ac.uk/empiar/download/
```

---

### Step 2: MRC Processing

**Input**: Raw MRC movie files (multi-frame)
**Output**: Odd/even frame pairs for Noise2Noise

**Process**:
```python
# tools/noise2noise_splitter.py

import mrcfile
import numpy as np

def split_movie_frames(mrc_path):
    """Split movie into odd/even frames for Noise2Noise."""
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        movie = mrc.data  # Shape: (n_frames, height, width)
        
        # Split frames
        odd_frames = movie[::2]   # Frames 0, 2, 4, ...
        even_frames = movie[1::2] # Frames 1, 3, 5, ...
        
        # Average frames (reduce noise)
        odd_avg = np.mean(odd_frames, axis=0).astype(np.float32)
        even_avg = np.mean(even_frames, axis=0).astype(np.float32)
        
        return odd_avg, even_avg

# odd_avg = noisy input
# even_avg = noisy target
# Both have same underlying signal, different noise realizations
```

---

### Step 3: Normalization

**95th Percentile Normalization** (standard in cryo-EM):
```python
def normalize_micrograph(img):
    """Normalize using 95th percentile of positive values."""
    positive_values = img[img > 0]
    if len(positive_values) == 0:
        return img
    
    p95 = np.percentile(positive_values, 95)
    normalized = img / p95
    
    # Clip outliers
    normalized = np.clip(normalized, 0, 1)
    
    return normalized.astype(np.float32)
```

---

### Step 4: Patch Extraction

**Extract 1024×1024 patches** from 7420×7676 micrographs:

```python
def extract_patches(img, patch_size=1024, stride=512):
    """Extract overlapping patches from large micrograph."""
    h, w = img.shape
    patches = []
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    
    return np.array(patches)

# Example: 7420×7676 → ~150 patches per micrograph
# 20 movies × 150 patches = ~3000 training patches
# With 50% overlap: ~12,000 patches
```

---

### Step 5: Binary Format Conversion

**Same format as climate project**:

```python
def create_streaming_binary(patches, output_path):
    """Convert patches to streaming binary format."""
    # Shape: (num_patches, height, width)
    # Output: sample-major, float32
    
    num_samples, height, width = patches.shape
    
    with open(output_path, 'wb') as f:
        for i in range(num_samples):
            # Write each patch as contiguous float32 array
            patches[i].astype(np.float32).tofile(f)
    
    print(f"Wrote {num_samples} patches to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**3:.2f} GB")
```

---

## Memory Estimates

### Managed Memory (8 GB Dataset)
```
Patches: 12,000 × 1024 × 1024 × 4 bytes = 48 GB
GPU managed memory handles overflow to system RAM

Training batch (size 4):
  Input: 4 × 1 × 1024 × 1024 = 16 MB
  Activations: ~500 MB (encoder/decoder)
  Gradients: ~500 MB
  Weights: 8 MB
  Total: ~1 GB per batch (fits easily in 8GB GPU)
```

### Streaming (2 TB Dataset)
```
Batches loaded on-demand from SSD
Same GPU memory as managed (1 GB)
Throughput limited by SSD speed (~500 MB/s sequential read)
```

---

## Benchmarking Plan

Once both loaders work, we'll benchmark:

| Metric | Managed Memory | Streaming | Comparison |
|--------|---------------|-----------|------------|
| Load time (startup) | ~30 sec (load 48 GB) | <1 sec | Managed slower start |
| Batch load time | ~0 ms (already in GPU) | ~50 ms (read from disk) | Managed faster |
| Throughput (samples/sec) | ~100-150 | ~80-100 | Managed 20-30% faster |
| GPU utilization | ~90% | ~85% | Streaming has I/O wait |
| RAM required | 48 GB | <1 GB | Streaming scales better |
| Scalability | Limited by RAM | Limited by disk space | Streaming wins for 2 TB |

**Expected Result**: Managed memory faster for 8 GB, streaming necessary for 2 TB

---

## Documentation Structure

### README.md Sections:

**Quick Start**:
```markdown
### Small Dataset (8 GB subset) - Recommended for Testing
Download EMPIAR-10025 subset and use managed memory:
./cryo_train_unet --epochs 15 --save

### Large Dataset (2 TB full) - Requires Streaming
Download full EMPIAR-10025 and use streaming:
./cryo_train_unet --stream --epochs 15 --save
```

**Data Preparation Guide** (`data/README_DATA.md`):
- Both subset and full dataset instructions
- Preprocessing steps for both
- When to use managed vs streaming
- Performance comparison

---

## Implementation Checklist (Updated)

### Today: Data Preparation
- [x] Finalize plan (managed + streaming)
- [ ] Download EMPIAR-10025 subset (8 GB)
- [ ] Write `tools/download_empiar_10025.sh`
- [ ] Write `tools/mrc_reader.py`
- [ ] Write `tools/noise2noise_splitter.py`
- [ ] Write `tools/patch_extractor.py`
- [ ] Write `tools/create_binary_format.py`
- [ ] Process all 20 movies → binary files
- [ ] Write `tools/visualize_patches.py`
- [ ] Verify data quality

**Deliverable**: `data/cryo_data_managed/` with train/test binary files

---

### Tomorrow: Training Implementation
- [ ] Copy U-Net from v28e_climate_cnn
- [ ] Write `common/managed_memory_loader.cuf`
- [ ] Write `common/streaming_image_loader.cuf`
- [ ] Write `common/cryo_unet.cuf` (1 channel)
- [ ] Write `data/cryo_config.cuf`
- [ ] Write `cryo_train_unet.cuf` (supports both loaders)
- [ ] Write `compile.sh`
- [ ] Test compilation
- [ ] Train 5 epochs (managed memory)
- [ ] Train 5 epochs (streaming - force mode)
- [ ] Compare performance

**Deliverable**: Working training with both memory strategies

---

### Session 3: PyTorch Verification + Full Training
- [ ] PyTorch U-Net implementation
- [ ] Verification script (max diff < 1e-6)
- [ ] Full training (15-30 epochs)
- [ ] Compute PSNR/SSIM
- [ ] Visual inspection

**Deliverable**: Baseline results vs Topaz-Denoise

---

### Sessions 4-6: Optimization
- [ ] Profile both loaders (nsys)
- [ ] Tensor Core acceleration
- [ ] Persistent memory pool
- [ ] Mixed precision (FP16)
- [ ] Multi-stream pipeline
- [ ] Final benchmarks

**Target**: 2-3x speedup over PyTorch

---

## Success Criteria (Final)

### Minimum (MVP)
- ✓ Works with 8 GB subset (managed memory)
- ✓ Works with streaming mode (tested on subset)
- ✓ PyTorch verification passes
- ✓ PSNR > 18 dB

### Target (Good)
- ✓ All MVP criteria
- ✓ PSNR 20-21 dB (match Topaz)
- ✓ SSIM > 0.8
- ✓ Documentation for both 8 GB and 2 TB
- ✓ Performance comparison (managed vs streaming)

### Stretch (Excellent)
- ✓ All Target criteria
- ✓ PSNR > 21 dB
- ✓ 2-3x faster than PyTorch
- ✓ Multiple optimization techniques demonstrated
- ✓ Clear scaling story (8 GB → 2 TB)

---

**Status**: Ready to start data preparation  
**Next Action**: Download EMPIAR-10025 subset and begin preprocessing
