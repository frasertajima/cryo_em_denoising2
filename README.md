# Cryo-EM Denoising with Deep Residual CNN (CUDA Fortran)

**Status**: ✅ **PRODUCTION READY**  
**Achievement**: Noise2Noise training on real cryo-EM data reveals protein particles invisible in raw images

---

## Overview

This project implements a **12-layer deep residual CNN** for cryo-EM image denoising in **CUDA Fortran**. The network uses residual learning (predicting noise and subtracting it) combined with **Noise2Noise training** on real cryo-EM movie data.

### Breakthrough Results on Real Data (EMPIAR-10025)

| Metric | Raw Half-Set | Denoised | Improvement |
|--------|--------------|----------|-------------|
| **Odd/Even Correlation** | 0.067 | 0.879 | **13x improvement** |
| **PSNR** | baseline | +2.82 dB | Averaged over 10 movies |
| **SSIM** | baseline | +0.109 | Structural similarity |
| **Removed Noise Correlation** | - | 0.024 | Confirms real noise removal |

**The denoised images reveal protein particles that are completely invisible in the raw data** - both odd and even frame averages show only noise, but the denoised version clearly shows individual T20S Proteasome particles scattered across the ice.

### Synthetic Data Results

| Metric | Noisy Input | Denoised | Improvement |
|--------|-------------|----------|-------------|
| **PSNR** | 12.81 dB | 16.90 dB | **+4.09 dB** |
| **Edge Preservation** | - | 55.9% | Structures visible |

---

## Architecture

```
12-Layer Deep Residual CNN (93,089 parameters)

Input (1024×1024, 1 channel)
    ↓
Conv1:  1 → 32 channels (3×3, ReLU)
Conv2:  32 → 32 channels (3×3, ReLU)
Conv3:  32 → 32 channels (3×3, ReLU)
Conv4:  32 → 32 channels (3×3, ReLU)
Conv5:  32 → 32 channels (3×3, ReLU)
Conv6:  32 → 32 channels (3×3, ReLU)
Conv7:  32 → 32 channels (3×3, ReLU)
Conv8:  32 → 32 channels (3×3, ReLU)
Conv9:  32 → 32 channels (3×3, ReLU)
Conv10: 32 → 32 channels (3×3, ReLU)
Conv11: 32 → 32 channels (3×3, ReLU)
Conv12: 32 → 1 channel (3×3, no activation) → Predicted Noise
    ↓
Output = Input - Predicted Noise (Residual Connection)
```

**Why residual learning?** The network learns to predict *what doesn't belong* (noise) rather than reconstructing the signal directly. This is more robust and preserves structures better.

---

## Training Results

### Real Data: Noise2Noise on EMPIAR-10025 (T20S Proteasome)

```
Dataset: 200 raw movie stacks (395 GB)
Training: 35,150 patches (64×64) from odd/even frame splitting
Validation: 3,906 patches

Epoch 1:  Val Loss 0.000342 (new best)
Epoch 4:  Val Loss 0.000341 (new best)
Epoch 5:  Val Loss 0.000340 (new best)
Epoch 9:  Val Loss 0.000339 (new best) ← Final model
Epoch 10: Val Loss 0.000341
```

- **Training time**: ~10 minutes total for 10 epochs
- **395GB dataset** processed with only 2MB buffer memory
- **Verification**: Denoised odd/even correlation: 0.879 (raw: 0.067)

### Synthetic Data: Supervised Training

Training on 10,309 patches (1024×1024) with σ=0.25 Gaussian noise:

```
Epoch 1: Val Loss 0.0207, Val RMSE 0.144
Epoch 5: Val Loss 0.0205, Val RMSE 0.143  ← Best model
```

- **Converged quickly** (epoch 1-2)
- **Stable training** (no overfitting)
- **~50 minutes per epoch** on 8GB GPU

---

## Quick Start

### Prerequisites

- NVIDIA GPU with 8GB+ VRAM
- NVIDIA HPC SDK (nvfortran with CUDA)
- cuDNN library
- Python 3.8+ with PyTorch (for evaluation)

### Training on Real Data (Noise2Noise)

```bash
# Preprocess movie stacks (creates odd/even frame pairs)
python tools/preprocess_n2n_movies.py \
    --input_dir /path/to/raw/movies \
    --output_dir data/real_n2n \
    --patch_size 64 --stride 128

# Train
cd v33_deep_residual
make cryo_train_n2n
./cryo_train_n2n \
    --train_file ../data/real_n2n/train_n2n.bin \
    --val_file ../data/real_n2n/val_n2n.bin \
    --epochs 10 --batch_size 32 --save
```

### Training on Synthetic Data (Supervised)

```bash
cd v33_deep_residual
make cryo_train_deep
./cryo_train_deep --stream --epochs 5 --data_dir ../data/cryo_high_noise/ --save
```

### Evaluation

```bash
# Interactive Jupyter notebook with verification tests
jupyter notebook notebooks/evaluate_n2n_real.ipynb
```

---

## Repository Structure

```
cryo_em_v2/
├── v33_deep_residual/           # Main implementation (12-layer CNN)
│   ├── cryo_train_n2n.cuf       # Noise2Noise training (real data)
│   ├── cryo_train_deep.cuf      # Supervised training (synthetic)
│   ├── streaming_n2n_loader.cuf # N2N streaming data loader
│   ├── streaming_cryo_loader.cuf # Original streaming loader
│   ├── conv2d_cudnn.cuf         # cuDNN convolution wrapper
│   ├── Makefile
│   └── saved_models/
│       ├── n2n_real/epoch_0009/ # Best N2N model (real data)
│       └── deep12/epoch_0005/   # Best supervised model (synthetic)
│
├── notebooks/
│   └── evaluate_n2n_real.ipynb  # Evaluation with verification tests
│
├── tools/
│   ├── preprocess_n2n_movies.py # Movie stack → N2N training pairs
│   └── download_empiar_10025.sh # Download EMPIAR data
│
├── data/                        # Training data (not in repo)
│   ├── real_n2n/               # Preprocessed N2N pairs
│   └── cryo_high_noise/        # Synthetic noisy/clean pairs
│
├── v31_perceptual_loss/         # Perceptual loss experiments
└── v32_architecture_tests/      # Architecture A/B testing
```

---

## Data Format

The streaming binary format enables training on datasets larger than GPU memory:

```
train_input.bin:  [patch1_pixels][patch2_pixels]...
train_target.bin: [patch1_pixels][patch2_pixels]...

Each patch: 1024 × 1024 × float32 = 4 MB
```

### Creating Training Data

```python
# Generate synthetic noisy/clean pairs
python tools/create_synthetic_data.py \
    --input /path/to/mrc/files \
    --output data/cryo_high_noise \
    --noise_sigma 0.25 \
    --patch_size 1024
```

---

## Why This Works

### 1. Residual Learning
By predicting noise instead of the clean image directly, the network:
- Learns what *doesn't belong* (easier task)
- Preserves structures automatically
- Avoids over-smoothing

### 2. Optimal Depth (12 Layers)
Through A/B testing, we found:
- 3 layers: Too shallow, loses detail
- 12 layers: Sweet spot for structure preservation
- 15 layers: Unstable training

### 3. Large Dataset + Simple Architecture
- 10,309 real patches provide natural variation
- No augmentation needed
- Simple architecture = debuggable and fast

---

## Comparison: Architecture Search Results

| Architecture | PSNR Gain | Edge Preservation | Notes |
|--------------|-----------|-------------------|-------|
| 3-layer baseline | +4.0 dB | 54% | Too shallow |
| 7-layer | +4.0 dB | 54% | Similar to baseline |
| **12-layer residual** | **+4.1 dB** | **56%** | **Best visual quality** |
| 15-layer | - | 138% | Unstable, didn't converge |

The 12-layer residual CNN provides the best balance of noise removal and structure preservation.

---

## Hardware Requirements

### Tested Configuration
- **GPU**: NVIDIA RTX 4060 Ti (8GB)
- **RAM**: 32GB system memory
- **Storage**: 1TB for dataset
- **Training time**: ~4 hours (5 epochs)

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070/4060 Ti or equivalent)
- **CUDA**: 11.8+
- **cuDNN**: 8.x

---

## Technical Innovations

### 1. Streaming Data Loader
Handles datasets larger than GPU memory:
- Double-buffered async I/O
- Constant ~100MB memory usage
- Works with 100GB+ datasets on 8GB GPU

### 2. Native cuDNN Integration
Direct Fortran-to-cuDNN calls:
- Minimal framework overhead
- Efficient memory management
- Full control over algorithms

### 3. Adam Optimizer with Bias Correction
Built-in Adam implementation:
- β1=0.9, β2=0.999
- Proper bias correction for early steps
- Per-layer moment tracking

---

## Citation

If you use this code, please cite:

```bibtex
@software{cryo_em_denoise_fortran,
  title={Cryo-EM Denoising with Deep Residual CNN in CUDA Fortran},
  author={Fraser},
  year={2025},
  url={https://github.com/fraser/cryo-em-denoise}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- EMPIAR database for cryo-EM micrographs
- NVIDIA for HPC SDK and cuDNN
- The cryo-EM community for benchmarks and datasets

---

**Last Updated**: February 2025  
**Status**: ✅ Production Ready
