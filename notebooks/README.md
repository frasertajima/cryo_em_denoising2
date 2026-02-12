# Cryo-EM CNN Evaluation Notebooks

Jupyter notebooks for analyzing and visualizing the Fortran/CUDA CNN denoising results.

## Current Notebooks

### `evaluate_n2n_real.ipynb` - Real Data Evaluation (Primary)

**Purpose:** Evaluate Noise2Noise trained model on real cryo-EM data (EMPIAR-10025)  
**Runtime:** ~5-10 minutes  
**Dataset:** T20S Proteasome movie stacks

**Features:**
- Load Fortran-trained weights into PyTorch
- Denoise full 7676×7420 micrographs
- Verification tests proving signal is real:
  - Cross-validation (odd/even frame independence)
  - Particle detection comparison
  - Fourier Ring Correlation analysis
  - Consistency check (both halves reveal same structures)
- Quantitative metrics (PSNR, SSIM)

**Key Results:**
| Metric | Raw | Denoised | Improvement |
|--------|-----|----------|-------------|
| Odd/Even Correlation | 0.067 | 0.879 | 13x |
| PSNR | baseline | +2.82 dB | - |
| Removed Noise Correlation | - | 0.024 | Confirms noise removal |

**Reveals protein particles that are completely invisible in raw half-set averages!**

## Quick Start

```bash
cd notebooks
jupyter notebook evaluate_n2n_real.ipynb
```

## Requirements

```bash
pip install numpy matplotlib torch scipy scikit-image mrcfile tqdm jupyter
```

## Model Weights

The current best models are:
- `v33_deep_residual/saved_models/n2n_real/epoch_0009/` - N2N trained on real data
- `v33_deep_residual/saved_models/deep12/epoch_0005/` - Supervised on synthetic data

## Archived Notebooks

Old notebooks from earlier versions (v28f) are in `archive/`. They reference model paths that no longer exist and are kept for historical reference only.

## Architecture

The 12-layer residual CNN:
```
Input (64×64 or 1024×1024)
  → Conv1 (1→32, ReLU)
  → Conv2-11 (32→32, ReLU) × 10 layers
  → Conv12 (32→1, no activation)
  → Output = Input - PredictedNoise (residual)
```

Total: 93,089 parameters
