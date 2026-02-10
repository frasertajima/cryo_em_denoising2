# Cryo-EM Denoising Project Roadmap

## Phase 1: Fourier Loss Foundation [COMPLETE]
- [x] Create minimal cuFFT test program (`test_cufft_minimal.cuf`)
- [x] Verify FFT and power spectrum computation
- [x] Create fourier_loss module (`fourier_loss.cuf`)
- [x] Integrate Fourier loss into training (`cryo_train_fourier.cuf`)
- [x] Add `--fourier_weight` command line argument
- [x] Use log power spectrum for stable loss scaling
- [x] Commit to GitHub

## Phase 2: Full Training Run & Validation [COMPLETE]
- [x] Fix data mismatch (regenerated 10,309 matched train patches)
- [x] Run full epoch with Fourier loss (fourier_weight=0.1)
- [x] Compare MSE-only vs MSE+Fourier results
- [ ] Measure impact on high-frequency detail preservation (visual inspection)
- [ ] Tune fourier_weight hyperparameter (try 0.05, 0.2)
- [x] Save best checkpoint with Fourier loss

### Phase 2 Results (1 epoch, 10,309 patches)

| Metric | MSE-Only | MSE + Fourier (10%) |
|--------|----------|---------------------|
| Train MSE | 0.00994 | 0.00993 |
| Train RMSE | 0.100 | 0.100 |
| Val MSE | 0.00697 | ~0.0063 |
| Val RMSE | 0.084 | ~0.079 |
| FFT Loss | N/A | 12.4 → 1.71 (86% reduction) |
| Time | 7.7 min | 7.8 min |

**Key Finding:** Fourier loss reduces frequency domain error by 86% without impacting pixel-wise MSE.

## Phase 3: Perceptual/Structural Losses [NEXT]
- [ ] Add SSIM loss module (structural similarity)
- [ ] Add gradient loss (edge preservation)
- [ ] Experiment with multi-scale losses
- [ ] Compare loss combinations

## Phase 4: Architecture Improvements
- [ ] Increase network depth (more conv layers)
- [ ] Add residual connections (ResNet-style)
- [ ] Experiment with U-Net architecture
- [ ] Add batch normalization

## Phase 5: Advanced Features
- [ ] Multi-resolution training (coarse-to-fine)
- [ ] Data augmentation (rotations, flips)
- [ ] Learning rate scheduling
- [ ] Mixed precision training (FP16)

## Phase 6: Inference & Deployment
- [ ] Create inference-only program
- [ ] Export model for Python/PyTorch loading
- [ ] Benchmark inference speed
- [ ] Create visualization tools

---

## Current Status

**Location:** `/var/home/fraser/cryo_em_v2/v29_fourier_loss/`

**Key Files:**
| File | Description |
|------|-------------|
| `fourier_loss.cuf` | Log power spectrum MSE loss module |
| `cryo_train_fourier.cuf` | Training with combined MSE + Fourier loss |
| `compile.sh` | Build script |

**Usage:**
```bash
cd v29_fourier_loss
./compile.sh
./cryo_train_fourier --stream                      # MSE only
./cryo_train_fourier --stream --fourier_weight 0.1 # 10% Fourier loss
./cryo_train_fourier --stream --epochs 5 --save    # Save checkpoints
```

**Data:** 
- Training: `../data/cryo_data_streaming/` (~41 GB, 10,309 patches)
- Test: 1,014 patches (~4 GB)
- Patch size: 1024 x 1024 pixels
- Source: 67 EMPIAR-10025 micrographs

**Hardware:**
- GPU: 8GB VRAM (streaming loader handles large datasets)
- Batch size: 4
- Throughput: ~5 samples/sec

---

## Notes

### Fourier Loss Design
The Fourier loss uses `log(1 + |F|^2)` instead of raw power spectra because:
1. Power spectra scale with N^4 (~10^12 for 1024x1024 images)
2. Log transform normalizes to ~0-15 range, comparable to MSE
3. Prevents gradient explosion during training

### Data Issue Fixed
Original data had mismatched input/target files (29,913 vs 7,355 patches).
Regenerated both files from source MRCs with matching 10,309 patches.
