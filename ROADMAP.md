# Cryo-EM Denoising Project Roadmap

## Phase 1: Fourier Loss Foundation [COMPLETE]
- [x] Create minimal cuFFT test program (`test_cufft_minimal.cuf`)
- [x] Verify FFT and power spectrum computation
- [x] Create fourier_loss module (`fourier_loss.cuf`)
- [x] Integrate Fourier loss into training (`cryo_train_fourier.cuf`)
- [x] Add `--fourier_weight` command line argument
- [x] Use log power spectrum for stable loss scaling
- [x] Commit to GitHub

## Phase 2: Full Training Run & Validation [CURRENT]
- [ ] Run full epoch with Fourier loss (fourier_weight=0.1)
- [ ] Compare MSE-only vs MSE+Fourier results
- [ ] Measure impact on high-frequency detail preservation
- [ ] Tune fourier_weight hyperparameter (try 0.05, 0.1, 0.2)
- [ ] Save best checkpoint with Fourier loss

## Phase 3: Perceptual/Structural Losses
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
- Training: `../data/cryo_data_streaming/` (~117 GB, 29,952 patches)
- Patch size: 1024 × 1024 pixels

**Hardware:**
- GPU: 8GB VRAM (streaming loader handles 117GB dataset)
- Batch size: 4

---

## Notes

### Fourier Loss Design
The Fourier loss uses `log(1 + |F|²)` instead of raw power spectra because:
1. Power spectra scale with N⁴ (~10¹² for 1024×1024 images)
2. Log transform normalizes to ~0-50 range, comparable to MSE
3. Prevents gradient explosion during training

### Next Steps
Ready to run Phase 2: Full training comparison between MSE-only and MSE+Fourier approaches.
