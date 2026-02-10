# Cryo-EM CNN Training - v28f

## Overview

This directory contains the **production training program** for Cryo-EM particle denoising using a simple 3-layer CNN adapted from the proven climate U-Net template.

## Architecture

**Simple 3-Layer CNN:**
- Input: 1024×1024 noisy particle image (1 channel)
- Conv1: 1 → 16 channels, 3×3 kernel, ReLU, same padding
- Conv2: 16 → 16 channels, 3×3 kernel, ReLU, same padding  
- Conv3: 16 → 1 channel, 3×3 kernel, same padding
- Output: 1024×1024 clean particle image (1 channel)

**Loss:** MSE (pixel-wise reconstruction)

**Optimizer:** Adam (β₁=0.9, β₂=0.999, ε=1e-8)

## Key Features

✅ **Bug-fixed conv2d_cudnn module** - Critical alpha/beta initialization fix applied  
✅ **Streaming data loader** - Handles large datasets efficiently  
✅ **Adapted from climate template** - Proven architecture with ACC 0.9851  
✅ **Checkpoint saving** - Best model saved automatically  
✅ **Validation split** - 10% holdout for model evaluation  

## Compilation

```bash
./compile.sh
```

Successfully compiles with warnings (TARGET attributes - can be ignored).

## Usage

### Basic Training (5 epochs, no checkpoints)
```bash
./cryo_train --stream --epochs 5
```

### Training with Checkpoint Saving
```bash
./cryo_train --stream --epochs 5 --save
```

### Custom Learning Rate
```bash
./cryo_train --stream --epochs 5 --lr 0.0001
```

### Custom Validation Split
```bash
./cryo_train --stream --epochs 5 --val_split 0.2
```

### Quick Test (first 10 batches only)
```bash
./cryo_train --stream --max_batches 10
```

## Expected Results

Based on PyTorch reference training:
- Initial loss: ~0.44
- Final loss (5 epochs): ~0.013
- Improvement: 34×

The Fortran/CUDA implementation should match these results within 1% if correctly implemented.

## Data Requirements

The training program expects data in:
```
../cryo_data_streaming/
  ├── noisy_train.bin   (noisy patches)
  └── clean_train.bin   (clean patches)
```

Each file should contain 1024×1024 float32 images in row-major format.

## Output Structure

When `--save` is used, checkpoints are saved to:
```
saved_models/cryo_cnn/
  └── epoch_NNNN/
      ├── loss.txt
      ├── conv1_weights.bin
      ├── conv1_bias.bin
      ├── conv2_weights.bin
      ├── conv2_bias.bin
      ├── conv3_weights.bin
      └── conv3_bias.bin
```

Only the best validation model is saved (lowest val loss).

## Training Parameters

- **Batch size:** 8 (hardcoded)
- **Image size:** 1024×1024 (hardcoded)
- **Default epochs:** 5
- **Default learning rate:** 0.001
- **Default val split:** 10%

## Progress Reporting

- Training: Every 100 batches
- Validation: After each epoch
- Metrics: Loss, RMSE, throughput (samples/sec)

## Next Steps

1. **Run training:** `./cryo_train --stream --epochs 5 --save`
2. **Monitor loss:** Should decrease from ~0.44 to ~0.013
3. **Compare with PyTorch:** Validate results match reference implementation
4. **Scale up:** Train on full dataset if initial results are good

## Notes

- The bug fix in `conv2d_cudnn.cuf` is **critical** for correct training
- This architecture is intentionally simple - proven to work in PyTorch
- Based on the climate template that achieved 98.51% ACC (28× improvement)
- Streaming loader handles datasets larger than GPU memory
