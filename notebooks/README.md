# Cryo-EM CNN Evaluation Notebooks

Jupyter notebooks for analyzing and visualizing the Fortran/CUDA CNN denoising results.

## Notebooks

### 1. `cryo_cnn_evaluation.ipynb` - Quick Metrics ⚡

**Purpose:** Fast evaluation of model performance  
**Runtime:** ~2-5 minutes  
**Output:** Key metrics and quick visualization

**Metrics provided:**
- MSE / RMSE
- PSNR (Peak Signal-to-Noise Ratio)
- Pearson correlation (mean, median, std)
- Comparison with training val loss
- Random sample visualization

**Use this for:**
- ✅ Quick validation after training
- ✅ Verifying model loaded correctly
- ✅ Checking test set performance

### 2. `cryo_cnn_analysis.ipynb` - Detailed Analysis 📊

**Purpose:** Comprehensive visual analysis  
**Runtime:** ~5-15 minutes  
**Output:** Detailed visualizations and statistical analysis

**Features:**
- Load Fortran weights into PyTorch model
- Full test set inference
- Multiple random sample visualizations
- Distribution analysis (RMSE, correlation, pixel values)
- Best/worst case analysis
- Predicted vs target scatter plots
- Error maps

**Use this for:**
- ✅ Publication-quality figures
- ✅ Deep dive into model behavior
- ✅ Understanding failure cases
- ✅ Statistical validation

## Quick Start

### Requirements

```bash
pip install numpy matplotlib torch scipy tqdm jupyter
```

### Run Evaluation

```bash
cd v28f_cryo_em/notebooks
jupyter notebook cryo_cnn_evaluation.ipynb
```

Then run all cells (Cell → Run All).

## Expected Results

Based on training performance (epoch 1):

| Metric | Expected Value |
|--------|---------------|
| Val Loss (training) | 0.00697 |
| Test MSE | ~0.007-0.008 |
| Test RMSE | ~0.083-0.089 |
| PSNR | >25 dB |
| Mean Correlation | >0.95 |

## File Structure

```
v28f_cryo_em/
├── v28f_e_final_training/
│   └── saved_models/
│       └── cryo_cnn/
│           └── epoch_0001/          # Best checkpoint (val loss 0.00697)
│               ├── loss.txt
│               ├── conv1_weights.bin
│               ├── conv1_bias.bin
│               ├── conv2_weights.bin
│               ├── conv2_bias.bin
│               ├── conv3_weights.bin
│               └── conv3_bias.bin
├── data/
│   └── cryo_data_streaming/
│       ├── test_input.bin           # Noisy test patches
│       └── test_target.bin          # Clean test patches
└── notebooks/                        # ← You are here
    ├── cryo_cnn_evaluation.ipynb    # Quick metrics
    └── cryo_cnn_analysis.ipynb      # Detailed analysis
```

## Weight Loading

Fortran saves weights in **(kH, kW, in_ch, out_ch)** format.  
PyTorch expects **(out_ch, in_ch, kH, kW)** format.

The notebooks handle this conversion automatically:

```python
# Load from Fortran binary
weights = np.fromfile('conv1_weights.bin', dtype=np.float32)

# Reshape and transpose
weights = weights.reshape(3, 3, 1, 16).transpose(3, 2, 0, 1)

# Load into PyTorch
model.conv1.weight.data = torch.from_numpy(weights)
```

## Troubleshooting

### Checkpoint not found
```
ERROR: Checkpoint not found at ../v28f_e_final_training/saved_models/cryo_cnn/epoch_0001/
```

**Solution:** Training must complete with `--save` flag:
```bash
cd ../v28f_e_final_training
./cryo_train --stream --epochs 5 --save
```

### Data files not found
```
FileNotFoundError: ../data/cryo_data_streaming/test_input.bin
```

**Solution:** Check data path is correct. Files should be in `v28f_cryo_em/data/cryo_data_streaming/`.

### Out of memory

**Solution:** Reduce batch size in inference loop:
```python
batch_size = 4  # Reduce from 8
```

Or process fewer test patches:
```python
test_noisy = test_noisy[:100]  # First 100 patches only
test_clean = test_clean[:100]
```

## Comparing Fortran vs PyTorch

The notebooks load **Fortran-trained weights** and run inference in PyTorch. This validates:

✅ Weight format conversion is correct  
✅ Model architecture matches  
✅ Fortran training produced valid weights  
✅ Results match training metrics  

## Next Steps

After validation:

1. **If results match expectations** (~0.007 test loss):
   - ✓ Model is production-ready
   - ✓ Can deploy for large-scale denoising
   - ✓ Bug fix confirmed working

2. **If results don't match**:
   - Check weight loading (print weight statistics)
   - Verify data normalization
   - Compare with PyTorch reference training

## Citation

If you use these notebooks, please cite the training framework:

```
Fortran/CUDA CNN for Cryo-EM Denoising (v28f)
Based on climate U-Net template (ACC 0.9851)
Bug fix: Explicit alpha/beta initialization in cuDNN convolutions
```

---

**Questions?** Check the main project README or SETUP_COMPLETE.md in the training directory.
