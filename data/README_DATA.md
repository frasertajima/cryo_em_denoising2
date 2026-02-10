# Data Setup Instructions

The training data is **not included** in this repository due to size (117GB+).

---

## Download Dataset

**Dataset**: EMPIAR-10025 (Electron Microscopy Public Image Archive)

**Source**: https://www.ebi.ac.uk/empiar/EMPIAR-10025/

**What to download**: 
- Frame-averaged micrographs (~41 GB compressed)
- Direct link: [EMPIAR-10025 Download](https://ftp.ebi.ac.uk/empiar/world_availability/10025/)

---

## Preprocessing

After downloading the MRC files, run the preprocessing script:

```bash
# Install dependencies
conda create -n cryo python=3.11
conda activate cryo
conda install numpy scipy matplotlib mrcfile

# Run preprocessing
cd tools
python preprocess_empiar_streaming.py \
    --input /path/to/downloaded/mrc/files \
    --output ../data/cryo_data_streaming \
    --patch_size 1024 \
    --stride 512 \
    --noise_type poisson_gaussian \
    --noise_level 0.05 \
    --test_split 0.1
```

**Output** (in `data/cryo_data_streaming/`):
- `train_input.bin` (117GB - noisy patches)
- `train_target.bin` (117GB - clean patches)
- `test_input.bin` (13GB - test noisy)
- `test_target.bin` (13GB - test clean)

**Total size**: ~260 GB (preprocessed)

---

## Expected Directory Structure

```
data/
├── README_DATA.md          (this file)
└── cryo_data_streaming/    (create this - empty initially)
    ├── train_input.bin     (create via preprocessing)
    ├── train_target.bin    (create via preprocessing)
    ├── test_input.bin      (create via preprocessing)
    └── test_target.bin     (create via preprocessing)
```

---

## Alternative: Use Your Own Data

You can also use your own cryo-EM micrographs:

1. Place `.mrc` files in a directory
2. Run preprocessing script (same command as above)
3. Adjust `--patch_size` and `--stride` as needed
4. Training code will work with any generated `.bin` files

---

## Storage Requirements

- **Downloaded data**: ~41 GB (MRC files)
- **Preprocessed data**: ~260 GB (binary files)
- **Training temp**: ~10 GB (model checkpoints)
- **Total**: ~311 GB

**Recommendation**: Use SSD for faster I/O during training

---

## Questions?

See main [README.md](../README.md) for complete setup instructions.
