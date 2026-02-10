# Tensor Layout: PyTorch ↔ Fortran Conversion

## The Critical Issue

PyTorch and Fortran have **different memory layouts**:

### PyTorch (NumPy)
- **Row-major** (C-order)
- **NCHW format**: (batch, channels, height, width)
- Element at `[n, c, h, w]` is at index: `n*CHW + c*HW + h*W + w`

### Fortran
- **Column-major** (Fortran-order)
- We use **(C, H, W, N)** format to match cuDNN
- Element at `(c, h, w, n)` is at index: `c + h*C + w*C*H + n*C*H*W`

## Example: 2×1×3×4 Tensor

PyTorch shape: `(2, 1, 3, 4)` - 2 samples, 1 channel, 3×4 images

```python
# PyTorch memory layout (row-major, NCHW):
[n0_c0_h0_w0, n0_c0_h0_w1, n0_c0_h0_w2, n0_c0_h0_w3,  # first row
 n0_c0_h1_w0, n0_c0_h1_w1, n0_c0_h1_w2, n0_c0_h1_w3,  # second row
 n0_c0_h2_w0, n0_c0_h2_w1, n0_c0_h2_w2, n0_c0_h2_w3,  # third row
 n1_c0_h0_w0, n1_c0_h0_w1, ...]                       # second sample
```

Fortran shape: `(1, 3, 4, 2)` - (C, H, W, N)

```fortran
! Fortran memory layout (column-major, CHWN):
! First all channels, then all heights, then all widths, then all batches
[n0_c0_h0_w0, n0_c0_h1_w0, n0_c0_h2_w0,  ! first column (h varies fastest)
 n0_c0_h0_w1, n0_c0_h1_w1, n0_c0_h2_w1,  ! second column
 n0_c0_h0_w2, n0_c0_h1_w2, n0_c0_h2_w2,  # third column
 n0_c0_h0_w3, n0_c0_h1_w3, n0_c0_h2_w3,  # fourth column
 n1_c0_h0_w0, ...]                       # second sample
```

## The Conversion

When exporting from PyTorch to Fortran:

```python
# PyTorch array (NCHW, row-major)
pytorch_array = torch.randn(2, 1, 3, 4)

# WRONG: Direct tofile() - wrong layout!
pytorch_array.numpy().tofile("wrong.bin")

# CORRECT: Transpose to (C, H, W, N) then use Fortran order
transposed = pytorch_array.permute(1, 2, 3, 0)  # NCHW -> CHWN
fortran_array = np.asfortranarray(transposed.numpy())
fortran_array.tofile("correct.bin")
```

When loading in Fortran:

```fortran
! Declare with Fortran dimensions: (C, H, W, N)
real(4) :: array(1, 3, 4, 2)

! Read directly - already in correct layout
open(unit=100, file="correct.bin", form='unformatted', access='stream')
read(100) array
close(100)

! Access element: array(channel, height, width, batch)
value = array(1, 2, 3, 1)  ! channel 1, row 2, col 3, sample 1
```

## Convolution Weights

Weights are even more tricky:

### PyTorch Conv2d weights:
- Shape: `(out_channels, in_channels, kernel_h, kernel_w)`
- Order: OIHW (row-major)

### cuDNN/Fortran expects:
- Shape in descriptor: `(out_channels, in_channels, kernel_h, kernel_w)`
- But memory layout is column-major!

### Conversion for weights:

```python
# PyTorch weight: (16, 1, 3, 3) for Conv2d(1, 16, kernel_size=3)
weight = model.conv1.weight.detach().cpu()

# Convert to Fortran column-major
weight_fortran = np.asfortranarray(weight.numpy())
weight_fortran.tofile("conv1_weight.bin")
```

In Fortran, read as:
```fortran
real(4) :: weights(out_ch, in_ch, kH, kW)  ! Same dimension order!
read(100) weights  ! But column-major storage
```

## Verification Strategy

1. **Export test pattern** - Use known values (0, 1, 2, 3, ...)
2. **Print first few values** - Compare PyTorch vs Fortran
3. **Run trivial operation** - e.g., element-wise add
4. **Compare results** - Should match exactly

## Our Current Bug

In `export_for_fortran.py`, we're doing:

```python
noisy.numpy().tofile(noisy_file)  # WRONG!
```

Should be:

```python
# Transpose NCHW -> CHWN and convert to Fortran order
noisy_fortran = np.asfortranarray(noisy.permute(1, 2, 3, 0).numpy())
noisy_fortran.tofile(noisy_file)
```

## Fix Required

Update `export_for_fortran.py` to properly export in Fortran column-major layout!
