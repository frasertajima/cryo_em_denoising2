#!/usr/bin/env python3
"""
Export PyTorch Model Weights and Test Batch for Fortran Validation

This script:
1. Initializes the same SimpleCNN architecture
2. Exports initial weights to binary format
3. Exports a test batch (noisy, clean)
4. Runs forward pass and saves output
5. Computes loss and gradients, saves them

The Fortran implementation can then load these files and verify
it produces identical results.
"""

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Same architecture as training script"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def export_conv_weights(conv_layer, prefix, output_dir):
    """Export convolution weights and biases to binary files

    PyTorch format: (Out, In, H, W) in row-major (C-order)
    Fortran format: (H, W, In, Out) in column-major (F-order)

    Conversion:
    1. Flip spatial dimensions: (Out, In, H, W) with H,W flipped
    2. Transpose axes: (Out, In, H, W) -> (H, W, In, Out)
    3. Convert to Fortran order
    """
    # Get weights: (out_channels, in_channels, kH, kW)
    weight = conv_layer.weight.detach().cpu().numpy().astype(np.float32)
    bias = conv_layer.bias.detach().cpu().numpy().astype(np.float32)

    print(f"  {prefix} weight original shape: {weight.shape} (Out, In, H, W)")

    # Step 1: Flip spatial dimensions (H and W axes)
    weight_flipped = np.flip(weight, axis=(2, 3)).copy()

    # Step 2: Transpose - inverse of (3,2,1,0) used in loading
    weight_transposed = weight_flipped.transpose(3, 2, 1, 0)

    print(f"  {prefix} weight Fortran shape: {weight_transposed.shape} (H, W, In, Out)")

    # Save weights with F-order flatten (Fortran column-major)
    weight_file = output_dir / f"{prefix}_weight.bin"
    weight_transposed.reshape(-1, order="F").tofile(weight_file)
    print(f"  {prefix} weight -> {weight_file}")

    # Save bias (1D array, no layout conversion needed)
    bias_file = output_dir / f"{prefix}_bias.bin"
    bias.tofile(bias_file)
    print(f"  {prefix} bias: {bias.shape} -> {bias_file}")

    return weight.shape, bias.shape


def main():
    print("=" * 70)
    print("  Export PyTorch Model for Fortran Validation")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = Path("pytorch_reference/fortran_validation")
    output_dir.mkdir(exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    print("Model Architecture:")
    print(f"  Conv1: 1 -> 16 channels (3×3)")
    print(f"  Conv2: 16 -> 16 channels (3×3)")
    print(f"  Conv3: 16 -> 1 channel (3×3)")
    print()

    # Export initial weights
    print("Exporting weights...")
    export_conv_weights(model.conv1, "conv1", output_dir)
    export_conv_weights(model.conv2, "conv2", output_dir)
    export_conv_weights(model.conv3, "conv3", output_dir)
    print()

    # Create a small test batch (2 samples, 1024×1024)
    batch_size = 2
    print(f"Creating test batch (size={batch_size})...")

    # Generate random test data
    noisy = torch.rand(batch_size, 1, 1024, 1024, dtype=torch.float32)
    clean = torch.rand(batch_size, 1, 1024, 1024, dtype=torch.float32)

    # Export test batch with layout conversion
    # PyTorch: (N, C, H, W) row-major -> Fortran: (C, H, W, N) column-major
    noisy_file = output_dir / "test_noisy.bin"
    clean_file = output_dir / "test_clean.bin"

    # Convert to Fortran layout: (N, C, H, W) -> (C, H, W, N)
    noisy_np = noisy.numpy()
    clean_np = clean.numpy()

    print(f"  Noisy original: {noisy_np.shape} (N, C, H, W)")

    # For data tensors: transpose (N,C,H,W) -> (C,H,W,N), then flatten with F-order
    noisy_transposed = noisy_np.transpose(1, 2, 3, 0)  # (N,C,H,W) -> (C,H,W,N)
    clean_transposed = clean_np.transpose(1, 2, 3, 0)

    print(f"  Noisy Fortran: {noisy_transposed.shape} (C, H, W, N)")

    # Flatten with F-order to write in Fortran column-major memory order
    noisy_transposed.reshape(-1, order="F").tofile(noisy_file)
    clean_transposed.reshape(-1, order="F").tofile(clean_file)

    print(f"  Noisy -> {noisy_file}")
    print(f"  Clean -> {clean_file}")
    print()

    # Run forward pass
    print("Running forward pass...")
    noisy_gpu = noisy.to(device)
    clean_gpu = clean.to(device)

    model.eval()
    with torch.no_grad():
        output = model(noisy_gpu)

    # Export forward pass output with layout conversion
    output_file = output_dir / "test_output.bin"
    output_np = output.cpu().numpy()

    # For data tensors: transpose (N,C,H,W) -> (C,H,W,N), then flatten with F-order
    output_transposed = output_np.transpose(1, 2, 3, 0)

    output_transposed.reshape(-1, order="F").tofile(output_file)
    print(f"  Output: {output.shape} -> {output_file}")
    print()

    # Compute loss
    criterion = nn.MSELoss()
    loss = criterion(output, clean_gpu)
    print(f"Forward pass loss: {loss.item():.6f}")
    print()

    # Run backward pass
    print("Running backward pass...")
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    output = model(noisy_gpu)
    loss = criterion(output, clean_gpu)

    optimizer.zero_grad()
    loss.backward()

    # Export gradients with layout conversion
    print("Exporting gradients...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_file = output_dir / f"{name.replace('.', '_')}_grad.bin"
            grad_np = param.grad.cpu().numpy().astype(np.float32)

            # Apply same layout conversion for weight gradients
            if "weight" in name and len(grad_np.shape) == 4:
                # Conv weight gradient: (Out, In, H, W) -> (H, W, In, Out)
                grad_flipped = np.flip(grad_np, axis=(2, 3)).copy()
                grad_transposed = grad_flipped.transpose(3, 2, 1, 0)
                grad_transposed.reshape(-1, order="F").tofile(grad_file)
                print(
                    f"  {name}: {grad_np.shape} -> {grad_transposed.shape} (Fortran) -> {grad_file}"
                )
            else:
                # Bias gradient: no conversion needed
                grad_np.tofile(grad_file)
                print(f"  {name}: {grad_np.shape} -> {grad_file}")
    print()

    # Create metadata file
    meta_file = output_dir / "metadata.txt"
    with open(meta_file, "w") as f:
        f.write("PyTorch Model Export for Fortran Validation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Image size: 1024 x 1024\n")
        f.write(f"Forward loss: {loss.item():.10f}\n")
        f.write(f"Random seed: 42\n")
        f.write(
            "\nFile Format: All binary files are float32, column-major (Fortran order)\n"
        )
        f.write("\nLayout Conversion Applied:\n")
        f.write("  - Conv weights: (Out,In,H,W) -> (H,W,In,Out) with spatial flip\n")
        f.write("  - Tensors: (N,C,H,W) -> (C,H,W,N) with spatial flip\n")
        f.write("  - All arrays stored in Fortran column-major order\n")
        f.write("\nWeight shapes:\n")
        f.write(f"  conv1.weight: {model.conv1.weight.shape}\n")
        f.write(f"  conv1.bias: {model.conv1.bias.shape}\n")
        f.write(f"  conv2.weight: {model.conv2.weight.shape}\n")
        f.write(f"  conv2.bias: {model.conv2.bias.shape}\n")
        f.write(f"  conv3.weight: {model.conv3.weight.shape}\n")
        f.write(f"  conv3.bias: {model.conv3.bias.shape}\n")

    print(f"Metadata saved to: {meta_file}")
    print()

    print("=" * 70)
    print("  Export Complete!")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    for f in sorted(output_dir.glob("*.bin")):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {f.name:30s} {size_mb:8.2f} MB")
    print()
    print("Next steps:")
    print("  1. Create Fortran test program")
    print("  2. Load these weights and test batch")
    print("  3. Run forward pass and compare output")
    print("  4. Verify loss matches PyTorch")
    print()


if __name__ == "__main__":
    main()
