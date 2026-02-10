#!/usr/bin/env python3
"""
PyTorch Reference Implementation: Simple CNN for Cryo-EM Denoising

This serves as a reference for the Fortran/CUDA implementation.
It uses the same architecture and training approach to:
  1. Validate the data pipeline
  2. Establish baseline performance
  3. Show expected loss curves
  4. Help identify cuDNN calls for Fortran

Architecture matches the Fortran version:
  - Conv1: 1 -> 16 channels, 3×3, ReLU, padding=1
  - Conv2: 16 -> 16 channels, 3×3, ReLU, padding=1
  - Conv3: 16 -> 1 channel, 3×3, no activation, padding=1

Usage:
    python simple_cnn_torch.py --epochs 2 --batch_size 4 --lr 0.001
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    """Simple 3-layer CNN for denoising"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Three convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # No activation on output
        return x

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CryoEMDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset for streaming cryo-EM patches"""

    def __init__(self, input_file, target_file, num_patches, patch_size=1024):
        self.input_file = input_file
        self.target_file = target_file
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_pixels = patch_size * patch_size

        # Open files in __init__ but seek per __getitem__
        self.input_fp = None
        self.target_fp = None

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        """Load a single patch from binary files"""
        # Open files on first access (lazy initialization)
        if self.input_fp is None:
            self.input_fp = open(self.input_file, "rb")
            self.target_fp = open(self.target_file, "rb")

        # Calculate file position
        offset = idx * self.patch_pixels * 4  # 4 bytes per float32

        # Read noisy patch
        self.input_fp.seek(offset)
        noisy = np.fromfile(self.input_fp, dtype=np.float32, count=self.patch_pixels)
        noisy = noisy.reshape(self.patch_size, self.patch_size)

        # Read clean patch
        self.target_fp.seek(offset)
        clean = np.fromfile(self.target_fp, dtype=np.float32, count=self.patch_pixels)
        clean = clean.reshape(self.patch_size, self.patch_size)

        # Convert to tensors and add channel dimension
        noisy = torch.from_numpy(noisy).unsqueeze(0)  # (1, H, W)
        clean = torch.from_numpy(clean).unsqueeze(0)

        return noisy, clean


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    start_time = time.time()

    for batch_idx, (noisy, clean) in enumerate(dataloader):
        # Move to device
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        epoch_loss += loss.item()
        batch_count += 1

        # Print progress
        if (batch_idx + 1) % 1000 == 0:
            avg_loss = epoch_loss / batch_count
            elapsed = time.time() - start_time
            batches_per_sec = batch_count / elapsed
            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss.item():.6f} | "
                f"Avg: {avg_loss:.6f} | "
                f"Speed: {batches_per_sec:.2f} batches/s"
            )

    avg_epoch_loss = epoch_loss / batch_count
    epoch_time = time.time() - start_time

    return avg_epoch_loss, epoch_time


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Simple CNN for Cryo-EM Denoising"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/cryo_data_streaming",
        help="Directory containing train_input.bin and train_target.bin",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num_patches",
        type=int,
        default=29913,
        help="Total number of training patches",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  PyTorch Simple CNN - Cryo-EM Denoising Reference")
    print("=" * 70)
    print()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print()

    # Create dataset
    input_file = Path(args.data_dir) / "train_input.bin"
    target_file = Path(args.data_dir) / "train_target.bin"

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Please run preprocessing first.")
        return

    print("Dataset:")
    print(f"  Input:  {input_file}")
    print(f"  Target: {target_file}")
    print(f"  Patches: {args.num_patches:,}")
    print(f"  Patch size: 1024 × 1024")
    print()

    dataset = CryoEMDataset(str(input_file), str(target_file), args.num_patches)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Create model
    model = SimpleCNN().to(device)
    num_params = model.count_parameters()

    print("Model Architecture:")
    print(f"  Conv1: 1 -> 16 channels (3×3, ReLU, padding=1)")
    print(f"  Conv2: 16 -> 16 channels (3×3, ReLU, padding=1)")
    print(f"  Conv3: 16 -> 1 channel (3×3, no activation, padding=1)")
    print(f"  Total parameters: {num_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    print("Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: SGD")
    print(f"  Loss: MSE")
    print(f"  Batches per epoch: {len(dataloader):,}")
    print()

    # Training loop
    print("=" * 70)
    print("  Starting Training")
    print("=" * 70)
    print()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 70)

        avg_loss, epoch_time = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch, args.epochs
        )

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(
            f"  Time: {epoch_time:.2f}s ({len(dataloader) / epoch_time:.2f} batches/s)"
        )
        print()

    print("=" * 70)
    print("  Training Complete")
    print("=" * 70)
    print()

    # Save model
    model_path = Path("simple_cnn_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "final_loss": avg_loss,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")
    print()


if __name__ == "__main__":
    main()
