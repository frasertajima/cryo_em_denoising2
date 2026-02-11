#!/usr/bin/env python3
"""
A/B Testing Framework for Denoising Architectures

Tests different CNN architectures on a small subset of data
and generates visual comparisons.

Architectures to test:
A. Baseline: 3-layer CNN (current)
B. Residual: Predict noise, subtract from input
C. Deeper: 7-layer CNN
D. Skip: Mini U-Net with skip connections
E. Wide: Fewer layers but more channels
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import ndimage

# ============================================================
# Data Loading
# ============================================================


def load_patches(filepath, num_patches, patch_size=1024, crop_size=256):
    """Load patches from binary file and center-crop to reduce memory."""
    patches = []
    bytes_per_patch = patch_size * patch_size * 4
    offset = (patch_size - crop_size) // 2
    with open(filepath, "rb") as f:
        for i in range(num_patches):
            data = f.read(bytes_per_patch)
            if len(data) < bytes_per_patch:
                break
            patch = np.frombuffer(data, dtype=np.float32).reshape(
                patch_size, patch_size
            )
            # Center crop to reduce memory usage
            cropped = patch[offset : offset + crop_size, offset : offset + crop_size]
            patches.append(cropped)
    return np.array(patches)


# ============================================================
# Architecture Definitions
# ============================================================


class BaselineCNN(nn.Module):
    """A. Original 3-layer CNN"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResidualCNN(nn.Module):
    """B. Residual learning - predict noise, subtract from input"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        noise = self.relu(self.conv1(x))
        noise = self.relu(self.conv2(noise))
        noise = self.conv3(noise)
        return identity - noise  # Predict and subtract noise


class DeeperCNN(nn.Module):
    """C. Deeper 7-layer CNN"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class DeeperResidualCNN(nn.Module):
    """C2. Deeper 7-layer with residual learning"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return x - self.layers(x)  # Residual: predict noise


class SkipCNN(nn.Module):
    """D. Mini U-Net with skip connections"""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        # Decoder with skip connections
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU()
        )  # 64+64=128
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1), nn.ReLU()
        )  # 32+32=64

        self.final = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)  # 32 channels
        e2 = self.enc2(e1)  # 64 channels

        # Bottleneck
        b = self.bottleneck(e2)  # 64 channels

        # Decoder with skips
        d2 = self.dec2(torch.cat([b, e2], dim=1))  # 64+64=128 -> 32
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # 32+32=64 -> 16

        return self.final(d1)


class WideCNN(nn.Module):
    """E. Wide but shallow - more channels, fewer layers"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# ============================================================
# Training and Evaluation
# ============================================================


def train_model(
    model, train_noisy, train_clean, epochs=50, lr=0.001, batch_size=4, device="cuda"
):
    """Train a model and return loss history."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors
    X = torch.tensor(train_noisy[:, np.newaxis, :, :]).to(device)
    Y = torch.tensor(train_clean[:, np.newaxis, :, :]).to(device)

    n_samples = len(X)
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Shuffle
        perm = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = X[idx]
            batch_y = Y[idx]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (n_samples // batch_size)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")

    return losses


def evaluate_model(model, test_noisy, test_clean, device="cuda"):
    """Evaluate model and return metrics."""
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X = torch.tensor(test_noisy[:, np.newaxis, :, :]).to(device)
        output = model(X).cpu().numpy()[:, 0, :, :]

    # Compute metrics
    mse_noisy = np.mean((test_noisy - test_clean) ** 2)
    mse_denoised = np.mean((output - test_clean) ** 2)

    psnr_noisy = 10 * np.log10(1.0 / mse_noisy)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised)

    # Edge preservation
    clean_edges = np.mean([np.abs(ndimage.sobel(c)).mean() for c in test_clean])
    denoised_edges = np.mean([np.abs(ndimage.sobel(d)).mean() for d in output])
    edge_ratio = denoised_edges / clean_edges

    # Structure preservation (std ratio)
    clean_std = np.mean([c.std() for c in test_clean])
    denoised_std = np.mean([d.std() for d in output])
    std_ratio = denoised_std / clean_std

    return {
        "psnr_noisy": psnr_noisy,
        "psnr_denoised": psnr_denoised,
        "psnr_improvement": psnr_denoised - psnr_noisy,
        "edge_ratio": edge_ratio,
        "std_ratio": std_ratio,
        "output": output,
    }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Main Testing
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/cryo_high_noise/")
    parser.add_argument("--train_patches", type=int, default=20)
    parser.add_argument("--test_patches", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("  Architecture A/B Testing")
    print("=" * 70)
    print(f"  Train patches: {args.train_patches}")
    print(f"  Test patches:  {args.test_patches}")
    print(f"  Epochs:        {args.epochs}")
    print()

    # Load data
    print("Loading data...")
    train_noisy = load_patches(f"{args.data_dir}/train_input.bin", args.train_patches)
    train_clean = load_patches(f"{args.data_dir}/train_target.bin", args.train_patches)
    test_noisy = load_patches(f"{args.data_dir}/test_input.bin", args.test_patches)
    test_clean = load_patches(f"{args.data_dir}/test_target.bin", args.test_patches)
    print(f"  Loaded {len(train_noisy)} train, {len(test_noisy)} test patches")
    print()

    # Define architectures to test
    architectures = {
        "A_Baseline": BaselineCNN(),
        "B_Residual": ResidualCNN(),
        "C_Deeper": DeeperCNN(),
        "C2_DeeperRes": DeeperResidualCNN(),
        "D_Skip": SkipCNN(),
        "E_Wide": WideCNN(),
    }

    results = {}

    for name, model in architectures.items():
        print("-" * 70)
        print(f"Testing: {name}")
        print(f"  Parameters: {count_parameters(model):,}")

        start = time.time()
        losses = train_model(
            model, train_noisy, train_clean, epochs=args.epochs, device=args.device
        )
        train_time = time.time() - start

        metrics = evaluate_model(model, test_noisy, test_clean, device=args.device)
        metrics["train_time"] = train_time
        metrics["parameters"] = count_parameters(model)
        metrics["final_loss"] = losses[-1]
        results[name] = metrics

        print(
            f"  PSNR: {metrics['psnr_noisy']:.2f} -> {metrics['psnr_denoised']:.2f} dB "
            + f"(+{metrics['psnr_improvement']:.2f})"
        )
        print(f"  Edge preservation: {metrics['edge_ratio']:.2%}")
        print(f"  Structure (std) ratio: {metrics['std_ratio']:.2%}")
        print(f"  Time: {train_time:.1f}s")
        print()

    # Summary table
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(
        f"{'Architecture':<15} {'Params':>10} {'PSNR↑':>8} {'Edge%':>8} {'Std%':>8} {'Time':>8}"
    )
    print("-" * 70)
    for name, m in results.items():
        print(
            f"{name:<15} {m['parameters']:>10,} {m['psnr_improvement']:>+7.2f} "
            f"{m['edge_ratio']:>7.1%} {m['std_ratio']:>7.1%} {m['train_time']:>7.1f}s"
        )
    print("=" * 70)
    print()
    print("Higher Edge% and Std% = better structure preservation")
    print("Higher PSNR↑ = better noise removal")
    print()

    # Visual comparison
    print("Generating visual comparison...")
    fig, axes = plt.subplots(
        2, len(architectures) + 2, figsize=(4 * (len(architectures) + 2), 8)
    )

    test_idx = 0
    c, s = 128, 64  # Center crop (for 256x256 patches)

    # Top row: full test image
    # Bottom row: center crop

    # Noisy
    axes[0, 0].imshow(test_noisy[test_idx], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Noisy\n{results['A_Baseline']['psnr_noisy']:.1f} dB")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(
        test_noisy[test_idx][c - s : c + s, c - s : c + s], cmap="gray", vmin=0, vmax=1
    )
    axes[1, 0].axis("off")

    # Each architecture
    for i, (name, m) in enumerate(results.items()):
        axes[0, i + 1].imshow(m["output"][test_idx], cmap="gray", vmin=0, vmax=1)
        axes[0, i + 1].set_title(
            f"{name}\n+{m['psnr_improvement']:.1f}dB, E:{m['edge_ratio']:.0%}"
        )
        axes[0, i + 1].axis("off")
        axes[1, i + 1].imshow(
            m["output"][test_idx][c - s : c + s, c - s : c + s],
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, i + 1].axis("off")

    # Clean target
    axes[0, -1].imshow(test_clean[test_idx], cmap="gray", vmin=0, vmax=1)
    axes[0, -1].set_title("Clean Target\n(100% edge)")
    axes[0, -1].axis("off")
    axes[1, -1].imshow(
        test_clean[test_idx][c - s : c + s, c - s : c + s], cmap="gray", vmin=0, vmax=1
    )
    axes[1, -1].axis("off")

    plt.tight_layout()
    plt.savefig("architecture_comparison.png", dpi=150)
    print("Saved architecture_comparison.png")


if __name__ == "__main__":
    main()
