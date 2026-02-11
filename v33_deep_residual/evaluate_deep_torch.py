#!/usr/bin/env python3
"""
Evaluate 12-Layer Deep Residual CNN using PyTorch (GPU accelerated)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Paths
DATA_DIR = Path("../data/cryo_high_noise")
MODEL_DIR = Path("saved_models/deep12/epoch_0005")
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Network architecture
NUM_LAYERS = 12
HIDDEN_CHANNELS = 32
IMG_SIZE = 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DeepResidualCNN(nn.Module):
    """12-layer residual CNN matching Fortran implementation."""

    def __init__(self):
        super().__init__()

        layers = []
        # Layer 1: 1 -> 32 with ReLU
        layers.append(nn.Conv2d(1, HIDDEN_CHANNELS, 3, padding=1))
        layers.append(nn.ReLU())

        # Layers 2-11: 32 -> 32 with ReLU
        for _ in range(10):
            layers.append(nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=1))
            layers.append(nn.ReLU())

        # Layer 12: 32 -> 1 (no activation)
        layers.append(nn.Conv2d(HIDDEN_CHANNELS, 1, 3, padding=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        noise_pred = self.layers(x)
        return x - noise_pred  # Residual: output = input - noise


def load_fortran_weights(model, model_dir):
    """Load weights from Fortran binary files into PyTorch model."""
    conv_idx = 0

    for i, layer in enumerate(model.layers):
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            w_file = model_dir / f"conv{conv_idx:02d}_weights.bin"
            b_file = model_dir / f"conv{conv_idx:02d}_bias.bin"

            w = np.fromfile(w_file, dtype=np.float32)
            b = np.fromfile(b_file, dtype=np.float32)

            # Reshape weights (Fortran: out, in, kH, kW)
            out_ch = layer.out_channels
            in_ch = layer.in_channels
            w = w.reshape(out_ch, in_ch, 3, 3)

            layer.weight.data = torch.from_numpy(w).to(device)
            layer.bias.data = torch.from_numpy(b).to(device)

    return model


def compute_psnr(img1, img2, data_range=1.0):
    """Compute PSNR."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def compute_ssim_simple(img1, img2):
    """Simple SSIM approximation."""
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()

    mu1 = np.mean(img1_np)
    mu2 = np.mean(img2_np)
    sigma1 = np.std(img1_np)
    sigma2 = np.std(img2_np)
    sigma12 = np.mean((img1_np - mu1) * (img2_np - mu2))

    C1 = 0.01**2
    C2 = 0.03**2

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
    )
    return ssim


def compute_edge_ratio(denoised, clean):
    """Compute edge preservation ratio."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    # Add batch and channel dims if needed
    if denoised.dim() == 2:
        denoised = denoised.unsqueeze(0).unsqueeze(0)
        clean = clean.unsqueeze(0).unsqueeze(0)

    dx_den = torch.nn.functional.conv2d(denoised, sobel_x, padding=1)
    dy_den = torch.nn.functional.conv2d(denoised, sobel_y, padding=1)
    edge_den = torch.sqrt(dx_den**2 + dy_den**2)

    dx_clean = torch.nn.functional.conv2d(clean, sobel_x, padding=1)
    dy_clean = torch.nn.functional.conv2d(clean, sobel_y, padding=1)
    edge_clean = torch.sqrt(dx_clean**2 + dy_clean**2)

    return (edge_den.mean() / edge_clean.mean() * 100).item()


def main():
    print("=" * 60)
    print("  12-Layer Deep Residual CNN Evaluation (PyTorch)")
    print("=" * 60)

    # Create and load model
    print(f"\nLoading model from {MODEL_DIR}...")
    model = DeepResidualCNN().to(device)
    model = load_fortran_weights(model, MODEL_DIR)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Load test data
    print(f"\nLoading test data from {DATA_DIR}...")
    test_input = np.fromfile(DATA_DIR / "test_input.bin", dtype=np.float32)
    test_target = np.fromfile(DATA_DIR / "test_target.bin", dtype=np.float32)

    pixels_per_image = IMG_SIZE * IMG_SIZE
    n_test = len(test_input) // pixels_per_image
    print(f"  Found {n_test} test images")

    test_input = test_input.reshape(n_test, IMG_SIZE, IMG_SIZE)
    test_target = test_target.reshape(n_test, IMG_SIZE, IMG_SIZE)

    # Evaluate
    print("\nRunning inference...")
    psnr_noisy_list = []
    psnr_denoised_list = []
    ssim_noisy_list = []
    ssim_denoised_list = []
    edge_ratio_list = []
    samples = []

    n_eval = min(n_test, 50)  # Evaluate up to 50 samples

    with torch.no_grad():
        for i in range(n_eval):
            noisy = torch.from_numpy(test_input[i]).to(device)
            clean = torch.from_numpy(test_target[i]).to(device)

            # Run inference
            noisy_input = noisy.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            denoised = model(noisy_input).squeeze()
            denoised = torch.clamp(denoised, 0, 1)

            # Compute metrics
            psnr_noisy = compute_psnr(noisy, clean)
            psnr_denoised = compute_psnr(denoised, clean)
            ssim_noisy = compute_ssim_simple(noisy, clean)
            ssim_denoised = compute_ssim_simple(denoised, clean)
            edge_ratio = compute_edge_ratio(denoised, clean)

            psnr_noisy_list.append(psnr_noisy)
            psnr_denoised_list.append(psnr_denoised)
            ssim_noisy_list.append(ssim_noisy)
            ssim_denoised_list.append(ssim_denoised)
            edge_ratio_list.append(edge_ratio)

            if i < 5:
                samples.append(
                    (noisy.cpu().numpy(), clean.cpu().numpy(), denoised.cpu().numpy())
                )

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{n_eval} images...")

    # Print results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Noisy Input:")
    print(f"    PSNR: {np.mean(psnr_noisy_list):.2f} dB")
    print(f"    SSIM: {np.mean(ssim_noisy_list):.4f}")

    print(f"\n  Denoised (12-Layer Residual CNN):")
    print(f"    PSNR: {np.mean(psnr_denoised_list):.2f} dB")
    print(f"    SSIM: {np.mean(ssim_denoised_list):.4f}")
    print(f"    Edge Preservation: {np.mean(edge_ratio_list):.1f}%")

    psnr_improvement = np.mean(psnr_denoised_list) - np.mean(psnr_noisy_list)
    ssim_improvement = np.mean(ssim_denoised_list) - np.mean(ssim_noisy_list)
    print(f"\n  Improvement:")
    print(f"    PSNR: +{psnr_improvement:.2f} dB")
    print(f"    SSIM: +{ssim_improvement:.4f}")

    print("\n" + "-" * 60)
    print("  Comparison to Python 12-Layer Results (from A/B test):")
    print("-" * 60)
    print("  Python 12-layer: PSNR +3.9 dB, Edge ~59%")
    print(
        f"  Fortran 12-layer: PSNR +{psnr_improvement:.1f} dB, Edge {np.mean(edge_ratio_list):.0f}%"
    )

    # Create visualization
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    for i, (noisy, clean, denoised) in enumerate(samples):
        c = IMG_SIZE // 2
        crop = 256
        s = slice(c - crop // 2, c + crop // 2)

        axes[0, i].imshow(noisy[s, s], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Noisy\nPSNR: {psnr_noisy_list[i]:.1f} dB")
        axes[0, i].axis("off")

        axes[1, i].imshow(denoised[s, s], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"Denoised\nPSNR: {psnr_denoised_list[i]:.1f} dB")
        axes[1, i].axis("off")

        axes[2, i].imshow(clean[s, s], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title("Clean Target")
        axes[2, i].axis("off")

    plt.suptitle(
        f"12-Layer Deep Residual CNN (CUDA Fortran)\n"
        f"PSNR: {np.mean(psnr_noisy_list):.1f} → {np.mean(psnr_denoised_list):.1f} dB "
        f"(+{psnr_improvement:.1f} dB)",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "denoising_results_deep12.png", dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'denoising_results_deep12.png'}")

    # Save metrics
    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write("12-Layer Deep Residual CNN Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_DIR}\n")
        f.write(f"Parameters: {total_params:,}\n\n")
        f.write("Metrics:\n")
        f.write(f"  Noisy PSNR:     {np.mean(psnr_noisy_list):.2f} dB\n")
        f.write(f"  Denoised PSNR:  {np.mean(psnr_denoised_list):.2f} dB\n")
        f.write(f"  PSNR Gain:      +{psnr_improvement:.2f} dB\n\n")
        f.write(f"  Noisy SSIM:     {np.mean(ssim_noisy_list):.4f}\n")
        f.write(f"  Denoised SSIM:  {np.mean(ssim_denoised_list):.4f}\n")
        f.write(f"  SSIM Gain:      +{ssim_improvement:.4f}\n\n")
        f.write(f"  Edge Preservation: {np.mean(edge_ratio_list):.1f}%\n")
    print(f"  Saved: {OUTPUT_DIR / 'metrics.txt'}")

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
