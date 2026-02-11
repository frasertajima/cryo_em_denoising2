#!/usr/bin/env python3
"""
Evaluate 12-Layer Deep Residual CNN for Cryo-EM Denoising

Loads trained Fortran model weights, runs inference, and computes metrics.
Compares to Python baseline results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

# Paths
DATA_DIR = Path("../data/cryo_high_noise")
MODEL_DIR = Path("saved_models/deep12/epoch_0005")
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Network architecture
NUM_LAYERS = 12
HIDDEN_CHANNELS = 32
IMG_SIZE = 1024


def load_fortran_weights(model_dir):
    """Load weights from Fortran binary files."""
    weights = []
    biases = []

    for i in range(1, NUM_LAYERS + 1):
        w_file = model_dir / f"conv{i:02d}_weights.bin"
        b_file = model_dir / f"conv{i:02d}_bias.bin"

        # Load weights
        w = np.fromfile(w_file, dtype=np.float32)
        b = np.fromfile(b_file, dtype=np.float32)

        # Reshape weights based on layer
        if i == 1:
            # Layer 1: 1 -> 32, shape (out, in, kH, kW) = (32, 1, 3, 3)
            w = w.reshape(32, 1, 3, 3)
        elif i == NUM_LAYERS:
            # Layer 12: 32 -> 1, shape (1, 32, 3, 3)
            w = w.reshape(1, 32, 3, 3)
        else:
            # Layers 2-11: 32 -> 32, shape (32, 32, 3, 3)
            w = w.reshape(32, 32, 3, 3)

        weights.append(w)
        biases.append(b)

    return weights, biases


def conv2d(x, weight, bias, padding=1):
    """Apply 2D convolution with padding."""
    out_channels, in_channels, kH, kW = weight.shape

    # Pad input
    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        x_padded = x

    H, W = x.shape[1], x.shape[2]
    out = np.zeros((out_channels, H, W), dtype=np.float32)

    for oc in range(out_channels):
        for ic in range(in_channels):
            out[oc] += ndimage.convolve(
                x_padded[ic], weight[oc, ic, ::-1, ::-1], mode="constant"
            )[padding : padding + H, padding : padding + W]
        out[oc] += bias[oc]

    return out


def relu(x):
    """ReLU activation."""
    return np.maximum(0, x)


def forward_pass(x, weights, biases):
    """Run forward pass through 12-layer residual network."""
    # x shape: (1, H, W) - single channel input
    input_img = x.copy()

    # Layer 1: 1 -> 32 with ReLU
    h = conv2d(x, weights[0], biases[0])
    h = relu(h)

    # Layers 2-11: 32 -> 32 with ReLU
    for i in range(1, NUM_LAYERS - 1):
        h = conv2d(h, weights[i], biases[i])
        h = relu(h)

    # Layer 12: 32 -> 1 (no activation) - predicts noise
    noise_pred = conv2d(h, weights[NUM_LAYERS - 1], biases[NUM_LAYERS - 1])

    # Residual: output = input - noise
    output = input_img - noise_pred

    return output[0], noise_pred[0]  # Remove channel dimension


def compute_psnr(img1, img2, data_range=1.0):
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(data_range**2 / mse)


def compute_ssim(img1, img2, data_range=1.0):
    """Compute SSIM between two images."""
    return ssim(img1, img2, data_range=data_range)


def compute_edge_ratio(denoised, clean):
    """Compute edge preservation ratio using Sobel gradients."""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = sobel_x.T

    # Compute gradients
    dx_den = ndimage.convolve(denoised, sobel_x)
    dy_den = ndimage.convolve(denoised, sobel_y)
    edge_den = np.sqrt(dx_den**2 + dy_den**2)

    dx_clean = ndimage.convolve(clean, sobel_x)
    dy_clean = ndimage.convolve(clean, sobel_y)
    edge_clean = np.sqrt(dx_clean**2 + dy_clean**2)

    return np.mean(edge_den) / np.mean(edge_clean) * 100


def main():
    print("=" * 60)
    print("  12-Layer Deep Residual CNN Evaluation")
    print("=" * 60)

    # Load model weights
    print(f"\nLoading model from {MODEL_DIR}...")
    weights, biases = load_fortran_weights(MODEL_DIR)
    total_params = sum(w.size + b.size for w, b in zip(weights, biases))
    print(f"  Total parameters: {total_params:,}")

    # Load test data
    print(f"\nLoading test data from {DATA_DIR}...")
    test_input = np.fromfile(DATA_DIR / "test_input.bin", dtype=np.float32)
    test_target = np.fromfile(DATA_DIR / "test_target.bin", dtype=np.float32)

    # Determine number of test samples
    pixels_per_image = IMG_SIZE * IMG_SIZE
    n_test = len(test_input) // pixels_per_image
    print(f"  Found {n_test} test images")

    test_input = test_input.reshape(n_test, IMG_SIZE, IMG_SIZE)
    test_target = test_target.reshape(n_test, IMG_SIZE, IMG_SIZE)

    # Evaluate on test set
    print("\nRunning inference...")
    psnr_noisy_list = []
    psnr_denoised_list = []
    ssim_noisy_list = []
    ssim_denoised_list = []
    edge_ratio_list = []

    # Store some samples for visualization
    samples = []

    for i in range(min(n_test, 10)):  # Evaluate up to 10 samples (CPU is slow)
        noisy = test_input[i]
        clean = test_target[i]

        # Run inference
        denoised, noise_pred = forward_pass(noisy[np.newaxis, :, :], weights, biases)

        # Clip to valid range
        denoised = np.clip(denoised, 0, 1)

        # Compute metrics
        psnr_noisy = compute_psnr(noisy, clean)
        psnr_denoised = compute_psnr(denoised, clean)
        ssim_noisy = compute_ssim(noisy, clean)
        ssim_denoised = compute_ssim(denoised, clean)
        edge_ratio = compute_edge_ratio(denoised, clean)

        psnr_noisy_list.append(psnr_noisy)
        psnr_denoised_list.append(psnr_denoised)
        ssim_noisy_list.append(ssim_noisy)
        ssim_denoised_list.append(ssim_denoised)
        edge_ratio_list.append(edge_ratio)

        # Save first few samples
        if i < 5:
            samples.append((noisy, clean, denoised, noise_pred))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{min(n_test, 50)} images...")

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

    # Compare to Python 12-layer results (from previous A/B testing)
    print("\n" + "-" * 60)
    print("  Comparison to Python 12-Layer Results:")
    print("-" * 60)
    print("  Python 12-layer (from A/B test):")
    print("    PSNR improvement: ~+3.9 dB")
    print("    Edge preservation: ~59%")
    print(f"\n  CUDA Fortran 12-layer:")
    print(f"    PSNR improvement: +{psnr_improvement:.2f} dB")
    print(f"    Edge preservation: {np.mean(edge_ratio_list):.1f}%")

    # Create visualization
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    for i, (noisy, clean, denoised, noise_pred) in enumerate(samples):
        # Show center crop for detail
        c = IMG_SIZE // 2
        crop_size = 256
        s = slice(c - crop_size // 2, c + crop_size // 2)

        axes[0, i].imshow(noisy[s, s], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Noisy\nPSNR: {psnr_noisy_list[i]:.1f} dB")
        axes[0, i].axis("off")

        axes[1, i].imshow(clean[s, s], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("Clean Target")
        axes[1, i].axis("off")

        axes[2, i].imshow(denoised[s, s], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title(f"Denoised\nPSNR: {psnr_denoised_list[i]:.1f} dB")
        axes[2, i].axis("off")

        # Show predicted noise (scaled for visibility)
        noise_display = noise_pred[s, s]
        vmax = np.percentile(np.abs(noise_display), 99)
        axes[3, i].imshow(noise_display, cmap="RdBu", vmin=-vmax, vmax=vmax)
        axes[3, i].set_title("Predicted Noise")
        axes[3, i].axis("off")

    axes[0, 0].set_ylabel("Noisy Input", fontsize=12)
    axes[1, 0].set_ylabel("Clean Target", fontsize=12)
    axes[2, 0].set_ylabel("Denoised", fontsize=12)
    axes[3, 0].set_ylabel("Noise Pred", fontsize=12)

    plt.suptitle(
        f"12-Layer Deep Residual CNN Denoising Results\n"
        f"PSNR: {np.mean(psnr_noisy_list):.1f} → {np.mean(psnr_denoised_list):.1f} dB "
        f"(+{psnr_improvement:.1f} dB) | "
        f"SSIM: {np.mean(ssim_noisy_list):.3f} → {np.mean(ssim_denoised_list):.3f}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "denoising_results_deep12.png", dpi=150, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'denoising_results_deep12.png'}")

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use first sample, full image
    noisy, clean, denoised, _ = samples[0]

    axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Noisy Input\nPSNR: {psnr_noisy_list[0]:.2f} dB, SSIM: {ssim_noisy_list[0]:.4f}"
    )
    axes[0].axis("off")

    axes[1].imshow(denoised, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Denoised (12-Layer ResNet)\nPSNR: {psnr_denoised_list[0]:.2f} dB, SSIM: {ssim_denoised_list[0]:.4f}"
    )
    axes[1].axis("off")

    axes[2].imshow(clean, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Clean Target")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_full.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'comparison_full.png'}")

    # Save metrics to file
    with open(OUTPUT_DIR / "metrics.txt", "w") as f:
        f.write("12-Layer Deep Residual CNN Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {MODEL_DIR}\n")
        f.write(f"Parameters: {total_params:,}\n\n")
        f.write("Metrics (averaged over test set):\n")
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
