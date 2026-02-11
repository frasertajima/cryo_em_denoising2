#!/usr/bin/env python3
"""
Evaluate trained cryo-EM denoiser on test set.
Computes PSNR, SSIM, and saves visualizations.
"""

import numpy as np
import struct
import os
from pathlib import Path

def load_patches(filepath, num_patches, patch_size=1024):
    """Load patches from binary file."""
    patches = []
    bytes_per_patch = patch_size * patch_size * 4  # float32

    with open(filepath, 'rb') as f:
        for i in range(num_patches):
            data = f.read(bytes_per_patch)
            if len(data) < bytes_per_patch:
                break
            patch = np.frombuffer(data, dtype=np.float32).reshape(patch_size, patch_size)
            patches.append(patch)

    return np.array(patches)

def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val**2 / mse)

def compute_ssim(img1, img2, window_size=11):
    """Compute Structural Similarity Index (simplified version)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Simple box filter instead of Gaussian for speed
    from scipy.ndimage import uniform_filter

    mu1 = uniform_filter(img1, window_size)
    mu2 = uniform_filter(img2, window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, window_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)

def load_fortran_weights(weights_dir):
    """Load weights saved by Fortran training."""
    weights = {}

    for layer in ['conv1', 'conv2', 'conv3']:
        w_file = os.path.join(weights_dir, f'{layer}_weights.bin')
        b_file = os.path.join(weights_dir, f'{layer}_bias.bin')

        with open(w_file, 'rb') as f:
            w_data = np.frombuffer(f.read(), dtype=np.float32)
        with open(b_file, 'rb') as f:
            b_data = np.frombuffer(f.read(), dtype=np.float32)

        weights[layer] = {'weights': w_data, 'bias': b_data}
        print(f"  {layer}: weights shape approx {len(w_data)}, bias shape {len(b_data)}")

    return weights

def simple_conv2d(x, weights, bias, out_channels, in_channels, kernel_size=3):
    """Simple 2D convolution with same padding."""
    h, w = x.shape[-2:]
    pad = kernel_size // 2

    # Reshape weights: (out_ch, in_ch, kH, kW)
    W = weights.reshape(out_channels, in_channels, kernel_size, kernel_size)

    # Pad input
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :]  # Add channel dim

    x_padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')

    output = np.zeros((out_channels, h, w), dtype=np.float32)

    for oc in range(out_channels):
        for ic in range(in_channels):
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    output[oc] += W[oc, ic, ki, kj] * x_padded[ic, ki:ki+h, kj:kj+w]
        output[oc] += bias[oc]

    return output

def relu(x):
    return np.maximum(0, x)

def denoise_patch(patch, weights):
    """Run CNN forward pass on a single patch."""
    # Conv1: 1 -> 16 channels
    x = simple_conv2d(patch, weights['conv1']['weights'], weights['conv1']['bias'],
                      out_channels=16, in_channels=1)
    x = relu(x)

    # Conv2: 16 -> 16 channels
    x = simple_conv2d(x, weights['conv2']['weights'], weights['conv2']['bias'],
                      out_channels=16, in_channels=16)
    x = relu(x)

    # Conv3: 16 -> 1 channel
    x = simple_conv2d(x, weights['conv3']['weights'], weights['conv3']['bias'],
                      out_channels=1, in_channels=16)

    return x[0]  # Remove channel dimension

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_input', default='../data/cryo_data_streaming/test_input.bin')
    parser.add_argument('--test_target', default='../data/cryo_data_streaming/test_target.bin')
    parser.add_argument('--weights', default='saved_models/mse_baseline/epoch_0001/')
    parser.add_argument('--num_patches', type=int, default=10)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    print("=" * 50)
    print("  Cryo-EM Denoiser Evaluation")
    print("=" * 50)
    print(f"  Test input:  {args.test_input}")
    print(f"  Test target: {args.test_target}")
    print(f"  Weights:     {args.weights}")
    print(f"  Num patches: {args.num_patches}")
    print()

    # Load weights
    print("Loading weights...")
    weights = load_fortran_weights(args.weights)
    print()

    # Load test data
    print(f"Loading {args.num_patches} test patches...")
    noisy_patches = load_patches(args.test_input, args.num_patches)
    clean_patches = load_patches(args.test_target, args.num_patches)
    print(f"  Loaded {len(noisy_patches)} patches, shape: {noisy_patches[0].shape}")
    print()

    # Evaluate
    print("Evaluating...")
    results = {
        'noisy_psnr': [], 'noisy_ssim': [],
        'denoised_psnr': [], 'denoised_ssim': []
    }

    denoised_patches = []

    for i in range(len(noisy_patches)):
        noisy = noisy_patches[i]
        clean = clean_patches[i]

        # Denoise
        denoised = denoise_patch(noisy, weights)
        denoised_patches.append(denoised)

        # Metrics for noisy
        results['noisy_psnr'].append(compute_psnr(noisy, clean))
        results['noisy_ssim'].append(compute_ssim(noisy, clean))

        # Metrics for denoised
        results['denoised_psnr'].append(compute_psnr(denoised, clean))
        results['denoised_ssim'].append(compute_ssim(denoised, clean))

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Patch {i+1}/{len(noisy_patches)}: "
                  f"Noisy PSNR={results['noisy_psnr'][-1]:.2f} dB, "
                  f"Denoised PSNR={results['denoised_psnr'][-1]:.2f} dB")

    print()
    print("=" * 50)
    print("  Results Summary")
    print("=" * 50)
    print(f"  Noisy PSNR:      {np.mean(results['noisy_psnr']):.2f} ± {np.std(results['noisy_psnr']):.2f} dB")
    print(f"  Denoised PSNR:   {np.mean(results['denoised_psnr']):.2f} ± {np.std(results['denoised_psnr']):.2f} dB")
    print(f"  PSNR Improvement: {np.mean(results['denoised_psnr']) - np.mean(results['noisy_psnr']):.2f} dB")
    print()
    print(f"  Noisy SSIM:      {np.mean(results['noisy_ssim']):.4f} ± {np.std(results['noisy_ssim']):.4f}")
    print(f"  Denoised SSIM:   {np.mean(results['denoised_ssim']):.4f} ± {np.std(results['denoised_ssim']):.4f}")
    print(f"  SSIM Improvement: {np.mean(results['denoised_ssim']) - np.mean(results['noisy_ssim']):.4f}")
    print("=" * 50)

    # Save sample images
    if args.save_images:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 3, figsize=(12, 12))

            for idx in range(min(3, len(noisy_patches))):
                axes[idx, 0].imshow(noisy_patches[idx], cmap='gray')
                axes[idx, 0].set_title(f'Noisy (PSNR: {results["noisy_psnr"][idx]:.1f} dB)')
                axes[idx, 0].axis('off')

                axes[idx, 1].imshow(denoised_patches[idx], cmap='gray')
                axes[idx, 1].set_title(f'Denoised (PSNR: {results["denoised_psnr"][idx]:.1f} dB)')
                axes[idx, 1].axis('off')

                axes[idx, 2].imshow(clean_patches[idx], cmap='gray')
                axes[idx, 2].set_title('Clean (Target)')
                axes[idx, 2].axis('off')

            plt.tight_layout()
            plt.savefig('denoising_results.png', dpi=150)
            print(f"\n  Saved visualization to denoising_results.png")
        except ImportError:
            print("\n  matplotlib not available, skipping visualization")

if __name__ == '__main__':
    main()
