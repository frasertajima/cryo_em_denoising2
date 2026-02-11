#!/bin/bash
# Compile SSIM test

set -e

echo "Compiling SSIM loss test..."

nvfortran -cuda -O3 -c ssim_loss.cuf -o ssim_loss.o
nvfortran -cuda -O3 -o test_ssim ssim_loss.o test_ssim.cuf -lcudnn

echo "Done. Run with: ./test_ssim"
