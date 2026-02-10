#!/bin/bash
#================================================================
# Compile Cryo-EM Training - v28f_d
#================================================================

set -e

echo "Compiling Cryo-EM denoising training..."

nvfortran -O2 -cuda -gpu=cuda13.0,cc89 -cudaforlibs -lcudnn \
    common/streaming_cryo_loader.cuf \
    common/conv2d_cudnn.cuf \
    cryo_train.cuf \
    -o cryo_train

echo "✓ Compilation successful!"
echo ""
echo "Run with: ./cryo_train"
echo ""
