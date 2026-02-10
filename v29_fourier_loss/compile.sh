#!/bin/bash
#================================================================
# Compile script for Cryo-EM CNN Training - v29 (Fourier Loss)
#================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Compiling Cryo-EM CNN Training (Fourier)"
echo "=============================================="

# Compiler settings
FC=nvfortran
CUDA_ARCH=cc80
CUDNN_LIB=/usr/local/cuda/lib64
CUDNN_INC=/usr/local/cuda/include

# Compiler flags
FFLAGS="-cuda -O3"
LDFLAGS="-lcudnn -lcublas -lcudart -lcufft"

# Module path
MODULE_PATH="../common"

echo ""
echo "Step 1: Compiling conv2d_cudnn module..."
${FC} ${FFLAGS} -c ${MODULE_PATH}/conv2d_cudnn.cuf -o conv2d_cudnn.o
echo "  conv2d_cudnn.o created"

echo ""
echo "Step 2: Compiling streaming_cryo_loader module..."
${FC} ${FFLAGS} -c ${MODULE_PATH}/streaming_cryo_loader.cuf -o streaming_cryo_loader.o
echo "  streaming_cryo_loader.o created"

echo ""
echo "Step 3: Compiling fourier_loss module..."
${FC} ${FFLAGS} -c fourier_loss.cuf -o fourier_loss.o
echo "  fourier_loss.o created"

echo ""
echo "Step 4: Compiling and linking main training program..."
${FC} ${FFLAGS} -o cryo_train_fourier \
    conv2d_cudnn.o \
    streaming_cryo_loader.o \
    fourier_loss.o \
    cryo_train_fourier.cuf \
    ${LDFLAGS}

echo "  cryo_train_fourier executable created"

echo ""
echo "=============================================="
echo "  Compilation Complete!"
echo "=============================================="
echo ""
echo "Run training with:"
echo "  ./cryo_train_fourier --stream                      # MSE only"
echo "  ./cryo_train_fourier --stream --fourier_weight 0.1 # 10% Fourier loss"
echo "  ./cryo_train_fourier --stream --epochs 5 --save    # Save checkpoints"
echo ""
