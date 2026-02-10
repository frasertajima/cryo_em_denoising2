#!/bin/bash
#================================================================
# Compile script for Cryo-EM CNN Training - v28f
#================================================================

set -e  # Exit on error

echo "=============================================="
echo "  Compiling Cryo-EM CNN Training"
echo "=============================================="

# Compiler settings
FC=nvfortran
CUDA_ARCH=cc80
CUDNN_LIB=/usr/local/cuda/lib64
CUDNN_INC=/usr/local/cuda/include

# Compiler flags
FFLAGS="-cuda -O3"
LDFLAGS="-lcudnn -lcublas -lcudart"

# Module path
MODULE_PATH="common"

echo ""
echo "Step 1: Compiling conv2d_cudnn module..."
${FC} ${FFLAGS} -c ${MODULE_PATH}/conv2d_cudnn.cuf -o conv2d_cudnn.o
echo "  ✓ conv2d_cudnn.o created"

echo ""
echo "Step 2: Compiling streaming_cryo_loader module..."
${FC} ${FFLAGS} -c ${MODULE_PATH}/streaming_cryo_loader.cuf -o streaming_cryo_loader.o
echo "  ✓ streaming_cryo_loader.o created"

echo ""
echo "Step 3: Compiling and linking main training program..."
${FC} ${FFLAGS} -o cryo_train \
    conv2d_cudnn.o \
    streaming_cryo_loader.o \
    cryo_train.cuf \
    ${LDFLAGS}

echo "  ✓ cryo_train executable created"

echo ""
echo "=============================================="
echo "  Compilation Complete!"
echo "=============================================="
echo ""
echo "Run training with:"
echo "  ./cryo_train --stream"
echo "  ./cryo_train --stream --epochs 5 --save"
echo ""
