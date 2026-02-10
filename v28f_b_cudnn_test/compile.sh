#!/bin/bash
#================================================================
# Compile cuDNN Forward Pass Test
#================================================================

set -e

echo "========================================"
echo "  Compiling cuDNN Forward Pass Test"
echo "========================================"
echo ""

NVFORTRAN="nvfortran"

if ! command -v $NVFORTRAN &> /dev/null; then
    echo "ERROR: nvfortran not found"
    exit 1
fi

# Flags
CUDA_FLAGS="-cuda -gpu=cc80 -Minfo=accel"
OPT_FLAGS="-O2"
CUDNN_FLAGS="-lcudnn"

# Files
CONV="common/conv2d_cudnn.cuf"
TEST="test_cudnn_forward.cuf"
OUTPUT="test_cudnn_forward"

echo "Compiling..."
echo "  1. $CONV"
echo "  2. $TEST"
echo ""

$NVFORTRAN $CUDA_FLAGS $OPT_FLAGS $CUDNN_FLAGS \
    -o $OUTPUT \
    $CONV \
    $TEST

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  SUCCESS!"
    echo "========================================"
    echo ""
    echo "Run with: ./$OUTPUT"
    echo ""
else
    echo ""
    echo "ERROR: Compilation failed"
    exit 1
fi
