#!/bin/bash
#================================================================
# Compile Simple CNN Training Program
#================================================================

set -e  # Exit on error

echo "========================================"
echo "  Compiling Simple CNN Trainer"
echo "========================================"
echo ""

# CUDA Fortran compiler
NVFORTRAN="nvfortran"

# Check compiler exists
if ! command -v $NVFORTRAN &> /dev/null; then
    echo "ERROR: nvfortran not found"
    echo "Please install NVIDIA HPC SDK"
    exit 1
fi

# Compiler flags
CUDA_FLAGS="-cuda -gpu=cc80 -Minfo=accel"
OPT_FLAGS="-O3 -fast"
OPENMP_FLAGS="-mp"
CUDNN_FLAGS="-lcudnn"

# Source files
LOADER="common/streaming_cryo_loader.cuf"
CONV="common/conv2d_cudnn.cuf"
CNN="common/simple_cnn.cuf"
MAIN="cryo_train_simple.cuf"

# Output executable
OUTPUT="cryo_train_simple"

echo "Source files:"
echo "  1. $LOADER"
echo "  2. $CONV"
echo "  3. $CNN"
echo "  4. $MAIN"
echo ""
echo "Compiling..."

$NVFORTRAN $CUDA_FLAGS $OPT_FLAGS $OPENMP_FLAGS $CUDNN_FLAGS \
    -o $OUTPUT \
    $LOADER \
    $CONV \
    $CNN \
    $MAIN

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Compilation SUCCESSFUL"
    echo "========================================"
    echo ""
    echo "Executable: ./$OUTPUT"
    echo ""
    echo "Run with:"
    echo "  ./$OUTPUT"
    echo ""
else
    echo ""
    echo "ERROR: Compilation failed"
    exit 1
fi
