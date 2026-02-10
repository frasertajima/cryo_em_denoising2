#!/bin/bash
#================================================================
# Compile Cryo-EM Data Loader Test
#================================================================

set -e  # Exit on error

echo "========================================"
echo "  Compiling Cryo-EM Data Loader Test"
echo "========================================"
echo ""

# CUDA Fortran compiler
NVFORTRAN="nvfortran"

# Check compiler
if ! command -v $NVFORTRAN &> /dev/null; then
    echo "ERROR: nvfortran not found"
    echo "Please install NVIDIA HPC SDK"
    exit 1
fi

# Flags
CUDA_FLAGS="-cuda -gpu=cc80 -Minfo=accel"
OPT_FLAGS="-O3 -fast"
OPENMP_FLAGS="-mp"
DEBUG_FLAGS="-g -traceback"

# Source files
LOADER="common/streaming_cryo_loader.cuf"
TEST="tests/test_cryo_loader.cuf"

# Output
OUTPUT="test_cryo_loader"

echo "Compiling..."
echo "  Loader: $LOADER"
echo "  Test:   $TEST"
echo ""

$NVFORTRAN $CUDA_FLAGS $OPT_FLAGS $OPENMP_FLAGS \
    -o $OUTPUT \
    $LOADER \
    $TEST

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Compilation SUCCESSFUL"
    echo "========================================"
    echo ""
    echo "Run with: ./test_cryo_loader"
    echo ""
else
    echo ""
    echo "ERROR: Compilation failed"
    exit 1
fi
