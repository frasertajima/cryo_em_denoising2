#!/bin/bash
# Compile minimal cuFFT test

nvfortran -cuda -cudalib=cufft test_cufft_minimal.cuf -o test_cufft

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Run with: ./test_cufft"
else
    echo "Compilation failed!"
    exit 1
fi
