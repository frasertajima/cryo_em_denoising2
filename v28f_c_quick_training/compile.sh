#!/bin/bash

echo "Compiling quick training test..."
echo ""

nvfortran -O2 -cuda -gpu=cuda13.0,cc89,ptxinfo -cudaforlibs -lcudnn \
    common/conv2d_cudnn.cuf \
    quick_train.cuf \
    -o quick_train

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Compilation successful: quick_train"
    echo ""
    echo "Run with: ./quick_train"
    echo ""
else
    echo ""
    echo "✗ Compilation failed"
    exit 1
fi
