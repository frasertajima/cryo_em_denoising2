#!/bin/bash

echo "========================================"
echo "  Compiling cuDNN Backward Pass Test"
echo "========================================"
echo ""

echo "Compiling..."
echo "  1. common/conv2d_cudnn.cuf"
echo "  2. test_cudnn_backward.cuf"
echo ""

nvfortran -O2 -cuda -gpu=cuda13.0,cc89,ptxinfo -cudaforlibs -lcudnn \
    common/conv2d_cudnn.cuf \
    test_cudnn_backward.cuf \
    -o test_cudnn_backward

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  SUCCESS!"
    echo "========================================"
    echo ""
    echo "Run with: ./test_cudnn_backward"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  COMPILATION FAILED"
    echo "========================================"
    echo ""
    exit 1
fi
