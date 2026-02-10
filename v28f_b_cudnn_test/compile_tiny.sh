#!/bin/bash

nvfortran -O2 -cuda -gpu=cuda13.0,cc89,ptxinfo -cudaforlibs -lcudnn \
    common/conv2d_cudnn.cuf \
    test_tiny_cudnn.cuf \
    -o test_tiny_cudnn

echo "Compilation complete: test_tiny_cudnn"
