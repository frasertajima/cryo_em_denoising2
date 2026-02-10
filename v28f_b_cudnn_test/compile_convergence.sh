#!/bin/bash

nvfortran -O2 -cuda -gpu=cuda13.0,cc89,ptxinfo -cudaforlibs -lcudnn \
    common/conv2d_cudnn.cuf \
    test_convergence.cuf \
    -o test_convergence

echo "Compilation complete: test_convergence"
