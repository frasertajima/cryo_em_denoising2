#!/bin/bash
nvfortran -cuda -gpu=cc89 -o test_loss_bug test_loss_bug.cuf
