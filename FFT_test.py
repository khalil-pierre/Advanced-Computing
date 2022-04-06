# -*- coding: utf-8 -*-
"""
This script is where I test the FFT functions that do not involve MPI. The 
method is simple I calculate the FFT for various sized arrays and compare the
answer to numpy using np.allclose
"""

from FFT import FFT_itt_c
from SerialFFT import DFT, FFT, FFT_vec, testFFT

testFFT(FFT_itt_c, 10)
testFFT(DFT, 10)
testFFT(FFT, 10)
testFFT(FFT_vec, 10)

    
