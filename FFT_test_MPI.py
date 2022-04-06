# -*- coding: utf-8 -*-
"""
This script is where I test the FFT functions involving the MPI. The 
"""

import sys
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from FFT import *
from mpi4py import MPI


comm = MPI.COMM_WORLD
taskid = comm.Get_rank()

    
for i in range(1,22):
    x = np.empty(2 ** i, dtype=complex)
    
    if taskid == 0:
        x = np.asarray(np.random.random(2**i), dtype=complex)
    
    comm.Bcast([x, MPI.COMPLEX], root=0)
    
    f_true = np.fft.fft(x)
    f_test = FFT_MPI_CBM(x)
    
    if taskid == 0:
        print('{}:{}'.format(i, np.allclose(f_true, f_test)))















