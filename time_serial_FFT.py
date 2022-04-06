# -*- coding: utf-8 -*-
"""
This is where I time the serial functions. The data is stored to a numpy array
for plotting later.
"""
from FFT import FFT_itt_c
from SerialFFT import DFT, FFT, FFT_vec, time_FFT
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time
import pandas as pd

columns = ['array_power', 'DFT_timing', 'DFT_error',
           'FFT_timing', 'FFT_error',
           'FFTv_timing', 'FFTv_error',
            'FFT_itt_timing', 'FFT_itt_error',
           'FFT_np_timing', 'FFT_np_error']

df = pd.DataFrame(columns=columns)

initial_time = time.time()
for i in range(1,23):
    row = []
    row.append(i)
    print(i)
    x = np.random.random(int(2 ** i))
    x = np.asarray(x, dtype=np.complex128)
    
    if i <= 10:
        DFT_mean, DFT_std = time_FFT(DFT, x, 5)
        row.append(DFT_mean)
        row.append(DFT_std)
    else:
        row.append(np.nan)
        row.append(np.nan)
    
    FFT_mean, FFT_std = time_FFT(FFT, x, 5)
    row.append(FFT_mean)
    row.append(FFT_std)
    
    FFTv_mean, FFTv_std = time_FFT(FFT_vec, x, 5)
    row.append(FFTv_mean)
    row.append(FFTv_std)
    
    FFT_itt_mean, FFT_itt_std = time_FFT(FFT_itt_c, x, 5)
    row.append(FFT_itt_mean)
    row.append(FFT_itt_std)
    
    np_FFT_mean, np_FFT_std = time_FFT(np.fft.fft, x, 5)
    row.append(np_FFT_mean)
    row.append(np_FFT_std)
    
    df_length = len(df)
    df.loc[df_length] = row
    
time_taken = time.time() - initial_time
print('It took {}s to perform this operation'.format(time_taken))

df.to_pickle('serial_fft_timing.pkl')