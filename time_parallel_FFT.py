# -*- coding: utf-8 -*-
"""
This is where I time the parallel FFT functions. The data is stored to pandas 
dataframes for plotting later.
"""

from FFT import FFT_itt_c, FFT_par_c, FFT_MPI, FFT_MPI_CBM

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI 
import pandas as pd

def time_FFT(func, x, repeat=5):
    '''
    This function times different FFT or DFT algorithms by repeatedly calling  
    said function on a test array, timing each call and returning the mean and 
    std time of each call.

    Parameters
    ----------
    func : function 
        The FFT or DFT function you want to time.
    x : np.darray
        Test array.
    repeat : int, optional
        The number of times you want the function to be called. The default is 5.

    Returns
    -------
    mean_time : float
        The mean time func takes to run.
    std_time : float
        The std time func takes to run.

    '''
    times = []
    for i in range(repeat): 
        initial_time = MPI.Wtime()
        func(x)
        Time = MPI.Wtime() - initial_time
        times += [Time]
                
    mean_time = np.mean(times)
    std_time = np.std(times)
        
    return mean_time, std_time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numtasks = comm.Get_size()

#The CBM parallel function only works if the number of cores is an integer power of 2 

if np.log2(numtasks) % 1 > 0:
    perform_CBM = False
else:
    perform_CBM = True

#Create pandas dataframe for data to be stored 
if rank == 0:
    columns = ['array_power', 'FFT_par_c_timing', 'FFT_par_c_error',
               'FFT_par_c_eff', 'FFT_par_c_eff_error',
               'FFT_MPI_timing', 'FFT_MPI_error',
               'FFT_MPI_eff', 'FFT_MPI_eff_error',
               'FFT_MPI_CBM_timing', 'FFT_MPI_CBM_error',
               'FFT_MPI_CBM_eff', 'FFT_MPI_CBM_eff_error']
    
    df = pd.DataFrame(columns=columns)

#Loop through different size arrays
for i in range(4,15):
    row = []
    row.append(i)
    
    print('Calculating times for 2^{} input array'.format(i))
    
    N = int(2 ** i)
    x = np.empty(N, dtype=complex)
    
    if rank == 0:
        x = np.asarray(np.random.random(N), dtype=np.complex128)
        FFT_itt_mean, FFT_itt_std = time_FFT(FFT_itt_c, x, 5)
        FFT_par_c_mean, FFT_par_c_std = time_FFT(lambda a: FFT_par_c(a, numtasks), x, 5)
        
        FFT_par_c_eff = FFT_itt_mean / FFT_par_c_mean
        FFT_par_c_eff_error = np.sqrt((1/FFT_par_c_mean) ** 2 * FFT_itt_std ** 2 + (FFT_itt_mean/(FFT_par_c_mean**2)) ** 2 * FFT_par_c_std ** 2)
        
        row.append(FFT_par_c_mean)
        row.append(FFT_par_c_std)
        row.append(FFT_par_c_eff)
        row.append(FFT_par_c_eff_error)

    
    comm.Bcast([x, MPI.COMPLEX], root=0)
    
    FFT_MPI_mean, FFT_MPI_std = time_FFT(FFT_MPI, x, 5)

    if perform_CBM == True:
    	FFT_CBM_mean, FFT_CBM_std = time_FFT(FFT_MPI_CBM, x, 5)
    else:
        FFT_CBM_mean = np.nan
        FFT_CBM_std = np.nan

    if rank == 0:
        
        FFT_MPI_eff = FFT_itt_mean / FFT_MPI_mean
        FFT_MPI_eff_error = np.sqrt((1/FFT_MPI_mean) ** 2 * FFT_itt_std ** 2 + (FFT_itt_mean/(FFT_MPI_mean**2)) ** 2 * FFT_MPI_std ** 2)
        
        row.append(FFT_MPI_mean)
        row.append(FFT_MPI_std)
        row.append(FFT_MPI_eff)
        row.append(FFT_MPI_eff_error)
        
        row.append(FFT_CBM_mean)
        row.append(FFT_CBM_std)
        
        if perform_CBM == True:
            FFT_CBM_eff = FFT_itt_mean / FFT_CBM_mean
            FFT_CBM_eff_error = np.sqrt((1/FFT_CBM_mean) ** 2 * FFT_itt_std ** 2 + (FFT_itt_mean/(FFT_CBM_mean**2)) ** 2 * FFT_CBM_std ** 2)
        else:
            FFT_CBM_eff = np.nan
            FFT_CBM_eff_error = np.nan
            
        row.append(FFT_CBM_eff)
        row.append(FFT_CBM_eff_error)
            
        df_length = len(df)
        df.loc[df_length] = row
        
if rank == 0:
    df.to_pickle('parallel_fft_timing_{}.pkl'.format(numtasks))  
