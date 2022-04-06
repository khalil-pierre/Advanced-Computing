# -*- coding: utf-8 -*-
"""
In this script I want to investegate how the number of threads effects the 
timing for each stage. I belive as the subproblem size gets smaller and the ratio
of threads to subproblem size gets larger the time taken to perform a single 
stage will increase due to bottlenecking.
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound
from mpi4py import MPI

@boundscheck(False)
@wraparound(False)
def stage_timing(double complex [::1] x, int threads=1, str method='1'):
    '''
    This function is used to see how the timing of each stage is effected by 
    the number of threads.

    Parameters
    ----------
    double [ : :1] x
        Input array.
    
    threads : int
        The number of threads avaliable to the FFT_par_c algorithm.
    
    method : str
        Method can either be 1 or 2 depending on which loop you want prange.
        If 1 the stage loop will be parallised if 2 the subproblem loop will be
        parallised.

    Returns
    -------
    stage_number : int [::1] 
        The stage number that is being timed
        
    stage_timing : double [::1] 
        An array of the timing of each stage

    '''
    cdef int N = x.shape[0]
    cdef int layers = int(np.log2(N))
    cdef double complex temp, temp_twiddle
    cdef int i,j,k
    cdef int subproblemsize,halfproblemsize
    cdef int number_of_problems
    cdef int index_start,index_end
    cdef int threads_per_stage
    #Create look up table for twiddle factor 
    cdef double complex [::1] twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2)/N)
    #Out array, copy so dosnt overide input
    cdef double complex [::1] X = x.copy()
    
    cdef int [::1] stage_number = np.arange(layers, dtype=np.int32)
    cdef double [::1] stage_timing = np.zeros(layers, dtype=np.double)
    cdef double start_time
    
    #Works out the number and size of subproblems
    for i in range(layers):
        start_time = MPI.Wtime()
        number_of_problems = 2**(i)
        subproblemsize = int(N / 2 ** (i))
        halfproblemsize = int(subproblemsize / 2)
        
        if method == '1':
            
            if number_of_problems < threads:
                threads_per_stage = number_of_problems
            else:
                threads_per_stage = threads
    
            #Works out the start and end index of each subproblem
            for j in prange(number_of_problems, nogil=True, num_threads=threads_per_stage):
                index_start = j * subproblemsize
                index_end = index_start + halfproblemsize
                
                #Performs Gentleman-Sande butterfly computation 
                for k in range(index_start, index_end):
                    temp = X[k]
                    temp_twiddle = twiddle_factor[number_of_problems * (k - index_start)]
                    X[k] = temp + X[k + halfproblemsize]
                    X[k + halfproblemsize] = temp_twiddle * (temp - X[k + halfproblemsize])   
                    
        elif method == '2':
            for j in range(number_of_problems):
                index_start = j * subproblemsize
                index_end = index_start + halfproblemsize
                
                #Performs Gentleman-Sande butterfly computation 
                for k in prange(index_start, index_end, nogil=True, num_threads=threads):
                    temp = X[k]
                    temp_twiddle = twiddle_factor[number_of_problems * (k - index_start)]
                    X[k] = temp + X[k + halfproblemsize]
                    X[k + halfproblemsize] = temp_twiddle * (temp - X[k + halfproblemsize])  
        
        else:
            raise ValueError('Method is either 1 or 2.')
        
        
        stage_timing[i] = MPI.Wtime() - start_time
    
    return stage_number, stage_timing
            




