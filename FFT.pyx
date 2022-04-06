# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 03:25:52 2021

@author: user
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from cython.parallel import parallel,prange
from cython import boundscheck, wraparound
cimport openmp
from mpi4py import MPI

def bit_reversed_array(double complex [::1] f, int N, int base):
    '''
    This function performs an inplace bit reversed shuffle on an input array.
    ----------
    x np.darray
        The input array.
    N : int
        Size of input array.
    base : int
        The number of leading order zeros in bit 

    '''
    def rev(int number, int length):
        '''
        This function calculates the reversed bit for a integer given some 
        bit length.

        Parameters
        ----------
        number : int
            Integer whose bit you want to revers.
        length : int
            bit length.

        Returns
        -------
        result : TYPE
            Bit reversed integer.

        '''
        cdef int result = 0
        cdef int l
        
        for l in range(length):
            result <<= 1
            result |= number & 1
            number >>= 1
        
        return result
    
    cdef int m
    cdef int n
    
    #Itterates through list swapping 
    for m in range(N):
        n = rev(m,base)
        if m < n:
            f[m],f[n] = f[n],f[m]

    return f
    
def FFT_itt_c(double complex [::1] x):
    '''
    This is a cythonised iterative FFT algorithm. It works by using a  
    Gentleman-Sande butterfly algorithm, more info can be found in report.

    Parameters
    ----------
    double [ : :1] x
        Input array.

    Returns
    -------
    x : TYPE
        Output fourier transform.

    '''
    cdef int N = x.shape[0]
    cdef int layers = int(np.log2(N))
    cdef double complex temp
    cdef int i,j,k
    cdef int subproblemsize,halfproblemsize
    cdef int number_of_problems
    cdef int index_start,index_end
    #Create look up table for twiddle factor 
    cdef int twiddle_index
    cdef double complex [::1] twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2)/N)
    #Out array, copy so dosnt overide input
    cdef double complex [::1] X = x.copy()
    
    #Works out the number and size of subproblems
    for i in range(layers):
        number_of_problems = 2**(i)
        subproblemsize = int(N / 2 ** (i))
        halfproblemsize = int(subproblemsize / 2)
        
        #Works out the start and end point of each subproblem
        for j in range(number_of_problems):
            index_start = int(j * subproblemsize)
            index_end = int(index_start + halfproblemsize)
            twiddle_index = 0
            
            #Performs Gentleman-Sande butterfly computation 
            for k in range(index_start,index_end):
                temp = X[k]
                X[k] = temp + X[k + halfproblemsize]
                X[k + halfproblemsize] = twiddle_factor[twiddle_index] * (temp - X[k + halfproblemsize])    
                twiddle_index = twiddle_index + number_of_problems
    
    bit_reversed_array(X,N,layers) #Output is scrambled in DIF FFT see report
    
    return X

@boundscheck(False)
@wraparound(False)
def FFT_par_c(double complex [::1] x, int threads=1):
    '''
    This is a cythonised iterative FFT algorithm. It works by using a  
    Gentleman-Sande butterfly algorithm, more info can be found in report.

    Parameters
    ----------
    double [ : :1] x
        Input array.
    
    threads : int
        The number of threads avaliable to the FFT_par_c algorithm.

    Returns
    -------
    x : TYPE
        Output fourier transform.

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
    
    #Works out the number and size of subproblems
    for i in range(layers):
        number_of_problems = 2**(i)
        subproblemsize = int(N / 2 ** (i))
        halfproblemsize = int(subproblemsize / 2)
        
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
                
    bit_reversed_array(X, N, layers) #Output is scrambled in DIF FFT see report
    
    return X

@boundscheck(False)
@wraparound(False)
def FFT_MPI(double complex [::1] x):
    '''
    This is a cythonised iterative FFT algorithm. It works by using a  
    Gentleman-Sande butterfly algorithm, more info can be found in report.

    Parameters
    ----------
    double complex [ : :1] x
        Input array.
    int threads : TYPE, optional
        The number of threads avaliable to the FFT_par_c algorithm. The default
        is 1.

    Returns
    -------
    x : TYPE
        Output fourier transform.

    '''
    comm = MPI.COMM_WORLD
    #Make sure all the cores are working on the same problem
    comm.Bcast([x, MPI.COMPLEX], root=0) 
    
    cdef int taskid = comm.Get_rank()
    cdef int numtasks = comm.Get_size()
    cdef int N = x.shape[0]
    cdef int layers = int(np.log2(N))
    cdef int i,j,k,l, source
    cdef int number_of_problems
    cdef int subproblemsize, halfproblemsize
    cdef int index_start, index_end, twiddle_index
    cdef double complex temp_twiddle
    cdef double complex [::1] twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2)/N)
    cdef int blocksize = N / numtasks
    cdef int block_index_start = taskid * blocksize
    cdef int block_index_end = block_index_start + blocksize
    cdef int [::1] block_indices = np.arange(block_index_start, block_index_end, dtype=np.int32)
    cdef double complex [::1] X = x.copy()
    #Each core saves its part of the answer to chunk
    cdef double complex [::1] chunk = np.zeros(blocksize, dtype=complex)
    
    #Works out the number and size of subproblems
    for i in range(layers):
        number_of_problems = 2 ** i
        subproblemsize = int(N / 2 ** (i))
        halfproblemsize = int(subproblemsize / 2)
        
        #Works out the start and end index of each subproblem
        for j in range(number_of_problems):
            index_start = j * subproblemsize
            index_end = index_start + subproblemsize
            
            for k in range(index_start, index_end):
                if k >= block_index_start and k < block_index_end:
                    if (k - index_start) < halfproblemsize:            
                        chunk[k - block_index_start] = X[k] + X[k + halfproblemsize]
                    else:
                        twiddle_index = number_of_problems * (k - index_start - halfproblemsize)
                        temp_twiddle = twiddle_factor[twiddle_index]
                        chunk[k - block_index_start] = temp_twiddle * (X[k - halfproblemsize] - X[k])
                else:
                    pass
                
        #After each layer the master node recives each chunk and updates the array
        if taskid == 0:
            #Update master node
            for l in range(blocksize):
                X[block_indices[l]] = chunk[l]
            
            #update worker nodes
            for source in range(1, numtasks):
                comm.Recv([chunk, MPI.COMPLEX], source=source, tag=1)
                comm.Recv([block_indices, MPI.INT], source=source, tag=2)
                
                for l in range(blocksize):
                    X[block_indices[l]] = chunk[l]
                
            block_indices = np.arange(block_index_start, block_index_end, dtype=np.int32)
            
        else:
            #send infomation to master node
            comm.Send([chunk, MPI.COMPLEX], dest=0, tag=1)
            comm.Send([block_indices, MPI.INT], dest=0, tag=2)
            
        #Send updated array to all worker nodes
        comm.Bcast([X, MPI.COMPLEX], root=0)
    
    bit_reversed_array(X, N, layers) #Output is scrambled in DIF FFT see report
    
    return X

@boundscheck(False)
@wraparound(False)
def FFT_MPI_CBM(double complex [::1] x):
    '''
    This is a cythonised iterative FFT algorithm. It works by using a  
    Gentleman-Sande butterfly algorithm, more info can be found in report.

    Parameters
    ----------
    double complex [ : :1] x
        Input array.

    Returns
    -------
    x : TYPE
        Output fourier transform.
    '''
    comm = MPI.COMM_WORLD
    #Make sure all the cores are working on the same problem
    comm.Bcast([x, MPI.COMPLEX], root=0) 
    
    cdef int taskid = comm.Get_rank()
    cdef int numtasks = comm.Get_size()
    cdef int N = x.shape[0]
    cdef int layers = int(np.log2(N))
    cdef int i, j, k, l, source
    cdef int number_of_problems
    cdef int subproblemsize, halfproblemsize, blocksize
    cdef int index_start, index_end, block_index_start, block_index_end
    cdef int twiddle_index, chunk_index
    cdef double complex temp_twiddle
    cdef double complex [::1] twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2)/N)
    cdef int chunksize = N / numtasks
    cdef double complex [::1] chunk = np.zeros(chunksize, dtype=complex)
    cdef int [::1] chunk_indices = np.zeros(chunksize, dtype=np.int32)
    cdef double complex [::1] X = x.copy()
    
    #Works out the number and size of subproblems
    for i in range(layers):
        number_of_problems = 2 ** i
        subproblemsize = int(N / 2 ** (i))
        halfproblemsize = int(subproblemsize / 2)
        blocksize = int(subproblemsize / (numtasks))
        
        #Itterates over each problem in the stage 
        for j in range(number_of_problems):
            
            #If blocksize is greater than one cyclic block mapping will be used
            if blocksize >= 1:
                #Works out the subproblem half range
                index_start = j * subproblemsize
                index_end = index_start + halfproblemsize
                
                #Works out the range for each core 
                block_index_start = index_start + taskid * blocksize
                block_index_end = block_index_start + blocksize
                
                #itterate over the core range 
                for k in range(block_index_start, block_index_end):
                    #Works out the storage index for each computation
                    chunk_index = k - block_index_start + j * blocksize
                    
                    #Checks if k is in top of Gentleman-Sande butterfly
                    if (k - index_start) < halfproblemsize:
                        chunk[chunk_index] = X[k] + X[k + halfproblemsize]
                        chunk_indices[chunk_index] = k
                    
                    #Performs the bottom of the Gentleman-Sande butterfly
                    else:
                        #Works out correct twiddle index
                        twiddle_index = number_of_problems * (k - block_index_start + int(taskid - numtasks / 2) * blocksize)
                        temp_twiddle = twiddle_factor[twiddle_index]
                        chunk[chunk_index] = temp_twiddle * (X[k - halfproblemsize] - X[k])
                        chunk_indices[chunk_index] = k
            
            #If blocksize is less than one static block mapping is used 
            else:
                #Works out subproblem full range
                index_start = j * subproblemsize
                index_end = index_start + subproblemsize 
                
                #Works out static core position 
                block_index_start = taskid * chunksize
                block_index_end = block_index_start + chunksize
                
                #Itterates over whole subproblem and works out whether k is in core range manually
                for k in range(index_start, index_end):
                    if k >= block_index_start and k < block_index_end:
                        chunk_index = k - block_index_start + j * blocksize
                        
                        #Checks if k is in top of Gentleman-Sande butterfly
                        if (k - index_start) < halfproblemsize:
                            chunk[chunk_index] = X[k] + X[k + halfproblemsize]
                            chunk_indices[chunk_index] = k
                        
                        #Performs the bottom of the Gentleman-Sande butterfly    
                        else:
                            twiddle_index = number_of_problems * (k - index_start - halfproblemsize)
                            temp_twiddle = twiddle_factor[twiddle_index]
                            chunk[chunk_index] = temp_twiddle * (X[k - halfproblemsize] + X[k])
                            chunk_indices[chunk_index] = k
                    else:
                        pass
        
        #After each stage the master core gathers all chunks and allocates the 
        #elements to the appropriate X position using the chunk_indices array.
        if taskid == 0:
            for l in range(chunksize):
                X[chunk_indices[l]] = chunk[l]
                
            for source in range(1, numtasks):
                comm.Recv([chunk, MPI.COMPLEX], source=source, tag=1)
                comm.Recv([chunk_indices, MPI.INT], source=source, tag=2)
                
                for l in range(chunksize):
                    X[chunk_indices[l]] = chunk[l]
                    
        else:
            comm.Send([chunk, MPI.COMPLEX], dest=0, tag=1)
            comm.Send([chunk_indices, MPI.INT], dest=0, tag=2)
            
        comm.Bcast([X, MPI.COMPLEX], root=0)
        
    bit_reversed_array(X, N, layers)
    
    return X                
 


        
    


    
    
    
    
    
    
    