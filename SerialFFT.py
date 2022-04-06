# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 07:05:22 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator

def testFFT(func, size=5):
    '''
    This function test if my DFT and FFT function work by comparing them to 
    the numpy fft function for arrays of varying lengths. The length of the 
    largest array will be given by 2**(size).

    Parameters
    ----------
    func : function
        DFT or FFT function you want to test.
    size : int, optional
        The size of the array you want to test to. The default is 5.
    '''
    for i in range(1, size + 1):
        N = 2 ** i
        x = np.asarray(np.random.random(N), dtype=complex)
        print('Array size {} : result {} '.format(N, np.allclose(np.fft.fft(x),func(x))))

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
        initial_time = time.time()
        func(x)
        final_time = time.time()
        times += [final_time - initial_time]
        
    mean_time = np.mean(times)
    std_time = np.std(times)
        
    return mean_time, std_time

def DFT(x, ortho=False):
    '''
    This function will calculate the discrete fourier transform of a 1d array 
    using matrix multiplication. The elements of the transformation matrix are
    given by M_mn = exp(-2j*pi*n*m*N^{-1}), where n and m are the coloumn and 
    row indices respectivly and N is the size of the array.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.
    ortho : bool
        The transformed array will be normalised

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N) #generate row indices
    m = n.reshape((N, 1)) #generate column indices
    M = np.exp(-2j * np.pi *m * n / N) #generate transformation matrix
    
    if ortho == True:
        norm = 1 / np.sqrt(N)
    else:
        norm = 1
    
    f = norm*np.dot(M, x)
    
    return  f

def FFT(x):
    '''
    This function calculates the DIT of a 1D array using the recursive 1D 
    Cooley-Tukey method.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 >0:
        raise ValueError('The size of input array must be an even number the '
                         + 'size of the input is N = {}'.format(N))
    elif N <= 32:
        n = np.arange(N) #generate row indices
        m = n.reshape((N, 1)) #generate column indices
        M = np.exp(-2j * np.pi *m * n / N) #generate transformation matrix
        
        return np.dot(M, x)
    else:
        even_x = x[::2] #Get all the even elements indexed of 
        odd_x = x[1::2] #Get all the odd elements indexed of 
        
        #If the split arrays have a size of less then 32 then the DFT of the 
        #split arrays will be returned. If not the split arrays will be passed 
        #back through the function until they meet this condition.
        
        even_f = FFT(even_x) #Calculate the DFT of even indexed elements
        odd_f = FFT(odd_x)   #Calculate the DFT of odd indexed elements 
        twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2) / N) 
        f = np.concatenate([even_f + twiddle_factor * odd_f,
                            even_f - twiddle_factor * odd_f])
        
        return f
    
def FFT_vec(x):
    '''
    Vectorised version of the Cooley-Tukey method.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if np.log2(N) % 1 > 0:
        raise ValueError('The size of x must be a power of 2, size is {}'.format(N))
    
    x = x.reshape(2,-1)
    M = np.array([[1,1],[1,-1]]) # Fourier transform matrix for 2x1 input
    f = np.dot(M, x) #Calculates Fourier transform for all pairs
    
    while f.shape[0] < N:
        f_even = f[:,:int(f.shape[1]/2)]
        f_odd = f[:,int(f.shape[1]/2):]
        
        twiddle_factor = np.exp(-1j * np.pi * np.arange(f.shape[0])
                        / f.shape[0])[:, None]
        
        f = np.vstack([f_even + twiddle_factor * f_odd,
                       f_even - twiddle_factor * f_odd])
    
    return f.ravel()

if __name__ == "__main__":
    #Diffraction pattern of a square appeture 
    
    N = 2 ** 9 #Create x array
    aperture_start = int((N / 2) - 1 - N / 8)
    aperture_end = int((N / 2) -1 + N / 8)
    x = np.zeros(N)
    x[aperture_start : aperture_end] = 1
    plt.plot(range(len(x)), x)
    
    X = FFT_vec(x) 
    X = np.fft.fftshift(X) # swaps first and second half of array
    
    power = X * np.conj(X)
    power = power / np.max(power)
    
    plt.plot(range(len(X)), power)
    # initial_time = time.time()
    # DFT_timing = []
    # DFT_error = []
    
    # FFT_timing = []
    # FFT_error = []
    
    # FFTv_timing = []
    # FFTv_error = []
    
    # for i in range(1,10):
    #     print(i)
    #     x = np.random.random(int(2 ** i))
    #     if i <= 14:
    #         DFT_mean, DFT_std = time_FFT(DFT, x, 5)
    #         DFT_timing += [DFT_mean]
    #         DFT_error += [DFT_std]
        
    #     FFT_mean, FFT_std = time_FFT(FFT, x, 5)
    #     FFT_timing += [FFT_mean]
    #     FFT_error += [FFT_std]
        
        
    #     FFTv_mean, FFTv_std = time_FFT(FFT_vec, x, 5)
    #     FFTv_timing += [FFTv_mean]
    #     FFTv_error += [FFTv_std]
        
        
    # time_taken = time.time() - initial_time
    # print('It took {}s to perform this operation'.format(time_taken))
    
    # fig, ax = plt.subplots()
    # ax.errorbar(range(1,len(DFT_timing)+1), DFT_timing, 
    #               yerr=DFT_error, label = 'DFT',
    #               capsize=3, lw=1, ls='--')
    
    # ax.errorbar(range(1,len(FFT_timing)+1), FFT_timing, 
    #           yerr=FFT_error, label = 'FFT',
    #           capsize=3, lw=1, ls='--')
    
    # ax.errorbar(range(1,len(FFTv_timing)+1), FFTv_timing, 
    #           yerr=FFTv_error, label = 'FFT_vector',
    #           capsize=3, lw=1, ls='--')
    
    # ax.legend()
    # ax.set_ylabel('time (s)')
    # ax.set_xlabel('Array size $2^{n}$')
    # ax.set_yscale('log')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
        