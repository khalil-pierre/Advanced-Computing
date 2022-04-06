# -*- coding: utf-8 -*-
"""
This script is used to collect the data from the stage timing function so that
I can compare the timing for the different stage using the 2 different methods.
"""
from stage_timing import stage_timing

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

N = 2 ** 22
threads = int(sys.argv[1])
timing_array_1 = []
timing_array_2 = []

for i in range(5):
    x = np.asarray(np.random.random(N), dtype=np.complex)
    
    stage_number_1, stage_timing_1 = stage_timing(x, threads, '1')
    stage_number_2, stage_timing_2 = stage_timing(x, threads, '2')
    
    timing_array_1.append(stage_timing_1)
    timing_array_2.append(stage_timing_2)
    
stage_timing_1 = np.mean(np.array(timing_array_1), axis=0)
stage_timing_2 = np.mean(np.array(timing_array_2), axis=0)

stage_error_1 = np.std(np.array(timing_array_1), axis=0)
stage_error_2 = np.std(np.array(timing_array_2), axis=0)

dictionary = {'stage_number' : stage_number_1, 
       'stage_timing_1': stage_timing_1,
       'stage_error_1': stage_error_1,
       'stage_timing_2': stage_timing_2,
       'stage_error_2': stage_error_2}

df = pd.DataFrame(dictionary)
df.to_pickle('stage_timing_thread_{}.pkl'.format(threads))


    
    