#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:13:29 2020

@author: leonard

Loading Data 
data is stored in variable 'res', consists of 399 participants, it has the
following columns:
sub == subject ID
connm == all possible temporal correlations (or functional connectivity)
for every region resulting in 12720 datapoints for each subject (160*159/2)
alff == Amplitude of Low Frequency Fluctuations,
falff == fractional Amplitude of Low Frequency Fluctuations,
reho == regional homogeneity
"""

### Importing Modules ########################################################

from scipy.io import loadmat
import pandas as pd
import numpy as np
import math

def load_data(data_path):
    '''input is path that leads to data as a string,
    Output is D = the whole datafile, D_sub = series of subject id numbers, 
    D_connectivity_data is the subjects concatenated connectivity data, last out-
    put is transposed connectivity data'''
    D = loadmat(
        data_path,
        squeeze_me = False,
        mat_dtype = True,
        struct_as_record = True,
        simplify_cells = True)
    
    D_connm = D['res']['connm']
    D_sub = D['res']['sub']
    D_connectivity_data = pd.DataFrame(D_connm, D_sub)
    D_cd_transposed = D_connectivity_data.transpose()
    
    return D, D_sub, D_connectivity_data, D_cd_transposed

def get_conn_matrix( X ):
    '''function for obtaining participants connectivity matrix input should be 
    a pandas series containing the connectivity data for one subject'''
    n = len(X)
    N = round((math.sqrt(8*n+1)+1)/2)
    square_zeros = np.zeros((N,N))
    indices = np.nonzero(np.triu(np.ones((N,N)), 1))
    square_zeros[indices] = X
    square_connm = square_zeros + np.transpose(square_zeros)
    return np.eye(N) + square_connm


