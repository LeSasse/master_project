#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:21:37 2020

@author: leonard
"""

from scipy.io import loadmat
import pandas as pd


## data is stored in variable 'res', consists of 399 participants, it has the
## following columns:
## sub == subject ID
## connm == all possible temporal correlations (or functional connectivity)
## for every region resulting in 12720 datapoints for each subject (160*159/2)
## alff == Amplitude of Low Frequency Fluctuations,
## falff == fractional Amplitude of Low Frequency Fluctuations,
## reho == regional homogeneity

D = loadmat(
    "/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat",
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)


## defining the variables
connm = D['res']['connm']
reho = D['res']['reho']
alff = D['res']['alff']
falff = D['res']['falff']
sub = D['res']['sub']




#print(connm.ndim)
#print(connm.shape)
#print(connm.size)
#print(connm.dtype)

#print(connm)

connectivity_data = pd.DataFrame(connm, sub)
re_homogeneity = pd.DataFrame(reho, sub)
df_alff = pd.DataFrame(alff, sub)
df_falff = pd.DataFrame(falff, sub)

#print(connectivity_data)
#print(re_homogeneity)
#print(df_alff)
#print(df_falff)