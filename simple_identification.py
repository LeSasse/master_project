#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:21:37 2020

@author: leonard
"""

### Importing Modules ########################################################

from scipy.io import loadmat
import pandas as pd

import seaborn as sns
from seaborn import load_dataset

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from julearn import run_cross_validation
from julearn.utils import configure_logging

### Loading Data
## data is stored in variable 'res', consists of 399 participants, it has the
## following columns:
## sub == subject ID
## connm == all possible temporal correlations (or functional connectivity)
## for every region resulting in 12720 datapoints for each subject (160*159/2)
## alff == Amplitude of Low Frequency Fluctuations,
## falff == fractional Amplitude of Low Frequency Fluctuations,
## reho == regional homogeneity


## Data "D" is the Database or "train" set
D = loadmat(
    "/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat",
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)


## defining the variables
D_connm = D['res']['connm']
D_reho = D['res']['reho']
D_alff = D['res']['alff']
D_falff = D['res']['falff']
D_sub = D['res']['sub']

#print(connm.ndim)
#print(connm.shape)
#print(connm.size)
#print(connm.dtype)

#print(connm)

D_connectivity_data = pd.DataFrame(D_connm, D_sub)
D_re_homogeneity = pd.DataFrame(D_reho, D_sub)
D_df_alff = pd.DataFrame(D_alff, D_sub)
D_df_falff = pd.DataFrame(D_falff, D_sub)


#print(connectivity_data)
#print(re_homogeneity)
#print(df_alff)
#print(df_falff)

D_cd_transposed = D_connectivity_data.transpose()
#print(cd_transposed)

## Data Y is the target set
Y = loadmat(
    "/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat",
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)

Y_connm = Y['res']['connm']
Y_reho = Y['res']['reho']
Y_alff = Y['res']['alff']
Y_falff = Y['res']['falff']
Y_sub = Y['res']['sub']

Y_connectivity_data = pd.DataFrame(Y_connm, Y_sub)
Y_re_homogeneity = pd.DataFrame(Y_reho, Y_sub)
Y_df_alff = pd.DataFrame(Y_alff, Y_sub)
Y_df_falff = pd.DataFrame(Y_falff, Y_sub)

Y_cd_transposed = Y_connectivity_data.transpose()
print(Y_cd_transposed)

### Identification Analysis ##################################################
## From Finn et al., 2005: First a database was created that consisted of 
## all subjects conn. matrix, in the identification step the identity of the 
## target matrix was predicted using a correlation mamtrix obtained from a 
## different session, to predict subject identity similarity between current 
## target matrix  and all other matrices was computed; similarity was defined as 
## the pearson correlation between two vectors of edge values taken from the target 
## matrix 

## I used the Identification Code from Finn et al., 2005 as an orientation
count1 = 0
count2 = 0




for i in enumerate(Y_cd_transposed):
    
    tt_corr = Y_cd_transposed.iloc[:,i[0]]
    tt_to_all = D_cd_transposed.corrwith(tt_corr)
    max_value = max(tt_to_all)
    max_index = tt_to_all.index(max_value)
    
    #if i == va_id:
    #    count1 = count1 +1

