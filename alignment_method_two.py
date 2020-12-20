#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:37:48 2020

@author: leonard
"""

### Importing Modules ########################################################

from scipy.io import loadmat
import pandas as pd
from brainspace.gradient import GradientMaps
import numpy as np
import math

## Data "D" is the Database or "train" set
D = loadmat(
    '../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat',
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)

## Data Y is the target set
Y = loadmat(
    '../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat',
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)


## defining the variables ####################################################
## for Database set D
D_connm = D['res']['connm']
D_sub = D['res']['sub']
D_connectivity_data = pd.DataFrame(D_connm, D_sub)
D_cd_transposed = D_connectivity_data.transpose()

## for Target Set Y
Y_connm = Y['res']['connm']
Y_sub = Y['res']['sub']
Y_connectivity_data = pd.DataFrame(Y_connm, Y_sub)
Y_cd_transposed = Y_connectivity_data.transpose()


## function for obtaining participants connectivity matrix
## input should be a pandas series containing the connectivity data
## for one subject
def get_conn_matrix( X ):
    n = len(X)
    N = round((math.sqrt(8*n+1)+1)/2)
    square_zeros = np.zeros((N,N))
    indices = np.nonzero(np.triu(np.ones((N,N)), 1))
    square_zeros[indices] = X
    square_connm = square_zeros + np.transpose(square_zeros)
    return np.eye(N) + square_connm


### Identification Analysis ##################################################
### Target Gradients and Identification from Database

count1 = 0
for i in enumerate(Y_sub):
    
    subject_connm = get_conn_matrix(Y_cd_transposed[i[1]])
    ## gradient construction using the matrices for each subject to be "identi-
    ## fied" in the "target set" Y
    gm = GradientMaps(n_components=1, kernel = "pearson", approach = "dm",
                      random_state=0, alignment = 'procrustes')
    gm.fit(subject_connm)
    subject_gradient = gm.gradients_[:,0]
    subject_gradient_dataframe = pd.DataFrame(subject_gradient)
    gradient_array_database = np.zeros((len(D_sub), 160))
    
    ### Database Gradient Construction #######################################
    ### This will make sure that the gradients from database are always aligned
    ### to the target gradient to be identified
    for k in enumerate(D_sub):
        subject_matrix = get_conn_matrix(D_cd_transposed[k[1]])
        # b = pd.DataFrame(subject_matrix) # as DataFrame
    
        database_gm = GradientMaps(n_components=1, kernel = "pearson", approach = "dm",
                      random_state=0, alignment = 'procrustes')
        database_gm.fit(subject_matrix, reference = gm.gradients_)
    
    
        gradient_array_database[k[0]] = database_gm.aligned_[:,0]


        ## So now we have a database consisting of gradients from all partici
        ## -pants in database D
        gradient_dataframe = pd.DataFrame(gradient_array_database)
        gradient_dataframe_transposed = gradient_dataframe.T
        gradient_dataframe_transposed.columns = list(D_sub)
    
    all_corr = gradient_dataframe_transposed.corrwith(subject_gradient_dataframe.iloc[:,0],
                                                      method = 'pearson')
    max_value = max(all_corr)
    max_index = all_corr.index[all_corr == max_value]
    
    if max_index[0] == k[1]:
        count1 = count1 +1
        print(max_index[0])
    
rate = count1/len(Y_cd_transposed.columns)

print(str(rate*100) + '% of subjects in were accurately predicted from data in '
      'dataset D.')