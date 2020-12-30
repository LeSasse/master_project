#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 12:37:48 2020

@author: leonard
"""

### Importing Modules ########################################################

import pandas as pd
from brainspace.gradient import GradientMaps
import numpy as np

## my own imports
import load_data_and_functions as ldf

### Global Variables/User input

database_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat'
target_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat'

kernel = "pearson"
dimension_reduction = "dm"
alignment = "procrustes"
atlas_size = 160


## Loading Data
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)


### Identification Analysis ##################################################
### Target Gradients and Identification from Database

count1 = 0
for index, subject in enumerate(Y_sub):
    
    subject_connm = ldf.get_conn_matrix(Y_cd_transposed[subject])
    ## gradient construction using the matrices for each subject to be "identi-
    ## fied" in the "target set" Y
    gm = GradientMaps(n_components=1, kernel = kernel, 
                      approach = dimension_reduction,
                      random_state=0, alignment = alignment)
    gm.fit(subject_connm)
    subject_gradient = gm.gradients_[:,0]
    subject_gradient_dataframe = pd.DataFrame(subject_gradient)
    gradient_array_database = np.zeros((len(D_sub), atlas_size))
    
    ### Database Gradient Construction #######################################
    ### This will make sure that the gradients from database are always aligned
    ### to the target gradient to be identified
    for index2, subject2 in enumerate(D_sub):
        subject_matrix = ldf.get_conn_matrix(D_cd_transposed[subject2])
    
        database_gm = GradientMaps(n_components=1, kernel = kernel, 
                                   approach = dimension_reduction,
                                   random_state=0, alignment = alignment)
        database_gm.fit(subject_matrix, reference = gm.gradients_)
    
        gradient_array_database[index2] = database_gm.aligned_[:,0]
        gradient_dataframe = pd.DataFrame(gradient_array_database)
        gradient_dataframe_transposed = gradient_dataframe.T
        gradient_dataframe_transposed.columns = list(D_sub)
    
    all_corr = gradient_dataframe_transposed.corrwith(subject_gradient_dataframe.iloc[:,0],
                                                      method = 'pearson')
    max_value = max(all_corr)
    max_index = all_corr.index[all_corr == max_value]
    
    if max_index[0] == subject:
        count1 = count1 +1
        #print(subject) to know who was correctly identified
    
rate = count1/len(Y_cd_transposed.columns)

print(str(rate*100) + '% of subjects in were accurately predicted from data in '
      'dataset D.')