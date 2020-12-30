#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:21:37 2020

@author: leonard
"""

### Importing Modules ########################################################

## my own imports
import load_data_and_functions as ldf

### Global Variables #########################################################
database_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat'
target_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat'


D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)

Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)


### Identification Analysis ##################################################
## I used the Identification Code from Finn et al., 2005 as an orientation
count1 = 0 #number of correct if D is Database

for index, subject in enumerate(Y_cd_transposed):
    
    tt_corr = Y_cd_transposed.iloc[:,index]
    tt_to_all = D_cd_transposed.corrwith(tt_corr, method = 'pearson')
    max_value = max(tt_to_all)
    max_index = tt_to_all.index[tt_to_all == max_value]
    
    if max_index[0] == subject:
        count1 = count1 +1

        
rate = count1/len(Y_cd_transposed.columns)

print(str(rate*100) + '% of subjects in were accurately predicted from data in '
      'dataset D.')