#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:56:21 2021

@author: leonard
"""

# metric learning for identification
import sys
import pickle
import pandas as pd
import numpy as np
import pdb
import random
from datetime import datetime
import cma
from sklearn.neighbors import NearestNeighbors


## my own imports
sys.path.append("../imports")
import load_data_and_functions as ldf


print("Starting...")


### Global Variables/User input ##############################################

### data file paths ##########################################################
session1_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)


## Loading Data ##############################################################
##############################################################################

(
    session1_data,
    session1_sub,
    session1_connectivity_data,
    session1_cd_transposed,
) = ldf.load_data(session1_path)
(
    session2_data,
    session2_sub,
    session2_connectivity_data,
    session2_cd_transposed,
) = ldf.load_data(session2_path)


# Data is an N by D Matrix
#N = session2_cd_transposed.shape[0]
#D = session2_cd_transposed.shape[1]
##############################################################################


RUNS = 10
MIN, MAX = -1.0, 1.0
MAXFEVALS = 500
TRAINPROP = 0.5


## Making sure that both dataframes have same participants ###################
for col in session1_cd_transposed.columns:
    if col not in session2_cd_transposed.columns:
        session1_cd_transposed.drop([col], axis = 1, inplace=True)
    
for col in session2_cd_transposed.columns:
    if col not in session1_cd_transposed.columns:
        session2_cd_transposed.drop([col], axis = 1, inplace=True)
##############################################################################


'''
## Identification ############################################################
##############################################################################
rate1 = ldf.identify(target=session2_cd_transposed, database=session1_cd_transposed)
rate2 = ldf.identify(target=session1_cd_transposed, database=session2_cd_transposed)

rate = (rate1 + rate2) / 2
'''

## initial random weights ####################################################
'''
vector w for weights
matrix A for session1 
matrix B for session2
'''


### summing data ... ##########################################
control1 = ldf.create_control_data(session1_cd_transposed, kind="sum", atlas_size=160)
control2 = ldf.create_control_data(session2_cd_transposed, kind="sum", atlas_size=160)



n = control1.shape[0]   #160
d = control1.shape[1]   #398

## generating some random weights to start with ... 
w1 = np.random.rand(int(n/2),)
w2 = np.random.rand(int(n/2),) * -1
w = np.append(w1, w2)
w = np.random.permutation(w)

## ... then weighting session 1
A = np.array(control1)
Aw = (A.T * w).T

## ... then weighting session 2
B = np.array(control2)
Bw = (B.T * w).T

## identification with randomly generated initial weights
data1 = pd.DataFrame(Aw)
data2 = pd.DataFrame(Bw)
N = data1.shape[0]
D = data1.shape[1]  
data1_ranked = data1.T.rank()
data2_ranked = data2.T.rank()
nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data1_ranked)
distances, indices = nbrs.kneighbors(data2_ranked)
rate1 = np.mean(indices.T == np.array(list(range(D))))
nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data2_ranked)
distances, indices = nbrs.kneighbors(data1_ranked)
rate2 = np.mean(indices.T == np.array(list(range(D))))
rate = (rate1 + rate2)/2


'''
## cma options 
cmaopts = cma.CMAOptions()
cmaopts.set('bounds', [[MIN]*d, [MAX]*D])
cmaopts.set('maxfevals', MAXFEVALS)
'''

def objectiveFUN(w, data1, data2):
    A = np.array(data1)
    B = np.array(data2)
    
    ## weighting columns
    Aw = (A.T * w).T
    Bw = (B.T * w).T
    #return pd.DataFrame(Aw)
    
    data1 = pd.DataFrame(Aw)
    data2 = pd.DataFrame(Bw)
    N = data1.shape[0]
    D = data1.shape[1]             
    
    ### identification method ###
    '''
    Depending on how tests on gradients go, I will have to implement the ident-
    fication method using pandas.corrwith():
    However, ranking data and then using pearson correlation, should be equiva-
    lent to spearman correlation on original data. Nonetheless identification 
    results are slightly different (+-1 or 2 percentage points), will have to 
    see how much it affects results overall
    '''
    
    data1_ranked = data1.T.rank()
    data2_ranked = data2.T.rank()
    
    ## first way
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data1_ranked)
    distances, indices = nbrs.kneighbors(data2_ranked)
    rate1 = np.mean(indices.T == np.array(list(range(D))))
    
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data2_ranked)
    distances, indices = nbrs.kneighbors(data1_ranked)
    rate2 = np.mean(indices.T == np.array(list(range(D))))
    
    rate = (rate1 + rate2)/2
    return rate