### Importing Modules ########################################################

from scipy.io import loadmat
import pandas as pd
from brainspace.gradient import GradientMaps
import numpy as np

### Loading Data #############################################################
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
    '/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelate'
    'd_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat',
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)

## Data Y is the target set
Y = loadmat(
    '/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelate'
    'd_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat',
    squeeze_me = False,
    mat_dtype = True,
    struct_as_record = True,
    simplify_cells = True)

## defining the variables ####################################################
## for Database set D
D_connm = D['res']['connm']
D_reho = D['res']['reho']
D_alff = D['res']['alff']
D_falff = D['res']['falff']
D_sub = D['res']['sub']

D_connectivity_data = pd.DataFrame(D_connm, D_sub)
D_re_homogeneity = pd.DataFrame(D_reho, D_sub)
D_df_alff = pd.DataFrame(D_alff, D_sub)
D_df_falff = pd.DataFrame(D_falff, D_sub)

D_cd_transposed = D_connectivity_data.transpose()

## for Target Set Y
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

### Identification Analysis

### Gradient Construction ####################################################
## 1. for loop to go through connectivity data for each participant; I have to re-
## shape data into a square n*n matrix to construct gradients and extract first
## component to use in identification; i am not sure if this is possible with 
## the data at hand

## 2. also align gradients using procrustes alignment (there will be two sep-
## implementations to investigate here: either all gradients will be aligned to
## one gradient or the gradient to be identified will be used to align all other
## gradients in each iteration) 
for i in enumerate(D_sub):
    a = D_cd_transposed[i[1]] # a is connectivity data atm but not as n**2 shape
    print(a)
    
    


