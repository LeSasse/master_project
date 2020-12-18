### Importing Modules ########################################################

from scipy.io import loadmat
import pandas as pd
from brainspace.gradient import GradientMaps
import numpy as np
import math

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



### Gradient Construction ####################################################

## So far gradients are NOT aligned in this code
## align gradients using procrustes alignment (there will be two sep-
## implementations to investigate here: either all gradients will be aligned to
## one gradient or the gradient to be identified will be used to align all other
## gradients in each iteration) 



### Database Gradient Construction ###########################################
gradient_array_database = np.zeros((len(D_sub), 160))
for k in enumerate(D_sub):
    conn_data = D_cd_transposed[k[1]]
    diagval = 0
    n = len(conn_data)
    N = round((math.sqrt(8*n+1)+1)/2)
    square_zeroes = np.zeros((N,N))
    ## sq_df = pd.DataFrame(sq) #if wanted as DataFrame
    indices = np.nonzero(np.triu(np.ones((N,N)), 1))
    square_zeroes[indices] = conn_data
    square_connm = square_zeroes + np.transpose(square_zeroes)
    square_connm[square_connm == 0] = 1
    # b = pd.DataFrame(square_connm) # as DataFrame
    
    gm = GradientMaps(n_components=1, kernel = "pearson", approach = "dm",
                      random_state=0)
    gm.fit(square_connm)
    
    
    gradient_array_database[k[0]] = gm.gradients_[:,0]


## So now we have a database consisting of gradients from all participants in 
## database D
### IMPORTANT: these gradients are not aliged
gradient_dataframe = pd.DataFrame(gradient_array_database)
gradient_dataframe_transposed = gradient_dataframe.T
gradient_dataframe_transposed.columns = list(D_sub)



### Identification Analysis ##################################################
### Target Gradients and Identification from Database

count1 = 0
for i in enumerate(Y_sub):
    
    conn_data = Y_cd_transposed[i[1]] # a is connectivity data atm but 
                                      # not as n**2 shape
    
    
    ## following code will construct connectivity matrices for each participant
    diagval = 0
    n = len(conn_data)
    N = round((math.sqrt(8*n+1)+1)/2)
    square_zeroes = np.zeros((N,N))
    ## sq_df = pd.DataFrame(sq) #if wanted as DataFrame
    indices = np.nonzero(np.triu(np.ones((N,N)), 1))
    square_zeroes[indices] = conn_data
    square_connm = square_zeroes + np.transpose(square_zeroes)
    square_connm[square_connm == 0] = 1
    # b = pd.DataFrame(square_connm) # as DataFrame
    
    ## gradient construction using the matrices for each subject to be "identi-
    ## fied" in the "target set" Y
    gm = GradientMaps(n_components=1, kernel = "pearson", approach = "dm",
                      random_state=0)
    gm.fit(square_connm)
    subject_gradient = gm.gradients_
    subject_gradient_dataframe = pd.DataFrame(subject_gradient)
    
    all_corr = gradient_dataframe_transposed.corrwith(subject_gradient_dataframe.iloc[:,0],
                                                      method = 'pearson')
    max_value = max(all_corr)
    max_index = all_corr.index[all_corr == max_value]
    
    if max_index[0] == k[1]:
        count1 = count1 +1
    
rate = count1/len(Y_cd_transposed.columns)

print(str(rate*100) + '% of subjects in were accurately predicted from data in '
      'dataset D.')
    
    


