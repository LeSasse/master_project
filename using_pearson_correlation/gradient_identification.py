### Importing Modules ########################################################

import pandas as pd
from brainspace.gradient import GradientMaps
import numpy as np

## my own imports
import load_data_and_functions as ldf

### Global Variables/User input ##############################################

database_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat'
target_path = '../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat'

kernel = "pearson"
dimension_reduction = "dm"
alignment = "procrustes"
atlas_size = 160


## Loading Data ##############################################################
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)


### Gradient Construction ####################################################

## So far gradients are NOT aligned in this code
## align gradients using procrustes alignment (there will be two sep-
## implementations to investigate here: either all gradients will be aligned to
## one gradient or the gradient to be identified will be used to align all other
## gradients in each iteration) 


### Database Gradient Construction ###########################################
gradient_array_database = np.zeros((len(D_sub), atlas_size))
for index, subject in enumerate(D_sub):
    subject_connm = ldf.get_conn_matrix(D_cd_transposed[subject])
    
    gm = GradientMaps(n_components=1,
                      kernel = kernel,
                      approach = dimension_reduction,
                      random_state=0)
    gm.fit(subject_connm)
    
    
    gradient_array_database[index] = gm.gradients_[:,0]

## So now we have a database consisting of gradients from all participants in 
## database D
### IMPORTANT: these gradients are not aligned
gradient_dataframe = pd.DataFrame(gradient_array_database)
gradient_dataframe_transposed = gradient_dataframe.T
gradient_dataframe_transposed.columns = list(D_sub)


### Identification Analysis ##################################################
### Target Gradients and Identification from Database

count1 = 0
for index, subject in enumerate(Y_sub):
    conn_data = Y_cd_transposed[subject]  
    subject_connm = ldf.get_conn_matrix(Y_cd_transposed[subject])
                                     
    ## gradient construction using the matrices for each subject to be "identi-
    ## fied" in the "target set" Y
    gm = GradientMaps(n_components=1,
                      kernel = kernel,
                      approach = dimension_reduction,
                      random_state=0)
    gm.fit(subject_connm)
    subject_gradient = gm.gradients_[:,0]
    subject_gradient_dataframe = pd.DataFrame(subject_gradient)
    
    all_corr = gradient_dataframe_transposed.corrwith(
        subject_gradient_dataframe.iloc[:,0], method = 'pearson'
        )
    max_value = max(all_corr)
    max_index = all_corr.index[all_corr == max_value]
    
    if max_index[0] == subject:
        count1 = count1 +1
        #print(subject) to know who was correctly identified
    
rate = count1/len(Y_cd_transposed.columns)

print(str(rate*100) + '% of subjects in were accurately predicted from data in '
      'dataset D.')
    
    


