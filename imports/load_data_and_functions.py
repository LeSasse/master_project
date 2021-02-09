
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
import brainspace
from brainspace.gradient import GradientMaps


def load_data(data_path):
    """input is path that leads to data as a string,
    Output is D = the whole datafile, D_sub = series of subject id numbers,
    D_connectivity_data is the subjects concatenated connectivity data, last out-
    put is transposed connectivity data"""
    D = loadmat(
        data_path,
        squeeze_me=False,
        mat_dtype=True,
        struct_as_record=True,
        simplify_cells=True,
    )

    D_connm = D["res"]["connm"]
    D_sub = D["res"]["sub"]
    D_connectivity_data = pd.DataFrame(D_connm, D_sub)
    D_cd_transposed = D_connectivity_data.transpose()

    return D, D_sub, D_connectivity_data, D_cd_transposed


def get_conn_matrix(X):
    """function for obtaining participants connectivity matrix; input should be
    a pandas series containing the connectivity data for one subject"""
    n = len(X)
    N = round((math.sqrt(8 * n + 1) + 1) / 2)
    square_zeros = np.zeros((N, N))
    indices = np.nonzero(np.triu(np.ones((N, N)), 1))
    square_zeros[indices] = X
    square_connm = square_zeros + np.transpose(square_zeros)
    return np.eye(N) + square_connm


def create_gradient_database(
    dataframe,
    subjects,
    atlas_size,
    reference=None,
    kernel="pearson",
    dimension_reduction="dm",
    alignment="procrustes",
    sparsity=0.9,
    n_gradients=1,
    concatenate=False,
):
    """dataframe = connectivity data to construct subjects gradient, subjects = list
    of subjects, atlas_size = size of the atlas, reference =  a reference gradient to use for
    alignment (default is none). For gradient construction pearson kernel and
    dm approach are default, if concatenate = False (default), only individual gradients are chosen, if concatenate = True, then gradients are aligned and then concatenate to use for analysis"""
    
    ### Gradient Database for just the first gradient
    if concatenate == False:

        gradient_array_database = np.zeros((len(subjects), atlas_size))

        for index, subject in enumerate(subjects):
            subject_matrix = get_conn_matrix(dataframe[subject])

            database_gm = GradientMaps(
                n_components=n_gradients,
                kernel=kernel,
                approach=dimension_reduction,
                random_state=0,
                alignment=alignment,
            )

            if np.all(reference == None):
                database_gm.fit(subject_matrix, sparsity=sparsity)
                gradient_array_database[index] = database_gm.gradients_[:, 0]
            else:
                
                ###############################################################
                ## for local alignment ########################################
                ##database_gm.fit(subject_matrix)
                ##current_grad = database_gm.gradients_[:,0].reshape((160,1))
                ##aligned = brainspace.gradient.alignment.procrustes(current_grad, reference)
                ##gradient_array_database[index] = aligned.reshape((160,))
                
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)
                gradient_array_database[index] = database_gm.aligned_[:, 0]

        gradient_dataframe = pd.DataFrame(gradient_array_database)
        gradient_dataframe_transposed = gradient_dataframe.T
        gradient_dataframe_transposed.columns = list(subjects)
        return gradient_dataframe_transposed
    
    ### Gradient Database if I want to concatenate
    elif concatenate == True:
        gradient_array_database = np.zeros((len(subjects), atlas_size * n_gradients))

        for index, subject in enumerate(subjects):
            subject_matrix = get_conn_matrix(dataframe[subject])
            
            database_gm = GradientMaps(
                n_components=n_gradients,
                kernel=kernel,
                approach=dimension_reduction,
                random_state=0,
                alignment=alignment,
            )
            if np.all(reference == None):
                database_gm.fit(subject_matrix, sparsity=sparsity)
                #gradient_array_database[index] = database_gm.gradients_[:,index]
            else:
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)
                #gradient_array_database[index] = database_gm.aligned_[:,0]

            grad = [None] * n_gradients
            for i in range(n_gradients):
                if np.all(reference == None):
                    # norm_array = np.linalg.norm(database_gm.aligned[:,i])
                    # normalised = database_gm.aligned_[:,i]/norm_array
                    grad[i] = database_gm.gradients_[:, i]  # normalised
                else:
                    # norm_array = np.linalg.norm(database_gm.gradients_[:,i])
                    # normalised = database_gm.gradients_[:,i]/norm_array
                    grad[i] = database_gm.aligned_[:, i]  # normalised

                grad = np.array(grad, dtype=object)
            grad_stacked = np.hstack(grad)

            gradient_array_database[index] = grad_stacked

        gradient_dataframe = pd.DataFrame(gradient_array_database)
        gradient_dataframe_transposed = gradient_dataframe.T
        gradient_dataframe_transposed.columns = list(subjects)
        return gradient_dataframe_transposed
    
    
def identify(target, database, id_method = "spearman"):
    '''

    Parameters
    ----------
    target : pandas dataframe
        holds all subject gradients from session that is to be identified.
        each column should correspond to one subject
    database : pandas dataframe
        holds all subject gradients from session from which to identify
        each column should correspond to one subject
    id_method : correlation method to use for identification. Default is 
    spearman
    Returns
    -------
    identification accuracy

    '''
    count = 0  # the count variable keeps track of iterations with
                            # accurate identification
    for index, subject in enumerate(target.columns):
        all_corr = database.corrwith(
            target.iloc[:, index], method=id_method
        )
                    
        max_value = max(all_corr)
        max_index = all_corr.index[all_corr == max_value]

        if max_index[0] == subject:
            count = count + 1
            
    return count / len(target.columns)