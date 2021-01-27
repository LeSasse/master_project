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
    which_gradient=0,
    n_gradients=1,
    concatenate=False,
):
    """dataframe = connectivity data to construct subjects gradient, subjects = list
    of subjects, atlas_size = size of the atlas, reference =  a reference gradient to use for
    alignment (default is none). For gradient construction pearson kernel and
    dm approach are default, if concatenate = False (default), only individual gradients are chosen, if concatenate = True, then gradients are aligned and then concatenate to use for analysis"""
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
                gradient_array_database[index] = database_gm.gradients_[
                    :, which_gradient
                ]
            else:
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)
                gradient_array_database[index] = database_gm.aligned_[:, which_gradient]

        gradient_dataframe = pd.DataFrame(gradient_array_database)
        gradient_dataframe_transposed = gradient_dataframe.T
        gradient_dataframe_transposed.columns = list(subjects)
        return gradient_dataframe_transposed

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
                # gradient_array_database[index] = database_gm.gradients_[:,0]
            else:
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)
                # gradient_array_database[index] = database_gm.aligned_[:,0]

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