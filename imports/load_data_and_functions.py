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
    """
    Parameters
    ----------
    data_path : TYPE
        DESCRIPTION.

    Returns
    -------
    D : Data.
    D_sub : array
        subject list
    D_connectivity_data : pandas dataframe
        connectivity data.
    D_cd_transposed : pandas dataframe
        transposed conenctivity data.

    """
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
    """
    Parameters
    ----------
    X : pandas series
        contains pandas series of concatenated connectivity data.

    Returns
    -------
    numpy ndarray
        square connectivity matrix.
    """

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
    global_alignment=True,
):
    """
    Parameters
    ----------
    dataframe : all participants concatenated connectivity data in a pandas dataframe,
    one column should be a participant
    subjects : np.ndarray or pandas series
        subject list.
    atlas_size : integer
        size of atlas
    reference : set of gradients to use as reference in alignment, optional
        DESCRIPTION. The default is None.
    kernel : string, optional
        Kernel to use in gradient construction. The default is "pearson".
    dimension_reduction : string, optional
        dimensionality reduction for brainspace. The default is "dm".
    alignment : string, optional
         The default is "procrustes".
    sparsity : TYPE, optional
        DESCRIPTION. The default is 0.9.
    n_gradients : TYPE, optional
        DESCRIPTION. The default is 1.
    concatenate : TYPE, optional
        DESCRIPTION. The default is False.
    global_alignment : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    gradient_dataframe_transposed : TYPE
        DESCRIPTION.

    """

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
            elif global_alignment == True:
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)
                gradient_array_database[index] = database_gm.aligned_[:, 0]
            elif global_alignment == False:
                ###############################################################
                ## for local alignment ########################################
                database_gm.fit(subject_matrix)
                current_grad = database_gm.gradients_[:, 0].reshape((160, 1))
                aligned = brainspace.gradient.alignment.procrustes(
                    current_grad, reference
                )
                gradient_array_database[index] = aligned.reshape((160,))

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
            else:
                database_gm.fit(subject_matrix, reference=reference, sparsity=sparsity)

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


def identify(target, database, id_method="spearman"):
    """

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

    """
    count = 0  # the count variable keeps track of iterations with
    # accurate identification
    for index, subject in enumerate(target.columns):
        all_corr = database.corrwith(target.iloc[:, index], method=id_method)

        max_value = max(all_corr)
        max_index = all_corr.index[all_corr == max_value]

        if max_index[0] == subject:
            count = count + 1

    return count / len(target.columns)


def create_control_data(connectivity_data, kind, atlas_size, roi=0):
    """

    Parameters
    ----------
    connectivity_data : pandas dataframe
        concatenated connectivity data, each column represents one subject
        subject id's should be column names'
    kind : string
        "mean": take mean from every row.
        "std": take std from every row.
        "max": take max from every row.
        "ROI": take one ROI connectivity profile at a time. Specify which row (index)
    atlas_size : integer
        number of rois in atlas

    Returns
    -------
    control_condition_data : pandas dataframe
        each column represents control vector for a subject, to be used in
        identification

    """
    control_condition_data = np.zeros((len(connectivity_data.columns), atlas_size))

    for index, subject in enumerate(connectivity_data.columns):
        subject_matrix = get_conn_matrix(connectivity_data[subject])
        subject_matrix[subject_matrix == 1] = 0

        if kind == "mean":
            sub_val = subject_matrix.mean(axis=1)
        elif kind == "std":
            sub_val = subject_matrix.std(axis=1)
        elif kind == "max":
            sub_val = subject_matrix.max(axis=1)
        elif kind == "ROI":
            sub_val = subject_matrix[roi]

        control_condition_data[index] = sub_val

    control_condition_data = pd.DataFrame(control_condition_data)
    control_condition_data = control_condition_data.T
    control_condition_data.columns = list(connectivity_data.columns)

    return control_condition_data
