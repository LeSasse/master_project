#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:36:29 2021

@author: leonard
"""

### Importing Modules ########################################################

import pandas as pd
from brainspace.gradient import GradientMaps
import sys

## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf

### Global Variables/User input ##############################################

database_path = (
    "../../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
target_path = (
    "../../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)

kernels = ["pearson", "spearman", "normalized_angle"]
dimension_reductions = ["pca", "dm", "le"]
alignment = "procrustes"
atlas_size = 160

## Loading Data ##############################################################
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)

kernel_used = []
dreduction_used = []
rates = []

for kernel in kernels:
    for dimension_reduction in dimension_reductions:

        ### Gradient Construction ####################################################
        ## Reference Gradient for Alignment
        ## In alignment method one I will align all gradients to one reference gradient.
        reference_participant = ldf.get_conn_matrix(D_cd_transposed.iloc[:, 3])
        gref = GradientMaps(
            n_components=1, kernel=kernel, approach=dimension_reduction, random_state=0
        )
        gref.fit(reference_participant)

        ### Database Gradient Construction ###########################################
        gradient_dataframe_transposed = ldf.create_gradient_database(
            dataframe=D_cd_transposed,
            subjects=D_sub,
            atlas_size=atlas_size,
            reference=gref.gradients_,
            kernel=kernel,
            dimension_reduction=dimension_reduction,
            alignment=alignment,
        )

        ### Identification Analysis ##################################################
        ### Target Gradients and Identification from Database
        subjects_correctly_identified = []
        count1 = 0
        for index, subject in enumerate(Y_sub):

            subject_connm = ldf.get_conn_matrix(Y_cd_transposed[subject])
            ## gradient construction using the matrices for each subject to be "identi-
            ## fied" in the "target set" Y
            gm = GradientMaps(
                n_components=1,
                kernel=kernel,
                approach=dimension_reduction,
                random_state=0,
                alignment=alignment,
            )
            gm.fit(subject_connm, reference=gref.gradients_)
            subject_gradient = gm.aligned_[:, 0]
            subject_gradient_dataframe = pd.DataFrame(subject_gradient)

            all_corr = gradient_dataframe_transposed.corrwith(
                subject_gradient_dataframe.iloc[:, 0], method="pearson"
            )
            max_value = max(all_corr)
            max_index = all_corr.index[all_corr == max_value]

            if max_index[0] == subject:
                count1 = count1 + 1
                subjects_correctly_identified.append(subject)

        rate = count1 / len(Y_cd_transposed.columns)
        kernel_used.append(kernel)
        dreduction_used.append(dimension_reduction)
        rates.append(rate)

accuracy = {
    "kernels": kernel_used,
    "dimension reduction": dreduction_used,
    "accuracy": rates,
}
df_accuracy = pd.DataFrame(accuracy)
df_accuracy.to_csv(
    "/home/leonard/projects/master_project_files/output_datafiles/gradient_identification_accuracy.csv",
    index=False,
)
