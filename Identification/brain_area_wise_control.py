#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 14:10:18 2021

@author: leonard
"""

### Importing Modules ########################################################

import pandas as pd
import numpy as np
import sys


## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf

### Global Variables/User input ##############################################

database_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
target_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)

kernel = "pearson"
dimension_reduction = "dm"
alignment = "procrustes"
atlas_size = 160

## Loading Data ##############################################################
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)


### Database Construction ####################################################
## In the control condition we take the correlation matrix for each participant
## and calculate the row-wise average correlation for each brain area, this
## "vector/gradient" of average correlations can then be used for identification

areas = []
rates = []
for area in range(atlas_size):

    control_condition_database = np.zeros((len(D_sub), atlas_size))

    for index, subject in enumerate(D_sub):
        subject_matrix = ldf.get_conn_matrix(D_cd_transposed[subject])
        subject_matrix[subject_matrix == 1] = 0

        subset = subject_matrix[area]

        control_condition_database[index] = subset

    control_condition_database = pd.DataFrame(control_condition_database)
    control_condition_database = control_condition_database.T
    control_condition_database.columns = list(D_sub)

    ### Identification Analysis ##################################################
    ### Target subjects and Identification from Database
    ### in this control condition subjects are identified using their average
    ### connectome (row wise average pearson correlation from subject matrix)
    subjects_correctly_identified = []
    count1 = 0
    for index, subject in enumerate(Y_sub):

        subject_connm = ldf.get_conn_matrix(Y_cd_transposed[subject])
        subject_connm[subject_connm == 1] = 0
        subset = pd.DataFrame(subject_connm[area])

        all_corr = control_condition_database.corrwith(
            subset.iloc[:, 0], method="spearman"
        )
        max_value = max(all_corr)
        max_index = all_corr.index[all_corr == max_value]

        if max_index[0] == subject:
            count1 = count1 + 1
            subjects_correctly_identified.append(subject)

    rate = count1 / len(Y_cd_transposed.columns)
    areas.append(area)
    rates.append(rate)

accuracy = {"areas": areas, "rates": rates}
df_accuracy = pd.DataFrame(accuracy)

df_accuracy.to_csv(
    "/home/leonard/projects/master_project_files/output_datafiles/accuracy_by_area_spearman.csv",
    index=False,
)

"""   
    print(
        "At Area " + str(area) + " " + str(rate * 100) + "% of subjects in were accurately predicted from data in "
        "dataset D."
    )
"""
