### Importing Modules ########################################################

import pandas as pd
import numpy as np
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

control_condition_database = np.zeros((len(D_sub), atlas_size))

for index, subject in enumerate(D_sub):
    subject_matrix = ldf.get_conn_matrix(D_cd_transposed[subject])
    subject_matrix[subject_matrix == 1] = 0
    sub_avg = subject_matrix.mean(axis=1)  # / subject_matrix.mean(axis = 1)

    # for random sample :D : control_condition_database[index] = np.random.choice(subject_matrix.ravel(),160, replace = False)
    control_condition_database[index] = sub_avg

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

    subject_dataframe = pd.DataFrame(
        subject_connm.mean(axis=1)
    )  # / subject_connm.mean(axis = 1))
    # for median subject_dataframe = pd.DataFrame(np.median(subject_connm, axis = 1))
    # for random sample: subject_dataframe = np.random.choice(subject_connm.ravel(),160, replace = False)
    # for random sample: subject_dataframe = pd.DataFrame(subject_dataframe)

    all_corr = control_condition_database.corrwith(
        subject_dataframe.iloc[:, 0], method="pearson"
    )
    max_value = max(all_corr)
    max_index = all_corr.index[all_corr == max_value]

    if max_index[0] == subject:
        count1 = count1 + 1
        subjects_correctly_identified.append(subject)

rate = count1 / len(Y_cd_transposed.columns)


print(
    str(rate * 100) + "% of subjects in were accurately predicted from data in "
    "dataset D."
)
