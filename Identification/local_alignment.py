#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 11:29:01 2021

@author: leonard
"""

### Importing Modules ########################################################

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
import sys
from datetime import datetime
import time
import brainspace

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


##############################################################################
### customise gradient settings ##############################################
##############################################################################
alignment = "procrustes"
atlas_size = 160


sparsity = [0.9]
kernels = ["pearson", "spearman", "normalized_angle", "gaussian", "cosine"]
dimension_reductions = ["pca", "dm", "le"]
concatenate = False

## Identification method (spearman or pearson)
id_method = "spearman"

## for num_grads, if concatenation = true, then this defines which gradients
## concatenated, if concatenation = false then principal gradient will still be chosen
## and only n_components argument goes up
## num_grads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
num_grads = [1, 2, 3, 4, 5, 6 ]
##############################################################################


##############################################################################
#### these lists track the settings for every iteration and then are used for
#### a dataframe with results
kernel_used = []
dreduction_used = []
rates = []
ngradients = []
gradient_used = []
sparsities = []
iteration_count = 0
concatenation = []
total_iterations = (
    len(sparsity) * len(num_grads) * len(kernels) * len(dimension_reductions)
)
##############################################################################


##############################################################################
### path and name for output data
### filename as follows atlasname + concatenated (or not) + identification method
output_file = "RS1_RS2_dosenbach_avg_connectivity_reference" + str(concatenate) + "_" + str(id_method) + ".csv"
##############################################################################


##############################################################################
## ask for confirmation of settings
print("total iterations == " + str(total_iterations))
print("atlas size == " + str(atlas_size))
print("number of gradients to iterate " + str(num_grads))
print("concatenate == " + str(concatenate))


##############################################################################
## Loading Data ##############################################################
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

##############################################################################
## Looping over different settings
totaltime = datetime.timestamp(datetime.now())
for spars in sparsity:
    for n_gradients in num_grads:
        for kernel in kernels:
            for dimension_reduction in dimension_reductions:

                starttime = datetime.timestamp(datetime.now())

                ##############################################################
                ### Gradient Construction

                ## Reference Gradient for Alignment ##########################
                ## In this alignment method I will align all gradients to one
                ## reference gradient from a reference participant.
                #reference_participant = ldf.get_conn_matrix(
                #    session1_cd_transposed.iloc[:, 0])
                
                reference_participant = ldf.get_conn_matrix(
                    session1_cd_transposed.iloc[:,0]
                )
                gref = GradientMaps(
                    n_components=10,
                    kernel=kernel,
                    approach=dimension_reduction,
                    random_state=0,
                )
                gref.fit(reference_participant)