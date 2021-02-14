#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:21:37 2020

@author: leonard
"""

### Importing Modules ########################################################
import sys

## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf

### Global Variables #########################################################
session1_path = (
    "../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)


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


### Identification Analysis ##################################################
## I used the Identification Code from Finn et al., 2005 as an orientation
rate1 = ldf.identify(target=session2_cd_transposed, database=session1_cd_transposed)
rate2 = ldf.identify(target=session1_cd_transposed, database=session2_cd_transposed)

rate = (rate1 + rate2) / 2
