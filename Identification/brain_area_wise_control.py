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
from datetime import datetime

## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf

### Global Variables/User input ##############################################

session1_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)

kernel = "pearson"
dimension_reduction = "dm"
alignment = "procrustes"
atlas_size = 160

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


areas = []
rates = []

iteration_count = 0
total_iterations = atlas_size
totaltime = datetime.timestamp(datetime.now())
for area in range(atlas_size):

    starttime = datetime.timestamp(datetime.now())

    control_condition_session1 = ldf.create_control_data(
        connectivity_data=session1_cd_transposed,
        kind="ROI",
        atlas_size=atlas_size,
        roi=area,
    )
    control_condition_session2 = ldf.create_control_data(
        connectivity_data=session2_cd_transposed,
        kind="ROI",
        atlas_size=atlas_size,
        roi=area,
    )

    rate1 = ldf.identify(
        target=control_condition_session2, database=control_condition_session1
    )
    rate2 = ldf.identify(
        target=control_condition_session1, database=control_condition_session2
    )

    rate = (rate1 + rate2) / 2

    ### dataframing relevant information
    iteration_count = iteration_count + 1
    areas.append(area)
    rates.append(rate)
    stoptime = datetime.timestamp(datetime.now())
    print(str(iteration_count))
    print(" out of " + str(total_iterations) + " iterations.")
    print("(this round took: " + str(stoptime - starttime) + " sec )")
    print("(     total took: " + str(stoptime - totaltime) + " sec )")
    print(
        "(  avg per round: " + str((stoptime - totaltime) / iteration_count) + " sec )"
    )

accuracy = {"areas": areas, "rates": rates}
df_accuracy = pd.DataFrame(accuracy)

df_accuracy.to_csv(
    "accuracy_by_area_spearman_FIXED.csv",
    index=False,
)
