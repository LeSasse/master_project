#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 18:22:47 2021

@author: leonard
"""

### Importing Modules ########################################################
from brainspace.datasets import load_group_fc, load_parcellation
import nilearn
from nilearn import datasets
import numpy as np
from nilearn import plotting

import pandas as pd
from brainspace.gradient import GradientMaps
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

kernel = "gaussian"
dimension_reduction = "dm"
alignment = "procrustes"
atlas_size = 160

## Loading Data ##############################################################
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)


### Gradient Construction ####################################################
## Reference Gradient for Alignment
## In alignment method one I will align all gradients to one reference gradient.
reference_participant = ldf.get_conn_matrix(D_cd_transposed.iloc[:, 3])
gref = GradientMaps(
    n_components=1, kernel=kernel, approach=dimension_reduction, random_state=0
)
gref.fit(reference_participant)
# grad = gref.gradients_[:,0]

gradient_dataframe_transposed = ldf.create_gradient_database(
    n_gradients=1,
    dataframe=D_cd_transposed,
    subjects=D_sub,
    atlas_size=atlas_size,
    reference=gref.gradients_,
    kernel=kernel,
    dimension_reduction=dimension_reduction,
    alignment=alignment,
    which_gradient=0,
)

gradient_dataframe_transposed2 = ldf.create_gradient_database(
    n_gradients=1,
    dataframe=Y_cd_transposed,
    subjects=Y_sub,
    atlas_size=atlas_size,
    reference=gref.gradients_,
    kernel=kernel,
    dimension_reduction=dimension_reduction,
    alignment=alignment,
    which_gradient=0,
)


grad = np.array(gradient_dataframe_transposed.T.mean())
####### load atlas
dosenbach = nilearn.datasets.fetch_coords_dosenbach_2010(ordered_regions=False)

grad2 = np.array(gradient_dataframe_transposed2.T.mean())

coords = np.vstack(
    (
        dosenbach.rois["x"],
        dosenbach.rois["y"],
        dosenbach.rois["z"],
    )
).T

## plot brain with gradients as marker on nodes

brain_db = plotting.plot_markers(
    grad,
    coords,
    title="(Database) Fifth Average Gradient",
    # node_vmin=0,
    # node_vmax=1,
    node_cmap="viridis_r",
    display_mode="lyrz",
    colorbar=False,
)

brain_target = plotting.plot_markers(
    grad2,
    coords,
    title="(Targets) Fifth Average Gradient",
    # node_vmin=0,
    # node_vmax=1,
    node_cmap="viridis_r",
    display_mode="lyrz",
    colorbar=True,
)


# brain_db.savefig("/home/leonard/projects/master_project_files/fifth_avg_grad_database_viridis.png", dpi = 300)
# brain_target.savefig("/home/leonard/projects/master_project_files/fifth_avg_grad_targets_viridis.png", dpi = 300)
