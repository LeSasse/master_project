#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:52:09 2021

@author: leonard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:24:43 2021

@author: leonard
"""
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nilearn
from nilearn import datasets
from nilearn import plotting
import seaborn as sns
import sys
import brainspace
from brainspace.gradient import GradientMaps
import scipy
from scipy.stats import random_correlation

## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf


### Global Variables/User input ##############################################

database_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)

dosenbach = nilearn.datasets.fetch_coords_dosenbach_2010()
labels = dosenbach["labels"]

"""
example_p = ldf.get_conn_matrix(D_cd_transposed.iloc[:, 3])
corr_plot = plotting.plot_matrix(
    example_p,
    figure=(15, 15),
    labels=labels,
    colorbar=False,
    reorder=True,
    vmin=-1,
    vmax=1,
    cmap = "gray"
)


affinity_matrix = brainspace.gradient.kernels.compute_affinity(
    example_p, kernel="pearson"
)


aff_plot = plotting.plot_matrix(
    affinity_matrix, labels=labels, figure=(15, 15), reorder=True, vmin=-1, vmax=1
)

"""

data = np.array([0.7, 0.4, -0.3, -0.9, 0.9, 0.5, 0.9, 0.76, 0.2, -0.3])

corr_matrix = ldf.get_conn_matrix(data)
labels = np.array(["one area", "another area", "yet another area", "area 4", "area 51"])

corr_plot = plotting.plot_matrix(
    corr_matrix, figure=(15, 15), colorbar=False, cmap="binary"
)

plt.savefig("controls_matrix.png", dpi=250)

"""

np.random.seed(514)
x = random_correlation.rvs((0.5, 0.8, 1.2, 1.5))
"""
