#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:19:04 2021

@author: leonard
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import nilearn
from nilearn import datasets
from nilearn import plotting


##############################################################################
### loading and processing data ##############################################

data_path = "/home/leonard/projects/master_project_files/output_datafiles/control_data/dosenbach_atlas/identification_spearman/accuracy_by_area_spearman_FIXED.csv"

data = pd.read_csv(data_path)

## loading dosenbach atlas
dosenbach = nilearn.datasets.fetch_coords_dosenbach_2010(ordered_regions=False)
labels = dosenbach.labels

###############################################################################
## cmap
cmap = plt.cm.get_cmap("seismic", 11)
###############################################################################


data.columns = ["brain area", "identification accuracy rate"]
data_sorted = data.sort_values(
    by="identification accuracy rate", axis=0, ascending=False, ignore_index=True
)

data["labels"] = labels

data_sorted = data_sorted.reset_index()


coords = np.vstack(
    (
        dosenbach.rois["x"],
        dosenbach.rois["y"],
        dosenbach.rois["z"],
    )
).T
##############################################################################


'''
##############################################################################
### plotting all controls ####################################################

sns.set_theme(style="white")
fig = sns.catplot(
    data=data_sorted,
    x="index",
    y="identification accuracy rate",
    color="lightskyblue",
    kind="bar",
    
    
    
    
    legend_out=True,
)

fig.set_xticklabels([])

fig.map(plt.axhline, y=0.96, c="gold", label="whole connectome")  # whole connectome
plt.text(60, 0.97, "whole connectome")

fig.map(
    plt.axhline, y=0.87, xmin=0, xmax=0.49, c="aqua"
)
plt.text(10, 0.88, "max from each ROI")

fig.map(
    plt.axhline,
    y=0.69,
    xmin=0,
    xmax=0.49,
    c="cadetblue",
    label="take sd from each node",
)  # standard deviation from each node
plt.text(10, 0.7, "sd from each ROI")

fig.map(plt.axhline, xmin=0.51, xmax=1, y=0.87, c="red")
plt.text(90, 0.88, "max gradients **")

fig.map(
    plt.axhline, y=0.40, xmin=0, xmax=0.49, c="blue"
)
plt.text(10, 0.41, "mean from each ROI")

fig.map(plt.axhline, y=0.68, xmin=0.51, xmax=1, c="firebrick")
plt.text(90, 0.69, "mean gradients **")

fig.map(plt.axhline, y=0.42, xmin=0.51, xmax=1, c="tomato")
plt.text(90, 0.43, "min gradients **")


fig.set_xlabels("ROI-wise connectivity")
fig.set_ylabels("identification accuracy")

#fig.savefig("control.png", dpi=500)
##############################################################################
'''

##############################################################################
### plotting identification accuracy of individual connectivity profiles back 
### on the brain
brain = plotting.plot_markers(
    data["identification accuracy rate"],
    coords,
    title="Areas by their identification rate",
    node_vmin=0,
    node_vmax=1,
    node_cmap=cmap,
    display_mode="lyrz",
    colorbar=True,
)

#brain.savefig("roi_wise.png", dpi=500)