#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:24:43 2021

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

"""


import sys
import brainspace
from brainspace.gradient import GradientMaps
from scipy.io import loadmat
"""
## loading dosenbach atlas
dosenbach = nilearn.datasets.fetch_coords_dosenbach_2010(ordered_regions=False)
labels = dosenbach.labels

###############################################################################
## cmap
cmap = plt.cm.get_cmap("seismic", 11)
###############################################################################


coords = np.vstack(
    (
        dosenbach.rois["x"],
        dosenbach.rois["y"],
        dosenbach.rois["z"],
    )
).T

spearman_data = "/home/leonard/projects/master_project_files/output_datafiles/control_data/dosenbach_atlas/identification_spearman/spearman_by_area.csv"
pearson_data = "/home/leonard/projects/master_project_files/output_datafiles/control_data/dosenbach_atlas/identification_pearson/area_specific_accuracy.csv"


## loading accuracy data
area_accuracy = pd.read_csv(spearman_data)
area_accuracy.columns = ["brain area", "identification accuracy rate"]
area_sorted = area_accuracy.sort_values(
    by="identification accuracy rate", axis=0, ascending=False, ignore_index=True
)

area_accuracy["labels"] = labels

area_sorted = area_sorted.reset_index()

"""
best = area_accuracy[(area_accuracy["identification accuracy rate"] > 0.5)]
worst = area_accuracy[(area_accuracy["identification accuracy rate"] < 0.08)]
best_indices = best["brain area"]
worst_indices = worst["brain area"]
all_indices = best_indices.append(worst_indices).sort_values()

# 

dmn_area_indices = [
    0,
    3,
    4,
    5,
    6,
    7,
    11,
    13,
    14,
    15,
    17,
    20,
    63,
    72,
    73,
    84,
    85,
    90,
    91,
    92,
    93,
    94,
    105,
    108,
    111,
    112,
    115,
    117,
    124,
    132,
    134,
    136,
    137,
    141,
    146,
]

## get coords for best and worsts areas and markers
dfcoords = pd.DataFrame(coords)

dfcoords = dfcoords[dfcoords.index.isin(best_indices)]
dfcoords = np.array(dfcoords)
df = area_accuracy[area_accuracy.index.isin(best_indices)]
markers = df["identification accuracy rate"]


### plots

## simple line
#plt.plot(area_accuracy["brain area"], area_accuracy["identification accuracy rate"]
## histogram
bins = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
histogram = sns.displot(data=area_accuracy, x = "identification accuracy rate", bins = 10, kde= True)

brain = plotting.plot_markers(
    # area_accuracy["identification accuracy rate"],
    markers,
    dfcoords,
    title="Areas above 50% accuracy",
    node_vmin=0,
    node_vmax=1,
    node_cmap=cmap,
    display_mode="lyrz",
    colorbar=True,
)



brain2 = plotting.plot_markers(
    # area_accuracy["identification accuracy rate"],
    area_accuracy["identification accuracy rate"],
    coords,
    title="Areas by their identification rate",
    node_vmin=0,
    node_vmax=1,
    node_cmap="viridis_r",
    display_mode="lyrz",
    colorbar=True,
)


brain3 = plotting.plot_markers(
    #area_accuracy["identification accuracy rate"],
    area_accuracy["identification accuracy rate"],
    coords,
    #title= "Areas by their identification rate",
    node_vmin=0,
    node_vmax=1,
    node_cmap = cmap,
    display_mode="lr",
    colorbar = False
)



brain.savefig("best_areas.png", dpi=500)
brain2.savefig("yz_acc_by_area.png", dpi=500)
# brain3.savefig("lr_acc_by_area.png", dpi = 250)

"""

# brain.savefig("10_bins_sample_lr.png", dpi = 500)
sns.set_theme(style="white")
fig = sns.catplot(
    data=area_sorted,
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
    plt.axhline, y=0.86, xmin=0, xmax=0.49, c="aqua", label="take max from each node"
)
plt.text(10, 0.87, "max from each ROI")

fig.map(
    plt.axhline,
    y=0.67,
    xmin=0,
    xmax=0.49,
    c="cadetblue",
    label="take sd from each node",
)  # standard deviation from each node
plt.text(10, 0.68, "sd from each ROI")

fig.map(plt.axhline, xmin=0.51, xmax=1, y=0.87, c="red", label="max gradients")
plt.text(90, 0.88, "max gradients **")

fig.map(
    plt.axhline, y=0.41, xmin=0, xmax=0.49, c="blue", label="take mean from each node"
)
plt.text(10, 0.42, "mean from each ROI")

fig.map(plt.axhline, y=0.69, xmin=0.51, xmax=1, c="firebrick", label="mean gradients")
plt.text(90, 0.7, "mean gradients **")

fig.map(plt.axhline, y=0.41, xmin=0.51, xmax=1, c="tomato", label="min gradients")
plt.text(90, 0.42, "min for gradients **")


fig.set_xlabels("ROI-wise connectivity")
fig.set_ylabels("identification accuracy")

# plt.legend(bbox_to_anchor=(1.7, 1), loc='upper right')


# histogram.savefig("10bin_hist.png", dpi = 500)
fig.savefig("control.png", dpi=500)

control = ["whole connectome", "max node", "sd node", "mean node"]
spearman = [0.96, 0.85, 0.67, 0.41]
pearson = [0.98, 0.88, 0.67, 0.46]


controls = {"control": control, "spearman": spearman, "pearson": pearson}

pd.DataFrame(controls).to_csv("controls.csv")
