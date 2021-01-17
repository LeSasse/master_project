#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 16:24:43 2021

@author: leonard
"""
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nilearn
from nilearn import datasets
from nilearn import plotting

## loading dosenbach atlas
dosenbach = nilearn.datasets.fetch_coords_dosenbach_2010()


coords = np.vstack(
    (
        dosenbach.rois["x"],
        dosenbach.rois["y"],
        dosenbach.rois["z"],
    )
).T


## loading accuracy data
area_accuracy = pd.read_csv("area_specific_accuracy.csv")
area_accuracy.columns = ["brain area", "identification accuracy rate"]

## statistics
descriptives = area_accuracy["identification accuracy rate"].describe()

plt.plot(area_accuracy["brain area"], area_accuracy["identification accuracy rate"])


nilearn.plotting.plot_markers(
    area_accuracy["identification accuracy rate"],
    coords,
    title="accuracy",
)
