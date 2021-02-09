#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:13:40 2021

@author: leonard

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/varying_n_components_1to32.csv"

n_extracts = pd.read_csv(path)

## replace normalised_angle with norm_angle
for index, string in enumerate(n_extracts["kernels"]):
    n_extracts["kernels"][(n_extracts["kernels"] == "normalized_angle")] = "norm_angle"


n_extracts = n_extracts.rename(columns={"accuracy": "identification accuracy"})
n_extracts = n_extracts.rename(columns={"n_gradients": "n_components"})

n_extracts = n_extracts[(n_extracts["n_components"] == 10)]


palette = sns.color_palette("bright")
## plot accuracy
sns.set_theme(style="white")
ax1 = sns.catplot(
    x="n_components",
    y="identification accuracy",
    hue="kernels",
    col="dimension reduction",
    # row= concatenation
    data=n_extracts,
    kind="point",
    palette=palette,
)
ax1.set(ylim=(0, 1))

ax1.set_xticklabels(rotation=30)


# ax1.savefig("n_components.png", dpi=500)

pca = n_extracts[(n_extracts["dimension reduction"] == "pca")]

le = n_extracts[(n_extracts["dimension reduction"] == "le")]

dm = n_extracts[(n_extracts["dimension reduction"] == "dm")]

pearson = n_extracts[(n_extracts["kernels"] == "pearson")]

spearman = n_extracts[(n_extracts["kernels"] == "spearman")]

norm = n_extracts[(n_extracts["kernels"] == "norm_angle")]

gaussian = n_extracts[(n_extracts["kernels"] == "gaussian")]

cosine = n_extracts[(n_extracts["kernels"] == "cosine")]
