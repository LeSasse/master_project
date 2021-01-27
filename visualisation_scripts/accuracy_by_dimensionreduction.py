#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:09:55 2021

@author: leonard
"""

import pandas as pd
import seaborn as sns

## loading data

spearman_oneg_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/spearman_one_gradient.csv"
spearman_n_gradients_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/spearman_n_gradients.csv"
which_gradient_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/which_gradient.csv"
to_beappended = "/home/leonard/projects/master_project_files/output_datafiles/spearman_n_gradients.csv"

accuracy_table = pd.read_csv(spearman_n_gradients_path)


accuracy_table2 = pd.read_csv(to_beappended)


"""
accuracy_table = accuracy_table[accuracy_table["sparsity"] == 0.9]


"""
# gradient = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# accuracy_table2["n_gradients"] = gradient

accuracy_table = accuracy_table.append(accuracy_table2)


## replace normalised_angle with norm_angle
for index, string in enumerate(accuracy_table["kernels"]):
    accuracy_table["kernels"][
        (accuracy_table["kernels"] == "normalized_angle")
    ] = "norm_angle"


palette = sns.color_palette("bright")
## plot accuracy
sns.set_theme(style="white")
ax1 = sns.catplot(
    x="kernels",
    y="accuracy",
    hue="n_gradients",
    col="dimension reduction",
    data=accuracy_table,
    kind="bar",
    palette=palette,
)
ax1.set(ylim=(0, 1))

ax1.set_xticklabels(rotation=30)

ax1.savefig("ide_gradient_accuracy_control.png", dpi=500)


# n_gradients = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# accuracy_table2["n_gradients"] = n_gradients
