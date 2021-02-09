#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:09:55 2021

@author: leonard
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## loading data

spearman_oneg_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/spearman_one_gradient.csv"
# spearman_n_gradients_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/spearman_n_gradients.csv"
# which_gradient_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/which_gradient.csv"
# to_beappended = "/home/leonard/projects/master_project_files/output_datafiles/spearman_n_gradients.csv"

# faulty = pd.read_csv("faulty?.csv")
# fault2 = pd.read_csv("spearman_not_concatenated_n_gradients(1to15).csv")
# fault2 = fault2[(fault2["n_gradients"] < 8)]
#
# fault2 = fault2.rename(columns={"n_gradients": "n_components"})
# fault2 = fault2[(fault2["dimension reduction"] == "pca")]
# faulty = fault2.append(faulty)
# accuracy_table = pd.read_csv(spearman_n_gradients_path)

# data1 = fault2[(fault2["kernels"] == "gaussian")]
# data2 = fault2[(fault2["kernels"] != "gaussian")]
#
# data3 = data2.append(data1)
# accuracy_table2 = pd.read_csv(to_beappended)

data = pd.read_csv(spearman_oneg_path)
data = data[(data["sparsity"] == 0.9)]
"""
accuracy_table = accuracy_table[accuracy_table["sparsity"] == 0.9]


"""
# gradient = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# accuracy_table2["n_gradients"] = gradient
"""
#accuracy_table = accuracy_table.append(accuracy_table2)
concatenation = []
for i in range(478):
    concatenation.append(False)

faulty["concatenation"] = concatenation

faulty = faulty[(faulty["n_gradients"] < 21)]
"""
## replace normalised_angle with norm_angle
for index, string in enumerate(data["kernels"]):
    data["kernels"][(data["kernels"] == "normalized_angle")] = "norm_angle"


data = data.rename(columns={"accuracy": "identification accuracy"})

palette = sns.color_palette("bright")
## plot accuracy
sns.set_theme(style="white")
ax1 = sns.catplot(
    x="kernels",
    y="identification accuracy",
    hue="dimension reduction",
    # col="dimension reduction",
    # row= concatenation
    data=data,
    kind="bar",
    palette=palette,
)
ax1.set(ylim=(0, 1))

ax1.set_xticklabels(rotation=30)


ax1.savefig("pca.png", dpi=750)


# n_gradients = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# accuracy_table2["n_gradients"] = n_gradients

# faulty.to_csv("varying_n_components_1to32.csv")
