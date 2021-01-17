#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 10:09:55 2021

@author: leonard
"""

import pandas as pd
import seaborn as sns

## loading data
accuracy_table = pd.read_csv(
    "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/gradient_identification_accuracy.csv"
)


## plot accuracy
sns.set_theme(style="whitegrid")
ax = sns.barplot(
    x="kernels", y="accuracy", hue="dimension reduction", data=accuracy_table
)
