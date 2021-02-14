#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 08:52:19 2021

@author: leonard
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


global_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/RS1_RS2_dosenbach_participant_ref_concatenation_True_global_alignment_True_spearman.csv"
#local_path = "/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/RS1_RS2_dosenbach_participant_ref_concatenation:True_global_alignment:False_spearman.csv"

all_data = pd.read_csv("/home/leonard/projects/master_project_files/output_datafiles/gradient_data/dosenbach_atlas/identification_spearman/ridiculous1.csv")

#global_data = pd.read_csv(global_path)
#local_data = pd.read_csv(local_path)

#alignment = []
#for i in range(150):
 #   alignment.append("global")
 #   
#global_data["alignment"] = alignment

#alignment = []
#for i in range(150):
#    alignment.append("local")
    
#local_data["alignment"] = alignment


#all_data = local_data.append(global_data)

for index, string in enumerate(all_data["kernels"]):
    all_data["kernels"][(all_data["kernels"] == "normalized_angle")] = "norm_angle"


all_data = all_data.rename(columns={"accuracy": "identification accuracy"})
all_data = all_data.rename(columns={"n_gradients": "n_components"})

#data1 = data[(data["n_components"] == 1)]
#data2 = data[(data["n_components"] == 10)]

#data3 = data1.append(data2)
#data3 = data3.rename(columns={"n_components": "alignment"})
#data3["alignment"][(data3["alignment"] == 1)] = "local"
#data3["alignment"][(data3["alignment"] == 10)] = "global"

#data = data[(data["kernels"] == "cosine")]


### for plotting
palette = sns.color_palette("bright")
## plot accuracy
sns.set_theme(style="white")
ax1 = sns.catplot(
    x="n_components",
    y="identification accuracy",
    hue="kernels",
    col="dimension reduction",
    #row= "alignment",
    data=all_data,
    kind="swarm",
    palette=palette,
)
ax1.set(ylim=(0.5, 1))
ax1.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation="horizontal")
#ax1.set_xticklabels(rotation=30)
ax1.savefig("ridiculous.png", dpi=300)



