#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:21:37 2020

@author: leonard
"""

from scipy.io import loadmat
import pandas as pd
from tabulate import tabulate


## data is stored in variable 'res', consists of 399 participants, it has the
## following columns:
## sub == subject ID
## connm == all possible temporal correlations (or functional connectivity)
## for every region resulting in 12720 datapoints for each subject (160*159/2)
## alff == Amplitude of Low Frequency Fluctuations,
## falff == fractional Amplitude of Low Frequency Fluctuations,
## reho == regional homogeneity

D = loadmat("/home/leonard/projects/master_project_files/dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat")




#print(type(D['res'][0][0]),D['res'][0][0].shape)

#print([item.flat[0] for item in D['res'][0][0]])

## converting data into a pandas data frame


data = [[row.flat[0] for row in line] for line in D['res'][0]]
columns = ['sub', 'connm', 'alff', 'falff', 'reho']
data_frame = pd.DataFrame(data, columns=columns)

print(tabulate(data_frame, headers = 'keys', tablefmt = 'psql'))



