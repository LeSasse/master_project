#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:45:06 2021

@author: leonard
"""

### Importing Modules ########################################################

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import ProcrustesAlignment
import sys
from datetime import datetime
import time
import brainspace

## my own imports
sys.path.append("master_project/imports")
import load_data_and_functions as ldf
import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator

from brainspace.gradient.alignment import ProcrustesAlignment
from brainspace.gradient.kernels import compute_affinity
from brainspace.gradient.embedding import PCAMaps, LaplacianEigenmaps, DiffusionMaps


### Global Variables/User input ##############################################

### data file paths ##########################################################
session1_path = (
    "dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)

##############################################################################
## Loading Data ##############################################################
(
    session1_data,
    session1_sub,
    session1_connectivity_data,
    session1_cd_transposed,
) = ldf.load_data(session1_path)
(
    session2_data,
    session2_sub,
    session2_connectivity_data,
    session2_cd_transposed,
) = ldf.load_data(session2_path)


'''
Let's first align the gradients using GradientMaps'

'''
n_grads = 3 

##############################################################################
## reference #################################################################

reference_participant = ldf.get_conn_matrix(
    session1_cd_transposed.iloc[:, 0]
)                        
gref = GradientMaps(
    n_components=n_grads,
    kernel="gaussian",
    approach="pca",
    random_state=0,
)
gref.fit(reference_participant)

##############################################################################
## source ####################################################################

source_participant = ldf.get_conn_matrix(
    session1_cd_transposed.iloc[:, 1]
)                        
gsour = GradientMaps(
    n_components=n_grads,
    kernel="gaussian",
    approach="pca",
    random_state=0,
    alignment="procrustes"
)
gsour.fit(source_participant, reference=gref.gradients_)
##############################################################################
'''
Let's now align gradients using procrustes()'

'''

ref_grad = gref.gradients_
source_grad = gsour.gradients_
aligned = brainspace.gradient.alignment.procrustes(
    source_grad, ref_grad
)

'''
Let's see if results are the same'

'''

print(np.mean(gsour.aligned_ == aligned))
##############################################################################

'''
Results are the same for both methods (actually not always but at 1 and 10
components they are the same), so let's take a look at the procrustes()
function, in order to understand brainspace alignment
the following is the code from procrustes():
'''

source = gsour.gradients_
target = gref.gradients_

a = target.T
b = a.dot(source).T
u, w, vt = np.linalg.svd(b)

t = u.dot(vt)

aligned1 = source.dot(t)

'''
Let's see if results are the same'

'''

print(np.mean(aligned == aligned1))
print(np.mean(gsour.aligned_ == aligned1))