#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:45:20 2021

@author: leonard
"""

import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import cma
from sklearn.neighbors import NearestNeighbors


## my own imports
sys.path.append("../imports")
import load_data_and_functions as ldf


print("Starting...")


### Global Variables/User input ##############################################

### data file paths ##########################################################
session1_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)


## Loading Data ##############################################################
##############################################################################

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


## Making sure that both dataframes have same participants ###################
for col in session1_cd_transposed.columns:
    if col not in session2_cd_transposed.columns:
        session1_cd_transposed.drop([col], axis = 1, inplace=True)
    
for col in session2_cd_transposed.columns:
    if col not in session1_cd_transposed.columns:
        session2_cd_transposed.drop([col], axis = 1, inplace=True)
##############################################################################

df1 = session1_cd_transposed.T
df2 = session2_cd_transposed.T

d = df1.shape[0]
n = 160 

##############################################################################


## cma options ###############################################################
RUNS = 50

MIN, MAX = -1, 1
MAXFEVALS = 500
SIGMA = 0.01
TRAINPROP = 0.5
SAVE = True

cmaopts = cma.CMAOptions()
cmaopts.set('bounds', [[MIN]*n, [MAX]*n])
cmaopts.set('maxfevals', MAXFEVALS)

w = [1.0/n]*n
##############################################################################





def corr(df1, df2):
    '''
    returns all possible correlations between columns from two dataframes.
    This is the differential identifiability matrix.
    '''    
    n = len(df1)
    #df1 = df1.rank()
    #df2 = df2.rank()
    v1, v2 = df1.values, df2.values
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                        df2.columns, df1.columns)


def weight_and_sum(df, w):
    w = np.array(w)
    n = len(w)
    
    weighted_summed = np.zeros((len(df.columns), n))
    for index, subject in enumerate(df.columns):
        subject_matrix = ldf.get_conn_matrix(df[subject])
        subject_matrix[subject_matrix == 1] = 0
        #subject_matrix[((subject_matrix < .25)) & ((subject_matrix > -.25))] = 0
    
        subject_weighted = subject_matrix * w
        subject_vector = subject_weighted.sum(axis=1)
    
        weighted_summed[index] = subject_vector
    
    weighted_summed = pd.DataFrame(weighted_summed)
    weighted_summed = weighted_summed.T
    weighted_summed.columns = list(df.columns)
    return weighted_summed
    
def get_idiff(df1, df2):
    ## getting idiff
    df1, df2 = df1.rank(), df2.rank()
    idiff_matrix = corr(df1, df2)
    idiff_matrix = np.array(idiff_matrix)
    
    ## corr_self
    diag = np.diagonal(idiff_matrix)
    
    ## corr_others
    mask = np.ones(idiff_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    off_diag = idiff_matrix[mask]

    obj = -100 * (np.mean(diag) - np.mean(off_diag))
    
    return obj
    

def eval_id(w, df1, df2):
    ''' 
    Evaluate weighted Identification
    --------------------------------
    Individuals FC Matrices are weighted and then summed up. Then these vectors
    are used for identification. Output is accuracy.
    ''' 
              
    
    df1 = df1.T
    df2 = df2.T
    
    df1 = weight_and_sum(df1, w)
    df2 = weight_and_sum(df2, w)
    
    D = df1.shape[1]
    
    df1 = df1.rank().T
    df2 = df2.rank().T
    
    
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(df1)
    distances, indices = nbrs.kneighbors(df2)
    rate1 = np.mean(indices.T == np.array(list(range(D))))
        
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(df2)
    distances, indices = nbrs.kneighbors(df1)
    rate2 = np.mean(indices.T == np.array(list(range(D))))
    
    obj = (rate1 + rate2)/2
    
    return obj


def objective_fun(w, df1, df2):
    '''
    Objective Function to optimise Idiff

    '''
    df1, df2 = df1.T, df2.T
    
    df1 = weight_and_sum(df1, w)
    df2 = weight_and_sum(df2, w)
    
    obj = get_idiff(df1, df2)
    
    return obj
    



defaccuracy = eval_id(w, df1, df2)
defidiff = objective_fun(w, df1, df2)

### display initial settings
print("Initial Settings")
print("-----------------------------------------------------------")
print("Identification with starting weights:      {}.".format(round(defaccuracy,4)))
print("Initial Idiff                              {}.".format(round(defidiff,4)))
print("MAXFEVALS                                  {}.".format(MAXFEVALS))
print("MIN, MAX                                   {}, {}.".format(MIN, MAX))
print("RUNS                                       {}.".format(RUNS))
print("SIGMA                                      {}.".format(SIGMA))
print("TRAINPROP                                  {}.".format(TRAINPROP))
#print("idiff with starting weights:               {}.".format(round(defidiff,4)))
print("-----------------------------------------------------------")
print("\n \n Datasets: \n")
print(df1)
print("\n")
print(df2)

print("\n \n initial weights \n " + str(w))



'''
running the optimisation:
    
'''

results = {}

seed = np.random.choice(99999, 1)[0]
print("\n seed: " + str(seed))
np.random.seed(seed)
results['seed'] = seed
bests = np.empty((0, n))
totaltime = datetime.timestamp(datetime.now())
### running iterative optimisation
for run in range(RUNS):
    starttime = datetime.timestamp(datetime.now())
    print("\n \n starting run " + str(run))
    print("-----------------------------------------------------------")
    
    ### train-test split
    
    trainidx = np.random.rand(len(df1)) < TRAINPROP
    train = df1[trainidx]
    train = train.reset_index(drop=True)
    test = df1[~trainidx]
    test = test.reset_index(drop=True)
    train2 = df2[trainidx]
    train2 = train2.reset_index(drop=True)
    test2 = df2[~trainidx]
    test2 = test2.reset_index(drop=True)
    
    trainidiff = objective_fun(w, train, train2)
    testidiff = objective_fun(w, test, test2)
    
    ### train
    x0 = w
    x0 += np.random.uniform(0.0, 0.01/n, n)
    
    res = cma.fmin(objective_fun, x0=x0, sigma0=SIGMA, options=cmaopts, args=(train, train2), eval_initial_x=True)
    
    bestw = np.array(res[0])
    bests = np.vstack((bests, bestw))
    #bestw *= 10.0
    
    
    ### test
    testwgtidiff = objective_fun(bestw, test, test2)
    
    ### store results
    results[str(run) + "fitness"] = np.float(res[1])
    results[str(run) + "solution"] = np.array(res[0])
    results[str(run) + 'trainidx'] = trainidx
    results[str(run) + 'trainidiff'] = trainidiff
    results[str(run) + 'testidiff'] = testidiff
    results[str(run) + 'testwgtidiff'] = testwgtidiff

    ## wrap up run
    acc_train = eval_id(bestw, train, train2)
    acc_test = eval_id(bestw, test, test2)
    acc = eval_id(bestw, df1, df2)
    
    stoptime = datetime.timestamp(datetime.now())
    runtime = stoptime-starttime
    avg_runtime = (stoptime - totaltime) / (run+1)
    time_left = (stoptime - totaltime) - (avg_runtime * RUNS)
    
    ### output for this run
    print("\n \n Run: " + str(run) + " out of " + str(RUNS) + " RESULTS")
    print("-----------------------------------------------------------")
    print("idiff training data:            {}.".format(trainidiff))
    print("best function value:            {}.".format(res[1]))
    print("idiff test data (initial):      {}.".format(testidiff))
    print("idiff test data (best weights): {}.".format(testwgtidiff))
    
    print("ident. acc. train (weighted)    {}.".format(round(acc_train,4)))
    print("ident. acc. test (weighted)     {}.".format(round(acc_test,4)))
    print("ident. acc. whole (weighted)    {}.".format(round(acc,4)))
    print("-----------------------------------------------------------")
    print("Run Timestamp (in seconds)")
    print("-----------------------------------------------------------")
    print("Time for this run:              {} seconds.".format(round(runtime, 4)))
    print("Avg run time:                   {} seconds.".format(round(avg_runtime,4)))
    print("Total time so far:              {} seconds.".format(round((stoptime - totaltime),4)))
    print("Estimated Time left             {} seconds.".format(round(time_left, 4)))
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------END RUN-----------------------------")
    

results['bestmean'] = np.mean(bests, 0)
print('average of bests solutions:')
print("-----------------------------------------------------------")
print( results['bestmean'] )

if SAVE:
    now = datetime.now()
    fname = 'metric_learning_cma_' + now.strftime("%m%d%Y_%H%M%S") + '.pickle'
    print("saving results in: " + fname)
    with open(fname,'wb') as f:
        pickle.dump(results, f)
