#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:56:21 2021

@author: leonard
"""

# metric learning for identification
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


# Data is an N by D Matrix
#N = session2_cd_transposed.shape[0]
#D = session2_cd_transposed.shape[1]
##############################################################################


RUNS = 50

MIN, MAX = -1, 1
MAXFEVALS = 50000
TRAINPROP = 0.5
SAVE = True

## Making sure that both dataframes have same participants ###################
for col in session1_cd_transposed.columns:
    if col not in session2_cd_transposed.columns:
        session1_cd_transposed.drop([col], axis = 1, inplace=True)
    
for col in session2_cd_transposed.columns:
    if col not in session1_cd_transposed.columns:
        session2_cd_transposed.drop([col], axis = 1, inplace=True)
##############################################################################
'''
the control dataframes have participants as rows, and brain areas as columns

'''
### summing data ... ##########################################
control1 = ldf.create_control_data(session1_cd_transposed, kind="sum", atlas_size=160).T
control2 = ldf.create_control_data(session2_cd_transposed, kind="sum", atlas_size=160).T

''' the above is a function from my supplementary script load_data_and function.py:
    
def create_control_data(connectivity_data, kind, atlas_size, roi=0):
    control_condition_data = np.zeros((len(connectivity_data.columns), atlas_size))

    for index, subject in enumerate(connectivity_data.columns):
        subject_matrix = get_conn_matrix(connectivity_data[subject])
        subject_matrix[subject_matrix == 1] = 0

        if kind == "mean":
            sub_val = subject_matrix.mean(axis=1)
        elif kind == "std":
            sub_val = subject_matrix.std(axis=1)
        elif kind == "max":
            sub_val = subject_matrix.max(axis=1)
        elif kind == "sum":
            #subject_matrix[subject_matrix < .4] = 0
            sub_val = subject_matrix.sum(axis=1)
        elif kind == "ROI":
            sub_val = subject_matrix[roi]
        elif kind == "count":
            sub_val = ((subject_matrix > .5) | (subject_matrix < -.5)).sum(axis=1)

        control_condition_data[index] = sub_val

    control_condition_data = pd.DataFrame(control_condition_data)
    control_condition_data = control_condition_data.T
    control_condition_data.columns = list(connectivity_data.columns)

    return control_condition_data
'''



'''
if random data is needed for testing
uncomment the next block

'''
#print('generating random data')
#control1 = pd.DataFrame(np.random.randn(398, 5)).T
#control2 = control1 + np.random.normal(0,1,[398, 5]).T


d = control1.shape[0]   #398
n = control1.shape[1]   #160


### Identification without weights
rate1 = ldf.identify(control1.T, control2.T)
rate2 = ldf.identify(control2.T, control1.T)
rate = (rate1 + rate2) / 2

''' the above is a function from my supplementary script load_data_and function.py:
    
def identify(target, database, id_method="spearman"):

     for col in target.columns:
        if col not in database.columns:
            target.drop([col], axis = 1, inplace=True)
    
    for col in database.columns:
        if col not in target.columns:
            database.drop([col], axis = 1, inplace=True)
            
    count = 0   # the count variable keeps track of iterations with
                # accurate identification
    for index, subject in enumerate(target.columns):
        all_corr = database.corrwith(target.iloc[:, index], method=id_method)
        
        max_value = max(all_corr)
        max_index = all_corr.index[all_corr == max_value]
        
        
        if max_index[0] == subject:
            count = count + 1

    return count / len(target.columns)

'''

## cma options 
cmaopts = cma.CMAOptions()
cmaopts.set('bounds', [[MIN]*n, [MAX]*n])
cmaopts.set('maxfevals', MAXFEVALS)


def eval_id(w, data1, data2):
    ''' Evaluate weighted Identification ''' 
    data1 = data1.T
    data2 = data2.T
    n = data1.shape[0]
    d = data1.shape[1]
    
    A = np.array(data1)
    B = np.array(data2)
    w = np.array(w)
    
    ## weighting columns
    Aw = (A.T * w).T
    Bw = (B.T * w).T

    
    #return pd.DataFrame(Aw)
    
    data1 = pd.DataFrame(Aw)
    data2 = pd.DataFrame(Bw)            
    
    data1_ranked = data1.rank().T
    data2_ranked = data2.rank().T

    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data1_ranked)
    distances, indices = nbrs.kneighbors(data2_ranked)
    rate1 = np.mean(indices.T == np.array(list(range(d))))
        
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data2_ranked)
    distances, indices = nbrs.kneighbors(data1_ranked)
    rate2 = np.mean(indices.T == np.array(list(range(d))))
    
    obj = -(rate1 + rate2)/2
    
    return obj

'''
### get the idiff correlation matrix
def corr(df1, df2):
   return pd.concat([df1, df2], axis=1, keys=['df1', 'df2']).corr(method = "spearman").loc['df1', 'df2']

faster implementation with numpy:
    
https://stackoverflow.com/questions/41823728/how-to-perform-correlation-between-two-dataframes-with-different-column-names/41823779
'''

def corr(df1, df2):    
    n = len(df1)
    #df1 = df1.rank()
    #df2 = df2.rank()
    v1, v2 = df1.values, df2.values
    sums = np.multiply.outer(v2.sum(0), v1.sum(0))
    stds = np.multiply.outer(v2.std(0), v1.std(0))
    return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                        df2.columns, df1.columns)

def objective_fun(w, df1, df2):
    ## transpose depends on input data, to get idiff, each column should corre-
    ## spond to a participant
    df1, df2 = df1.T, df2.T
    

    A, B, w = np.array(df1), np.array(df2), np.array(w)

    
    ## weighting columns
    Aw, Bw = (A.T * w).T, (B.T * w).T

    
    df1, df2 = pd.DataFrame(Aw), pd.DataFrame(Bw)
    
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



### generating some weights
weights = [-1 + 1.0/n]*n
#defidiff = objectiveFUN(weights, control1_T, control2_T)
defaccuracy = eval_id(weights, control1, control2)
defidiff = objective_fun(weights, control1, control2)



### display initial settings
print("Initial Settings")
print("-----------------------------------------------------------")
print("Identification without weights:            {}.".format(round(rate,4)))
print("Identification with starting weights:      {}.".format(round(defaccuracy,4)))
print("Initial Idiff                              {}.".format(round(defidiff,4)))
print("MAXFEVALS                                  {}.".format(MAXFEVALS))
print("MIN, MAX                                   {}, {}.".format(MIN, MAX))
print("RUNS                                       {}.".format(RUNS))
print("TRAINPROP                                  {}.".format(TRAINPROP))
#print("idiff with starting weights:               {}.".format(round(defidiff,4)))
print("-----------------------------------------------------------")
print("\n \n Datasets: \n")
print(control1)
print("\n")
print(control2)

print("\n \n initial weights \n " + str(weights) )



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
    
    trainidx = np.random.rand(len(control1)) < TRAINPROP
    train = control1[trainidx]
    train = train.reset_index(drop=True)
    test = control1[~trainidx]
    test = test.reset_index(drop=True)
    train2 = control2[trainidx]
    train2 = train2.reset_index(drop=True)
    test2 = control2[~trainidx]
    test2 = test2.reset_index(drop=True)
    
    
    trainidiff = objective_fun(weights, train, train2)
    ### train
    x0 = weights
    x0 += np.random.uniform(0.0, 0.1/n, n)
    
    res = cma.fmin(objective_fun, x0=x0, sigma0=0.001, options=cmaopts, args=(train, train2), eval_initial_x=True)
    
    bestw = np.array(res[0])
    bests = np.vstack((bests, bestw))
    #bestw *= 10.0
    
    
    ### test
    testidiff = objective_fun(weights, test, test2)
    testwgtidiff = objective_fun(bestw, test, test2)
    
    ### store results
    results[str(run) + "fitness"] = np.float(res[1])
    results[str(run) + "solution"] = np.array(res[0])
    results[str(run) + 'trainidx'] = trainidx
    results[str(run) + 'trainidiff'] = trainidiff
    results[str(run) + 'testidiff'] = testidiff
    results[str(run) + 'testwgtidiff'] = testwgtidiff

    ## wrap up run
    acc = eval_id(bestw, control1, control2)
    stoptime = datetime.timestamp(datetime.now())
    runtime = stoptime-starttime
    avg_runtime = (stoptime - totaltime) / (run+1)
    time_left = (stoptime - totaltime) - (avg_runtime * RUNS)
    
    ### output for this run
    print("\n \n Run: " + str(run) + " out of " + str(RUNS) + " RESULTS")
    print("-----------------------------------------------------------")
    print("idiff training data:            {}.".format(trainidiff))
    print("idiff (weighted) test data:     {}.".format(testwgtidiff))
    print("best function value:            {}.".format(res[1]))
    print("identification accuracy         {}.".format(round(acc,4)))
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
