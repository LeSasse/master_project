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
MIN, MAX = -1.0, 1.0
MAXFEVALS = 500
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


### summing data ... ##########################################
control1 = ldf.create_control_data(session1_cd_transposed, kind="sum", atlas_size=160)
control2 = ldf.create_control_data(session2_cd_transposed, kind="sum", atlas_size=160)

'''
if random data is needed for testing
uncomment the next block

'''
#print('generating random data')
#control1 = pd.DataFrame(np.random.randn(100, 5)).T
#control2 = control1 + np.random.normal(0,1,[100, 5]).T



control1_T = control1.T
control2_T = control2.T

n = control1.shape[0]   #398
d = control1.shape[1]   #160


### Identification without weights
rate1 = ldf.identify(control1, control2)
rate2 = ldf.identify(control2, control1)
rate = (rate1 + rate2) / 2


## cma options 
cmaopts = cma.CMAOptions()
cmaopts.set('bounds', [[MIN]*n, [MAX]*n])
cmaopts.set('maxfevals', MAXFEVALS)


def eval_id(w, data1, data2):
    ''' Evaluate weighted Identification ''' 
    A = np.array(data1)
    B = np.array(data2)
    w = np.array(w)
    
    ## weighting columns
    Aw = (A.T * w).T
    Bw = (B.T * w).T
    #return pd.DataFrame(Aw)
    
    data1 = pd.DataFrame(Aw)
    data2 = pd.DataFrame(Bw)
    N = data1.shape[0]
    D = data1.shape[1]             
    
    data1_ranked = data1.rank().T
    data2_ranked = data2.rank().T

    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data1_ranked)
    distances, indices = nbrs.kneighbors(data2_ranked)
    rate1 = np.mean(indices.T == np.array(list(range(D))))
        
    nbrs = NearestNeighbors(n_neighbors=1, metric='correlation', n_jobs= 1).fit(data2_ranked)
    distances, indices = nbrs.kneighbors(data1_ranked)
    rate2 = np.mean(indices.T == np.array(list(range(D))))
    
    obj = (rate1 + rate2)/2
    
    return obj


def objective_fun(w, data1, data2):
   data1 = data1.T
   data2 = data2.T
   w = np.array(w)
   A = np.array(data1)
   B = np.array(data2)
    
   ## weighting columns
   Aw = (A.T * w).T
   Bw = (B.T * w).T
   #return pd.DataFrame(Aw)
    
   data1 = pd.DataFrame(Aw)
   data2 = pd.DataFrame(Bw)
   N = data1.shape[0]
   D = data1.shape[1]
   
   data1_ranked = data1.T.rank()
   data2_ranked = data2.T.rank()
   w = np.array(w)
   
   crrsii = np.array([])
   crrsij = np.array([])
   
   for index, row in data1_ranked.iterrows():
       crrmax = -np.inf
       mycrrs = np.array([])
       x1 = row.to_numpy()
       y1 = data2_ranked.iloc[index].to_numpy()
       for index2, row2 in data2.iterrows():
           y2 = row2.to_numpy()
           x2 = data1_ranked.iloc[index2].to_numpy()
           
           if index == index2:
               crrself = np.correlate(x1, y1)
               crrsii = np.append(crrsii, crrself)
               
           else:
               mycrrs = np.append(mycrrs, np.correlate(x1, x2))
               mycrrs = np.append(mycrrs, np.correlate(x1, y2))
               mycrrs = np.append(mycrrs, np.correlate(y1, x2))
               mycrrs = np.append(mycrrs, np.correlate(y1, y2))
               crrsij = np.append(crrsij, mycrrs)
               crr = np.max(mycrrs)
               
               if crr > crrmax:
                   crrmax = crr
                   
   obj = -(np.mean(crrsii) - np.mean(crrsij)) 
                   
   return obj


### generating some random weights
weights = [1.0/n]*n
#defidiff = objectiveFUN(weights, control1_T, control2_T)
defaccuracy = eval_id(weights, control1, control2)


#weights += np.random.uniform(0.0, 0.1/n, n )



### display initial settings
print("Initial Settings")
print("-----------------------------------------------------------")
print("Identification without weights:            {}.".format(round(rate,4)))
print("Identification with starting weights:      {}.".format(round(defaccuracy,4)))
print("MAXFEVALS                                  {}.".format(MAXFEVALS))
print("MIN, MAX                                   {}, {}.".format(MIN, MAX))
print("RUNS                                       {}.".format(RUNS))
print("TRAINPROP                                  {}.".format(TRAINPROP))
#print("idiff with starting weights:               {}.".format(round(defidiff,4)))
print("-----------------------------------------------------------")
print("Datasets:")
print(control1_T)
print(control2_T)


results = {}

seed = np.random.choice(99999, 1)[0]
print("seed: " + str(seed))
np.random.seed(seed)
results['seed'] = seed
bests = np.empty((0, n))
totaltime = datetime.timestamp(datetime.now())
### running iterative optimisation
for run in range(RUNS):
    starttime = datetime.timestamp(datetime.now())
    print("starting run " + str(run))
    print("-----------------------------------------------------------")
    
    ### train-test split
    trainidx = np.random.rand(len(control1_T)) < TRAINPROP
    train = control1_T[trainidx]
    train = train.reset_index(drop=True)
    test = control1_T[~trainidx]
    test = test.reset_index(drop=True)
    train2 = control2_T[trainidx]
    train2 = train2.reset_index(drop=True)
    test2 = control2_T[~trainidx]
    test2 = test2.reset_index(drop=True)
    
    
    trainmar = objective_fun(weights, train, train2)
    ### train
    x0 = weights
    x0 += np.random.uniform(0.0, 0.1/n, n)
    
    res = cma.fmin(objective_fun, x0=x0, sigma0=np.mean(x0), options=cmaopts, args=(train, train2), eval_initial_x=True)
    
    bestw = np.array(res[0])
    bests = np.vstack((bests, bestw))
    bestw *= 10.0
    
    
    ### test
    testmar = objective_fun(weights, test, test2)
    testwgtmar = objective_fun(bestw, test, test2)
    
    ### store results
    results[str(run) + "fitness"] = np.float(res[1])
    results[str(run) + "solution"] = np.array(res[0])
    results[str(run) + 'trainidx'] = trainidx
    results[str(run) + 'trainmar'] = trainmar
    results[str(run) + 'testmar'] = testmar
    results[str(run) + 'testwgtmar'] = testwgtmar

    ## wrap up run
    stoptime = datetime.timestamp(datetime.now())
    runtime = stoptime-starttime
    avg_runtime = (stoptime - totaltime) / (run+1)
    time_left = (stoptime - totaltime) - (avg_runtime * RUNS)
    
    ### output for this run
    print("Run: " + str(run) + " out of " + str(RUNS) + " RESULTS")
    print("-----------------------------------------------------------")
    print("idiff training data:            {}.".format(trainmar))
    print("idiff test data:                {}.".format(testmar))
    print("best function value:            {}.".format(res[1]))
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



