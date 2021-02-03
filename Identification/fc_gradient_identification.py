### Importing Modules ########################################################

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
import sys
from datetime import datetime
import time

## my own imports
sys.path.append("../imports")
import load_data_and_functions as ldf


print( "Starting..." )


### Global Variables/User input ##############################################

### data file paths ##########################################################
database_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
target_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)


##############################################################################
### customise gradient settings ##############################################
##############################################################################
alignment = "procrustes"
atlas_size = 160


sparsity = [0.9]
kernels = ["pearson", "spearman", "normalized_angle", "gaussian","cosine"]
dimension_reductions = ["pca", "dm", "le"]
concatenate = False

## Identification method (spearman or pearson)
id_method = "spearman"

## for num_grads, if concatenation = true, then this defines which gradients
## concatenated, if concatenation = false then principal gradient will still be chosen
## and only n_components argument goes up
## num_grads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
num_grads = [10]
##############################################################################


##############################################################################
#### these lists track the settings for every iteration and then are used for
#### a dataframe with results
kernel_used = []
dreduction_used = []
rates = []
ngradients = []
gradient_used = []
sparsities = []
iteration_count = 0
concatenation = []
total_iterations = len(sparsity) * len(num_grads) * len(kernels) * len(dimension_reductions)
##############################################################################


##############################################################################
### path and name for output data
### filename as follows atlasname + concatenated (or not) + identification method
output_file = "dosenbach_" + str(concatenate) + "_" +str(id_method) + ".csv"
##############################################################################


##############################################################################
## ask for confirmation of settings
print("total iterations == " + str(total_iterations))
print("atlas size == " + str(atlas_size))
print("number of gradients to iterate " + str(num_grads))
print("concatenate == " + str(concatenate))


##############################################################################
## Loading Data ##############################################################
D, D_sub, D_connectivity_data, D_cd_transposed = ldf.load_data(database_path)
Y, Y_sub, Y_connectivity_data, Y_cd_transposed = ldf.load_data(target_path)

## D contains all available data
## D_sub is subject list
## D_connectivity_data has all connectivity data with subjects as rows
## D_cd_transposed has all connectivity data with subjects as columns
## the transpose is necessary to use pandas.corrwith() method for identification
##############################################################################

##############################################################################
## Looping over different settings
totaltime =datetime.timestamp( datetime.now() )
for spars in sparsity:
    for n_gradients in num_grads:
        for kernel in kernels:
            for dimension_reduction in dimension_reductions:

                starttime =datetime.timestamp( datetime.now() )
                
                ##############################################################
                ### Gradient Construction
                ##        
                ## Reference Gradient for Alignment ##########################
                ## In this alignment method I will align all gradients to one 
                ## reference gradient from a reference participant.
                reference_participant = ldf.get_conn_matrix(D_cd_transposed.iloc[:, 0])
                
                gref = GradientMaps(
                   n_components=n_gradients,
                   kernel=kernel,
                   approach=dimension_reduction,
                   random_state=0,
                   )
                gref.fit(reference_participant)
                #print("reference participants gradient:")
                #print(gref.gradients_[:,0].shape)
                
               
                ### Database Gradient Construction ###########################
                gradient_dataframe_transposed = ldf.create_gradient_database(
                    dataframe=D_cd_transposed,
                    subjects=D_sub,
                    atlas_size=atlas_size,
                    reference=gref.gradients_,
                    kernel=kernel,
                    dimension_reduction=dimension_reduction,
                    alignment=alignment,
                    n_gradients=n_gradients,
                    concatenate=concatenate,
                )
                #print("gradient database shape:")
                #print(gradient_dataframe_transposed.shape)
                #print("waiting 10 seconds for inspection")
                #time.sleep(10)
                
                
                ### Identification Analysis ##################################
                ### Target Gradients and Identification from Database ########

                count1 = 0  # the count variable keeps track of iterations with
                            # accurate identification
                for index, subject in enumerate(Y_sub):
                    subject_connm = ldf.get_conn_matrix(Y_cd_transposed[subject])
                    #print("target subject shape:")
                    #print(subject_connm.shape)
                    gm = GradientMaps(
                        n_components=n_gradients,
                        kernel=kernel,
                        approach=dimension_reduction,
                        random_state=0,
                        alignment=alignment,
                    )
                    gm.fit(subject_connm, reference=gref.gradients_, sparsity=spars)
                    
                    ### stacking subject gradients ###########################
                    ### or just take principal gradient
                    if concatenate == True:
                        #print("concatenate for target subject is true")
                        grad = [None] * n_gradients
                        for i in range(n_gradients):
                            #norm_array = np.linalg.norm(gm.aligned_[:,i])
                            grad[i] = gm.aligned_[:, i]  #/norm_array
                            grad = np.array(grad, dtype=object)
                        grad_stacked = np.hstack(grad)
                        subject_gradient_dataframe = pd.DataFrame(grad_stacked)
                        #print("target participant has been concatenated")
                        #print(subject_gradient_dataframe.shape)
                    elif concatenate == False:
                        grad = gm.aligned_[:,0]
                        subject_gradient_dataframe = pd.DataFrame(grad)
                        #print("target participant has not been concatenated")
                        #print(subject_gradient_dataframe.shape)
                    
                    #print(subject_gradient_dataframe)
                    
                    ### identification #######################################
                    all_corr = gradient_dataframe_transposed.corrwith(
                        subject_gradient_dataframe.iloc[:, 0], method=id_method
                    )
                    
                    max_value = max(all_corr)
                    max_index = all_corr.index[all_corr == max_value]

                    if max_index[0] == subject:
                        count1 = count1 + 1
                        #print("subject correctly identified with correlation of")
                        #print("r == " + str(max_value))
                        #time.sleep(2)
                    #else:
                     #   print("subject not correctly identified with correlaiton of")
                     #   print("r == " + str(max_value))
                     #   time.sleep(2)
                rate = count1 / len(Y_cd_transposed.columns)

                ### dataframing relevant information #########################
                kernel_used.append(kernel)
                dreduction_used.append(dimension_reduction)
                rates.append(rate)
                ngradients.append(str(n_gradients))
                iteration_count = iteration_count + 1
                sparsities.append(spars)
                concatenation.append(concatenate)
                stoptime =datetime.timestamp( datetime.now() )
                
                ## get output to see where loop at
                print(str(iteration_count))
                print(" out of " + str(total_iterations) + " iterations.")
                print("(this round took: " + str(stoptime-starttime) + " sec )")
                print("(     total took: " + str(stoptime-totaltime) + " sec )")
                print("(  avg per round: " + str((stoptime-totaltime)/iteration_count) + " sec )")


                print(
                    "Settings were " + str(kernel) + " " + str(dimension_reduction) + ", sparsity == " + str(spars) + "."
                )
                print("number of Gradients: " + str(n_gradients))
                print("Accuracy was " + str(rate))
                print("Sparsity was " + str(spars))
                print("count1 was " + str(count1))
                
                #time.sleep(10)
## uniting dataframes
accuracy = {
    "kernels": kernel_used,
    "dimension reduction": dreduction_used,
    "accuracy": rates,
    "n_gradients": ngradients,
    "sparsity": sparsities,
    "concatenation": concatenation
}

df_accuracy = pd.DataFrame(accuracy)
df_accuracy.to_csv(
    output_file,
    index=False,
)
 