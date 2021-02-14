### Importing Modules ########################################################

import numpy as np
import pandas as pd
from brainspace.gradient import GradientMaps
import sys
from datetime import datetime
import time
import brainspace

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


##############################################################################
### customise gradient settings ##############################################
##############################################################################
alignment = "procrustes"
atlas_size = 160


sparsity = [0.9]
kernels = ["gaussian"]
dimension_reductions = ["le"]
concatenate = True
global_alignment = False
## if concatente == True and global alignment == True, reference gradient must have
## 50 components (rather than n_gradients components) and values in num_grads cannot
## be more than 50


## Identification method (spearman or pearson)
id_method = "spearman"

## for num_grads, if concatenation = true, then this defines which gradients
## concatenated, if concatenation = false then principal gradient will still be chosen
## and only n_components argument goes up
## num_grads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
num_grads = [3]
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
total_iterations = (
    len(sparsity) * len(num_grads) * len(kernels) * len(dimension_reductions)
)
##############################################################################


##############################################################################
### path and name for output data
### filename as follows atlasname + concatenated (or not) + identification method
output_file = "RS1_RS2_dosenbach_participant_ref_concatenation:" + str(concatenate) + "_global_alignment:" + str(global_alignment) + "_" + str(id_method) + ".csv"
##############################################################################


##############################################################################
## ask for confirmation of settings
print("total iterations == " + str(total_iterations))
print("atlas size == " + str(atlas_size))
print("number of gradients to iterate " + str(num_grads))
print("concatenate == " + str(concatenate))
print("global_alignment == " + str(global_alignment))


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

## _data contains all available data
## _sub is subject list
## _connectivity_data has all connectivity data with subjects as rows
## _cd_transposed has all connectivity data with subjects as columns
## the transpose is necessary to use pandas.corrwith() method for identification
##############################################################################

#avg_connectivity = session1_cd_transposed.mean(axis=1)

##############################################################################
## Looping over different settings
totaltime = datetime.timestamp(datetime.now())
for spars in sparsity:
    for n_gradients in num_grads:
        for kernel in kernels:
            for dimension_reduction in dimension_reductions:

                starttime = datetime.timestamp(datetime.now())

                ##############################################################
                ### Gradient Construction

                ## Reference Gradient for Alignment ##########################
                ## In this alignment method I will align all gradients to one
                ## reference gradient from a reference participant.
                
                reference_participant = ldf.get_conn_matrix(
                    session1_cd_transposed.iloc[:, 0]
                )
                
                
                gref = GradientMaps(
                    n_components=n_gradients,
                    kernel=kernel,
                    approach=dimension_reduction,
                    random_state=0,
                )
                gref.fit(reference_participant)

                ### Session1 Gradient Construction ###########################
                session1_gradients = ldf.create_gradient_database(
                    dataframe=session1_cd_transposed,
                    subjects=session1_sub,
                    atlas_size=atlas_size,
                    reference=gref.gradients_,
                    kernel=kernel,
                    dimension_reduction=dimension_reduction,
                    alignment=alignment,
                    n_gradients=n_gradients,
                    concatenate=concatenate,
                    global_alignment=global_alignment
                )

                ## Session2 Gradient Construction ###########################
                session2_gradients = ldf.create_gradient_database(
                    dataframe=session2_cd_transposed,
                    subjects=session2_sub,
                    atlas_size=atlas_size,
                    reference=gref.gradients_,
                    kernel=kernel,
                    dimension_reduction=dimension_reduction,
                    alignment=alignment,
                    n_gradients=n_gradients,
                    concatenate=concatenate,
                    global_alignment=global_alignment
                )

                ### Identification Analysis ##################################
                ## identification (default = spearman)
                rate1 = ldf.identify(
                    target=session2_gradients, database=session1_gradients
                )
                rate2 = ldf.identify(
                    target=session1_gradients, database=session2_gradients
                )

                rate = (rate1 + rate2) / 2

                ### dataframing relevant information #########################
                kernel_used.append(kernel)
                dreduction_used.append(dimension_reduction)
                rates.append(rate)
                ngradients.append(str(n_gradients))
                iteration_count = iteration_count + 1
                sparsities.append(spars)
                concatenation.append(concatenate)
                stoptime = datetime.timestamp(datetime.now())

                ## get output to see where loop at
                print(str(iteration_count))
                print(" out of " + str(total_iterations) + " iterations.")
                print("(this round took: " + str(stoptime - starttime) + " sec )")
                print("(     total took: " + str(stoptime - totaltime) + " sec )")
                print(
                    "(  avg per round: "
                    + str((stoptime - totaltime) / iteration_count)
                    + " sec )"
                )

                print(
                    "Settings were "
                    + str(kernel)
                    + " "
                    + str(dimension_reduction)
                    + ", sparsity == "
                    + str(spars)
                    + "."
                )
                print("number of Gradients: " + str(n_gradients))
                print("       Accuracy was: " + str(rate))
                print("       Sparsity was: " + str(spars))


## uniting dataframes
accuracy = {
    "kernels": kernel_used,
    "dimension reduction": dreduction_used,
    "accuracy": rates,
    "n_gradients": ngradients,
    "sparsity": sparsities,
    "concatenation": concatenation,
}

df_accuracy = pd.DataFrame(accuracy)
df_accuracy.to_csv(
    output_file,
    index=False,
)
