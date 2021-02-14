#!/usr/bin/python3
### Importing Modules ########################################################


## my own imports
import sys
sys.path.append("../imports")
from gradient import gradient



# Instantiate a gradient object
Gradiator = gradient()


##############################################################################
### customise gradient settings ##############################################
##############################################################################
Gradiator.set_kernels( ["pearson", "spearman", "normalized_angle", "gaussian","cosine"] )

Gradiator.set_alignment( "procrustes" )
Gradiator.set_atlas_size( 160)
Gradiator.set_sparsity( [0.9] )
Gradiator.set_dimension_reductions( ["pca", "dm", "le"] )

Gradiator.set_concatenate(False)

# We want some output
Gradiator.set_verbose(True)
Gradiator.set_debug(False)

## Identification method (spearman or pearson)
Gradiator.set_id_method("spearman")
Gradiator.set_num_grads( range(1,11) )



Gradiator.add_session( "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat" )
Gradiator.add_session( "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat" )

##############################################################################
### path and name for output data
### filename as follows atlasname + concatenated (or not) + identification method
Gradiator.set_output_file("RS1_RS2_dosenbach_{}_{}.csv" )
##############################################################################



# That's what we have so far
Gradiator.dump_settings()

Gradiator.run_multi()
#Gradiator.run()

# And output
Gradiator.store_result()


