### Importing Modules ########################################################

import pandas as pd
import numpy as np
import sys


## my own imports
sys.path.append("/home/leonard/projects/master_project_files/master_project/imports")
import load_data_and_functions as ldf

### Global Variables/User input ##############################################

session1_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP1LRFIX_dos160_roi_atlas_2x2x2.mat"
)
session2_path = (
    "../../dotmat_data/hcp_conn_unrelated_FULLHCP2LRFIX_dos160_roi_atlas_2x2x2.mat"
)


atlas_size = 160

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


### control data
control_condition_session1 = ldf.create_control_data(
    connectivity_data=session1_cd_transposed, kind="max", atlas_size=atlas_size
)
control_condition_session2 = ldf.create_control_data(
    connectivity_data=session2_cd_transposed, kind="max", atlas_size=atlas_size
)

### Identification Analysis ##################################
## identification (default = spearman)
rate1 = ldf.identify(
    target=control_condition_session2, database=control_condition_session1
)
rate2 = ldf.identify(
    target=control_condition_session1, database=control_condition_session2
)

rate = (rate1 + rate2) / 2
