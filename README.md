My master project: 
Using global gradients for subject identification

I am using Python v.3.7.9. Gradients are computed using Brainspace
(https://www.nature.com/articles/s42003-020-0794-7).
Other libraries used are scipy, pandas, numpy, and math as well as spyder.

The Identification method is based on Finn et al., 2015:
(https://www.researchgate.net/publication/282812326_Functional_connectome_fingerprinting_Identifying_individuals_using_patterns_of_brain_connectivity) 

using_pearson_correlation
Scripts in this folder use pearson correlation as an identification method.

"simple_identification.py" is an example of subject identification using 
participants concatenated connectivity matrices. It uses the identification 
method from Finn et al., 2015 
(https://www.researchgate.net/publication/282812326_Functional_connectome_fingerprinting_Identifying_individuals_using_patterns_of_brain_connectivity) 
where the pearson correlation is calculated between the connectivity data that 
is to be identified and all of the connectivity data from the database.
The connectome that has the highest correlation with the target connectivity data is chosen.
It leads to 98% accuracy in identification with the data I have.

Folder "imports":

"load_data_and_functions.py" handles all functions that are essentially the same 
for each different identification method script.


Folder "Identification"
All identification scripts:

1. "individual_gradients.py": Only individual gradients are selected for 
identification, i.e. the principal gradient. 

2. "concatenated_gradients.py": Gradients are aligned, then concatenated, then 
used for identification.

3. "control_condition.py": Provides alternative dimension reduction to control
against gradients.

4. "brain_area_wise_control.py": Uses individual brain areas for identification.