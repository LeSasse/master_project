My master project: 
Using global gradients for subject identification

I am using Python v.3.7.9. because Brainspace is not compatible with versions after that.
Gradients are computed using Brainspace (https://www.nature.com/articles/s42003-020-0794-7).
Other libraries used are scipy, pandas, numpy, and math as well as spyder.


"simple_identification.py" is an example of subject identification using participants concatenated connectivity matrices. It uses theidentification method from Finn et al., 2005 (https://www.researchgate.net/publication/282812326_Functional_connectome_fingerprinting_Identifying_individuals_using_patterns_of_brain_connectivity) where the pearson correlation is calculated between the connectivity data that is to be identified and all of the connectivity data from the database. The connectome that has the highest correlation with the target connectivity data is chosen.
It leads to 98% accuracy in identification with the data I have.

"gradient_identification.py" is an example of subject identification using participants main global gradients.
There is no gradient alignment implemented yet, therefore gradient comparison meaningless. The same method is used as in "simple_identification.py" except correlations are calculated between the gradients rather than the edges from the connectivity matrix.

When gradients are computed using a "pearson" kernel, and dimensionality reduction is based on diffusion map  embedding, it identifies 0.25% of participants correctly (1).

"alignment_method_one.py" aligns all gradients according to one reference gradient using procrustes rotation. When gradients are computed using a "pearson" kernel and diffusion map embedding, it leads to 0.25% accuracy. Gradient alignment does not seem to change anything. Am I doing something wrong with the alignment?? 
When gradients are computed using a "normalised angle" kernel and diffusion map embedding it leads to 0% accuracy. When using normalised angle and pca, its also 0%.

"alignment_method_two.py" aligns all gradients from the database to the gradient of the person to be identified in the target data using procrustes rotation. Therefore this takes a while longer to compute. It also identifies only one person correctly (the same one as in the previous two attempts) using "pearson" kernel and diffusion map embedding. I am still not sure if the alignment is working properly or maybe there is something wrong with gradient construction (or maybe they are not so good with this kind of identification method/maybe use sth like canonical correlation??)
