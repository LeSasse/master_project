My master project: 
Using global gradients for subject identification

I am using Python v.3.7.9. because Brainspace is not compatible with versions after that.
Gradients are computed using Brainspace (https://www.nature.com/articles/s42003-020-0794-7).
Other libraries used are scipy, pandas, numpy, and math as well as spyder.


"simple_identification.py" is an example of subject identification using participants concatenated connectivity matrices.
Using Pearson Correlation leads to 98% accuracy in identification with the data I have.

"gradient_identification.py" is an example of subject identification using participants main global gradients.
There is no gradient alignment implemented yet, therefore gradient comparison/identification is quite meaningless at the moment.
It identifies 0.25% of participants correctly (1).

"alignment_method_one.py" aligns all gradients according to one reference gradient. It leads to 0.25% accuracy. Gradient alignment does not seem to change anythin. Am I doing it wrong??   
