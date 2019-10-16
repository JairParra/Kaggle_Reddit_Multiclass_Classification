[![Generic badge](https://img.shields.io/badge/Reddit_Classification-Building-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Contributors-3-<COLOR>.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/COMP551-Applied_Machine_Learning-red.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Neat_level-OVER_8000-green.svg)](https://shields.io/)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kaggle_Reddit_Multiclass_Classification</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/JairParra/Kaggle_Reddit_Multiclass_Classification" property="cc:attributionName" rel="cc:attributionURL">Hair Albeiro Parra Barrera</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.</a>. Every unauthorized infraction will be legally prosecuted.

# Mini-Project 2: Kaggle Reddit multi-class classification
- Project for implementing multi-output classification of reddit data. 

## IMPORTANT/ANNOUNCEMENTS:  
- Finish the whole thing tomorrow, including the report. 


## TASKS: 
-The following are the task divisions. These will be updated accordingly. 

### Jair: 
- Finish the report

### Ashray: 
- **Finish Naive Bayes** 
- **Finish the report**

### Hamza: 
- **Finish Naive Bayes** 
- **Finish the report**

## Our paper: 
- https://www.overleaf.com/6192466927dckyydkjvmct

## Naive Bayes Formulation 
See https://sklearn.org/modules/naive_bayes.html
https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/NB.pdf

![](figs/Bernoulli_NB.png)

![](figs/Naive_Bayes_formulation.png)

## Dataset labels distribution  
- We observe that the labels have a very well balanced distribution. 

![](figs/labels_countplot.png)

## Current Best Model: Scikit-learn Multinomial NB (Kaggle acc: 55.15%, local cv acc: 54.94 %)

- **Note:** This accuracy comes from the artificially split training and testing sets we created, i.e., using the actual training and test data, we further split it into `X_train` (63000 samples), `X_test` (7000 samples), `y_train` (63000 samples) and `y_test`(7000) samples.  

![](figs/Multinomial_NB_Confussion_matrix.png)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Kaggle_Reddit_Multiclass_Classification</span> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/JairParra/Kaggle_Reddit_Multiclass_Classification" property="cc:attributionName" rel="cc:attributionURL">Hair Albeiro Parra Barrera</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.</a>. Every unauthorized infraction will be legally prosecuted.
