[![Generic badge](https://img.shields.io/badge/Reddit_Classification-Building-blue.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Contributors-3-<COLOR>.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/COMP551-Applied_Machine_Learning-red.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Neat_level-OVER_8000-green.svg)](https://shields.io/)


# Mini-Project 2: Kaggle Reddit multi-class classification
- Project for implementing multi-output classification of reddit data. 

## IMPORTANT/ANNOUNCEMENTS:  
- **Don't forget** to start registration on Kaggle https://www.kaggle.com/t/cbd6c7bc66394bd682983a6daeefe759
- nltk/spacy basic usage &  preprocessing tutorial ready at "notebooks" 
- Ideally, we would like to have data pre-processing, Naive bayes implementation, 
as well as a fitted Pipeline with results by **next wednesday** We will then use the rest of the time 
to polish up and write-up the project. 
- Short meeting after class on **Monday** to discuss progress.
- Please check `issues`, consult  
- Our current best current classifier is **Multinomial Naive Bayes** and it's only accuracte to like 54% , which is utterly crap. We need to think of better features as well as perform hyperparameter tunning. 


## TASKS: 
-The following are the task divisions. These will be updated accordingly. 

### Jair: 
- Finish uploadin the nltk/spacy tutorials and setting up. **done**
- Naive Bayes implementation (compatible with sklearn). **Started, in progress**
- Data-preprocessing script and outputs. **done**
- Start implementing classification pipelines. **In progress**
- get added to the kaggle group (give ashray your username)

### Ashray: 
- **Fix Naive Bayes running time issue/ finish implementation of other functions**
- Start working on improving classification pipelines and parameter Grid Parameter search
- Make kaggle group and join competiton (done)

### Hamza: 
- **Fix Naive Bayes running time issue/ finish implementation of other functions**
- Start working on improving classification pipelines and parameter Grid Parameter search 
- get added to the kaggle group (give ashray your username)

## Our paper: 
- https://www.overleaf.com/6192466927dckyydkjvmct

## Naive Bayes Formulation 
See https://sklearn.org/modules/naive_bayes.html
https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/NB.pdf

![](figs/Bernoulli_NB.png)

![](figs/Naive_Bayes_formulation.png)

## Dataset labels distribution  
![](figs/labels_countplot.png)

## Current Best Model: Scikit-learn Multinomial NB (52.9571 %)

![](figs/Multinomial_NB_Confussion_matrix.png)
