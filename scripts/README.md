# scripts 
- main scripts go here (like data_preprocessing, algorithm classes and other final implementations)
- There should be a "main" script that calls any algorithmic class 
- Common functions may be implemented inside main 
- Data-preprocessing may be implemented inside main 
- Informal tests should **not** be included in main

## main.py
- Calls to Bernoulli_NB , testing all the other models and all the Scikit-learn packages or whatever go here. If the script becomes too big, or if we decide to implement some other extra class, we might create it in a different step. 
- The basic implementation idea goes as follows: 
  - i. Get the clean data  
  - ii. Apply the ```Pipeline``` described below. 
  - iii. CV_search 
  
### Pipeline 

We will emply a Pipeline that follows the following format: 

`vectorizer` -> `tfidf-transformer` -> `normalizer` -> `classifier` 

Where: 
 - **Vectorizer** : transforms the input text into one hot-encoded vectors 
 - **Transformer** : assigns a tfidf based weight to the vectors 
 - **Normalizer** : normalizes the vectors so they have the same range 
 - **Classifier** : the classifier algorithm

## data_preprocessing.py 
- As it name indicates, this script is dedicated solely to clean the data, up to a format that sklearn can understand, and process. 
- Note that some of the preprocessing actually goes along main (since it's more convenient for sklearn),  but we keep these separeated since it makes part of the Pipeline. 

## Bernoulli_NB.py 
- Implementation of the Bernoulli Naive Bayes model as a class. 
- Functions will be briefly dscribed below as implementation progresses 

