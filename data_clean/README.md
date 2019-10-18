# data_clean 
- Data that has been pre-processed. 
- Note that the following train and test sets are actually further splits of the original labeled set we were given. 
The actual test set has also been preprocessed, but before we use it to output the predictions, we will first test with 
the new train and test sets , splitted at a ratio of 95:5 . When we finish model selection, we will retrain the best model 
with the full data , and make predictions on the real training test set. 

## X_train (Size=63000)
- artificial trianing set pre-processed using lemmatization 

## X_test (Size=7000) 
- artificial testing set pre-processed using lemmatization 

## real_X_train (Size=70000) 
- original, full pre-processed training set using lemmatization 

## real_X_test (Size=30000) 
- original, full pre-processed testing set using lemmatization  


## X_train_STEM (Size=63000)
- artificial trianing set pre-processed using stemming 

## X_test_STEM (Size=7000) 
- artificial testing set pre-processed using stemming

## real_X_train_STEM (Size=70000) 
- original, full pre-processed training set using stemming

## real_X_test_STEM (Size=30000) 
- original, full pre-processed testing set using stemming


## y_train (Size=63000) 
- labels training set 

## y_test (Size=7000) 
- labels test set 

## real_y_train (Size=70000) 
- original labels test set 


## test.csv 
- These are our predictions from our best model coming from main.py (currently Multinomial NB) 

## labels.txt 
- A text file with the target labels.

