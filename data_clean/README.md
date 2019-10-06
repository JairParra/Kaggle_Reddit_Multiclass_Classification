# data_clean 
- Data that has been pre-processed. 
- Note that the following train and test sets are actually further splits of the original labeled set we were given. 
The actual test set has also been preprocessed, but before we use it to output the predictions, we will first test with 
the new train and test sets , splitted at a ratio of 95:5 . When we finish model selection, we will retrain the best model 
with the full data , and make predictions on the real training test set. 

## X_train 
- feature training set  

## y_train 
- labels training set 

## X_test 
- features test set 

## y_test 
= labels test set 

## real_X_test 
- real test set on which the best model will be re-trained. 


