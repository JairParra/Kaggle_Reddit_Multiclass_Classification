# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:26:34 2019

@ COMP 551: Applied Machine Learning (Winter 2019) 
@ Mini-project 1 : Implementing Logistic Regression and LDA from scratch 
# Team Members: 

@ Hair Albeiro Parra Barrera 
@ ID: 260738619 

@ Ashray Mallesh
@ ID: 260838256
    
@ Hamza Khan
@ ID: 
"""

# *****************************************************************************

### *** Reddit Multi-output classification *** #### 

# *****************************************************************************

### 1. Imports ### 

import random 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm

# Trasnformers
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer 

# Algorithms & Pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

# Metrics & Model selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score, confusion_matrix 


# ****************************************************************************

### 2. Load the clean data ### 

### Lemmatized version 
with open("../data_clean/X_train.txt",'r',  encoding='utf-8') as file: 
    X_train = [x[:-1] for x in file.readlines()] 
    file.close() 
    
with open("../data_clean/X_test.txt",'r',  encoding='utf-8') as file: 
    X_test = file.readlines()
    file.close() 
    
with open("../data_clean/real_X_train.txt",'r',  encoding='utf-8') as file: 
    real_X_train = file.readlines() 
    file.close() 
    
with open("../data_clean/real_X_test.txt",'r',  encoding='utf-8') as file: 
    real_X_test = file.readlines() 
    file.close()   
    
    
### Stemmed version  
#with open("../data_clean/X_train_STEM.txt",'r',  encoding='utf-8') as file: 
#    X_train = file.readlines() 
#    file.close() 
#    
#with open("../data_clean/X_test_STEM.txt",'r',  encoding='utf-8') as file: 
#    X_test = file.readlines()
#    file.close() 
#    
#with open("../data_clean/real_X_train_STEM.txt",'r',  encoding='utf-8') as file: 
#    real_X_train = file.readlines() 
#    file.close() 
#    
#with open("../data_clean/real_X_test_STEM.txt",'r',  encoding='utf-8') as file: 
#    real_X_test = file.readlines() 
#    file.close()   
    
    
# TAGETS
with open("../data_clean/y_train.txt",'r',  encoding='utf-8') as file: 
    y_train = [y[:-1] for y in file.readlines()]
    file.close() 
    
with open("../data_clean/y_test.txt",'r',  encoding='utf-8') as file: 
    y_test = [y[:-1] for y in file.readlines()]
    file.close()    

    
with open("../data_clean/real_y_train.txt",'r',  encoding='utf-8') as file: 
    real_y_train = [y[:-1] for y in file.readlines()] 
    file.close() 
    
tags_df = pd.read_csv('../data_clean/labels.txt', index_col=0) #tags dataframe
tags = list(tags_df['0']) # tags list
tags_nums = [str(tag) for tag in tags_df.index] # number encodings
clf_accuracies = {}
clf_cv_accuracies = {} # to store accuracies for each model 
best_estimators = {} # to store best parameter configurations 

# To convert back the prediction tags: 

labels = sorted(set(real_y_train)) 
label_to_num = { label:i for i, label in enumerate(labels)}
num_to_label = { i:label for i, label in enumerate(labels)}

# function to map 
f_lab2num = lambda x: label_to_num[x]
f_num2lab = lambda x: num_to_label[x]


# *****************************************************************************
# *****************************************************************************

### *** BEST MODELS (in order) *** ### 

# NOTE: Here goes the code for the best model only, using the full training 
# and test set, so there is no actual accuracy except for cross-validation. 
# This chunk of code will also output the predictions that will be submitted to Kaggle. 

###                ###
## 1. Multinomial NB##
###                ###

MN_pipe = Pipeline([('vect', CountVectorizer(min_df=5, # 5
                                                 max_df=0.95, # 0.95  
                                                 ngram_range=(1,2), # 1,2
                                                 max_features = 50000, # 40000
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize 
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=False)), # smoothing 
                 ('clf', MultinomialNB(alpha=1.0, # 1.0  
                                     fit_prior=True)),  
                 ])
                 

# Fit the dataset with the full set
MN_pipe.fit(real_X_train,real_y_train) 


# Get predictions on the test set and save as csv

real_y_pred = MN_pipe.predict(real_X_test) # get predictions
real_y_pred_df = pd.DataFrame(np.zeros((30000,2)), columns=['Id','Category'])  # empty df template
real_y_pred_df['Id'] = range(0,30000) # assign indices
real_y_pred_df['Category'] = real_y_pred # assign predictions 
real_y_pred_df.to_csv('../data_clean/test.csv', index=False) # to save predictions in clean data 

# Get the 5-folds cross_validation_score with the real training data
MN_cv_scores = cross_val_score(MN_pipe, real_X_train, real_y_train, cv=10)
MN_cv_score = round(MN_cv_scores.mean()*100, 4)
clf_cv_accuracies['Multinomial NB'] = MN_cv_score

print("Multinomial Naive Bayes cv-accuracy {}%".format(MN_cv_score))
""" Multinomial Naive Bayes cv-accuracy  55.0186%% """

###                       ###
## 2. Logistic Regresssion ## 
###                       ###

logreg_pipe = Pipeline([('vect', CountVectorizer(min_df=0, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=False)), # smoothing 
                 ('clf', LogisticRegression() )q
                 ])
                 
                 
logreg_pipe.fit(real_X_train, real_y_train)

real_y_pred = logreg_pipe.predict(real_X_test) # get predictions 
real_y_pred_df = pd.DataFrame(np.zeros((30000,2)), columns=['Id','Category'])  # empty df template
real_y_pred_df['Id'] = range(0,30000) # assign indices
real_y_pred_df['Category'] = real_y_pred # assign predictions 
real_y_pred_df.to_csv('../data_clean/test.csv', index=False) # to save predictions in clean data 

# Get the 5-folds cross_validation_score with the real training data
logreg_cv_scores = cross_val_score(logreg_pipe, real_X_train, real_y_train, cv=10)
logreg_cv_score = round(logreg_cv_scores.mean()*100, 4)
clf_cv_accuracies['Logistic Regression'] =logreg_cv_score


print("Logistic Regression cv-accuracy {}%".format(logreg_cv_score))
""" """

###             ###
## 3. Linear SVM ##
###             ###

linear_SVC_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1,1), # kinds of ngrams
                                              max_df=1.0, # ignore corpus words
                                              min_df = 0.0,  # ignore low frequency words
                                              max_features=10000)), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            smooth_idf=True)), # smoothing 
                 ('clf', LinearSVC(penalty='l2', # 
                                   loss='hinge', # hinge loss
                                   dual=True,  # dual formulation
                                   C = 1.0, 
                                   max_iter=3000) )
                 ])
                 
                 
linear_SVC_pipe.fit(real_X_train, real_y_train)

real_y_pred = linear_SVC_pipe.predict(real_X_test) # get predictions 
real_y_pred_df = pd.DataFrame(np.zeros((30000,2)), columns=['Id','Category'])  # empty df template
real_y_pred_df['Id'] = range(0,30000) # assign indices
real_y_pred_df['Category'] = real_y_pred # assign predictions 
real_y_pred_df.to_csv('../data_clean/test.csv', index=False) # to save predictions in clean data 

# Get the 5-folds cross_validation_score with the real training data
linear_SVC_cv_scores = cross_val_score(linear_SVC_pipe, real_X_train, real_y_train, cv=5)
linear_SVC_cv_score = round(linear_SVC_cv_scores.mean()*100, 4)
clf_cv_accuracies['Linear SVM'] = linear_cv_score


print("linear_SVC cv-accuracy {}%".format(linear_SVC_cv_score))

###                 ###
## 4. Decision Tress ## 
###                 ###

DT_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
#                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
                 ('clf', DecisionTreeClassifier(criterion='gini', 
                                                splitter='best', 
                                                min_samples_split=10, # min 10% 
                                                min_samples_leaf=5, 
                                                max_features=None, # consider all of them 
                                                class_weight='balanced')), # approximatedly balanced
                 ])
                 
# trian the model 
DT_pipe.fit(real_X_train, real_y_train)

real_y_pred = DT_pipe.predict(real_X_test) # get predictions 
real_y_pred_df = pd.DataFrame(np.zeros((30000,2)), columns=['Id','Category'])  # empty df template
real_y_pred_df['Id'] = range(0,30000) # assign indices
real_y_pred_df['Category'] = real_y_pred # assign predictions 
real_y_pred_df.to_csv('../data_clean/test.csv', index=False) # to save predictions in clean data 

# Get the 5-folds cross_validation_score with the real training data
DT_cv_scores = cross_val_score(DT_pipe, real_X_train, real_y_train, cv=5)
DT_cv_score = round(DT_cv_scores.mean()*100, 4)
clf_cv_accuracies['Decision Tree'] = DT_cv_score


print("Decision Trees score: {}%".format(DT_cv_score))       


###                      ###
## 5. Naive Bayes (ours) ### 
###                      ###


### NAIVE BAYES MODEL HERE ### 

                 

# *****************************************************************************
# *****************************************************************************

### TESTING PART OF THE SCRIPT ### 

# This section is dedicated to implementing CV-search for hyperparameter tunning. 

# *****************************************************************************
# *****************************************************************************
    
### 3.1 Multinomial Naive Bayes Classifier ### 

MN_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
                 ('clf', MultinomialNB(alpha=1.0, 
                                     fit_prior=True)),  
                 ])
                 

# Fit the dataset 
MN_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = MN_pipe.predict(X_test) 

# Get the accuracy classification report 
MN_acc = round(accuracy_score(y_pred, y_test)*100, 4)
print("Multinomial Naive Bayes accuracy {}%".format(MN_acc))
print(classification_report(y_test, y_pred, target_names=tags)) 
clf_accuracies['Stem Multinomial NB'] = MN_acc

### 3.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.93, 0.95, 0.97], 
          "vect__min_df" : [5,20,50], 
          "vect__max_features" :[10000, 20000, 30000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False],
          "tfidf__sublinear_tf":[True, False], 
          "clf__alpha": [1.5,2.0,2.5], 
          "clf__fit_prior" : [True, False] 
          }
          

# Set up a random seed  
seed = 42 

# Create a randoized search cross-validation with 5 folds 
# and 10 repetitions. 
random_search_MN = RandomizedSearchCV(MN_pipe, 
                                        param_distributions = params, 
                                        cv = 5, # 5  Cross folds
                                        verbose=10,  
                                        random_state = seed, 
                                        n_iter=10, 
                                        n_jobs = -1 # run in parallel using all processors 
                                        )

random_search_MN.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_MN.best_estimator_
y_pred = random_search_MN.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags_nums) 

# Obtain accuracies
MN_acc = round(accuracy_score(y_pred, y_test)*100, 4)
print("Multinomial Naive Bayes accuracy {}%".format(MN_acc))
clf_accuracies['Stem Multinomial NB'] = MN_acc
    
# Display reports 
print("CV report: \n", CV_report)
print("Best estimator: \n", best_estimator)
best_estimators['MultinomialNB'] = best_estimator

# Display confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(2, figsize= (15,15)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Multinomial NB Confussion matrix")
plt.savefig('../figs/Multinomial_NB_Confussion_matrix.png')

# *******************************************************************************

### 4.3 Logistic Regression Classifier   
    
logreg_pipe = Pipeline([('vect', CountVectorizer(min_df=0, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=False)), # smoothing 
                 ('clf', LogisticRegression() )
                 ])
                 
# Fit the dataset 
logreg_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = logreg_pipe.predict(X_test) 

# Get the accuracy classification report 
logreg_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Logistic Regression accuracy %s" % logreg_acc)
print(classification_report(y_test, y_pred, target_names=tags)) 
clf_accuracies['Logreg'] = logreg_acc

### 4.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.93, 0.95, 0.97], 
          "vect__min_df" : [0, 5,10], 
          "vect__max_features" :[15000, 17000, 20000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False], 
          "tfidf__sublinear_tf":[True, False], 
          "clf__penalty": ['l1','l2'], 
          "clf__C": [1.0, 2.0, 2.5], # inverse regularization parameter
          "clf__max_iter": [200,300, 400]
          }
          
# Set up a random seed  
seed = 42 

# Create a randomized search cross-validation with 5 folds 
# and 10 repetitions. 
random_search_logreg = RandomizedSearchCV(logreg_pipe, 
                                        param_distributions = params, 
                                        cv = 5, # 5  Cross folds
                                        verbose=10,  
                                        random_state = seed, 
                                        n_iter = 10, # number of repetitions
                                        n_jobs = -1 # run in parallel using all processors
                                        )

random_search_logreg.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_logreg.best_estimator_
y_pred = random_search_logreg.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags_nums) 

print("CV report: \n", CV_report)
print("Best estimator: \n", best_estimator)
best_estimators['LogisticRegression'] = best_estimator

# Obtain accuracies
logreg_acc = round(accuracy_score(y_pred, y_test)*100, 4)
print("Logistic Regression accuracy {}%".format(logreg_acc))
clf_accuracies['Logreg'] = logreg_acc

# Display confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(3, figsize= (15,10)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("../figs/Logistic_Regression_Confusion_Matrix.png")
    
# *******************************************************************************

### 4.3 Linear Kernel Support Vector Machine

linear_SVC_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1,1), # kinds of ngrams
                                              max_df=1.0, # ignore corpus words
                                              min_df = 0.0,  # ignore low frequency words
                                              max_features=10000)), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            smooth_idf=True)), # smoothing 
                 ('clf', LinearSVC(penalty='l2', # 
                                   loss='hinge', # hinge loss
                                   dual=True,  # dual formulation
                                   C = 1.0, 
                                   max_iter=1000) )
                 ])
                 
# Fit the dataset 
linear_SVC_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = linear_SVC_pipe.predict(X_test) 

# Get the accuracy classification report 
linear_SVC_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Logistic Regression accuracy %s" % linear_SVC_acc)
print(classification_report(y_test, y_pred, target_names=tags))
clf_accuracies['Linear Kernel SVM'] = linear_SVC_acc 

### 4.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [0.0, 0.05, 0.1], 
          "vect__max_features" :[8000, 9000, 10000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False], 
          "clf__C": [1.0,2.0,3.0], 
          "clf__max_iter": [1000,2000,3000], 
          }
          

# Set up a random seed  
seed = 42 

# Create a randoized search cross-validation with 5 folds 
# and 10 repetitions. 
random_search_linear_SVC = RandomizedSearchCV(linear_SVC_pipe, 
                                        param_distributions = params, 
                                        cv = 5, # 5  Cross folds
                                        verbose=10,  
                                        random_state = seed, 
                                        n_iter = 10
                                        )

random_search_linear_SVC.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_linear_SVC.best_estimator_
y_pred = random_search_linear_SVC.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags) 

print("CV report: \n", CV_report)
print("Best Estimator: \n", best_estimator)

linear_SVC_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Linear Kernel SVM accuracy {}%".format(linear_SVC_acc))
clf_accuracies['Linear Kernel SVM'] = linear_SVC_acc

# Print confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(4, figsize= (5,3)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Linear Kernel SVM Confusion Matrix")

    
# *******************************************************************************

   
### 3.1 Decision Tree CLassifier ### 


# NOTE: This shit requires a lot of hyperparameter tunning  
DT_pipe = Pipeline([('vect', CountVectorizer(min_df=0.0, 
                                                 max_df=0.9, 
                                                 ngram_range=(1,1), 
                                                max_features = 20000,
                                                
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalizer
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
                 ('clf', DecisionTreeClassifier(criterion='gini', 
                                                splitter='best', 
                                                min_samples_split=10, # min 10% 
                                                min_samples_leaf=5, 
                                                max_features=None, # consider all of them 
                                                class_weight=None)), # approximatedly balanced
                 ])
                 

# Fit the dataset 
DT_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = DT_pipe.predict(X_test) 

# Get the accuracy classification report 
DT_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Decision Tree accuracy {}%".format(DT_acc))
print(classification_report(y_test, y_pred, target_names=tags_nums)) 
clf_accuracies['Decision Tree acc'] = DT_acc

### 3.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [0.0, 0.05, 0.1], 
          "vect__max_features" :[10000, 15000, 20000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False], 
          "tfidf__smooth_idf":[True, False], 
          "tfidf__sublinear_tf":[True, False], 
          "clf__criterion": ["gini","entropy"], 
          "clf__max_features":[None, "auto","log2"], 
          "clf__min_samples_split":[8,10,12], 
          "clf__min_samples_leaf":[8,10,12], 
          "clf__class_weight":["balanced", None]
          }
          

# Set up a random seed  
seed = 42 

# Create a randoized search cross-validation with 5 folds 
# and 10 repetitions. 
random_search_DT = RandomizedSearchCV(DT_pipe, 
                                        param_distributions = params, 
                                        cv = 5, # 5  Cross folds
                                        verbose=10,  
                                        random_state = seed, 
                                        n_iter=10, 
                                        n_jobs = -1 # run in parallel using all processors 
                                        )

random_search_DT.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_DT.best_estimator_
y_pred = random_search_DT.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags_nums) 

# Obtain accuracies
DT_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Decision Tree accuracy {}%".format(DT_acc))
clf_accuracies['Decision Tree'] = DT_acc

# Display reports 
print("CV report: \n", CV_report)
print("Best estimator: \n", best_estimator)
best_estimators['Decision Tree'] = best_estimator

# Display confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(2, figsize= (15,15)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Decision Tree Confussion matrix")
plt.savefig('../figs/Decision Tree Confussion matrix.png')



# ******************************************************************************
#   
#### 4.1 AdaBoost Decision Tree CLassifier ### 
#
## NOTE: This shit requires a lot of hyperparameter tunning  
#
#Ada_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
#                                                 max_df=0.95, 
#                                                 ngram_range=(1,2), 
##                                                 max_features = 20000, 
#                                                 )), # max size of vectors
#                 ('tfidf', TfidfTransformer(norm='l2', # normalize
#                                            use_idf = True, 
#                                            smooth_idf=True, 
#                                            sublinear_tf=True)), # smoothing 
#                 ('clf', AdaBoostClassifier(
#                                            learning_rate=1.0, # default tradeoff  
#                                            n_estimators=50, # max number of estimators
#                                            algorithm='SAMME', # discrete ada boost
#                                            ))
#                 ])
#                 
#        
#        
#count_vect = CountVectorizer().fit(X_train) # fit input data 
#X_train_counts = count_vect.transform(X_train)# get train counts
#X_test_counts = count_vect.transform(X_test) # get test counts
#
## apply tfidf 
#tfidf_transformer = TfidfTransformer().fit(X_train_counts)
#X_train_tfidf = tfidf_transformer.transform(X_train_counts)
#X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
### Normalization (L2)
#normalizer_transformer = Normalizer().fit(X=X_train_tfidf)
#X_train_normalized = normalizer_transformer.transform(X_train_tfidf)
#X_test_normalized = normalizer_transformer.transform(X_test_tfidf)
#
#ada = AdaBoostClassifier(n_estimators=100, random_state=0) 
#ada.fit(X_train_normalized, y_train)
#
#y_pred = ada.predict(X_test_normalized)
#
## Get the accuracy classification report 
#ada_acc = round(accuracy_score(y_pred, y_test)*100, 2)
#print("AdaBoost accuracy {}%".format(ada_acc))
#print(classification_report(y_test, y_pred, target_names=tags_nums)) 
#clf_accuracies['Ada Boost acc'] = ada_acc






