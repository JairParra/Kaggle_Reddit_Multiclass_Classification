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

import re
import nltk 
import random 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score, confusion_matrix 


# ****************************************************************************

### 2. Load the clean data ### 

with open("../data_clean/X_train.txt",'r') as file: 
    X_train = file.readlines() 
    file.close() 
    
with open("../data_clean/X_test.txt",'r') as file: 
    X_test = file.readlines()
    file.close() 
    
with open("../data_clean/y_train.txt",'r') as file: 
    y_train = [y[:-1] for y in file.readlines()]
    file.close() 
    
with open("../data_clean/y_test.txt",'r') as file: 
    y_test = [y[:-1] for y in file.readlines()]
    file.close() 
    
with open("../data_clean/real_X_test.txt",'r') as file: 
    real_X_train = file.readlines() 
    file.close()    
    
tags_df = pd.read_csv('../data_clean/labels.txt', index_col=0) #tags dataframe
tags = list(tags_df['0']) # tags list
tags_nums = [str(tag) for tag in tags_df.index] # number encodings
clf_accuracies = {} # to store accuracies for each model 
best_estimators = {} # to store best parameter configurations 

# *****************************************************************************
    
### 3.1 Multinomial Naive Bayes Classifier ### 

MN_pipe = Pipeline([('vect', CountVectorizer()), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            smooth_idf=True, # smoothing 
                                            sublinear_tf=False)),
                 ('clf', MultinomialNB(alpha=1.0, 
                                     fit_prior=True)),  
                 ])
                 

# Fit the dataset 
MN_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = MN_pipe.predict(X_test) 

# Get the accuracy classification report 
MN_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Multinomial Naive Bayes accuracy {}%".format(MN_acc))
print(classification_report(y_test, y_pred, target_names=tags)) 

### 3.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [0.0, 0.05, 0.1], 
          "vect__max_features" :[10000, 10000, 20000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False],
          "tfidf__sublinear_tf":[True, False], 
          "clf__alpha": [1.0,2.0,3.0], 
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
                                        n_jobs = 5
                                        )

random_search_MN.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_MN.best_estimator_
y_pred = random_search_MN.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags) 

# Obtain accuracies
MN_acc = round(accuracy_score(y_pred, y_test)*100, 2)
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
plt.savefig('../figs/Multinomial NB Confussion matrix.png')

# *******************************************************************************

### 4.3 Logistic Regression Classifier

logreg_pipe = Pipeline([('vect', CountVectorizer()), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            smooth_idf=True)), # smoothing 
                 ('clf', LogisticRegression()) 
                 ])
                 
# Fit the dataset 
logreg_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = logreg_pipe.predict(X_test) 

# Get the accuracy classification report 
logreg_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Logistic Regression accuracy %s" % logreg_acc)
print(classification_report(y_test, y_pred, target_names=tags)) 

### 4.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [0.0, 0.05, 0.1], 
          "vect__max_features" :[5000, 10000, 15000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False], 
          "clf__penalty": ['l1','l2'], 
          "clf__C": [1.0,2.0,3.0], 
          "clf__max_iter": [100,200,300]
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
                                        n_jobs = 5 # run in parallel
                                        )

random_search_logreg.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_logreg.best_estimator_
y_pred = random_search_logreg.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags) 

print("CV report: \n", CV_report)
print("Best estimator: \n", best_estimator)
best_estimators['LogsiticRegression'] = best_estimator

# Obtain accuracies
logreg_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Logistic Regression accuracy {}%".format(logreg_acc))
clf_accuracies['Stem Logreg'] = logreg_acc

# Display confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(3, figsize= (5,3)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("../figs/Logistic Regression Confusion Matrix.png")
    
    
    
