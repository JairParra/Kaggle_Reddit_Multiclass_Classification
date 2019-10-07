# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:26:34 2019

@ COMP 551: Applied Machine Learning (Winter 2019) 
@ Mini-project 1 : Implementing Logistic Regression and LDA from scratch 
# Team Members: 

@ Hair Albeiro Parra Barrera 
@ ID: 260738619 

@ Ashray Malleshachari
@ ID: 260838256
    
@ Hamza Rizwan
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, NuSVC
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
MN_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Multinomial Naive Bayes accuracy {}%".format(MN_acc))
print(classification_report(y_test, y_pred, target_names=tags)) 
clf_accuracies['Stem Multinomial NB'] = MN_acc

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
                                        n_jobs = -1 # run in parallel using all processors 
                                        )

random_search_MN.fit(X_train, y_train) 

# Obtain report 
best_estimator = random_search_MN.best_estimator_
y_pred = random_search_MN.predict(X_test) 
CV_report = classification_report(y_test, y_pred, 
                                 target_names=tags_nums) 

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
    
logreg_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
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
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [10, 30, 50], 
#          "vect__max_features" :[5000, 15000, None], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],
          "tfidf__smooth_idf":[True, False], 
          "tfidf__sublinear_tf":[True, False], 
          "clf__penalty": ['l1','l2'], 
          "clf__C": [0.5,1.0,2.0], # inverse regularization parameter
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
logreg_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("Logistic Regression accuracy {}%".format(logreg_acc))
clf_accuracies['Logreg'] = logreg_acc

# Display confusion matrix 
confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
df_cm = pd.DataFrame(confusion_mat, index=tags, 
                     columns=tags) 
plt.figure(3, figsize= (15,10)) 
sns.heatmap(df_cm, annot=True, fmt='g') 
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("../figs/Logistic Regression Confusion Matrix.png")
    
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

### WARNING: SVC IS EXTRMELY SLOW!!!!! Impossible to complete. 


#### 4.4 Linear Kernel Support Vector Machine (WARNING: EXTREMELY SLOW!!!! )
#
#Nu_SVC_pipe = Pipeline([('vect', CountVectorizer(ngram_range = (1,2), # kinds of ngrams
#                                              max_df=1.0, # ignore corpus words
#                                              min_df = 0.0,  # ignore low frequency words
#                                              max_features=20000)), # max size of vectors
#                 ('tfidf', TfidfTransformer(norm='l2', # normalize
#                                            smooth_idf=True)), # smoothing 
#                 ('clf', NuSVC(kernel = 'linear', # radial basis Kernel
#                               degree=2, 
#                                   gamma='auto', # kernel coefficient
#                                   max_iter=100,  # no limit of iterations 
#                                   shrinking=True, # heuristic
#                                   cache_size=8000
#                                   ))
#                 ])
#                 
## Fit the dataset 
#Nu_SVC_pipe.fit(X_train,y_train) 
#
## Get predictions on the test set 
#y_pred = Nu_SVC_pipe.predict(X_test) 
#
## Get the accuracy classification report 
#SVC_acc = round(accuracy_score(y_pred, y_test)*100, 2)
#print("Logistic Regression accuracy %s" % SVC_acc)
#print(classification_report(y_test, y_pred, target_names=tags))
#clf_accuracies['SVC'] = SVC_acc 
#
#### 4.3.1 Hyperparameter tunning 
#
## Set up the parameters to search 
#params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
#          "vect__max_df" : [0.9, 0.95, 1.0], 
#          "vect__min_df" : [0.0, 0.05, 0.1], 
#          "vect__max_features" :[8000, 9000, 10000], 
#          "tfidf__norm" : ['l1','l2'], 
#          "tfidf__use_idf" :[True, False],
#          "tfidf__smooth_idf":[True, False], 
#          "clf__kernel":['poly','rbf','sigmoid'], 
#          "clf__C":[0.5, 1.0, 2.0]
#          }
#          
#
## Set up a random seed  
#seed = 42 
#
## Create a randoized search cross-validation with 5 folds 
## and 10 repetitions. 
#random_search_Nu_SVC = RandomizedSearchCV(Nu_SVC_pipe, 
#                                        param_distributions = params, 
#                                        cv = 5, # 5  Cross folds
#                                        verbose=10,  
#                                        random_state = seed, 
#                                        n_iter = 10
#                                        )
#
#random_search_Nu_SVC.fit(X_train, y_train) 
#
## Obtain report 
#best_estimator = random_search_Nu_SVC.best_estimator_
#y_pred = random_search_NU_SVC.predict(X_test) 
#CV_report = classification_report(y_test, y_pred, 
#                                 target_names=tags) 
#
#print("CV report: \n", CV_report)
#print("Best Estimator: \n", best_estimator)
#
#SVC_acc = round(accuracy_score(y_pred, y_test)*100, 2)
#print("Linear Kernel SVM accuracy {}%".format(SVC_acc))
#clf_accuracies['Linear Kernel SVM'] = SVC_acc
#
## Print confusion matrix 
#confusion_mat = confusion_matrix(y_test, y_pred, labels=tags_nums).tolist() 
#df_cm = pd.DataFrame(confusion_mat, index=tags, 
#                     columns=tags) 
#plt.figure(4, figsize= (5,3)) 
#sns.heatmap(df_cm, annot=True, fmt='g') 
#plt.title("Linear Kernel SVM Confusion Matrix")

# ******************************************************************************

   
### 3.1 Decision Tree CLassifier ### 

# NOTE: This shit requires a lot of hyperparameter tunning  

DT_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
#                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
                 ('clf', DecisionTreeClassifier(criterion='entropy', 
                                                splitter='best', 
                                                min_samples_split=10, # min 10% 
                                                min_samples_leaf=5, 
                                                max_features=None, # consider all of them 
                                                class_weight='balanced')), # approximatedly balanced
                 ])
                 

# Fit the dataset 
DT_pipe.fit(X_train,y_train) 

# Get predictions on the test set 
y_pred = DT_pipe.predict(X_test) 

# Get the accuracy classification report 
DT_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("DEcision Tree accuracy {}%".format(DT_acc))
print(classification_report(y_test, y_pred, target_names=tags_nums)) 
clf_accuracies['Decision Tree acc'] = DT_acc

### 3.3.1 Hyperparameter tunning 

# Set up the parameters to search 
params = {"vect__ngram_range" :[(1,1),(1,2),(1,3)], 
          "vect__max_df" : [0.9, 0.95, 1.0], 
          "vect__min_df" : [0.0, 0.05, 0.1], 
          "vect__max_features" :[10000, 10000, 20000], 
          "tfidf__norm" : ['l1','l2'], 
          "tfidf__use_idf" :[True, False],  ### TODO: set up a grid-search for Decision tree 
          "tfidf__smooth_idf":[True, False], #         hyperparameter tunning ### 
          "tfidf__sublinear_tf":[True, False], 
          "clf__alpha": [1.0,2.0,3.0], 
          "clf__fit_prior" : [True, False] 
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



# ******************************************************************************
   
### 4.1 AdaBoost Decision Tree CLassifier ### 

# NOTE: This shit requires a lot of hyperparameter tunning  

Ada_pipe = Pipeline([('vect', CountVectorizer(min_df=5, 
                                                 max_df=0.95, 
                                                 ngram_range=(1,2), 
#                                                 max_features = 20000, 
                                                 )), # max size of vectors
                 ('tfidf', TfidfTransformer(norm='l2', # normalize
                                            use_idf = True, 
                                            smooth_idf=True, 
                                            sublinear_tf=True)), # smoothing 
                 ('clf', AdaBoostClassifier(
                                            learning_rate=1.0, # default tradeoff  
                                            n_estimators=50, # max number of estimators
                                            algorithm='SAMME', # discrete ada boost
                                            ))
                 ])
                 
        
        
count_vect = CountVectorizer().fit(X_train) # fit input data 
X_train_counts = count_vect.transform(X_train)# get train counts
X_test_counts = count_vect.transform(X_test) # get test counts

# apply tfidf 
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

## Normalization (L2)
normalizer_transformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_transformer.transform(X_train_tfidf)
X_test_normalized = normalizer_transformer.transform(X_test_tfidf)

ada = AdaBoostClassifier(n_estimators=100, random_state=0) 
ada.fit(X_train_normalized, y_train)

y_pred = ada.predict(X_test_normalized)

# Get the accuracy classification report 
ada_acc = round(accuracy_score(y_pred, y_test)*100, 2)
print("AdaBoost accuracy {}%".format(ada_acc))
print(classification_report(y_test, y_pred, target_names=tags_nums)) 
clf_accuracies['Ada Boost acc'] = ada_acc



