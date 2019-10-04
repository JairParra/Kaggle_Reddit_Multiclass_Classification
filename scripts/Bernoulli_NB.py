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

### Bernoulli Naive Bayes class implementation ### 

# ******************************************************************************

### 1. Imports ### 

import math
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass
from collections import Counter # to count classes 


from sklearn.preprocessing import OneHotEncoder # will convert to a matrix of one-hot encoded vectors


# *****************************************************************************


class BernoulliNaiveBayes(): 
    
    # WARNING: This is just a template from the previous implementation 
    # Changes will be made to fit the actual model 
    
    def __init__(self,X=np.array([[0]]), y=np.array([0]), alpha=1.0, 
                 fit_prior=True, one_hot_encode = False,  binarize=False, 
                 class_prior=np.array([0]), fit=True): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. (in np matrix format)
        y: (n x 1) vector of targets (as Series)
        
        NOTE: 
            - Features must be binary inputs
        """     
        
        self.binarize = binarize
        
        if self.binarize: 
            try: 
                # count vectorizer approach 
                count_vect = CountVectorizer(binary=True).fit(X) # fit input data 
                X = count_vect.transform(X) # get train counts
            except Exception as e: 
                print(e.with_traceback)
                print(e.__traceback__) 
                print(e.args) 
                print(e.__context__) 
           
        # Verify dimensions
        if X.shape[0] != y.shape[0]: 
            message = "Input dimensions don't match" 
            message += "\n X is {} but y is {}".format(X.shape, y.shape)
            raise ValueError(message) 
            
            
        self.n = X.shape[0] # number of observations
        self.m = X.shape[1] # number of features?? 
        self.X = X
        self.y = y 
        self.class_counts = [] # keeps tracks of class frequencies
        self.classes_ = sorted(set(y)) # ordered list of class names in y
        self.alpha = alpha # Laplace smoothing 
        self.fit_prior = fit_prior   
        self.class_prior = []
        
        # fit the prior probabilities estimates 
        if fit_prior: 
            counter = Counter(self.y)  # instantiate counter on train labels 
            self.class_counts = sorted([count for count in counter.items()], key=lambda x: x[0])# list of (count)
            self.class_prior = [freq/self.n for label, freq in self.class_counts] # fit prior probabilities for each class 
            
            
        self.K = len(self.class_counts) # number of classes
        self.params = np.zeros((self.m, self.K)) # (m parameters x K classes)
        
        # fit parameters if necessary 
        if fit: 
            # transform to pd format 
            X = pd.DataFrame(self.X) 
            y = pd.DataFrame(self.y) 
            
            # for each class 
            for k in self.classes_:
                # obtain count of y=k for that class 
                print("class {}".format(k))
            
            X = np.array(X)
            y = np.array(y)
                
        if self.X[0,0] == 0 and self.y[0] == 0: 
            print("Default initialization") 
            print("Initialized with dimensions\n X:({}) y:({})".format(self.X.shape, self.y.shape)) 
            print("Number of features: m={}".format(self.m)) 
            print("Number of observations: n={}".format(self.n)) 
            print("Class priors:\n {}".format(self.class_prior))
            print("Class counts:\n {}".format(self.class_counts))
            print("Classes:\n {}".format(self.classes_))
        
        else: 
            print("Initialized with dimensions\n X:({}) y:({})".format(self.X.shape, self.y.shape)) 
            print("Number of features: m={}".format(self.m)) 
            print("Number of observations: n={}".format(self.n)) 
            print("Class priors:\n {}".format(self.class_prior))
            print("Class counts:\n {}".format(self.class_counts))
            print("Classes:\n {}".format(self.classes_))

        
    def sigmoid(self, z): 
        return 1/(1+ np.exp(-z))
        
    
    def fit(self, X, y, fit_prior=True, class_prior=np.array([0]), binarize=False, verbose=False): 
        """ 
        Initializes the model with given parameters
        """
        self.__init__(X,y, fit_prior=fit_prior, class_prior=class_prior, binarize=binarize) # Initialize with input 
        
        
    def transform_parameters(self, tranformer): 
        """
        Converts the according to the input transformers, in a Pipeline. 
        Example usage: 
            - BernoulliNB.transform_parameters('vec'=CountVectorizer, 
                                               'tfidf'=TfidfTransformer, 
                                               'norm'=Normalizer) 
        """
        
        transf = tranformer().fit(self.X) 
        self.X = transf.transform(self.X)
            
        
    def predict(): 
        
        print("Functio for predictions")
        

        
    def predict_probabilities(): 
        
        print("Predict probabilities") 
        
        
        
    def predict_log_probabilities(): 
        
        print("Log probabilities") 
        
        
    
        
        
# ********************************************************************************       
        
### TESTS ### 


# Temporary imports for testing purposes 


from sklearn.naive_bayes import BernoulliNB # SkLearn model to compare 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.preprocessing import Normalizer
from sklearn.datasets import fetch_20newsgroups 
from sklearn.metrics import classification_report 


newsgroups = fetch_20newsgroups(subset='all') # data to test 
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, 
                                                    newsgroups.target,
                                                    train_size=0.8, 
                                                    test_size=0.2)


# convert to count vectors
count_vect = CountVectorizer().fit(X_train) # fit input data 
X_train_counts = count_vect.transform(X_train) # get train counts
X_test_counts = count_vect.transform(X_test) # get test counts

# apply tfidf 
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

## Normalization (L2)
normalizer_transformer = Normalizer().fit(X=X_train_tfidf)
X_train_normalized = normalizer_transformer.transform(X_train_tfidf)
X_test_normalized = normalizer_transformer.transform(X_test_tfidf)


# This is how you should fit the model 
clf = BernoulliNB().fit(X_train_normalized, y_train)


# Obtain metrics
y_pred = clf.predict(X_test_normalized)
report = classification_report(y_test,y_pred, 
                               target_names=newsgroups.target_names)

print(report)


# obtain a list of unique classes

classes = sorted(set(y_train))

count_classes = Counter(y_train) # instantiate Counter object 

counts = sorted([count for count in count_classes.items()], key=lambda x: x[0])
print(counts)
print(len(counts))


# *****************************************************************************

### Model Tests ### 

nb = BernoulliNaiveBayes() 
nb.fit(X_train, y_train, binarize=True)
nb.transform_parameters(TfidfTransformer) # yesss
nb.transform_parameters(Normalizer)




X = nb.X[0:100,0:100].toarray()
X = nb.X[0:1000][100] # can access this way  
print(X)


mat = np.zeros((10,10))
