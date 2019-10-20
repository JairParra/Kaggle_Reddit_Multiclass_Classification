# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:26:34 2019

@ COMP 551: Applied Machine Learning (Winter 2019) 
# Team Members: 

@ Hair Albeiro Parra Barrera 
@ ID: 260738619 

@ Ashray Malleshachari
@ ID: 260838256
    
@ Hamza Rizwan
@ ID: 260816900
"""

# Define some notation:
# m = number of features (number of columns)
# n = number of examples in train or test set (number of rows)
# k = number of classes (k = 20 for this miniproject dataset)

# *****************************************************************************

### Bernoulli Naive Bayes class implementation ### 

# ******************************************************************************

### 1. Imports ### 

import time
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from crossvalidation import kfold_accuracy

# *****************************************************************************

### 2. Implementation ### 

class BernoulliNB():
    """
    Implementing a Bernoulli Naive Bayes model for multiclass classification from scratch
    """

    def fit(self, X_train, y_train, k):
        """

        Parameters: 
        X_train     = feature matrix (numpy ndarray) of dataset
        y_train     = labels vector
        k           = number of classes in dataset

        X           = X_train
        y           = y_train
        n           = number of training examples (= number of rows of X)
        m           = number of features (= number of columns of X)

        Fit training data and calculate 2 class variables:

        1.  (k,)-shaped numpy array theta_k of prior values for each class --> P(y=i) for every class i
        2.  kxm parameterMatrix (k rows, m columns) where each row is the conditional probability 
            of features in that class "how often is feature j equal to 1 for examples from class i?"

            Vectorized implementation with laplace smoothing by using np.mean to calculate the conditional 
            probability matrix of features for each class (theta_j_k).

        Conditional Probabilities (theta_j_k) calculation equation originally from COMP 551 Fall 2019 Slides, Lecture 10, Page 4
        at https://cs.mcgill.ca/~wlh/comp551/slides/10-ensembles.pdf. We've applied a column wise sum using np.mean
        to achieve the same result for each class.
        """

        X = X_train
        y = y_train
        n,m = X.shape

        self.theta_k = np.zeros(k)
        self.parameterMatrix = np.zeros(shape=(k,m))

        for i in range(k): #for each class:
            # we find the conditional probability for each feature given each class (theta_j_k) using np.mean)

            currentClassData = X[ (y==i).flatten(), : ] # select all rows belonging to class i
            self.theta_k[i]= currentClassData.shape[0]/n #number of examples in class i / total # of rows

            #### LAPLACE SMOOTHING ####
            # add 2 rows so numRows in each class increases by 2 and numRows in each class where feature j is on increases by 1
            ones = np.ones(shape=(1,m)) #row of ones
            zeros = np.zeros(shape=(1,m)) #row of zeros
            currentClassData = np.vstack([currentClassData, ones, zeros]) #add ones and zeros rows to currentClassData
            #### LAPLACE SMOOTHING ####

            self.parameterMatrix[i] = np.mean(currentClassData, axis=0) #update each row of the parameter matrix


    def predict(self, X_test):
        """
        X_test = nxm matrix with n test examples (rows) and m features (columns)
        Returns a nx1 labels vector that assigns a class number (from 0 to k-1) for each row in X_test.

        Implementation details:

        -We're trying to calculate the probability of each test example (each row of X_test) belonging
        to class 1,2,3,..,k.
        -We can represent this as a 1xk row vector[ P(y=1|X) P(y=2|X) P(y=3|X) ... P(y=k|X) ]
        -Stack n of these 1xk row vectors to create a nxk matrix 
        -After that, calculate the argmax for each row to find the class with the highest probability value,
        this should return a nx1 vector of labels

        X_test (nxm) multiplied by parameterMatrix.T (mxk) gives a nxk matrix in which every row contains k values
        representing the probabilities of that row belonging to each of the k classes. This vectorized implementation
        uses no loops and runs expontentially faster than iterating over every feature, example, and class to do calculations.

        Posterior probability calculation equation originally from COMP 551 Fall 2019 Slides, Lecture 10, Page 6
        at https://cs.mcgill.ca/~wlh/comp551/slides/10-ensembles.pdf.

        We have vectorized this equation to be able to make predictions by performing just 2 matrix multiplications 
        which reduced our training time on the reddit dataset to about 10 seconds. By performing operations on the
        entire dataset at once, we can make predictions much faster than iterating through the dataset and performing
        operations on each row 

        """
        matrix = (X_test @ np.log(self.parameterMatrix.T)) + ((1-X_test)@(np.log(1-self.parameterMatrix.T))) + np.log(self.theta_k)
        return np.argmax(matrix, axis=1)
        
# ********************************************************************************               

# load training  data 
X_train = pd.read_csv('../data_clean/X_train.txt', header=None)
y_train = pd.read_csv('../data_clean/y_train.txt', header=None).to_numpy()

# load testing data 
X_test = pd.read_csv('../data_clean/X_test.txt', header=None)
y_test = pd.read_csv('../data_clean/y_test.txt', header=None).to_numpy()

#Some Data Cleaning
def alpha(string):
    """return only alphabetic chars"""
    return re.sub("[^a-zA-Z]", " ", string)
X_train.iloc[:,0] = X_train[0].apply(alpha)
X_test.iloc[:,0] = X_test[0].apply(alpha)

#sklearn tfidfvectorizer = countvectorizer + tfidftransformer
vectorizer = TfidfVectorizer(max_features=5000, 
                            min_df=5, 
                            max_df=0.9,
                            binary=True)

#vectorize train and test sets & convert sparse matrices to regular numpy arrays
X_train = vectorizer.fit_transform(X_train[0]).A
X_test = vectorizer.transform(X_test[0]).A

normalizer = Normalizer(norm='l2') #normalizing doesn't seem to improve accuracy
normalizer.fit_transform(X_train, y_train)
normalizer.transform(X_test)

nb = BernoulliNB()

print(kfold_accuracy(model=nb, X=X_train, y=y_train))

