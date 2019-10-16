# The code in this file should be merged into Bernoulli_NB.py
# I just wanted to write code from scratch in a new file

# Define some notation:
# m = number of features (number of columns)
# n = number of examples in train or test set (number of rows)
# k = number of classes (k = 20 for this miniproject dataset)

import numpy as np
import pandas as pd

class BernoulliNB():
    """Implementing a Bernoulli Naive Bayes model for multiclass classification from scratch"""

    def __init__(self, X_train, y_train, k):
        """
        Constructor to create a new BernoulliNB instance

        X_train     = feature matrix (numpy ndarray) of dataset
        y_train     = labels vector
        k           = number of classes in dataset
        X           = X_train
        y           = y_train
        n           = number of training examples (= number of rows of X)
        m           = number of features (= number of columns of X)

        """
        self.X = X_train
        self.y = y_train
        self.k = k
        self.n, self.m = self.X.shape

    def fit(self):
        """
        Fit training data and calculate 2 class variables:

        1.  (k,)-shaped numpy array theta_k of prior values for each class --> P(y=i) for every class i
        2.  kxm parameterMatrix (k rows, m columns) where each row is the conditional probability 
            of features in that class "how often is feature j equal to 1 for examples from class i?"

        Adapted from COMP 551 Fall 2019 Slides, Lecture 10, Page 4
        from https://cs.mcgill.ca/~wlh/comp551/slides/10-ensembles.pdf
        """
        self.theta_k = np.zeros(self.k)
        self.parameterMatrix = np.zeros(shape=(self.k,self.m))

        for i in range(self.k): #for each class:
            # we find the conditional probability for each feature given each class (theta_j_k) using np.mean)

            currentClassData = self.X[self.y == i]#select all rows belonging to class i
            self.theta_k[i]= currentClassData.shape[0]/self.n #number of examples in class i / total # of rows
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

        """
        n = X_test.shape[0]
        matrix = np.zeros(shape=(n,self.k))

        for i in range(n):
            X = X_test[i,:][:,np.newaxis] #take current row
            matrix[i] = np.log(self.theta_k) + (np.log(self.parameterMatrix) @ X).flatten() + ( np.log(1-self.parameterMatrix) @ (1-X) ).flatten()
            
        return np.argmax(matrix, axis=1)

###### TESTING ##########

# load training  data 
X_train = pd.read_csv('../data_clean/X_train.txt', header=None)
y_train = pd.read_csv('../data_clean/y_train.txt', header=None)

# load testing data 
X_test = pd.read_csv('../data_clean/X_test.txt', header=None)

print(y_train)
