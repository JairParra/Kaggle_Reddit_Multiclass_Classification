# The code in this file should be merged into Bernoulli_NB.py
# I just wanted to write code from scratch in a new file

# Define some notation:
# m = number of features (number of columns)
# n = number of examples in train or test set (number of rows)
# k = number of classes (k = 20 for this miniproject dataset)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class BernoulliNB():
    """
    Implementing a Bernoulli Naive Bayes model for multiclass classification from scratch
    Parameters: X_train, y_train, k (# classes)
    """

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
            Calculated using a vectorized implementation by using np.mean to calculate the conditional 
            probability matrix of features for each class

        Adapted from COMP 551 Fall 2019 Slides, Lecture 10, Page 4
        from https://cs.mcgill.ca/~wlh/comp551/slides/10-ensembles.pdf
        """
        self.theta_k = np.zeros(self.k)
        self.parameterMatrix = np.zeros(shape=(self.k,self.m))

        for i in range(self.k): #for each class:
            # we find the conditional probability for each feature given each class (theta_j_k) using np.mean)

            currentClassData = self.X[(self.y==i).flatten(), :]#select all rows belonging to class i
            self.theta_k[i]= currentClassData.shape[0]/self.n #number of examples in class i / total # of rows

            #### LAPLACE SMOOTHING ####
            # add 2 rows so num of rows in class increase by 2 and num of rows where feature j is on increases by 1
            ones = np.ones(shape=(1,self.m)) #row of ones
            zeros = np.zeros(shape=(1,self.m)) #row of zeros
            currentClassData = np.vstack([currentClassData, ones, zeros]) #add these rows to currentClassData
            #### LAPLACE SMOOTHING ####

            self.parameterMatrix[i] = np.mean(currentClassData, axis=0) #update each row of the parameter matrix


    def predict(self, X_test):
        """
        X_test = nxm matrix with n test examples (rows) and m features (columns)
        Returns a nx1 labels vector that assigns a class number (from 0 to k-1) for each row in X_test.

        Implementation details:

        This is a vectorized implementation of calculating the probability of each test example belonging to
        all k classes. After we calculate P(y=1|X), P(y=2|X), ..., P(y=k|X) we pick the class that has the highest
        probability.

        -We're trying to calculate the probability of each test example (each row of X_test) belonging
        to class 1,2,3,..,k.
        -We can represent this as a 1xk row vector[ P(y=1|X) P(y=2|X) P(y=3|X) ... P(y=k|X) ]
        -Stack n of these 1xk row vectors to create a nxk matrix 
        -After that, calculate the argmax for each row to find the class with the highest probability value,
        this should return a nx1 vector of labels

        Adapted from COMP 551 Fall 2019 Slides, Lecture 10, Page 6
        from https://cs.mcgill.ca/~wlh/comp551/slides/10-ensembles.pdf

        """
        matrix = (X_test @ np.log(self.parameterMatrix.T)) + ((1-X_test)@(np.log(1-self.parameterMatrix.T))) + np.log(self.theta_k)
        return np.argmax(matrix, axis=1)


###### TESTING ##########

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


#one hot encoding
vectorizer = CountVectorizer(max_features=5000, 
                            min_df=5, 
                            max_df=0.9,
                            binary=True)
X_train = vectorizer.fit_transform(X_train[0]).A
X_test = vectorizer.transform(X_test[0]).A

#train model
nb = BernoulliNB(X_train=X_train, y_train=y_train, k=20)
nb.fit()
y_pred = nb.predict2(X_test)


print(accuracy_score(y_test, y_pred))



