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

import math
import scipy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # easier & prettier visualization 
from tqdm import tqdm # to display progress bar
from numpy import transpose as T # because it is a pain in the ass


class BernoulliNaiveBayes(): 
    
    # WARNING: This is just a template from the previous implementation 
    # Changes will be made to fit the actual model 
    
    def __init__(self,X=np.array([[0]]), y=np.array([0]) ): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. (in np matrix format)
        y: (n x 1) vector of targets (as Series)
        
        """     
        np.random.seed(42) 
    
        if X[0,0] == 0 and y[0] == 0: 
            print("Default initialization") 
            
        # Verify dimensions
        if X.shape[0] != y.shape[0]: 
            message = "Input dimensions don't match" 
            message += "\n X is {} but y is {}".format(X.shape, y.shape)
            raise ValueError(message) 
            
        self.n = X.shape[0]
        self.m = X.shape[1] + 1 # Because of intercept term 
        X_0 = np.ones(self.n) # intercept features 
        self.X = np.c_[X_0,X] # concatenate 
        self.w = np.random.rand(self.m, 1) # randomly initialize weights 
        self.y = y 
        self.last_train_losses = [0] # this will be used to plot training loss
        
        print("Initialized with dimensions\n X:({}) y:({})".format(self.X.shape, self.y.shape)) 
        print("Number of features: m={}".format(self.m)) 
        print("Number of observations: n={}".format(self.n)) 
        print("Number of weights: len(w)={}".format(len(self.w)))
            
        
        
