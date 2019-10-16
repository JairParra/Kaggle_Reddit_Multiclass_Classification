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
from sklearn.feature_extraction.text import CountVectorizer 



# *****************************************************************************


class BernoulliNaiveBayes(): 
    
    # WARNING: This is just a template from the previous implementation 
    # Changes will be made to fit the actual model 
    
    def __init__(self,X=np.array([[0]]), y=np.array([0]), alpha=1.0, 
                 fit_prior=True,  fit=False, binarize=False, 
                 class_prior=np.array([0]), ): 
        """
        X: (n x m) matrix of features with n observations 
         and m features. (in np matrix format)
        y: (n x 1) vector of numeric targets (as Series)
        
        NOTE: 
            - Features must be binary inputs
        """     
        
        self.binarize = binarize # parameter to binarize the input 
        
        # if the model receives text input
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
            
                    
            
        if self.X[0,0] == 0 and self.y[0] == 0: 
            print("Default initialization") 
            print("Initialized with dimensions\n X:{} y:{}".format(self.X.shape, self.y.shape)) 
            print("Number of features: m={}".format(self.m)) 
            print("Number of observations: n={}".format(self.n)) 
            print("Class priors:\n {}".format(self.class_prior))
            print("Class counts:\n {}".format(self.class_counts))
            print("Classes:\n {}".format(self.classes_))
        
        else: 
            print("Initialized with dimensions\n X:{} y:{}".format(self.X.shape, self.y.shape)) 
            print("Number of features: m={}".format(self.m)) 
            print("Number of observations: n={}".format(self.n)) 
            print("Class priors:\n {}".format(self.class_prior))
            print("Class counts:\n {}".format(self.class_counts))
            print("Classes:\n {}".format(self.classes_))
            
            
        self.K = len(self.class_counts) # number of classes
        self.params = np.zeros((self.m, self.K)) # (m parameters x K classes)
        
        # fit parameters if call is made
        if fit: 
            
            self.fit_params()
                       
        
    def fit_params(self): 
        
        # transform to pd format 
        print("X self shape = {}".format(self.X.shape))
        print("type: ", type(self.X))
                    
        X = self.X
        y = self.y 
        
        print("X shape = {}".format(X.shape))
        print("y_shape {}".format(y.shape))
        
        # for each class 
        for k in tqdm(self.classes_):
            # obtain count of y=k for that class 
            n = self.class_counts[k][1]
            # display counts
            print("class {}, count ={}\n".format(k,n)) 
            
            # for each feature
            for j in range(self.m): 
                
                summation = 0 # initialize summation 
                
                # for each observation 
                for i in range(self.n): 
                    
                    # if word j of observation i appears and its label is k 
                    if X[i,j] != 0 and y[i] == k: 
                        summation += 1 # increase the summation 
                        
                # assign the value 
                self.params[j,k] = summation 
                        
                            
    def sigmoid(self, z): 
        return 1/(1+ np.exp(-z))
        
    
    def fit(self, X, y, fit_prior=True, class_prior=np.array([0]), 
            binarize=False, fit_params=True, verbose=False): 
        """ 
        Initializes the model with given parameters
        """
        self.__init__(X,y, fit_prior=fit_prior, class_prior=class_prior, binarize=binarize) # Initialize with input 
        
        if fit_params: 
            self.fit_params() 
        
        
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

import re        
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer 

# Initialize spaCy's English model 
nlp = spacy.load('es_core_news_md')
    
### 3. Data Preprocessing ### 
 
# Re-assign tokenizer 
tokenizer = word_tokenize
    
# Initialize English Snowball stemmer 
stemmer = SnowballStemmer(language='english')    

# Re-assign lemmatizer 
lemmatizer = WordNetLemmatizer()

# Obtain English stopwords 
stopwords = set(stopwords.words('english'))

# Ex. 
stemmer.stem("Cats")
lemmatizer.lemmatize("corpora")   
    

def preprocess_text(sentence, stem=False, lemmatize=False): 
    """
    Cleans text list by applying the following steps: 
        1. Tokenize the input sentence 
        2. Remove punctuation, symbols and unwanted characters
        3. Convert the tokens to lowercase 
        4. Stem or lemmatize (according to input)
        5. Remove stopwords and empty strings
    """
    # Tokenize
    tokens = tokenizer(sentence) 
    
    # Remove punctuation & symbols
    tokens = [re.sub(r"[^a-zA-Z]","", token) for token in tokens ]
    
    # convert to lowercase 
    tokens = [token.lower() for token in tokens]
    
    # Stem or lemmatize
    if stem: 
        tokens = [stemmer.stem(token) for token in tokens] 
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens] 
    
    # remove stopwords and empty strings 
    tokens = [token for token in tokens if token not in stopwords
              and len(token) > 1] 
    
    return ' '.join(tokens)


def preprocess_texts(text_list, stem=False, lemmatize=False): 
    """ 
    Applies preprocess text on a list of texts. 
    """ 
    return [preprocess_text(text, stem=stem, lemmatize=lemmatize) for text in text_list] 
    

# Ex. with tokenization 
preprocess_text("This is a sentence, it isn't a cake!! @@ .", stem=True) 
""" Out[25]: 'sentenc nt cake' """ 



# ******************************************************************************       
        
        
### TESTS ### 

# Temporary imports for testing purposes 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.preprocessing import Normalizer


# load raw training  data 
data_train_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id',axis=1) 

# load raw testing data 
data_test_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id', axis=1)

# obtain train and test data  
X_train = data_train_raw['comments'] 
y_train = data_train_raw['subreddits']

X_test = data_test_raw['comments']
y_test=  data_test_raw['subreddits']  

# list of targets 
targets = sorted(set(y_train))

# perprocess the texts 
X_train = preprocess_texts(X_train, lemmatize=True) # this thing will be slow 


# obtain labels mapping
labels = sorted(set(y_train)) 
label_to_num = { label:i for i, label in enumerate(labels)}
num_to_label = { i:label for i, label in enumerate(labels)}

# function to map 
f = lambda x: label_to_num[x]

# map every string label to its respective number 
y_train = pd.Series([f(label) for label in y_train])
y_test = pd.Series([f(label) for label in y_test])

# *****************************************************************************

### Model Tests ### 

# use just a portion for testing whether the model is actually doing anything at all 
X_slice = X_train[0:100]
y_slice = y_train[0:100]

nb = BernoulliNaiveBayes() # initialize the model 
nb.fit(X_slice, y_slice, binarize=True, fit_params=True) # binarize and fit parameters
nb.transform_parameters(TfidfTransformer) # transform parameters 
nb.transform_parameters(Normalizer) # transform again  
nb.fit_params() # fit parameters again  
print(nb.class_counts) # display class counts 
params = nb.params # observe the parameters matrix. 


# Note nb.X is a sparse matrix!!! 
X_slice = nb.X[0:100,0:100].toarray() # This will conver to array, but if too big, RIP memory 
y_slice = nb.y[0:100] # slice 
X = nb.X[0:1000][100] # can access this way  
print(type(nb.X)) # sparse matrix 


# *****************************************************************************

## The following code shows how to apply the different transformers, 
# This is what is happening inside the Bernoulli NB class. 
# Just for demonstration, this part will be remouved. 

# convert to count vectors
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


