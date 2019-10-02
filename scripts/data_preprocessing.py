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

#### 1. Imports ### 

import re 
import spacy
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm # to display progress bar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# *****************************************************************************

### 2. Reading, preprocessing and statistics ### 

# load raw training  data 
data_train_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id',axis=1) 

# load raw testing data 
data_test_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id', axis=1)


##  Number of stuff we want to do to preprocess: 
    # Tokenize 
    # Lemmatize
    # Remove stopwords
    # convert to lowercase 
    # Remove punctuation 
    # Remove bad characters
    # Remove links ??? 
    # remove specific crap 


# Create a function for data preprocessing: 
def clean_text(text): 
    
    print("Holy shit, this function is smart af ")
    print("AHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    
    
