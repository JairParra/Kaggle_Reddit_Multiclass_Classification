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
from tqdm import tqdm # to display progress bar
from nltk.corpus import stopwords as stw
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split 


# *****************************************************************************

### 2. Reading, preprocessing and statistics ### 

# load raw training and test data 
data_train_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id',axis=1) 
data_test_raw = pd.read_csv('../data_raw/reddit_test.csv').drop('id', axis=1)

# split into appropriate groups  
X_train = data_train_raw['comments'] 
y_train = data_train_raw['subreddits']

# ******* NOTE!!!! ******* 
real_X_test = data_test_raw['comments']
# real_y_test doesn't exist!!!! 

# Since we want to actually test using a trianing and testing set, but we don't 
# have the labels for the test set, we will further split into our own train/test
# sets for now. 
# ******* NOTE!!!! *******

    
# The data is randomly shuffled before splitting. 
X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                     y_train, 
                                                     train_size=0.95, 
                                                     test_size=0.05, 
                                                     shuffle=True, 
                                                     random_state=42)

### 3. Data Preprocessing ### 

## 3.1  Feature Preprocessing 

##  Number of stuff we want to do to preprocess: 
    # Tokenize (by default )
    # Lemmatize 
    # Remove stopwords
    # convert to lowercase 
    # Remove punctuation 
    # Remove bad characters
    # Remove links ??? 
    # remove specific crap 

 
tokenizer = word_tokenize # Re-assign tokenizer 
stemmer = SnowballStemmer(language='english')   # Initialize English Snowball stemmer 
lemmatizer = WordNetLemmatizer() # Re-assign lemmatizer 
stopwords = list(set(stw.words('english'))) # Obtain English stopwords 

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

# Apply preprocessing to featture train and test sets 
X_train = preprocess_texts(X_train)
X_test = preprocess_texts(X_test) 
real_X_test = preprocess_texts(real_X_test) 


## 3.2 Classes encoding 

# To ease computations later for the NaiveBayes implementation, we will also 
# map our unique numbers. 

labels = sorted(set(y_train)) 
label_to_num = { label:i for i, label in enumerate(labels)}
num_to_label = { i:label for i, label in enumerate(labels)}

# function to map 
f = lambda x: label_to_num[x]

# map every string label to its respective number 
y_train = [f(label) for label in y_train]
y_test = [f(label) for label in y_test]

# convert mapping to a table and save
labels_df = pd.DataFrame(zip(label_to_num))
labels_df.to_csv('../data_clean/labels.txt')


## 3.3 Save the clean data 

with open('../data_clean/X_train.txt','w') as file: 
    for line in X_train: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test.txt','w') as file: 
    for line in X_test: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test.txt','w') as file: 
    for line in real_X_test: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/y_train.txt','w') as file: 
    for line in y_train: 
        file.write(str(line) + "\n")
    file.close()
    
with open('../data_clean/y_test.txt','w') as file: 
    for line in y_test: 
        file.write(str(line) + "\n")
    file.close()
    
        
