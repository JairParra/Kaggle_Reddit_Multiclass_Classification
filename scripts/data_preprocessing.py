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
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm # to display progress bar
from nltk.corpus import stopwords as stw
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split 

sns.set()

# intialize large web cropus trained langauge model 
# loading is a bit slow but it is more powerful 
nlp = spacy.load('en_core_web_lg')


# *****************************************************************************

### 2. Reading, preprocessing and statistics ### 

# load raw training and test data 
data_train_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id',axis=1) 
data_test_raw = pd.read_csv('../data_raw/reddit_test.csv').drop('id', axis=1)

# split into appropriate groups  
X_train_raw = data_train_raw['comments'] 
y_train_raw = data_train_raw['subreddits']

# ******* NOTE!!!! ******* 
real_X_test = data_test_raw['comments']
# real_y_test doesn't exist!!!! 

# Since we want to actually test using a trianing and testing set, but we don't 
# have the labels for the test set, we will further split into our own train/test
# sets for now. 
# ******* NOTE!!!! *******

    
# The data is randomly shuffled before splitting. 
X_train, X_test, y_train, y_test = train_test_split(X_train_raw, 
                                                     y_train_raw, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
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
    
# NOTE: Tags to remove are are follows: 
    # https://spacy.io/api/annotation
    
tags = ["PRON","DET","ADP","PUNCT","CONJ","CCONJ","SCONJ","NUM","SYM","X","PART","SPACE"] # tags to filter 
stemmer = SnowballStemmer(language='english')   # Initialize English Snowball stemmer 
stopwords = list(set(stw.words('english'))) # Obtain English stopwords 
tokenizer = word_tokenize


def preprocess_texts(texts, 
                     lemmatize=False,  
                     stem=False, 
                     lemma_all=False, 
                     stem_all=False,
                     lowercase=True, 
                     remove_stopwords=True, 
                     filter_all=True, 
                     filter_tags=[], 
                     verbose=False):
    
    if verbose: 
        print(texts)
    
#    docs= nlp.pipe(texts, disable=['ner','textcat','entity_linker', 'entity_ruler' ]) # fit all the texts to the spaCy language model 
    clean_texts = [] # to store the result 
    
    with nlp.disable_pipes('ner'): 
    
        for text in tqdm(texts): 
            
            doc = nlp(text)
            
    #    # for each processed document (text in the input list) 
    #    for doc in tqdm(docs, desc="Document"): 
            
            # To store the final tokens for the given text
            tokens = []
            
            
            # stem_all or lemma_all will apply: 
                # 1. stemming/lemmatizing
                # 2. lowercase 
                #.3. filter POS tags 
                # 4. filter stopwords 
                # 5. remove single letters 
            
            if lemma_all: 
                
                tokens = [re.sub(r"[^a-zA-Z0-9]","", token.lemma_.lower()) for token in doc if token.pos_ not in filter_tags 
                          and token.text not in stopwords and token.lemma_ not in stopwords
                          and len(token.lemma_) > 1]
                          
            # apply all preprocessing with stemmin g
            elif stem_all: 
                
                tokens = [re.sub(r"[^a-zA-Z0-9]","", stemmer(token.text.lower())) for token in doc if token.pos_ not in filter_tags 
                          and token.text not in stopwords and token.lemma_ not in stopwords
                          and len(token.lemma_) > 1 and token.text.isalpha() ]  
                
            # else consider case by case 
            else: 
                    
                if stem: 
                    # stem with nltk
                    tokens = [stemmer(token.text) for token in doc if token.pos_ not in filter_tags
                              and len(token.text) >1 ] 
                    
                if lemmatize: 
                    # lemmatize with spacy
                    # NOTE: pronouns are lemmatized as "PRON" 
                    tokens = [token.lemma_ for token in doc if token.pos_ not in filter_tags
                              and len(token.text) > 1]
                    
                if lowercase: 
                    tokens = [token.lower() for token in tokens]
                    
                if remove_stopwords: 
                    tokens = [token for token in tokens if token not in stopwords] 
                    
                if filter_all: 
                    tokens = [token for token in tokens if token.isalpha()]
                    
            
            # after any preprocessing
            clean_text = ' '.join(tokens)
            
            if len(clean_text) <= 2: 
                clean_text = ' '.join([re.sub(r"[^a-zA-Z0-9]","", token) for token in tokenizer(text)])
                
            
            clean_texts.append(clean_text)  
            
    return clean_texts


# EXAMPLE: 
sentences = ["""&gt;Oh man. All I know is that if you're story involves you being in 8th grade using MSN messenger then you are not old enough to be able to say "years ago"
Why not?""", 
"""👏 United 👏 in 👏 for 👏 Lukaku 👏 T H I C C 👏 and 👏 ready 👏 to 👏 score 👏 goals 👏 Chelsea 👏 still 👏 in 👏 for 👏 him 👏 too 👏 Everton 👏 want 👏 dat 👏 P 👏"""]

ex = [re.sub(r"[^a-zA-Z0-9]","",token) for token in tokenizer(sentences[0])] 
print(ex)

result = preprocess_texts(sentences,lemma_all = True,filter_tags=tags)
print(result)

print(len(sentences[1])) 
print(type(result[0]))
# EXAMPLE end


# Apply preprocessing to featture train and test sets (SLOW!!!)
X_train = preprocess_texts(X_train, lemma_all = True,filter_tags=tags)
X_test = preprocess_texts(X_test, lemma_all = True,filter_tags=tags) 
real_X_test = preprocess_texts(real_X_test, lemma_all = True,filter_tags=tags) 

# NOTE: If you wish to make any changes, you have to make sure that you re-load the original 
# data in memory once again. Failing to do so may cause unwanted bugs. 

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

# Ensure proper data type
X_train = pd.Series(X_train[0])
X_test = pd.Series(X_test)

## 3.3 Save the clean data 

with open('../data_clean/X_train.txt','w', encoding='utf-8') as file: 
    for line in X_train: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test.txt','w', encoding='utf-8') as file: 
    for line in X_test: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test.txt','w', encoding='utf-8') as file: 
    for line in real_X_test: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/y_train.txt','w', encoding='utf-8') as file: 
    for line in y_train: 
        file.write(str(line) + "\n")
    file.close()
    
with open('../data_clean/y_test.txt','w', encoding='utf-8') as file: 
    for line in y_test: 
        file.write(str(line) + "\n")
    file.close()
    
        
###############################################################################
    
### 3. Stastics ### 
    
# Label distribution plot
plt.figure(figsize=(20,10))
sns.countplot(y_train_raw)
plt.savefig('../figs/labels_countplot.png')


