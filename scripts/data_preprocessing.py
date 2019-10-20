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
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import chi2 # to find out most related terms

sns.set()

# intialize large web cropus trained langauge model 
# loading is a bit slow but it is more powerful 
nlp = spacy.load('en_core_web_lg')


# *****************************************************************************

### 2. Reading the data ### 

# load raw training and test data 
data_train_raw = pd.read_csv('../data_raw/reddit_train.csv').drop('id',axis=1) 
data_test_raw = pd.read_csv('../data_raw/reddit_test.csv').drop('id', axis=1)

# split into appropriate groups  
real_X_train = data_train_raw['comments'] 
real_y_train = data_train_raw['subreddits']

# ******* NOTE!!!! ******* 
real_X_test = data_test_raw['comments']
# real_y_test doesn't exist!!!! 


## 2.1 Stastics
    
# Label distribution plot
plt.figure(figsize=(20,10))
sns.countplot(real_y_train)
plt.savefig('../figs/labels_countplot.png')

# ******* NOTE!!!! *******
# Since we want to actually test using a trianing and testing set, but we don't 
# have the labels for the test set, we will further split into our own train/test
# sets for now. 

# Therefore, we will also split another model which is the full preprocessed 
# training dataset. Then, we will output predictions with it, and these will be 
# the ones we will submit. 
# ******* NOTE!!!! *******


## 2.2 Splitting the artifical data 

# The data is randomly shuffled before splitting. 
X_train, X_test, y_train, y_test = train_test_split(real_X_train, 
                                                     real_y_train, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
                                                     shuffle=True, 
                                                     random_state=42)


# Create a copy of these 
X_train_raw, X_test_raw, y_train_raw, y_test_raw = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()


# Save raw data: 

# RAW
with open('../data_clean/X_train_raw.txt','w', encoding='utf-8') as file: 
    for line in X_train_raw: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test_raw.txt','w', encoding='utf-8') as file: 
    for line in X_test_raw: 
        file.write(line + "\n")
    file.close()  
    
with open('../data_clean/real_X_train_raw.txt','w', encoding='utf-8') as file: 
    for line in real_X_train: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test_raw.txt','w', encoding='utf-8') as file: 
    for line in real_X_test: 
        file.write(line + "\n")
    file.close()

# *****************************************************************************

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
tags_2 = ["NUM","SYM","X","PART","SPACE"] # tags to filter 
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
                     min_len = 1, 
                     verbose=False):
    
    if verbose: 
        print(texts)
    
    clean_texts = [] # to store the result 
    
    # using language model without Named Entity Recognition 
    with nlp.disable_pipes('ner'): 
    
        # for each text in the input corpus 
        for text in tqdm(texts): 
            
            # fit the language model 
            doc = nlp(text)
            
            # To store the final tokens for the given text
            tokens = []
            
            
            # stem_all or lemma_all will apply: 
                # 1. stemming/lemmatizing
                # 2. lowercase 
                #.3. filter POS tags 
                # 4. filter stopwords 
                # 5. remove single letters 
            
            if lemma_all: 
                
                tokens = [re.sub(r"[^a-zA-Z0-9]","", token.lemma_.lower()) for token in doc 
                          if token.pos_ not in filter_tags 
                          and token.text not in stopwords 
                          and token.lemma_ not in stopwords
                          and len(token.lemma_) > min_len
                          and len(token.text) > min_len
                          and token.text.isalpha() ]
                          
            # apply all preprocessing with stemmin g
            elif stem_all: 
                
                tokens = [re.sub(r"[^a-zA-Z0-9]","", stemmer.stem(token.text.lower())) for token in doc 
                          if token.pos_ not in filter_tags 
                          and token.text not in stopwords
                          and token.lemma_ not in stopwords
                          and len(token.lemma_) > min_len
                          and len(token.text) > min_len
                          and token.text.isalpha() ]  
                
            # else consider case by case 
            else: 
                    
                if stem: 
                    # stem with nltk
                    tokens = [stemmer(token.text) for token in doc 
                              if token.pos_ not in filter_tags
                              and len(token.text) >1 ] 
                    
                if lemmatize: 
                    # lemmatize with spacy
                    # NOTE: pronouns are lemmatized as "PRON" 
                    tokens = [token.lemma_ for token in doc 
                              if token.pos_ not in filter_tags
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

sentences += list(data_train_raw['comments'][10:20])


ex = [re.sub(r"[^a-zA-Z0-9]","",token) for token in tokenizer(sentences[0])] 
print(ex)

# lemmatize + all tags
result = preprocess_texts(sentences,lemma_all = True,filter_tags=tags)
print("Lemmatization + all tags: \n", result)

# stemming + less tags 
result = preprocess_texts(sentences,stem_all = True,filter_tags=tags_2)
print("Lemmatization +  less tags: \n", result)

print(len(sentences[1])) 
print(type(result[0])) # type of each item in the list
# EXAMPLE end


# Apply LEMMA preprocessing to featture train and test sets (SLOW!!!)
real_X_train_lemma = preprocess_texts(real_X_train, lemma_all = True, filter_tags=tags)
real_X_test_lemma = preprocess_texts(real_X_test, lemma_all = True,filter_tags=tags) 

X_train_lemma, X_test_lemma, y_train, y_test = train_test_split(real_X_train_lemma, 
                                                     real_y_train, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
                                                     shuffle=True, 
                                                     random_state=42)

# LEMMA with less restricted tag filter
real_X_train_lemma2 = preprocess_texts(real_X_train, lemma_all = True, filter_tags=tags_2) 
real_X_test_lemma2 = preprocess_texts(real_X_test, lemma_all = True,filter_tags=tags_2) 

X_train_lemma2, X_test_lemma2, y_train, y_test = train_test_split(real_X_train_lemma2, 
                                                     real_y_train, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
                                                     shuffle=True, 
                                                     random_state=42)


# Apply STEM preprocessing to all of these 
real_X_train_stem = preprocess_texts(real_X_train, stem_all = True, filter_tags=tags)
real_X_test_stem = preprocess_texts(real_X_test, stem_all = True,filter_tags=tags) 

X_train_stem, X_test_stem, y_train, y_test = train_test_split(real_X_train_stem, 
                                                     real_y_train, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
                                                     shuffle=True, 
                                                     random_state=42)


# Apply STEM preprocessing to all of these 
real_X_train_stem2 = preprocess_texts(real_X_train, stem_all = True, filter_tags=tags_2)
real_X_test_stem2 = preprocess_texts(real_X_test, stem_all = True,filter_tags=tags_2) 


X_train_stem2, X_test_stem2, y_train, y_test = train_test_split(real_X_train_stem2, 
                                                     real_y_train, 
                                                     train_size=0.9, 
                                                     test_size=0.1, 
                                                     shuffle=True, 
                                                     random_state=42)


### Bad examples removal 

length_text = lambda x: len(x)
df = pd.DataFrame(list(zip(real_X_train, real_y_train)), columns=['comments','subreddits'])
df['length_comment'] = df['comments'].apply(length_text) # obtain length of each row 
df_filtered = df.loc[df['length_comment'] > 2] # exclude rows with length less than 3 

### 

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

## 3.3 Save the clean data  

# LEMMATIZED
with open('../data_clean/X_train.txt','w', encoding='utf-8') as file: 
    for line in X_train_lemma: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test.txt','w', encoding='utf-8') as file: 
    for line in X_test_lemma: 
        file.write(line + "\n")
    file.close()  
    
with open('../data_clean/real_X_train.txt','w', encoding='utf-8') as file: 
    for line in real_X_train_lemma: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test.txt','w', encoding='utf-8') as file: 
    for line in real_X_test_lemma: 
        file.write(line + "\n")
    file.close()
    
# LEMMATIZED 2
with open('../data_clean/X_train2.txt','w', encoding='utf-8') as file: 
    for line in X_train_lemma2: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test2.txt','w', encoding='utf-8') as file: 
    for line in X_test_lemma2: 
        file.write(line + "\n")
    file.close()  
    
with open('../data_clean/real_X_train2.txt','w', encoding='utf-8') as file: 
    for line in real_X_train_lemma2: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test2.txt','w', encoding='utf-8') as file: 
    for line in real_X_test_lemma2: 
        file.write(line + "\n")
    file.close()
    
    
    
# STEMMED 1 
with open('../data_clean/X_train_STEM.txt','w', encoding='utf-8') as file: 
    for line in X_train_stem: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test_STEM.txt','w', encoding='utf-8') as file: 
    for line in X_test_stem: 
        file.write(line + "\n") 
    file.close()  
    
with open('../data_clean/real_X_train_STEM.txt','w', encoding='utf-8') as file: 
    for line in real_X_train_stem: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test_STEM.txt','w', encoding='utf-8') as file: 
    for line in real_X_test_stem: 
        file.write(line + "\n")
    file.close()
    
# STEMMED 2 
with open('../data_clean/X_train_STEM2.txt','w', encoding='utf-8') as file: 
    for line in X_train_stem2: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/X_test_STEM2.txt','w', encoding='utf-8') as file: 
    for line in X_test_stem2: 
        file.write(line + "\n") 
    file.close()  
    
with open('../data_clean/real_X_train_STEM2.txt','w', encoding='utf-8') as file: 
    for line in real_X_train_stem2: 
        file.write(line + "\n")
    file.close()
    
with open('../data_clean/real_X_test_STEM2.txt','w', encoding='utf-8') as file: 
    for line in real_X_test_stem2: 
        file.write(line + "\n")
    file.close()
    

# TARGETS    
with open('../data_clean/y_train.txt','w', encoding='utf-8') as file: 
    for line in y_train: 
        file.write(str(line) + "\n")
    file.close()
    
with open('../data_clean/y_test.txt','w', encoding='utf-8') as file: 
    for line in y_test: 
        file.write(str(line) + "\n")
    file.close()

with open('../data_clean/real_y_train.txt','w', encoding='utf-8') as file: 
    for line in real_y_train: 
        file.write(line + "\n")
    file.close()
    
# ****************************************************************************

    
    
    
    
    
    
    
