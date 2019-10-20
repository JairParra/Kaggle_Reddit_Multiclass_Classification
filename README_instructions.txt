***README*** 

The present project has been organized in terms of a repository, which can be found at 
https://github.com/JairParra/Kaggle_Reddit_Multiclass_Classification

- data_clean: contains all pre-processed data , coming from the data_preprocessing.py script 
		as well as predictions, coming from the main script. 
- data_raw: original data 
- figs: any output images or other images to display 
- literature: a list of some of the papers we used in the report. 
- notebooks: custom tutorials for using specific tools (not too important, can ignore) 
- scripts: the main scripts used in this project are located here. i.e. 
	1. Bernoulli_NB: our implementation of Bernoulli Naive bayes. It also contains test of it. 
	2. crossvalidaton.py: script that implements cross-validation used in testing the Bernoulli Naive Bayes
	3. data_preprocessing.py: this script should be called first if starting from scratch; it pre-process the data. 
	   Since this takes some time, we suggest directly using the outputs in data_clean, which are called in the main.py script. 
	4. main.py: main file of the script whcih implements a variety of models including multinomial NB, SVM's etc.  and stores 
	 the result. We suggest running this code chunk by chunk since some parts might take some time. In particular, 	
	please follow the instructions below. 
- tests: tests scripts, to be ignored. 

*** instructions for main.py *** 

1. Run the 1. Imports section 
2. Run the Load and clean data section 
*** instructions for main.py *** 
3. The *** BEST MODELS *** section contains the different models that were used. 
Each model first implements a pipeline, then it fits the model while it times it, the obtains 
the predictions on the test dataset and saves in the csv format for the competition. 
EACH TIME THE """real_y_pred_df.to_csv""" IS CALLED, THE text.csv FILE IS OVERWRITTEN. 
Our best performant models are in fact AdaBoost based on the linear kernel SVM, then linear kernel SVM 
and then Multinomial NB.  
4. The part of the script that contains the following: 


# *****************************************************************************
# *****************************************************************************

### TESTING PART OF THE SCRIPT ### 

# This section is dedicated to implementing CV-search for hyperparameter tunning. 

# *****************************************************************************
# *****************************************************************************
    

is just as its description implies. Some of this may take a long more running time since 
Grig paramter search is applied. If only interested in final results, please ignore most of
the code in this section. One part of importance is the SVM section , on which we use a 
splitted version of the training set, creating a mini-test set, and therefore using it for 
hyperparameter tuning. We also produce a confusion matrix from this part, which is showed 
in the paper. 


*** instructions for main.py *** 

Please refer to the "README" files in each of the subfolders for futher descriptions. 
https://github.com/JairParra/Kaggle_Reddit_Multiclass_Classification

***README*** 