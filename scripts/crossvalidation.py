import numpy as np
from sklearn.metrics import accuracy_score

def kfold_accuracy(model, X, y, numFolds=5):
    """
    Parameters:

    model                   = instance of model to run k-fold cross validation on
    X                       = numpy array of preprocessed data
    y                       = labels vector
    numFolds                = number of folds (default: 5)

    Split X_train into numFolds (default: 5) equal sections, train model on the other
    numFolds - 1 folds and return average accuracy using the BernoulliNB classifier.
    """

    data = np.c_[X,y] #add labels as last col of X
    foldsList = np.array_split(data, numFolds)
    totalAccuracy = 0

    for currentFoldIndex in range(numFolds):

        data_test = foldsList[currentFoldIndex]
        del foldsList[currentFoldIndex]
        data_train = np.vstack(foldsList)
        foldsList.insert(currentFoldIndex, data_test)

        X_train = data_train[:,:-1]
        y_train = data_train[:,-1][:,np.newaxis]

        X_test = data_test[:,:-1]
        y_test = data_test[:,-1][:,np.newaxis]

        #Apply model
        model.fit(X_train=X_train, y_train=y_train, k=20)
        y_pred = model.predict(X_test)

        totalAccuracy += accuracy_score(y_true=y_test, y_pred=y_pred)

    return totalAccuracy / numFolds
