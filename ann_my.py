# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 12:17:02 2018

@author: ssurya200
"""

# Artificial neural network with keras ( tensorflow)
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the categorical Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()                
X = X[:, 1:]              


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fitting classifier to the Training set
# Create your classifier here


# importing keras library for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# initializing ANN

'''classifier = Sequential()
# first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
classifier.add(Dropout(rate = 0.1)) # used not to get over fitting

# second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.1))

# output layer

classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))


# create ANN

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# fit to training set

classifier.fit(x = X_train, y = y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred >0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

test = classifier.predict(sc.transform(np.array([[0.0,0,600,1,42,3,60000,2,1,1,50000]])))
test = (test > 0.5)'''


# K4 cross validation for evaluating the model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_classifer():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "softmax"))
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifer, batch_size = 25, epochs = 500)

if __name__ == "__main__":
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)

mean = accuracies.mean()
variance = accuracies.std()

# Improve the classifier/model using grid search
'''def build_classifer(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu", input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = "glorot_uniform", activation = "relu"))
    classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifer)
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_'''





