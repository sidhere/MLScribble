# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:55:50 2018

@author: ssurya200
"""

#RNN
# Data Pre-processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values



# feature scaling - normalization if RNN uses sigmoid in output layer
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)


# RNN -  60 timesteps with 1 output

X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Building the RNN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

# Initializing RNN

regressor = Sequential()

# Adding LSTM layer and some dropouts
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1) ))
regressor.add(Dropout(0.2))

# adding another LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding output layer

regressor.add(Dense(units = 1))


#compiling RNN

regressor.compile(optimizer = "adam" , loss = "mean_squared_error")


# fitting the RNN

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# prediction

# loading test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values


# Pre-processing - test set
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []


for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualizing the data
plt.plot(real_stock_price, color = "red", label = "real google stock price")
plt.plot(predicted_stock_price, color = "purple", label = "predicted google stock price")
plt.title("Google stock price prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()



import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

from keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



def build_regressor(optimizer, neurons):
    regressor = Sequential()
    regressor.add(LSTM(units = neurons, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = neurons, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = neurons, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = neurons))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = "mean_squared_error")
    return regressor

regressor = KerasRegressor(build_fn = build_regressor)
parameters = {'batch_size' : [32, 64, 128],
              'epochs' : [50, 100],
              'optimizer' : ['adam', 'rmsprop'],
              'neurons' : [50, 100, 200]}
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = "neg_mean_squared_error",
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_