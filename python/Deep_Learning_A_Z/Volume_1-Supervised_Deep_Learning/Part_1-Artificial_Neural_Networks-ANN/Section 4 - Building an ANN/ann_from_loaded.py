#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:37:30 2018

@author: abhay
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
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

# import keras 
# from keras.models import Sequential
# from keras.layers import Dense

#
## If you know the model, then you can create the model and load the weights
## You must know the model, its type, depth etc. Otherwise load the model from a 
## file
#
#classifier = Sequential()
## Adding the input layer and the first hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#
## Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#
## Adding the output layer
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
#classifier.load_weights("/tmp/saved_model.weights")

# load json and create model

from keras.models import model_from_json

json_file = open('/tmp/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

classifier = model_from_json(loaded_model_json)
classifier.load_weights("/tmp/saved_model.weights.hdf")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

