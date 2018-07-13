#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:06:51 2018

@author: abhay
"""
# Data preprocessing

# importing the libraries

import numpy as np # Mostly to work on arrays
import matplotlib.pyplot as plt # Mostly to plot graphs
import pandas as pd

# importing the dataset
dataset = pd.read_csv("Data.csv")
independentVars = dataset.iloc[:,:-1].values
dependentVars = dataset.iloc[:,3].values

"""
This may not be required in all the data preprocessing

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer.fit(independentVars[:, 1:3])
independentVars[:,1:3] = imputer.transform(independentVars[:, 1:3])
"""

"""
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="most_frequent", axis=0)
imputer.fit(independentVars[:, 1:3])
independentVars[:,1:3] = imputer.transform(independentVars[:, 1:3])
"""

"""
# May not be required in all the datapreprocessing,
# Hence, excluded from template

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_independentVars = LabelEncoder()
independentVars[:,0] = labelEncoder_independentVars.fit_transform(independentVars[:,0])
oneHotEncoder_independentVars = OneHotEncoder(categorical_features=[0], dtype=np.int8)
independentVars = oneHotEncoder_independentVars.fit_transform(independentVars).toarray()

labelEncoder_dependentVars = LabelEncoder()
dependentVars = labelEncoder_dependentVars.fit_transform(dependentVars)
"""

# Splitting the dataset into trainingset and testset
from sklearn.cross_validation import train_test_split
indep_train, indep_test, dep_train, dep_test = train_test_split(independentVars, dependentVars, test_size = 0.2, random_state = 0 )


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_deps = StandardScaler()
indep_train = sc_deps.fit_transform(indep_train)
indep_test = sc_deps.transform(indep_test)

dep_train = sc_deps.fit_transform(dep_train)
dep_test = sc_deps.transform(dep_test)

