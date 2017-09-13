import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from helperFunctions import *

train_features, train_labels = format_data("/Users/yazen/Desktop/datasets/HouseCostData/train.csv")


train_features_numeric = train_features.select_dtypes(exclude=['object'])
columns = train_features_numeric.columns
train_features_numeric = train_features_numeric.fillna(0)
min_max_scaler = preprocessing.MinMaxScaler()
train_features_numeric = min_max_scaler.fit_transform(train_features_numeric)
train_features_numeric = pd.DataFrame(train_features_numeric,columns = columns)

#one hot objects
train_features_object = train_features.select_dtypes(include=['object'])
train_features_object = pd.get_dummies(train_features_object)

train_features = pd.concat([train_features_object, train_features_numeric],1)

test_features = pd.read_csv("/Users/yazen/Desktop/datasets/HouseCostData/test.csv")
test_features.drop("Id", axis=1, inplace=True)

test_features_numeric = test_features.select_dtypes(exclude=['object'])
columns = test_features_numeric.columns
test_features_numeric = test_features_numeric.fillna(0)
min_max_scaler = preprocessing.MinMaxScaler()
test_features_numeric = min_max_scaler.fit_transform(test_features_numeric)
test_features_numeric = pd.DataFrame(test_features_numeric,columns = columns)

#one hot objects
test_features_object = test_features.select_dtypes(include=['object'])
test_features_object = pd.get_dummies(test_features_object)

test_features = pd.concat([test_features_object, test_features_numeric],1)

train_features.columns
