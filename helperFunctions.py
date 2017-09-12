import tensorflow as tf
import numpy as np
import pandas as pd


def format_data(file_path):
    data = pd.read_csv(file_path)
    data.drop("Id", axis=1, inplace=True)
    features = data.drop("SalePrice", axis=1)
    labels = data["SalePrice"]
    labels = np.expand_dims(labels, 1)
    return features, labels


def create_layer(layer, biases, weights, activation_function=None):
    new_layer = tf.matmul(layer, weights) + biases

    if activation_function is None:
        return new_layer

    return activation_function(new_layer)
