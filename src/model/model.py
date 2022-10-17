
import gc
import math
import random
import time
import warnings

import numpy as np
import pandas as pd
# import forestci as fci
import tensorflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz  # with pydot
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from src.data_handeling.Preprocess import *
from src.data_handeling.Preprocess import *
#from src.model.model import *

my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)
import matplotlib.pyplot as plt

# Neural networks
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Add, Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras import regularizers
from keras.regularizers import l1
from keras.regularizers import l2

# Wrapper to make neural network compitable with StackingRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
warnings.filterwarnings("ignore")
import yaml
optimisers = ['Adam']
im = 0
from mlinsights.mlmodel import IntervalRegressor
precision = []
recall = []
Accuracy = []
F1 = []
force_gc = True
import logging
import sys

with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
# Load the data
    config_path= experiment_config["experiment_settings"]["config_path"]



def generate_rnn(hidden_layers):
    """
    Generates a RNN using an array of hidden layers including the number of neurons for each layer
    :param hidden_layers:
    :return:
    """

    # Create and fit the RNN
    model = Sequential()
    # Add input layer
    model.add(Dense(1, input_shape=(1, experiment_config['data_parameters']['look_back'])))
    # pdb.set_trace()
    # Add hidden layers
    for i in range(len(hidden_layers)):

        if i == 0:
            neurons_layer = hidden_layers[i]
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters']['max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            #         model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            #        model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        #       model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        elif i == 1:
            neurons_layer = int(hidden_layers[0] / 4)
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters']['max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking
            #            pdb.set_trace()

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
        else:
            neurons_layer = int(hidden_layers[1] / 4)
            # Randomly select rnn type of layer
            rnn_type_index = random.randint(0, len(rnn_types) - 1)
            rnn_type = rnn_types[rnn_type_index]

            dropout = random.uniform(0, experiment_config['ml_parameters']['max_dropout'])  # dropout between 0 and max_dropout
            return_sequences = i < len(hidden_layers) - 1  # Last layer cannot return sequences when stacking

            # Select and add type of layer
            if rnn_type == 'LSTM':
                model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(LSTM(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'GRU':
                model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(GRU(neurons_layer, dropout=dropout, return_sequences=return_sequences))
            elif rnn_type == 'SimpleRNN':
                model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))
                # model.add(SimpleRNN(neurons_layer, dropout=dropout, return_sequences=return_sequences))

    # Add output layer
    model.add(Dense(1))
    # Compile the RNN
    model.compile(loss='mean_squared_error', optimizer= optimisers[0])
    return model


def generate_rf(estimators):
    """
    Generates a Random Forest with the number of estimators to use
    :param estimators:
    :return:
    """
    # Create and fit the RF
    model = RandomForestRegressor(n_estimators=estimators, criterion='mse', max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=4, max_features='auto', max_leaf_nodes=None,
                                   bootstrap=2,
                                   oob_score=False, n_jobs=4, random_state=None, verbose=0)

    return model


def generate_LinearRegression():
    """
    Generates a Linear Regression"
    """
    model =  IntervalRegressor(LinearRegression(), n_estimators=50, alpha=0.05)
    return model


def create_neural_network(input_shape, depth=5, batch_mod=2, num_neurons=20, drop_rate=0.1, learn_rate=.01,
                          r1_weight=0.02,
                          r2_weight=0.02):
    '''A neural network architecture built using keras functional API'''
    act_reg = l1(r2_weight)
    kern_reg = l1(r1_weight)

    inputs = Input(shape=(input_shape,))
    batch1 = BatchNormalization()(inputs)
    hidden1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(batch1)
    dropout1 = Dropout(drop_rate)(hidden1)
    #lstm1=LSTM(num_neurons,  dropout=drop_rate, recurrent_dropout=drop_rate)(dropout1)
    batch2 = BatchNormalization()(dropout1)

    hidden2 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
        dropout1)

    skip_list = [batch1]
    last_layer_in_loop = hidden2
    pdb.set_trace()

    for i in range(depth):
        added_layer = concatenate(skip_list + [last_layer_in_loop])
        skip_list.append(added_layer)
        b1 = None
        # Apply batch only on every i % N layers
        if i % batch_mod == 2:
            b1 = BatchNormalization()(added_layer)
        else:
            b1 = added_layer

        h1 = Dense(num_neurons, activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(b1)
        d1 = Dropout(drop_rate)(h1)
        h2 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d1)
        d2 = Dropout(drop_rate)(h2)
        h3 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d2)
        d3 = Dropout(drop_rate)(h3)
        h4 = Dense(int(num_neurons / 2), activation='relu', kernel_regularizer=kern_reg, activity_regularizer=act_reg)(
            d3)
        last_layer_in_loop = h4
        c1 = concatenate(skip_list + [last_layer_in_loop])
        output = Dense(1, activation='sigmoid')(c1)

    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam()
    optimizer.learning_rate = learn_rate

    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['accuracy'])
    return model