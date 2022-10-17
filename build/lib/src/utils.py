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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz  # with pydot
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from src.data_handeling.Preprocess import *
from src.data_handeling.Preprocess import *
from src.model.model import *
my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)
from src.evaluation.evaluation import *
import matplotlib.pyplot as plt

#optimisers = ['SGD', 'Adam']

rnn_types = ['LSTM', 'GRU', 'SimpleRNN']
warnings.filterwarnings("ignore")

optimisers = ['Adam']
im = 0

precision = []
recall = []
Accuracy = []
F1 = []
force_gc = True


def voting(anomalies_rf, anomalies_rnn, anomalies_merged):

    anomalies_rnn1 = anomalies_rnn.values.tolist()
    anomalies_rf1 = anomalies_rf.values.tolist()

    anomalies = set(anomalies_rf1 + anomalies_rnn1)
    anomalies = list(anomalies)
    anomalies = set(anomalies + anomalies_merged)
    # anomalies=pd.DataFrame(anomalies)
    anomalies = list(anomalies)

    append_anomalies = []

    for i in range(len(anomalies)):
        if anomalies[i] in anomalies_rf and anomalies[i] in anomalies_rnn and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])
        elif anomalies[i] in anomalies_rnn and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])
        elif anomalies[i] in anomalies_rf and anomalies[i] in anomalies_merged:
            append_anomalies.append(anomalies[i])

        elif anomalies[i] in anomalies_rf and anomalies[i] in anomalies_rnn:
            append_anomalies.append(anomalies[i])
    return append_anomalies




def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False



def save_plot_model_rnn(model):
    """
    Saves the plot of the RNN model
    :param model:
    :return:
    """
    plot_model(model, show_shapes=True)




def save_plot_model_rf(model):
    """
    Saves the plot of the Random Forest model
    :param model:
    :return:
    """
    for i in range(len(model.estimators_)):
        estimator = model.estimators_[i]
        out_file = open("trees/tree-" + str(i) + ".dot", 'w')
        export_graphviz(estimator, out_file=out_file)
        out_file.close()


