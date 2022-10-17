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
import gc # garbage collector
my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
#
import yaml
import pdb
class Preprocess:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, "r") as yaml_config_file:
            self.experiment_config = yaml.safe_load(yaml_config_file)


    def collect_gc(self):
        """
        Forces garbage collector
        :return:
        """
        if self.experiment_config['ga_parameters']['force_gc']:
            gc.collect()


    def build_df(self,data, start=0):
        #    pdb.set_trace()
        index = np.array(range(start, start + len(data)))
        timestamp = index * 86400 + 1022819200

        return pd.DataFrame({'timestamp': timestamp.astype(int), 'value': data[:, 0], 'index': index.astype(int)})



    def create_sliding_window(self,data, sequence_length, stride=1):
        X_list, y_list = [], []
        for i in range(len(data)):
            if (i + sequence_length) < len(data):
                X_list.append(data.iloc[i:i + sequence_length:stride, :].values)
                y_list.append(data.iloc[i + sequence_length, -1])
        return np.array(X_list), np.array(y_list)


    def inverse_transform(self,y):
        return scaler.inverse_transform(y.reshape(-1, 1))

    def create_dataset(self,dataset, look_back):
        """
        Converts an array of values into a dataset matrix
        :param dataset:
        :param look_back:
        :return:
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, -1])
        self.collect_gc()

        return np.array(dataX), np.array(dataY)

    def load_dataset(self,dataset_path):
        """
        Loads a dataset with training and testing arrays
        :param dataset_path:
        :return:
        """
        # Load dataset
        self.dataset = pd.read_csv(dataset_path, parse_dates=False, index_col=0)

        self.dataset = np.array(self.dataset.value)

        # dataset = dataset.value  # as numpy array
        self.dataset = self.dataset.astype('float64')
        # pdb.set_trace()

        # Normalise the dataset
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.dataset = scaler.fit_transform(self.dataset.reshape(-1, 1))
        train_size_percentage = 0.7

        # split into train and test sets
        train_size = int(len(self.dataset) * self.experiment_config['data_parameters']['train_size_percentage'])
        train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]
        # reshape into X=t and Y=t+1

        train_x, train_y = self.create_dataset(train, self.experiment_config['data_parameters']['look_back'])
        test_x, test_y = self.create_dataset(test, self.experiment_config['data_parameters']['look_back'])
        # reshape input to be [samples, time steps, features]
        train_x_rf, train_y_rf = self.create_dataset(train,self.experiment_config['data_parameters']['look_back2'])
        test_x_rf, test_y_rf = self.create_dataset(test, self.experiment_config['data_parameters']['look_back2'])
        train_x_rf_stf = train_x_rf.reshape(train_x_rf.shape[0], train_x_rf.shape[1], 1)
        test_x_rf_stf = test_x_rf.reshape(test_x_rf.shape[0], test_x_rf.shape[1], 1)

        train_x_stf = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x_stf = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
        train_x_st = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
        test_x_st = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        # pdb.set_trace()
        train_x_rf_st = np.reshape(train_x_rf, (train_x_rf.shape[0], train_x_rf.shape[1]))
        test_x_rf_st = np.reshape(test_x_rf, (test_x_rf.shape[0], test_x_rf.shape[1]))

        return self.dataset, scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y, train_x_rf_stf, train_x_rf, train_y_rf, test_x_rf_stf, test_x_rf, test_y_rf, train_x_rf_st, test_x_rf_st
