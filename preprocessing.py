import pickle
import os
import struct
import logging
import pdb
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
import gc
import urllib
import urllib.request
import io
import zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz  # with pydot
from sklearn.metrics import confusion_matrix
class Preprocessor():
    def __init__(self, settings):
        self.input_file = settings['input_file']
        self.label_file = settings['label_file']
        self.features = None
        self.labels = None
        self.data_url = settings['data_url']
        self.label_url = settings['data_url']
        self.selected_columns = 'value'
        self.label = pd.read_csv(self.label_file)
        self.look_back = settings['look_back']

        self.MSL = self.label[self.label.spacecraft == 'MSL']['chan_id']
        self.SMAP = self.label[self.label.spacecraft == 'SMAP']['chan_id']
        self.im=0
        self.force_gc=True



    def check_existing_dataset(self):
        """ Checks whether the dataset we are trying to create already exists
        :returns: Boolean flag, with `True` meaning that the dataset already exists
        """
        if not os.path.exists(self.input_file):
            response = urllib.request.urlopen(self.data_url)
            bytes_io = io.BytesIO(response.read())
            with zipfile.ZipFile(bytes_io) as zf:
                zf.extractall()
            os.makedirs('csv', exist_ok=True)

    def collect_gc(self):
        """
        Forces garbage collector
        :return:
        """
        if self.force_gc:
            gc.collect()

    def build_df(self,data,start=0):
        index = np.array(range(start, start + len(data)))
        return pd.DataFrame({'value': data[:, 0], 'index': index.astype(int)})

    def save_dataset_csv(self):

        im = 0

        for name in self.SMAP:

            label_row = self.label[self.label.chan_id == name]

            labels = label_row['anomaly_sequences'][label_row['anomaly_sequences'].index]

            appended_data = []

            labels = eval(labels[im])

            for i in range(len(labels)):
                anom = labels[i]
                start = anom[0]
                end = anom[1]

                index = np.array(range(start, end))

                anomalies = pd.DataFrame({'value': 1, 'index': index})
                appended_data.append(anomalies)

            label_data = pd.concat(appended_data)
            label_data = label_data[self.selected_columns]
            label_data.to_csv('csv/' + name + '.csv', index=False)

            signal = name
            train_np = np.load('data/train/' + signal + '.npy')
            test_np = np.load('data/test/' + signal + '.npy')
            data = self.build_df(np.concatenate([train_np, test_np]))
            data['name'] = name
            data['index'] = data['index'].astype(int)
            data = data[self.selected_columns]

            data.to_csv('csv/' + name + '.csv', index=True)
            train = self.build_df(train_np)
            train['name'] = name
            train['index'] = train['index'].astype(int)
            train = train[self.selected_columns]
            # train.to_csv('csv/' + name + '.csv', index=False)
            train.to_csv('csv/' + name + '-train.csv', index=True)

            test = self.build_df(test_np, start=len(train))
            test['name'] = name
            test['index'] = test['index'].astype(int)
            test = test[self.selected_columns]
            test.to_csv('csv/' + name + '-test.csv', index=True)

    def create_dataset(self,dataset):
        """
        Converts an array of values into a dataset matrix
        :param dataset:
        :param look_back:
        :return:
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back - 1):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])

        self.collect_gc()

        return np.array(dataX), np.array(dataY)

    def load_dataset(self,name):
        """ Loads the previously stored dataset, returning it

        :returns: previously stored features and labels
        """
        self.label = pd.read_csv(self.label_file)
        label_row = self.label[self.label.chan_id == name]

        labels = label_row['anomaly_sequences'][label_row['anomaly_sequences'].index]

        appended_data = []
        im = self.im

        labels = eval(labels[im])

        for i in  range(len(self.label)):

         train = pd.read_csv('csv/' + name+'-train' + '.csv', index_col=0)
         test = pd.read_csv('csv/' + name + '-test' + '.csv', index_col=0)
         train = np.array(train.value)
         train = train.astype('float64')
         test = np.array(test.value)
         test = test.astype('float64')


        # Normalise the dataset
         scaler = MinMaxScaler(feature_range=(-1, 1))
         train = scaler.fit_transform(train.reshape(-1, 1))
         test=scaler.fit_transform(test.reshape(-1, 1))

        # split into train and test sets
        # train_size = int(len(dataset) * train_size_percentage)
         #train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        # reshape into X=t and Y=t+1

         train_x, train_y = self.create_dataset(train)
         test_x, test_y = self.create_dataset(test)
        # reshape input to be [samples, time steps, features]

         train_x_stf = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
         test_x_stf = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
         train_x_st = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
         test_x_st = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
        # pdb.set_trace()

        return scaler, train_x_stf, train_x_st, train_y, test_x_stf, test_x_st, test_y
