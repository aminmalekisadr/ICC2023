
import sys
import pdb
import pandas as pd
from sklearn import tree
from sklearn import ensemble
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import os
import urllib.request
import io
import zipfile
import datetime
## Reading Data
from functions import *
DATA_URL = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'

Label_URL = 'https://github.com/khundman/telemanom/raw/master/labeled_anomalies.csv'

df_label = pd.read_csv(Label_URL)

MSL=df_label[df_label.spacecraft=='MSL']['chan_id']
SMAP=df_label[df_label.spacecraft=='SMAP']['chan_id']
train_signals=SMAP
print(SMAP)

if not os.path.exists('data'):
    response = urllib.request.urlopen(DATA_URL)
    bytes_io = io.BytesIO(response.read())

    with zipfile.ZipFile(bytes_io) as zf:
        zf.extractall()

train_signals = os.listdir('data/train')
test_signals = os.listdir('data/test')

# fix random seed for reproducibility
# np.random.seed(0)
os.makedirs('csv', exist_ok=True)
im=0
# %%
selected_columns = [  'value']
SMAP = ['P-1']
for name in SMAP:

    label_row = df_label[df_label.chan_id == name]

    labels = label_row['anomaly_sequences'][label_row['anomaly_sequences'].index]

    appended_data = []

    labels = eval(labels[im])

    for i in range(len(labels)):
        anom = labels[i]
        start = anom[0]
        end = anom[1]

        index = np.array(range(start, end))

        timestamp = index * 86400 + 1022819200

        anomalies = pd.DataFrame({'timestamp': timestamp.astype(int), 'value': 1, 'index': index})
        appended_data.append(anomalies)

    label_data = pd.concat(appended_data)

    label_data['date'] = pd.to_datetime(label_data['timestamp'], unit='s')
    label_data['month'] = label_data['date'].dt.month.astype(int)
    label_data['name'] = name
    label_data['day_of_week'] = label_data['date'].dt.dayofweek.astype(int)
    label_data['hour_of_day'] = label_data['date'].dt.hour.astype(int)
    label_data = label_data[selected_columns]
    label_data.to_csv('csv/' + name + '.csv', index=False)

    signal = name
    train_np = np.load('data/train/' + signal + '.npy')
    test_np = np.load('data/test/' + signal + '.npy')

    data = build_df(np.concatenate([train_np, test_np]))
    data['date'] = pd.to_datetime(data['timestamp'], unit='s')
    data['month'] = data['date'].dt.month.astype(int)
    data['name'] = name
    data['index'] = data['index'].astype(int)
    data['day_of_week'] = data['date'].dt.dayofweek.astype(int)
    data['hour_of_day'] = data['date'].dt.hour.astype(int)
    data = data[selected_columns]
    data.to_csv('csv/' + name + '.csv', index=True)

    train = build_df(train_np)
    train['date'] = pd.to_datetime(train['timestamp'], unit='s')
    train['month'] = train['date'].dt.month.astype(int)
    train['day_of_month'] = train['date'].dt.day.astype(int)
    train['name'] = name
    train['day_of_week'] = train['date'].dt.dayofweek.astype(int)
    train['hour_of_day'] = train['date'].dt.hour.astype(int)
    train['index'] = train['index'].astype(int)
    train = train[selected_columns]
   # train.to_csv('csv/' + name + '.csv', index=False)
    train.to_csv('csv/' + name + '-train.csv', index=True)

    test = build_df(test_np, start=len(train))
    test['date'] = pd.to_datetime(test['timestamp'], unit='s')
    test['month'] = test['date'].dt.month.astype(int)
    test['name'] = name
    test['day_of_week'] = test['date'].dt.dayofweek.astype(int)
    test['hour_of_day'] = test['date'].dt.hour.astype(int)
    test['index'] = test['index'].astype(int)
    test = test[selected_columns]
    #test.to_csv('csv/' + name + '.csv', index=False)
    #test.to_csv('csv/' + name + '-train.csv', index=False)
    test.to_csv('csv/' + name + '-test.csv', index=True)

    #datetime_columns = ['timestamp', 'index']
   # target_column = 'value'

  #  feature_columns = datetime_columns + ['value']
    # pdb.set_trace()

#    resample_df = train[feature_columns]
 #   resample_df_test = test[feature_columns]


datasets = ['csv/P-1.csv']

evaluate_ga(datasets[0])
