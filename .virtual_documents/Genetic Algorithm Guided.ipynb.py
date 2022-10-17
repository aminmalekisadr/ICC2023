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



