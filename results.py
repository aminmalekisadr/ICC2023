import math
import os
import pandas as pd
import pdb
from numpy.ma.core import mean

path = './results'
files = os.listdir(path)
# list to store files
mean_F1_rnn = []
mean_F1_rf = []
mean_F1_lr = []
mean_F1_RNNRF = []
mean_F1_RNNLR = []
mean_F1_RFLR = []
mean_F1_RNNRFLR = []
mean_accuracy_rnn = []
mean_accuracy_rf = []
mean_accuracy_LR = []
mean_accuracy_RNNRF = []
mean_accuracy_RNNLR = []
mean_accuracy_RFLR = []
mean_accuracy_RNNRFLR = []
mean_precision_rnn = []
mean_precision_rf = []
mean_precision_LR = []
mean_precision_RNNRF = []
mean_precision_RNNLR = []
mean_precision_RFLR = []
mean_precision_RNNRFLR = []
mean_recall_rnn = []
mean_recall_rf = []
mean_recall_LR = []
mean_recall_RNNRF = []
mean_recall_RNNLR = []
mean_recall_RFLR = []
mean_recall_RNNRFLR = []
list_mean_F1_rnn = []
list_mean_F1_rf = []
list_mean_F1_lr = []
list_mean_F1_RNNRF = []
list_mean_F1_RNNLR = []
list_mean_F1_RFLR = []
list_mean_F1_RNNRFLR = []
list_mean_accuracy_rnn = []
list_mean_accuracy_rf = []
list_mean_accuracy_LR = []
list_mean_accuracy_RNNRF = []
list_mean_accuracy_RNNLR = []
list_mean_accuracy_RFLR = []
list_mean_accuracy_RNNRFLR = []
list_mean_precision_rnn = []
list_mean_precision_rf = []
list_mean_precision_LR = []
list_mean_precision_RNNRF = []
list_mean_precision_RNNLR = []
list_mean_precision_RFLR = []
list_mean_precision_RNNRFLR = []
list_mean_recall_rnn = []
list_mean_recall_rf = []
list_mean_recall_LR = []
list_mean_recall_RNNRF = []
list_mean_recall_RNNLR = []
list_mean_recall_RFLR = []
list_mean_recall_RNNRFLR = []
SMAP = ['P-1', 'S-1', 'E-1', 'E-2', 'E-5', 'E-6', 'E-7',
        'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'P-3',
        'A-2', 'A-4', 'G-2',
        'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8', 'D-9',
        'G-4', 'T-3', 'D-12', 'B-1', 'G-7', 'P-7',
        'A-5', 'A-7', 'D-13', 'A-9']

MSL = [
    'M-1', 'F-7', 'T-5', 'M-4',
    'M-5', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14',
    'T-9', 'T-8', 'D-15', 'M-7', 'F-8']
MSL = [name + '.csv' for name in MSL]
SMAP = [name + '.csv' for name in SMAP]

SMAP = [name for name in files if name in SMAP]
MSL = [name for name in files if name in MSL]
for i, data in enumerate(MSL):
    df = pd.read_csv(path + '/' + data)

    mean_F1_rnn.append(df.best_F1_rnn[0])
    # print(dir(df))
    mean_F1_rf.append(df.best_F1_rf[0])
    mean_F1_lr.append(df.best_F1_lr[0])
    mean_accuracy_rnn.append(df.best_Accuracy_rnn[0])
    mean_accuracy_rf.append(df.best_Accuracy_rf[0])
    mean_accuracy_LR.append(df.best_Accuracy_lr[0])
    mean_precision_rnn.append(df.best_precision_rnn[0])
    mean_precision_rf.append(df.best_precision_rf[0])
    mean_precision_LR.append(df.best_precision_lr[0])
    mean_recall_rnn.append(df.best_recall_rnn[0])
    mean_recall_rf.append(df.best_recall_rf[0])

    mean_recall_LR.append(df.best_recall_lr[0])
    mean_F1_RNNRF.append(df.F112[0])
    mean_F1_RNNLR.append(df.F113[0])
    mean_F1_RFLR.append(df.F123[0])
    mean_F1_RNNRFLR.append(df.F1123[0])
    mean_accuracy_RNNRF.append(df.Accuracy12[0])
    mean_accuracy_RNNLR.append(df.Accuracy13[0])
    mean_accuracy_RFLR.append(df.Accuracy23[0])
    mean_accuracy_RNNRFLR.append(df.Accuracy123[0])
    mean_precision_RNNRF.append(df.precision12[0])
    mean_precision_RNNLR.append(df.precision13[0])
    mean_precision_RFLR.append(df.precision23[0])
    mean_precision_RNNRFLR.append(df.precision123[0])
    mean_recall_RNNRF.append(df.recall12[0])
    mean_recall_RNNLR.append(df.recall13[0])
    mean_recall_RFLR.append(df.recall23[0])
    mean_recall_RNNRFLR.append(df.recall123[0])
    pdb.set_trace()
    variance_RNN = df.variance_rnn[0]
    variance_RF = df.variance_rf[0]
    variance_LR = df.variance_lr[0]
    variance_RNNRF = df.variance_rnnrf[0]
    variance_RNNLR = df.variance_rnnlr[0]
    variance_RFLR = df.variance_rflr[0]
    variance_RNNRFLR = df.variance_rnnrflr[0]

mean_F1_rf = [0 if math.isnan(x) else x for x in mean_F1_rf]
mean_recall_rf = [0 if math.isnan(x) else x for x in mean_recall_rf]
mean_precision_rf = [0 if math.isnan(x) else x for x in mean_precision_rf]
mean_accuracy_rf = [0 if math.isnan(x) else x for x in mean_accuracy_rf]

final_mean_F1_RNNRF = mean(mean_F1_RNNRF)
final_mean_F1_RNNLR = mean(mean_F1_RNNLR)
final_mean_F1_RFLR = mean(mean_F1_RFLR)
final_mean_F1_RNNRFLR = mean(mean_F1_RNNRFLR)
final_mean_accuracy_RNNRF = mean(mean_accuracy_RNNRF)
final_mean_accuracy_RNNLR = mean(mean_accuracy_RNNLR)
final_mean_accuracy_RFLR = mean(mean_accuracy_RFLR)
final_mean_accuracy_RNNRFLR = mean(mean_accuracy_RNNRFLR)
final_mean_precision_RNNRF = mean(mean_precision_RNNRF)
final_mean_precision_RNNLR = mean(mean_precision_RNNLR)
final_mean_precision_RFLR = mean(mean_precision_RFLR)
final_mean_precision_RNNRFLR = mean(mean_precision_RNNRFLR)
final_mean_recall_RNNRF = mean(mean_recall_RNNRF)
final_mean_recall_RNNLR = mean(mean_recall_RNNLR)
final_mean_recall_RFLR = mean(mean_recall_RFLR)
final_mean_recall_RNNRFLR = mean(mean_recall_RNNRFLR)
final_mean_F1_rnn = mean(mean_F1_rnn)
final_mean_F1_rf = mean(mean_F1_rf)
final_mean_F1_LR = mean(mean_F1_lr)
final_mean_accuracy_rnn = mean(mean_accuracy_rnn)
final_mean_accuracy_rf = mean(mean_accuracy_rf)
final_mean_accuracy_LR = mean(mean_accuracy_LR)
final_mean_precision_rnn = mean(mean_precision_rnn)
final_mean_precision_rf = mean(mean_precision_rf)
final_mean_precision_LR = mean(mean_precision_LR)
final_mean_recall_rnn = mean(mean_recall_rnn)
final_mean_recall_rf = mean(mean_recall_rf)
final_mean_recall_LR = mean(mean_recall_LR)

print(f'mean value of F1 for Logistic Regression Algorithm  is {final_mean_F1_LR}')
print(f'mean value of F1 for Random Forest Algorithm  is {final_mean_F1_rf}')
print(f'mean value of F1 for RNN Algorithm  is {final_mean_F1_rnn}')
print(f'mean value of F1 for RNN + Random Forest Algorithm  is {final_mean_F1_RNNRF}')
print(f'mean value of F1 for RNN + Logistic Regression Algorithm  is {final_mean_F1_RNNLR}')
print(f'mean value of F1 for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_F1_RNNRFLR}')
print(f'mean value of F1 for Random Forest + Logistic Regression Algorithm  is {final_mean_F1_RFLR}')
print(f'mean value of Accuracy for Logistic Regression Algorithm  is {final_mean_accuracy_LR}')
print(f'mean value of Accuracy for Random Forest Algorithm  is {final_mean_accuracy_rf}')
print(f'mean value of Accuracy for RNN Algorithm  is {final_mean_accuracy_rnn}')
print(f'mean value of Accuracy for RNN + Random Forest Algorithm  is {final_mean_accuracy_RNNRF}')
print(f'mean value of Accuracy for RNN + Logistic Regression Algorithm  is {final_mean_accuracy_RNNLR}')
print(
    f'mean value of Accuracy for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_accuracy_RNNRFLR}')
print(f'mean value of Accuracy for Random Forest + Logistic Regression Algorithm  is {final_mean_accuracy_RFLR}')
print(f'mean value of Precision for Logistic Regression Algorithm  is {final_mean_precision_LR}')
print(f'mean value of Precision for Random Forest Algorithm  is {final_mean_precision_rf}')
print(f'mean value of Precision for RNN Algorithm  is {final_mean_precision_rnn}')
print(f'mean value of Precision for RNN + Random Forest Algorithm  is {final_mean_precision_RNNRF}')
print(f'mean value of Precision for RNN + Logistic Regression Algorithm  is {final_mean_precision_RNNLR}')
print(
    f'mean value of Precision for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_precision_RNNRFLR}')
print(f'mean value of Precision for Random Forest + Logistic Regression Algorithm  is {final_mean_precision_RFLR}')
print(f'mean value of Recall for Logistic Regression Algorithm  is {final_mean_recall_LR}')
print(f'mean value of Recall for Random Forest Algorithm  is {final_mean_recall_rf}')
print(f'mean value of Recall for RNN Algorithm  is {final_mean_recall_rnn}')
print(f'mean value of Recall for RNN + Random Forest Algorithm  is {final_mean_recall_RNNRF}')
print(f'mean value of Recall for RNN + Logistic Regression Algorithm  is {final_mean_recall_RNNLR}')
print(f'mean value of Recall for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_recall_RNNRFLR}')
print(f'mean value of Recall for Random Forest + Logistic Regression Algorithm  is {final_mean_recall_RFLR}')
print(f'mean value of F1 for RNN Algorithm  is {final_mean_F1_rnn}')
print(f'mean value of F1 for Random Forest Algorithm  is {final_mean_F1_rf}')
print(f'mean value of F1 for Logistic Regression Algorithm  is {final_mean_F1_LR}')
print(f'mean value of F1 for RNN + Random Forest Algorithm  is {final_mean_F1_RNNRF}')
print(f'mean value of F1 for RNN + Logistic Regression Algorithm  is {final_mean_F1_RNNLR}')
print(f'mean value of F1 for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_F1_RNNRFLR}')
print(f'mean value of F1 for Random Forest + Logistic Regression Algorithm  is {final_mean_F1_RFLR}')
print(f'mean value of Accuracy for RNN Algorithm  is {final_mean_accuracy_rnn}')
print(f'mean value of Accuracy for Random Forest Algorithm  is {final_mean_accuracy_rf}')
print(f'mean value of Accuracy for Logistic Regression Algorithm  is {final_mean_accuracy_LR}')
print(f'mean value of Accuracy for RNN + Random Forest Algorithm  is {final_mean_accuracy_RNNRF}')
print(f'mean value of Accuracy for RNN + Logistic Regression Algorithm  is {final_mean_accuracy_RNNLR}')
print(
    f'mean value of Accuracy for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_accuracy_RNNRFLR}')
print(f'mean value of Accuracy for Random Forest + Logistic Regression Algorithm  is {final_mean_accuracy_RFLR}')
print(f'mean value of Precision for RNN Algorithm  is {final_mean_precision_rnn}')
print(f'mean value of Precision for Random Forest Algorithm  is {final_mean_precision_rf}')
print(f'mean value of Precision for Logistic Regression Algorithm  is {final_mean_precision_LR}')
print(f'mean value of Precision for RNN + Random Forest Algorithm  is {final_mean_precision_RNNRF}')
print(f'mean value of Precision for RNN + Logistic Regression Algorithm  is {final_mean_precision_RNNLR}')
print(
    f'mean value of Precision for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_precision_RNNRFLR}')
print(f'mean value of Precision for Random Forest + Logistic Regression Algorithm  is {final_mean_precision_RFLR}')
print(f'mean value of Recall for RNN Algorithm  is {final_mean_recall_rnn}')
print(f'mean value of Recall for Random Forest Algorithm  is {final_mean_recall_rf}')
print(f'mean value of Recall for Logistic Regression Algorithm  is {final_mean_recall_LR}')
print(f'mean value of Recall for RNN + Random Forest Algorithm  is {final_mean_recall_RNNRF}')
print(f'mean value of Recall for RNN + Logistic Regression Algorithm  is {final_mean_recall_RNNLR}')
print(f'mean value of Recall for RNN + Random Forest + Logistic Regression Algorithm  is {final_mean_recall_RNNRFLR}')
print(f'mean value of Recall for Random Forest + Logistic Regression Algorithm  is {final_mean_recall_RFLR}')
print(f'mean value of F1 for RNN Algorithm  is {final_mean_F1_rnn}')
