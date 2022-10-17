import matplotlib.pyplot as plt
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix
from statistics import mean


def Nmax_gen(predictedanomaly, test_uncertainty2, truth_uncertainty_plot_df, label_data, name):
    rho1 = []
    rrr = []
    avg = []
    precision = []
    recall = []
    Accuracy = []
    rho = []
    F1 = []
    predictedanomaly = sorted(predictedanomaly)

    for N in range(2, 22):
        newarr = []

        if N == 2:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = sorted(list(set(newarr)))
            # pdb.set_trace()

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 3:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 4:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)

        elif N == 5:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)

            ''' and predictedanomaly[i+3]+1==predictedanomaly[i+4] and predictedanomaly[i+4]+1==predictedanomaly[i+5]
            and  predictedanomaly[i+5]+1==predictedanomaly[i+6] and predictedanomaly[i+6]+1==predictedanomaly[i+7]
            and predictedanomaly[i+7]+1==predictedanomaly[i+8]
            and predictedanomaly[i+8]+1==predictedanomaly[i+9]
            and predictedanomaly[i+9]+1==predictedanomaly[i+10]
            and predictedanomaly[i+10]+1==predictedanomaly[i+11]
            and predictedanomaly[i+11]+1==predictedanomaly[i+12]
            and predictedanomaly[i + 12] + 1 == predictedanomaly[i + 13]
            and predictedanomaly[i + 13] + 1 == predictedanomaly[i + 14]
            and predictedanomaly[i + 14] + 1 == predictedanomaly[i + 15]):'''

        #        newarr.append(predictedanomaly[i + 1])
        #       newarr.append(predictedanomaly[i + 2])
        elif N == 6:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)

        elif N == 7:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 9:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)

        elif N == 10:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 11:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 12:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 13:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 14:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 15:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 16:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)

        elif N == 17:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 18:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16] and predictedanomaly[i + 16] + 1 ==
                        predictedanomaly[
                            i + 17]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 19:
            newarr = []
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16] and predictedanomaly[i + 16] + 1 ==
                        predictedanomaly[
                            i + 17] and predictedanomaly[i + 17] + 1 ==
                        predictedanomaly[
                            i + 18]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 20:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and
                        predictedanomaly[i + 11] + 1 ==
                        predictedanomaly[
                            i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16] and predictedanomaly[i + 16] + 1 ==
                        predictedanomaly[
                            i + 17] and predictedanomaly[i + 17] + 1 ==
                        predictedanomaly[
                            i + 18] and predictedanomaly[i + 18] + 1 ==
                        predictedanomaly[
                            i + 19]
                ):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 21:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and predictedanomaly[i + 11] + 1 == predictedanomaly[i + 12] and predictedanomaly[
                            i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16] and predictedanomaly[i + 16] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 17] + 1 ==
                        predictedanomaly[
                            i + 18] and predictedanomaly[i + 18] + 1 ==
                        predictedanomaly[
                            i + 19] and predictedanomaly[i + 19] + 1 ==
                        predictedanomaly[
                            i + 20]):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
        elif N == 22:
            for i in range(len(predictedanomaly) - N):
                if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 ==
                        predictedanomaly[
                            i + 2] and predictedanomaly[i + 2] + 1 ==
                        predictedanomaly[
                            i + 3] and predictedanomaly[i + 3] + 1 ==
                        predictedanomaly[
                            i + 4] and predictedanomaly[i + 4] + 1 ==
                        predictedanomaly[
                            i + 5] and predictedanomaly[i + 5] + 1 ==
                        predictedanomaly[
                            i + 6] and predictedanomaly[i + 6] + 1 ==
                        predictedanomaly[
                            i + 7] and predictedanomaly[i + 7] + 1 ==
                        predictedanomaly[
                            i + 8] and predictedanomaly[i + 8] + 1 ==
                        predictedanomaly[
                            i + 9] and predictedanomaly[i + 9] + 1 ==
                        predictedanomaly[
                            i + 10] and predictedanomaly[i + 10] + 1 ==
                        predictedanomaly[
                            i + 11] and
                        predictedanomaly[i + 11] + 1 == predictedanomaly[i + 12] and predictedanomaly[i + 12] + 1 ==
                        predictedanomaly[
                            i + 13] and predictedanomaly[i + 13] + 1 ==
                        predictedanomaly[
                            i + 14] and predictedanomaly[i + 14] + 1 ==
                        predictedanomaly[
                            i + 15] and predictedanomaly[i + 15] + 1 ==
                        predictedanomaly[
                            i + 16] and predictedanomaly[i + 16] + 1 ==
                        predictedanomaly[
                            i + 17] and predictedanomaly[i + 17] + 1 ==
                        predictedanomaly[
                            i + 18] and predictedanomaly[i + 18] + 1 ==
                        predictedanomaly[
                            i + 19] and predictedanomaly[i + 19] + 1 ==
                        predictedanomaly[
                            i + 20] and predictedanomaly[i + 20] + 1 ==
                        predictedanomaly[
                            i + 21]
                ):
                    newarr.append(predictedanomaly[i])
            predicteddanomaly = list(set(newarr))

            realanomaly = label_data['index']
            pdb.set_trace()

            predicter = list(range(len(test_uncertainty2)))

            a1 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a1.columns = ['index', 'value']

            a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
            a2.columns = ['index', 'value']

            for i in range(len(predicter)):
                if i in predicteddanomaly:
                    a1.iloc[i, 1] = 1
                else:
                    a1.iloc[i, 1] = 0

            for i in range(len(predicter)):
                if i in realanomaly:
                    a2.iloc[i, 1] = 1
                else:
                    a2.iloc[i, 1] = 0

            y_real = a2.value
            y_real = y_real.astype(int)
            y_predi = a1.value
            y_predi = y_predi.astype(int)

            cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
            #       cm_plot_labels = ['no_anomaly', 'had_anomaly']
            #        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

            # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
            # fp = len(predicteddanomaly) - tp
            # fn = 0
            # tn = len(truth_uncertainty_plot_df) - tp - fp - fn

            tp = cm[0][0]
            fp = cm[0][1]
            fn = cm[1][0]
            tn = cm[1][1]

            rho1 = tp + tn - fp - fn

            precision1 = tp / (tp + fp)
            recall1 = tp / (tp + fn)
            Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
            F11 = 2 / ((1 / precision1) + (1 / recall1))
            print('precision', precision1, 'Signal', name, 'N_max', N)
            print('recall', recall1, 'Signal', name, 'N_max', N)
            print('Accuracy', Accuracy1, 'Signal', name, 'N_max', N)
            print('F1', F11, 'Signal', name, 'N_max', N)
            print('rho', rho1, 'Signal', name, 'N_max', N)
            precision.append(precision1)
            F1.append(F11)
            Accuracy.append(Accuracy1)
            recall.append(recall1)
            rho.append(rho1)
    rho_new = (rho) / max(rho)
    Nbest = [N for N in range(len(F1)) if (F1[N] - F1[N - 1]) < 0.01]
    #   plt.plot(range(2, 21), rho_new, label=r'$\rho $', linewidth=3.5)
    plt.plot(range(2, 21), F1, label='F1', linewidth=3.5)
    plt.plot(range(2, 21), Accuracy, label='Accuracy', linewidth=3.5)
    plt.plot(range(2, 21), precision, label='Precision', linewidth=3.5)
    plt.grid()
    plt.xlabel('$T_{max}$', fontsize=25)
    plt.ylabel(f'Normalized Measurement Metrics for signal A2 for GARNN', fontsize=15)

    plt.rc('font', size=20)  # controls default text sizes
    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=40)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)  # fontsize of the tick labels
    plt.rc('legend', fontsize=20)  # legend fontsize
    plt.rc('figure', titlesize=20)  # fontsize of the figure title
    plt.legend()
    plt.show()
    return Nbest

#    im = im + 1

# matched_indices = list(i_anom_predicted & true_indices_flat)

# recall_final = mean(recall)
# precision_final = mean(precision)
# F1_final = mean(F1)
# Accuracy_final = mean(Accuracy)
# cm = confusion_matrix(y_true=test_labels, y_pred=predicteddanomaly)

################################################################################

# %%
