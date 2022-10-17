import gc
import math
import random
import time
import warnings
from mlinsights.sklapi import SkBaseTransformStacking
import numpy as np
import pandas as pd
# import forestci as fci
import tensorflow
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz  # with pydot
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
import  sys
my_devices = tensorflow.config.experimental.list_physical_devices(device_type='GPU')
tensorflow.config.experimental.set_visible_devices(devices=my_devices, device_type='GPU')
# To find out which devices your operations and tensors are assigned to
# tensorflow.debugging.set_log_device_placement(True)
from mlinsights.mlmodel import IntervalRegressor
import matplotlib.pyplot as plt
import logging
import yaml
with open(sys.argv[1], "r") as yaml_config_file:
    logging.info("Loading simulation settings from %s", sys.argv[1])
    experiment_config = yaml.safe_load(yaml_config_file)
# Load the data


rnn_epochs= experiment_config['ml_parameters']['rnn_epochs']
optimiser= experiment_config['ml_parameters']['optimizer_type']

def score(REAL_LABELS, PREDICTED_LABELS, var_rf, var_rnn, var_merge):
    predicter = list(range(len(var_rf)))

    a1 = pd.DataFrame(index=range(len(var_rf)), columns=range(2))
    a1.columns = ['index', 'value']

    a2 = pd.DataFrame(index=range(len(var_rf)), columns=range(2))
    a2.columns = ['index', 'value']

    for i in range(len(predicter)):
        if i in PREDICTED_LABELS:
            a1.iloc[i, 1] = 1
        else:

            a1.iloc[i, 1] = 0

    for i in range(len(predicter)):
        if i in REAL_LABELS:
            a2.iloc[i, 1] = 1
        else:
            a2.iloc[i, 1] = 0

    y_real = a2.value
    y_real = y_real.astype(int)
    y_predi = a1.value
    y_predi = y_predi.astype(int)
    try:
        cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
        tp = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tn = cm[1][1]
        precision1 = tp / (tp + fp)
        recall1 = tp / (tp + fn)
        Accuracy1 = (tp + tn) / len(var_rf)
        F11 = 2 / ((1 / precision1) + (1 / recall1))
    except:
        precision1 = None
        recall1 = None
        Accuracy1 = None
        F11 = None
    return precision1, recall1, Accuracy1, F11


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)




def predict_dist(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.hstack(preds)


def predict_point(X, model, num_samples):
    pred_dist = predict_dist(X, model, num_samples)
    return pred_dist.mean(axis=1), pred_dist.var(axis=1)


def evaluate_lr(model, train_x, test_x, train_y, test_y, scaler, label_data):
    """
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :label_data:
    :return:
    """
    # Evaluate the model
    from statsmodels.stats.outliers_influence import OLSInfluence
    test_uncertainty_df = pd.DataFrame()

    # LR

    # train_x = [train_x[i][0] for i in range(train_x.shape[0])]
    # train_x=  np.array(train_x).reshape(-1,1)
    train_x = np.array(train_x)
    # test_x = [test_x[i][0] for i in range(test_x.shape[0])]
    test_x = np.array(test_x)  # .reshape(-1,1)
    train_y = np.array(train_y)  # .reshape(-1, 1)
    test_y = np.array(test_y)  # .reshape(-1, 1)
#    pdb.set_trace()

    lr_types = ['LinearRegression', 'LinearRegression_Higherconfidence_region', 'DecisionTreeRegressor']
    lr_type_index = random.randint(0, len(lr_types) - 1)
    lr_type = lr_types[lr_type_index]
    if lr_type == 'LinearRegression':
        model = IntervalRegressor(LinearRegression(normalize=True))
        model .fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
       # test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'LinearRegression_Higherconfidence_region':
#    model = IntervalRegressor(LinearRegression(), alpha=0.1)
 #   model.fit(train_x, train_y)
        model = IntervalRegressor(LinearRegression(), n_estimators=50, alpha=0.01)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
    #pred2 = lin2.predict(train_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'DecisionTreeRegressor':
        model = IntervalRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=10), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'RandomForestRegressor':
        model = IntervalRegressor(RandomForestRegressor(n_estimators=50, max_depth=5, min_samples_leaf=10), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
       # test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'SVR':
        model = IntervalRegressor(SVR(kernel='rbf', C=1e3, gamma=0.1), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'KNN':
        model = IntervalRegressor(KNeighborsRegressor(n_neighbors=5), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'AdaBoostRegressor':
        model = IntervalRegressor(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5, min_samples_leaf=10), n_estimators=50), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'GradientBoostingRegressor':
        model = IntervalRegressor(GradientBoostingRegressor(max_depth=5, min_samples_leaf=10, n_estimators=50), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    elif lr_type == 'XGBRegressor':
        model = IntervalRegressor(XGBRegressor(max_depth=5, min_samples_leaf=10, n_estimators=50), alpha=0.1)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
        #test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
        bootstrapped_pred = model.predict_sorted(test_x)
        min_pred = bootstrapped_pred[:, 0]
        max_pred = bootstrapped_pred[:, bootstrapped_pred.shape[1] - 1]
    else:
        print('Invalid LR type')

    variance=1-model.alpha








    #from mlinsights.mlmodel import QuantileLinearRegression
   # m = QuantileLinearRegression()
   # q1 = QuantileLinearRegression(quantile=0.05)
   # q2 = QuantileLinearRegression(quantile=0.95)
   # for model in [m, q1, q2]:
   #     model.fit(train_x, train_y)
    #pdb.set_trace()


    #for label, model in [('med', m), ('q0.05', q1), ('q0.95', q2)]:
    #    p = model.predict(test_x)
    #from sklearn.gaussian_process import GaussianProcessRegressor
    #from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct, WhiteKernel

   # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel()
    #gp = GaussianProcessRegressor( n_restarts_optimizer=10, kernel=RBF(1.0, (1e-2, 1e2)))
    #gp.fit(train_x, train_y)
    #y_pred, sigma = gp.predict(test_x, return_std=True)


  #  trans = SkBaseTransformStacking([LinearRegression(),
#                                     DecisionTreeRegressor()])


   # trans.fit(train_x, train_y)
   # pred_combine = trans.transform(test_x)
   # pdb.set_trace()
   # b_pred_combine = trans.predict_sorted(test_x)
   # min_pred_combine = b_pred_tree[:, 0]
   # max_pred_combine = b_pred_tree[:, b_pred_tree.shape[1] - 1]

    #from imblearn.pipeline import Pipeline as Pipeline

   # from sklearn.pipeline import make_pipeline
   # from imblearn.pipeline import make_pipeline
    #from sklearn.metrics import accuracy_score



 #   pipeline = make_pipeline(LinearRegression(), DecisionTreeRegressor())

  #  pipeline.fit(train_x, train_y)
   # pred = pipeline.predict(test_x)
    #score = accuracy_score(test_y, pred)
    #print(score)
    train_predict = model.predict(train_x)  # .reshape(-1,1) #mpg x_hat)
    test_predict = model.predict(test_x)  # .reshape(-1,1)
    # mpg y_hat)
   # train_predict = trans.transform(train_x)  # .reshape(-1,1) #mpg x_hat)
   # test_predict = pred_combine

    train_predict = scaler.inverse_transform([train_predict])
    train_y = scaler.inverse_transform([train_y])
    #test_predict = scaler.inverse_transform([test_predict])
    #test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test

    test_uncertainty_df['index'] = test_uncertainty_df.index
    import plotly.graph_objects as go

    test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = pd.DataFrame()

    truth_uncertainty_plot_df['value'] = test_y
    truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index




    test_uncertainty_df['value_mean'] = test_predict
#    test_uncertainty_df['value_std'] = result.bse.mean(axis=0)

    test_uncertainty_df = test_uncertainty_df[['value_mean']]
    test_uncertainty_df['lower_bound'] = min_pred
        #test_uncertainty_df['value_mean'] - 3 * test_uncertainty_df['value_std']
    test_uncertainty_df['upper_bound'] =max_pred
        #test_uncertainty_df['value_mean'] + 3 * test_uncertainty_df['value_std']


    # fig.show()


    bounds_df = pd.DataFrame()



    # Using 99% confidence bounds
    bounds_df['lower_bound'] = test_uncertainty_df['lower_bound']
    bounds_df['prediction'] = test_uncertainty_df['value_mean']
    bounds_df['real_value'] = test_y
    bounds_df['upper_bound'] = test_uncertainty_df['upper_bound']

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                              (bounds_df['real_value'] <= bounds_df['upper_bound']))

    print("Proportion of points contained within 99% confidence interval:",
          bounds_df['contained'].mean())
    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    print("Number of points predicted to be anomalies:", len(predictedanomaly))


#    test_predict = scaler.inverse_transform(test_predict)
    #  test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    #train_score = mean_squared_error(train_y[0], train_predict[:, 0])
    #train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
    #test_score_MC = mean_squared_error(test_y, test_MC_predict_point[0, :])
    # print('Train Score: %.2f RMSE' % (train_score))
    #test_score = mean_squared_error(test_y[0], test_predict[:, 0])
    # print('Test Score: %.2f RMSE' % (test_score))
    # pdb.set_trace()

    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()
    training_truth_df = pd.DataFrame()
    testing_truth_df = pd.DataFrame()
    training_df['value'] = train_predict[:,0]
    training_df['index'] = training_df.index
    training_df['source'] = 'Training Prediction'

    testing_df['value'] = test_predict
    testing_df['index'] = testing_df.index
    testing_df['source'] = 'Test Prediction'

    training_truth_df['value'] = train_y[0]
    training_truth_df['index'] = training_truth_df.index
    training_truth_df['source'] = 'True Value Training'
    testing_truth_df['value'] = test_y
    testing_truth_df['index'] = testing_truth_df.index
    testing_truth_df['source'] = 'True Value Testing'
    # pdb.set_trace()

    evaluation = pd.concat([training_df,
                            testing_df,
                            training_truth_df,
                            testing_truth_df
                            ], axis=0)

    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    pdb.set_trace()

    N = 15
    newarr = []
    predictedanomaly = predictedanomaly.sort_values()

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2] and predictedanomaly[i + 3] + 1 == predictedanomaly[i + 4] and predictedanomaly[i + 4] + 1 ==
                predictedanomaly[i + 5]
                and predictedanomaly[i + 5] + 1 == predictedanomaly[i + 6] and predictedanomaly[i + 6] + 1 ==
                predictedanomaly[i + 7]):
            newarr.append(predictedanomaly[i])

    predicteddanomaly = list(set(newarr))

    predicter = list(range(len(test_uncertainty_df)))

    precision, recall, Accuracy, F1 = score(label_data, predicteddanomaly, test_predict, test_predict, test_predict)


    return   train_predict, test_predict, variance, predicteddanomaly, precision, \
        recall, Accuracy, F1



def evaluate_rf(model, train_x, test_x, train_y, test_y, scaler, label_data):
    """
    Evaluates the Random Forest with training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :label_data:
    :return:
    """
    test_uncertainty_df = pd.DataFrame()

    # Forecast

    # train_x = [train_x[i][0] for i in range(train_x.shape[0])]
    # train_x=  np.array(train_x).reshape(-1,1)
    train_x = np.array(train_x)
    # test_x = [test_x[i][0] for i in range(test_x.shape[0])]
    test_x = np.array(test_x)  # .reshape(-1,1)
    train_y = np.array(train_y)  # .reshape(-1, 1)
    test_y = np.array(test_y)  # .reshape(-1, 1)

    # pdb.set_trace()
    model.fit(train_x, train_y)

    train_predict = model.predict(train_x)  # .reshape(-1,1) #mpg x_hat)
    test_predict = model.predict(test_x)  # .reshape(-1,1)
    # mpg y_hat)

    train_predict = scaler.inverse_transform([train_predict])
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform([test_predict])
    test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test

    n_experiments = 25
    test_uncertainty_df = pd.DataFrame()
    model1 = RandomForestRegressor(n_estimators=model.n_estimators, criterion='mse', max_depth=None,
                                   min_samples_split=2,
                                   min_samples_leaf=4, max_features='auto', max_leaf_nodes=None,
                                   bootstrap=2,
                                   oob_score=False, n_jobs=4, random_state=None, verbose=0)

    for i in range(n_experiments):
        model1.fit(train_x, train_y[0])
        I = np.random.choice(train_x.shape[0], train_x.shape[0], replace=True)
        J = np.random.choice(test_x.shape[0], test_x.shape[0], replace=True)
        bootstrap_train_x = train_x[I]
        bootstrap_train_y = train_y[0][I]
        bootstrap_train_x = np.array(bootstrap_train_x)
        bootstrap_train_y = np.array(bootstrap_train_y)
        model1.fit(bootstrap_train_x, bootstrap_train_y)
        bootstrap_test_x = test_x
        bootstrap_test_y = test_y[0]
        bootstrap_test_x = np.array(bootstrap_test_x)
        bootstrap_test_y = np.array(bootstrap_test_y)
        bootstrap_test_predict = model1.predict(bootstrap_test_x)
        # bootstrap_test_predict = model1.predict(test_x)
        bootstrap_test_predict = scaler.inverse_transform([bootstrap_test_predict])[0]
        bootstrap_test_y = scaler.inverse_transform([bootstrap_test_y])[0]

        bootstrap_test_rmse = np.sqrt(mean_squared_error(bootstrap_test_predict, bootstrap_test_y))

        test_uncertainty_df['value_{}'.format(i)] = bootstrap_test_predict


    log_energy_consumption_df = test_uncertainty_df.filter(like='value', axis=1)
    test_uncertainty_df['value_mean'] = log_energy_consumption_df.mean(axis=1)
    test_uncertainty_df['value_std'] = log_energy_consumption_df.std(axis=1)


    test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - 3 * test_uncertainty_df['value_std']
    test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + 3 * test_uncertainty_df['value_std']
    test_uncertainty_df['index'] = pd.DataFrame(test_predict[0]).index
    import plotly.graph_objects as go

    test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = pd.DataFrame()
    truth_uncertainty_plot_df['value'] = test_y[0]
    truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

    # .copy(deep=True)
    # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

    upper_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['upper_bound'],
        mode='lines',
        fill=None,
        name='99% Upper Confidence Bound   '
    )
    lower_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['lower_bound'],
        mode='lines',
        fill='tonexty',
        name='99% Lower Confidence Bound',
        fillcolor='rgba(255, 211, 0, 0.1)',
    )
    real_trace = go.Scatter(
        x=truth_uncertainty_plot_df['index'],
        y=truth_uncertainty_plot_df['value'],
        mode='lines',
        fill=None,
        name='Real Values'
    )

    data = [upper_trace, lower_trace, real_trace]

    fig = go.Figure(data=data)
    fig.update_layout(title='RF Uncertainty',
                      xaxis_title='index',
                      yaxis_title='value',
                      legend_font_size=14,
                      )
    # fig.show()
    bounds_df = pd.DataFrame()

    # Using 99% confidence bounds
    bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
    bounds_df['prediction'] = test_uncertainty_plot_df['value_mean']
    bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                              (bounds_df['real_value'] <= bounds_df['upper_bound']))

    print("Proportion of points contained within 99% confidence interval:",
          bounds_df['contained'].mean())
    predictedanomaly = bounds_df.index[~bounds_df['contained']]
    # pdb.set_trace()

    train_y = scaler.inverse_transform(train_y)
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform(test_y)
    predictedanomaly = predictedanomaly.sort_values()

    N = 5
    newarr = []

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2] and predictedanomaly[i + 3] + 1 == predictedanomaly[i + 4] and predictedanomaly[i + 4] + 1 ==
                predictedanomaly[i + 5]):
            newarr.append(predictedanomaly[i])

    predicteddanomaly = list(set(newarr))

    # realanomaly = label_data['index']

    predicter = list(range(len(test_uncertainty_df)))
    # pdb.set_trace()
    precision, recall, Accuracy, F1 = score(label_data, predictedanomaly, test_predict[0], test_predict[0],
                                            test_predict[0])

    # plt.errorbar(test_y, test_predict, yerr=np.sqrt(variance_rf.mean()), fmt='o')

    train_score = mean_squared_error(train_y[0], train_predict[0])
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = mean_squared_error(test_y[0], test_predict[0])
    # print('Test Score: %.2f RMSE' % (test_score))
    model.train_score = train_score
    model.test_score = test_score
    variance_rf = test_uncertainty_df['value_std'].values
    model.variance = variance_rf
    model.anomalies = predictedanomaly
    model.test_predict = test_predict
    model.train_predict = train_predict
    #    pdb.set_trace()

    return train_score, test_score, train_predict, test_predict, variance_rf, predictedanomaly, precision, recall, Accuracy, F1



def evaluate_rnn(model, train_x, test_x, train_y, test_y, scaler, optimiser, name, label_data):
    """
    Evaluates the RNN model using the training and testing data
    :param model:
    :param train_x:
    :param test_x:
    :param train_y:
    :param test_y:
    :param scaler:
    :param optimiser:
    :return:
    """
    model.compile(loss='mean_squared_error', optimizer='Adam')

    model.fit(train_x, train_y, epochs=rnn_epochs, shuffle=False, batch_size=128, verbose=2)
    # Forecast
    #  pdb.set_trace()

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    # Invert forecasts

    train_MC_predict = predict_dist(train_x, model, num_samples=100)

    test_MC_predict = predict_dist(test_x, model, num_samples=100)

    train_MC_predict_point, train_MC_predict_var = predict_point(train_x, model, num_samples=100)
    test_MC_predict_point, test_MC_predict_var = predict_point(test_x, model, num_samples=100)
    n_experiments = 20
    test_uncertainty_df = pd.DataFrame()
    # test_uncertainty_df['date'] = testing_df['date']
    indices = [i for i in range(len(test_y))]

    test_uncertainty_df['index'] = indices
    for i in range(n_experiments):
        experiment_predictions = np.stack([model(test_x, training=True)
                                           for sample in range(100)])
    test_uncertainty_df['value_mean'] = experiment_predictions.mean(axis=0)
    test_uncertainty_df['value_std'] = experiment_predictions.std(axis=0)
    test_uncertainty_df = test_uncertainty_df[['index', 'value_mean', 'value_std']]
    test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - 3 * test_uncertainty_df['value_std']
    test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + 3 * test_uncertainty_df['value_std']
    # pdb.set_trace()
    test_MC_predict_var = test_uncertainty_df['value_std']
    test_MC_predict_point = test_uncertainty_df['value_mean']

    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_y = scaler.inverse_transform([test_y])
    test_predict = scaler.inverse_transform(test_predict)
    train_MC_predict_point = scaler.inverse_transform([train_MC_predict_point])
    test_MC_predict_point = scaler.inverse_transform([test_MC_predict_point])
    # test_uncertainty_df = pd.DataFrame()
    #    pdb.set_trace()

    # test_uncertainty_df = test_uncertainty_df[['index',  'value_mean', 'value_std']]
    # test_uncertainty_df['value_mean']=test_MC_predict_point[0]
    # test_uncertainty_df['lower_bound'] = test_MC_predict_point[0] - 3 * abs(test_MC_predict_var)
    # test_uncertainty_df['upper_bound'] = test_MC_predict_point[0] + 3 * abs(test_MC_predict_var)
    test_uncertainty_df['index'] = test_uncertainty_df.index
    import plotly.graph_objects as go

    test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = pd.DataFrame()

    truth_uncertainty_plot_df['value'] = test_y[0]
    truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

    # .copy(deep=True)
    # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

    upper_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['upper_bound'],
        mode='lines',
        fill=None,
        name='99% Upper Confidence Bound'
    )
    lower_trace = go.Scatter(
        x=test_uncertainty_plot_df['index'],
        y=test_uncertainty_plot_df['lower_bound'],
        mode='lines',
        fill='tonexty',
        name='99% Lower Confidence Bound',
        fillcolor='rgba(255, 211, 0, 0.1)',
    )
    real_trace = go.Scatter(
        x=truth_uncertainty_plot_df['index'],
        y=truth_uncertainty_plot_df['value'],
        mode='lines',
        fill=None,
        name='Real Values'
    )

    data = [upper_trace, lower_trace, real_trace]

    fig = go.Figure(data=data)
    fig.update_layout(title='Uncertainty MCDropout Test Data',
                      xaxis_title='index',
                      yaxis_title='value',
                      legend_font_size=14,
                      )
    # fig.show()

    bounds_df = pd.DataFrame()

    # Using 99% confidence bounds
    bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
    bounds_df['prediction'] = test_uncertainty_plot_df['value_mean']
    bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                              (bounds_df['real_value'] <= bounds_df['upper_bound']))

    print("Proportion of points contained within 99% confidence interval:",
          bounds_df['contained'].mean())
    predictedanomaly = bounds_df.index[~bounds_df['contained']]

    test_predict = scaler.inverse_transform(test_predict)
    #  test_y = scaler.inverse_transform([test_y])
    # Calculate RMSE for train and test
    train_score = mean_squared_error(train_y[0], train_predict[:, 0])
    train_score_MC = mean_squared_error(train_y[0], train_MC_predict_point[0, :])
    test_score_MC = mean_squared_error(test_y[0], test_MC_predict_point[0, :])
    # print('Train Score: %.2f RMSE' % (train_score))
    test_score = mean_squared_error(test_y[0], test_predict[:, 0])
    # print('Test Score: %.2f RMSE' % (test_score))
    # pdb.set_trace()
    model.train_score = train_score
    model.test_score = test_score
    model.train_score_MC = train_score_MC
    model.test_score_MC = test_score_MC
    training_df = pd.DataFrame()
    testing_df = pd.DataFrame()
    training_truth_df = pd.DataFrame()
    testing_truth_df = pd.DataFrame()

    training_df['value'] = train_MC_predict_point[0]
    training_df['index'] = training_df.index
    training_df['source'] = 'Training Prediction'
    testing_df['value'] = test_MC_predict_point[0]
    testing_df['index'] = testing_df.index
    testing_df['source'] = 'Test Prediction'
    training_truth_df['value'] = train_y[0]
    training_truth_df['index'] = training_truth_df.index
    training_truth_df['source'] = 'True Value Training'
    testing_truth_df['value'] = test_y[0]
    testing_truth_df['index'] = testing_truth_df.index
    testing_truth_df['source'] = 'True Value Testing'
    # pdb.set_trace()

    evaluation = pd.concat([training_df,
                            testing_df,
                            training_truth_df,
                            testing_truth_df
                            ], axis=0)

    predictedanomaly = bounds_df.index[~bounds_df['contained']]


    N = 15
    newarr = []
    predictedanomaly = predictedanomaly.sort_values()

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2] and predictedanomaly[i + 3] + 1 == predictedanomaly[i + 4] and predictedanomaly[i + 4] + 1 ==
                predictedanomaly[i + 5]
                and predictedanomaly[i + 5] + 1 == predictedanomaly[i + 6] and predictedanomaly[i + 6] + 1 ==
                predictedanomaly[i + 7]):
            newarr.append(predictedanomaly[i])

    predicteddanomaly = list(set(newarr))

    predicter = list(range(len(test_uncertainty_df)))

    precision, recall, Accuracy, F1 = score(label_data, predicteddanomaly, test_predict, test_predict, test_predict)

    model.train_score_MC = train_score_MC
    model.test_score_MC = test_score_MC
    model.train_score = train_score
    model.test_score = test_score
    model.train_predict = train_predict
    model.test_predict = test_predict
    model.test_MC_predict = test_MC_predict
    model.test_MC_predict_point = test_MC_predict_point
    model.train_MC_predict = train_MC_predict
    model.train_MC_predict_point = train_MC_predict_point
    model.test_MC_predict_var = test_MC_predict_var
    model.train_MC_predict_var = train_MC_predict_var
    model.predictedanomaly = predictedanomaly

    return train_score_MC, test_score_MC, train_score, test_score, train_predict, test_predict, test_MC_predict, test_MC_predict_point, train_MC_predict, train_MC_predict_point, test_MC_predict_var, train_MC_predict_var, predictedanomaly, precision, recall, Accuracy, F1
