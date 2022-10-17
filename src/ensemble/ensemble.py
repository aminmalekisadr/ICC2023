import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.utils import score


def ensemble_stacking(rnn_model, lr_model, rf_model, model_1_values, best_rmse_rnn, model_3_values, model_2_values,
                      best_rmse_rf, test, scaler, train_x, train_y, test_x, test_y, *args, **kwargs):
    """
    Ensemble result of 2 models using stacking and averaging.
    Takes both model predictions, averages them and calculates the new RMSE
    :param model_1_values:
    :param model_2_values:
    :return:
    """

    if lr_model is not None:
        var_rnn = rnn_model.test_MC_predict_var
        var_rf = rf_model.test_MC_predict_var
        var_lr = lr_model.test_MC_predict_var
        model_3_values = model_3_values.squeeze()
        model_1_values = model_1_values.squeeze()
        model_2_values = model_2_values.squeeze()

        # Generates the stacking values by averaging both predictions
        # model1 rnn   model2 rf    model3 lr
        stacking_values12 = []
        stacking_values13 = []
        stacking_values23 = []
        stacking_values123 = []
        var12 = []
        var13 = []
        var23 = []
        var_total = []

        for i in range(len(model_1_values)):
            #   print(i)
            w_rnn = abs(1 / var_rnn[i])
            w_lr = abs(1 / var_lr[i])
            w_rf = abs(1 / var_rf[i])
            # stacking_values.append((model_1_values[i][0] + model_2_values[0][i]) / 2)
            stacking_values123.append(
                (w_rnn * model_1_values[i] + w_lr * model_3_values[i] + w_rf * model_2_values[i]) / (
                        w_rf + w_lr + w_rnn))
            stacking_values13.append((w_rnn * model_1_values[i] + w_lr * model_3_values[i]) / (w_rnn + w_lr))
            stacking_values12.append((model_1_values[i] * w_rnn + model_2_values[i] * w_rf) / (w_rnn + w_rf))
            stacking_values23.append((model_2_values[i] * w_rf + model_3_values[i] * w_lr) / (w_lr + w_rf))
            var_total.append(1 / (w_rf + w_lr + w_rnn))
            var13.append(1 / (w_rnn + w_lr))
            var12.append(1 / (w_rnn + w_rf))
            var23.append(1 / (w_rf + w_lr))
            # Calculates the new RMSE
            # pdb.set_trace()
        test = scaler.inverse_transform([test])
        # Calculates the new RMSE
        n_values = min(len(test), len(model_1_values))

        stacking_values12 = np.nan_to_num(stacking_values12)
        stacking_values23 = np.nan_to_num(stacking_values23)
        stacking_values13 = np.nan_to_num(stacking_values13)
        stacking_values123 = np.nan_to_num(stacking_values123)

        # stacking_values_uncertainty = stacking_values123
        rmse12 = mean_squared_error(test[0], stacking_values12)
        rmse13 = mean_squared_error(test[0], stacking_values13)
        rmse23 = mean_squared_error(test[0], stacking_values23)
        rmse123 = mean_squared_error(test[0], stacking_values123)
        test_uncertainty_df12 = pd.DataFrame()
        test_uncertainty_df13 = pd.DataFrame()
        test_uncertainty_df23 = pd.DataFrame()
        test_uncertainty_df123 = pd.DataFrame()

        test_uncertainty_df123['lower_bound'] = np.array(stacking_values123) - 3 * np.array(var_total)
        test_uncertainty_df123['upper_bound'] = np.array(stacking_values123) + 3 * np.array(var_total)
        test_uncertainty_df123['index'] = pd.DataFrame(test[0]).index.values

        test_uncertainty_df12['lower_bound'] = np.array(stacking_values12) - 3 * np.array(var12)
        test_uncertainty_df12['upper_bound'] = np.array(stacking_values12) + 3 * np.array(var12)
        test_uncertainty_df12['index'] = pd.DataFrame(test[0]).index.values
        test_uncertainty_df13['lower_bound'] = np.array(stacking_values13) - 3 * np.array(var13)
        test_uncertainty_df13['upper_bound'] = np.array(stacking_values13) + 3 * np.array(var13)
        test_uncertainty_df13['index'] = pd.DataFrame(test[0]).index.values
        test_uncertainty_df23['lower_bound'] = np.array(stacking_values23) - 3 * np.array(var23)
        test_uncertainty_df23['upper_bound'] = np.array(stacking_values23) + 3 * np.array(var23)
        test_uncertainty_df23['index'] = pd.DataFrame(test[0]).index.values
        # test_uncertainty_df12.set_index('index', inplace=True)
        # test_uncertainty_df13.set_index('index', inplace=True)
        # test_uncertainty_df23.set_index('index', inplace=True)
        test_uncertainty_plot_df123 = test_uncertainty_df123  # .copy(deep=True)
        test_uncertainty_plot_df12 = test_uncertainty_df12  # .copy(deep=True)
        test_uncertainty_plot_df13 = test_uncertainty_df13  # .copy(deep=True)
        test_uncertainty_plot_df23 = test_uncertainty_df23  # .copy(deep=True)

        # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
        truth_uncertainty_plot_df = pd.DataFrame()
        truth_uncertainty_plot_df['value'] = test[0]
        truth_uncertainty_plot_df['index'] = truth_uncertainty_plot_df.index

        # .copy(deep=True)
        # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

        # upper_trace = go.Scatter(
        #       x=test_uncertainty_plot_df['index'],
        #      y=test_uncertainty_plot_df['upper_bound'],
        #     mode='lines',
        #    fill=None,
        #   name='99% Upper Confidence Bound   '
        # )
        # lower_trace = go.Scatter(
        #       x=test_uncertainty_plot_df['index'],
        #      y=test_uncertainty_plot_df['lower_bound'],
        #     mode='lines',
        #    fill='tonexty',
        #   name='99% Lower Confidence Bound',
        #  fillcolor='rgba(255, 211, 0, 0.1)',
        # )
        # real_trace = go.Scatter(
        #       x=truth_uncertainty_plot_df['index'],
        #      y=truth_uncertainty_plot_df['value'],
        #     mode='lines',
        #    fill=None,
        #   name='Real Values'
        # )

        # data = [upper_trace, lower_trace, real_trace]

        # fig = go.Figure(data=data)
        # fig.update_layout(title='RF Uncertainty',
        #                      xaxis_title='index',
        #                     yaxis_title='value',
        #                    legend_font_size=14,
        #                   )
        # fig.show()
        bounds_df123 = pd.DataFrame()
        bounds_df12 = pd.DataFrame()
        bounds_df13 = pd.DataFrame()
        bounds_df23 = pd.DataFrame()
        bounds_df123['lower_bound'] = test_uncertainty_plot_df123['lower_bound']
        bounds_df123['upper_bound'] = test_uncertainty_plot_df123['upper_bound']
        bounds_df123['index'] = test_uncertainty_plot_df123['index']
        bounds_df123['prediction'] = stacking_values123
        bounds_df123['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df123['contained'] = ((bounds_df123['real_value'] >= bounds_df123['lower_bound']) &
                                     (bounds_df123['real_value'] <= bounds_df123['upper_bound']))
        predictedanomaly123 = bounds_df123.index[~bounds_df123['contained']]

        bounds_df12['lower_bound'] = test_uncertainty_plot_df12['lower_bound']
        bounds_df12['upper_bound'] = test_uncertainty_plot_df12['upper_bound']
        bounds_df12['index'] = test_uncertainty_plot_df12['index']
        bounds_df12['prediction'] = stacking_values12
        bounds_df12['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df12['contained'] = ((bounds_df12['real_value'] >= bounds_df12['lower_bound']) & (
                bounds_df12['real_value'] <= bounds_df12['upper_bound']))
        predictedanomaly12 = bounds_df12.index[~bounds_df12['contained']]
        bounds_df13['lower_bound'] = test_uncertainty_plot_df13['lower_bound']
        bounds_df13['upper_bound'] = test_uncertainty_plot_df13['upper_bound']
        bounds_df13['index'] = test_uncertainty_plot_df13['index']
        bounds_df13['prediction'] = stacking_values13
        bounds_df13['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df13['contained'] = ((bounds_df13['real_value'] >= bounds_df13['lower_bound']) & (
                bounds_df13['real_value'] <= bounds_df13['upper_bound']))
        predictedanomaly13 = bounds_df13.index[~bounds_df13['contained']]

        bounds_df23['lower_bound'] = test_uncertainty_plot_df23['lower_bound']
        bounds_df23['upper_bound'] = test_uncertainty_plot_df23['upper_bound']
        bounds_df23['index'] = test_uncertainty_plot_df23['index']
        bounds_df23['prediction'] = stacking_values23
        bounds_df23['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df23['contained'] = ((bounds_df23['real_value'] >= bounds_df23['lower_bound']) & (
                bounds_df23['real_value'] <= bounds_df23['upper_bound']))
        predictedanomaly23 = bounds_df23.index[~bounds_df23['contained']]

        N = 5
        newarr123 = []
        newarr12 = []
        newarr13 = []
        newarr23 = []

        for i in range(len(predictedanomaly123) - N):
            if (predictedanomaly123[i] + 1 == predictedanomaly123[i + 1] and predictedanomaly123[i + 1] + 1 ==
                    predictedanomaly123[
                        i + 2] and predictedanomaly123[i + 3] + 1 == predictedanomaly123[i + 4]):
                newarr123.append(predictedanomaly123[i])

        for i in range(len(predictedanomaly12) - N):
            if (predictedanomaly12[i] + 1 == predictedanomaly12[i + 1] and predictedanomaly12[i + 1] + 1 ==
                    predictedanomaly12[
                        i + 2] and predictedanomaly12[i + 3] + 1 == predictedanomaly12[i + 4]):
                newarr12.append(predictedanomaly12[i])

        for i in range(len(predictedanomaly13) - N):
            if (predictedanomaly13[i] + 1 == predictedanomaly13[i + 1] and predictedanomaly13[i + 1] + 1 ==
                    predictedanomaly13[
                        i + 2] and predictedanomaly13[i + 3] + 1 == predictedanomaly13[i + 4]):
                newarr13.append(predictedanomaly13[i])

        for i in range(len(predictedanomaly23) - N):
            if (predictedanomaly23[i] + 1 == predictedanomaly23[i + 1] and predictedanomaly23[i + 1] + 1 ==
                    predictedanomaly23[
                        i + 2] and predictedanomaly23[i + 3] + 1 == predictedanomaly23[i + 4]):
                newarr23.append(predictedanomaly23[i])

        newarr123 = np.array(newarr123)
        newarr12 = np.array(newarr12)
        newarr13 = np.array(newarr13)
        newarr23 = np.array(newarr23)
        newarr123 = newarr123.astype(int)
        newarr12 = newarr12.astype(int)
        newarr13 = newarr13.astype(int)
        newarr23 = newarr23.astype(int)

        #        newarr.append(predictedanomaly[i + 1])
        #       newarr.append(predictedanomaly[i + 2])

        predicteddanomaly123 = list(set(newarr123))
        predicteddanomaly12 = list(set(newarr12))
        predicteddanomaly13 = list(set(newarr13))
        predicteddanomaly23 = list(set(newarr23))

        # realanomaly = label_data['index']
        predicter = list(range(len(test_uncertainty_plot_df123)))
        precision123, recall123, Accuracy123, F1123 = score(test, predicteddanomaly123, stacking_values123)
        precision12, recall12, Accuracy12, F112 = score(test, predicteddanomaly12, stacking_values12)
        precision13, recall13, Accuracy13, F113 = score(test, predicteddanomaly13, stacking_values13)
        precision23, recall23, Accuracy23, F123 = score(test, predicteddanomaly23, stacking_values23)



    # pdb.set_trace()

    else:
        raise NotImplementedError('Not Implemented')

    predicteddanomaly123, stacking_values12, stacking_values13, stacking_values23, stacking_values123, rmse12, rmse13, rmse23, rmse123, predicteddanomaly12, predictedanomaly13, predictedanomaly123, var_total, var12, var13, var23, precision12, precision13, precision23, precision123, recall12, recall13, recall23, recall123, Accuracy12, Accuracy13, Accuracy23, Accuracy123, F112, F113, F123, F1123
    dict_to_return = {'predicteddanomaly123': predicteddanomaly123, 'stacking_values12': stacking_values12,
                      'stacking_values13': stacking_values13, 'stacking_values23': stacking_values23,
                      'stacking_values123': stacking_values123, 'rmse12': rmse12, 'rmse13': rmse13, 'rmse23': rmse23,
                      'rmse123': rmse123, 'predicteddanomaly12': predicteddanomaly12,
                      'predictedanomaly13': predictedanomaly13,
                      'predictedanomaly123': predictedanomaly123, 'var_total': var_total, 'var12': var12,
                      'var13': var13,
                      'var23': var23, 'precision12': precision12, 'precision13': precision13,
                      'precision23': precision23,
                      'precision123': precision123, 'recall12': recall12, 'recall13': recall13, 'recall23': recall23,
                      'recall123': recall123, 'Accuracy12': Accuracy12, 'Accuracy13': Accuracy13,
                      'Accuracy23': Accuracy23,
                      'Accuracy123': Accuracy123, 'F112': F112, 'F113': F113, 'F123': F123, 'F1123': F1123}
    return dict_to_return
