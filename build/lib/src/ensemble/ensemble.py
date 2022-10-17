import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
def ensemble_stacking(rnn_model, rf_model, model_1_values, best_rmse_rnn, model_2_values, best_rmse_rf, test, scaler,
                      name):
    """
    Ensemble result of 2 models using stacking and averaging.
    Takes both model predictions, averages them and calculates the new RMSE
    :param model_1_values:
    :param model_2_values:
    :return:
    """
    var1 = rnn_model.test_MC_predict_var
    var2 = rf_model.variance

    # Generates the stacking values by averaging both predictions
    stacking_values = []
    stacking_values_uncertainty = []

    var_total = []

    for i in range(len(model_1_values)):
        w1 = abs(1 / var1[i])
        w2 = abs(1 / var2[i])
        stacking_values.append((model_1_values[i][0] + model_2_values[0][i]) / 2)
        stacking_values_uncertainty.append((model_1_values[i][0] * w1 + model_2_values[0][i] * w2) / (w1 + w2))
        var_total.append(1 / (w1 + w2))
    # pdb.set_trace()
    test = scaler.inverse_transform([test])
    # Calculates the new RMSE
    stacking_values = np.nan_to_num(stacking_values)
    stacking_values_uncertainty = np.nan_to_num(stacking_values_uncertainty)
    rmse1 = mean_squared_error(test[0], stacking_values)
    rmse2 = mean_squared_error(test[0], stacking_values_uncertainty)
    test_uncertainty_df = pd.DataFrame()
    # pdb.set_trace()

    test_uncertainty_df['lower_bound'] = np.array(stacking_values_uncertainty) - 3 * np.array(var_total)
    test_uncertainty_df['upper_bound'] = np.array(stacking_values_uncertainty) + 3 * np.array(var_total)
    test_uncertainty_df['index'] = pd.DataFrame(test[0]).index.values
    import plotly.graph_objects as go

    test_uncertainty_plot_df = test_uncertainty_df  # .copy(deep=True)
    # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
    truth_uncertainty_plot_df = pd.DataFrame()
    truth_uncertainty_plot_df['value'] = test[0]
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

    # pdb.set_trace()

    # Using 99% confidence bounds
    bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
    bounds_df['prediction'] = stacking_values_uncertainty
    bounds_df['real_value'] = truth_uncertainty_plot_df['value']
    bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

    bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                              (bounds_df['real_value'] <= bounds_df['upper_bound']))


    predictedanomaly = bounds_df.index[~bounds_df['contained']]


    N = 15
    newarr = []

    for i in range(len(predictedanomaly) - N):
        if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
            i + 2] and predictedanomaly[i + 3] + 1 == predictedanomaly[i + 4] and predictedanomaly[i + 4] + 1 ==
                predictedanomaly[i + 5]
                and predictedanomaly[i + 5] + 1 == predictedanomaly[i + 6] and predictedanomaly[i + 6] + 1 ==
                predictedanomaly[i + 7]
                and predictedanomaly[i + 7] + 1 == predictedanomaly[i + 8]
                and predictedanomaly[i + 8] + 1 == predictedanomaly[i + 9]
                and predictedanomaly[i + 9] + 1 == predictedanomaly[i + 10]
                and predictedanomaly[i + 10] + 1 == predictedanomaly[i + 11]
                and predictedanomaly[i + 11] + 1 == predictedanomaly[i + 12]
                and predictedanomaly[i + 12] + 1 == predictedanomaly[i + 13]
                and predictedanomaly[i + 13] + 1 == predictedanomaly[i + 14]
                and predictedanomaly[i + 14] + 1 == predictedanomaly[i + 15]):
            newarr.append(predictedanomaly[i])
    #        newarr.append(predictedanomaly[i + 1])
    #       newarr.append(predictedanomaly[i + 2])

    predicteddanomaly = list(set(newarr))

    # realanomaly = label_data['index']

    predicter = list(range(len(test_uncertainty_df)))




    return stacking_values, stacking_values_uncertainty, rmse1, rmse2, predicteddanomaly, var_total, var_total
