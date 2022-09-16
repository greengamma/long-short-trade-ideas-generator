import numpy as np


def mean_absolute_percentage_error(y_true, predictions):
    '''Calculates MAPE for predictions'''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def determine_best_model(lstm_mape, arima_mape, cnn_mape, xgb_reg_mape):
    '''
This function receives as input the MAPE of the four models (order is important!!)
It returns the name of the model with the lowest
'''
    # list of model names to print
    model_names = ['LSTM', 'ARIMA', 'CNN', 'XGB']
    mape_list = [lstm_mape, arima_mape, cnn_mape, xgb_reg_mape]
    model_dict = {}

    # sort the MAPE from best (lowest) to worst (highest) and get the 5 best; round result to 4 digits
    for i in range(0, len(mape_list)):
        model_dict[model_names[i]] = round(
            mape_list[i].sort_values('MAPE').iloc[0:5]['MAPE'].mean(), 4)

    best_mape = min(model_dict.items(), key=lambda x: x[1])
    return best_mape
