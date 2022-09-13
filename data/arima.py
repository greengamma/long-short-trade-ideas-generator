import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pmdarima as pm
from sklearn.metrics import r2_score
#from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import figure
from statsmodels.tsa.arima.model import ARIMA
import statsmodels as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


class Arima_model:

    def __init__(self):
        pass

    # import data
    def get_data(self):
        df = pd.read_excel('raw_data/ratios.xlsx')
        return df

    def run_model(self, df_init, test_size):
        df = df_init.copy()
        ratio_list = list(df.columns)
        ratio_list.remove('Date')
        predictions = pd.DataFrame()
        #         predictions['Date'] = df_init['Date'].iloc[-30:]

        for i in range(len(ratio_list)):

            #train and test split

            ratio = [ratio_list[i]]

            df = df_init[ratio_list[i]]
            train_size = df.shape[0] - test_size
            index = train_size

            df_train = df.iloc[:index]
            df_test = df.iloc[index:]

            #auto - ARIMA
            model = pm.auto_arima(df,
                                  start_p=1,
                                  max_p=6,
                                  start_q=1,
                                  max_q=6,
                                  trend='t',
                                  max_d=1,
                                  seasonal=False,
                                  trace=True,
                                  verbose=0)

            # Build model
            arima = ARIMA(df_train,
                          order=model.get_params().get("order"),
                          trend='t')
            arima = arima.fit()

            # Forecast values
            forecast = arima.forecast(30, alpha=0.05)  # 95% confidence
            predictions[ratio_list[i]] = forecast

        predictions.reset_index(inplace=True, drop=True)
        return predictions


if __name__ == '__main__':
    data = Arima_model()
    df = data.get_data()
    print('received data')
    mape_arima = data.run_model(df, 30)
    print('received mape_arima')
    print(mape_arima)
