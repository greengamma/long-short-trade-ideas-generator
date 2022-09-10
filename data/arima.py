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


class ModelArima:
    def __init__(self):
        pass


    # import data
    def get_data(self):
        df = pd.read_csv('../../long_short_local/raw_data/cleaned_data_2y.csv', index_col=0)
        return df


    def run_model(self, df_init):

        ratio_list = list(df_init.columns)

        results = []

        for i in range(len(ratio_list)):

            #train and test split

            ratio = [ratio_list[i]]

            df = df_init[ratio_list[i]]
            train_size = 0.6
            index = round(train_size*df.shape[0])

            df_train = df.iloc[:index]
            df_test = df.iloc[index:]

            #auto - ARIMA
            model = pm.auto_arima(df,
                            start_p=1, max_p=6,
                            start_q=1, max_q=6,
                            trend='t',
                            max_d = 2,
                            seasonal=False,
                            trace=True)

            # Build model
            arima = ARIMA(df_train, order=model.get_params().get("order"), trend='t')
            arima = arima.fit()

            ## Forecast
            # Forecast values
            forecast = arima.forecast(len(df_test), alpha=0.05)  # 95% confidence

            forecast_df = pd.DataFrame(forecast)
            forecast_df["id"] = list(df_test.index)
            forecast_df.set_index("id",inplace=True)

            #append to MAPE list
            mape_arima = np.mean(np.abs(forecast_df["predicted_mean"] - df_test)/np.abs(df_test))
            results.append(mape_arima)

        summary = pd.DataFrame()
        summary["ratio"] =  ratio_list
        summary["MAPE"] = results
        return summary


if __name__ == '__main__':
    data = ModelArima()
    df = data.get_data()
    print('received data')
    summary = data.run_model(df)
    print('received summary')
    print(summary)
