from ctypes import get_errno
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import pandas_ta as ta
import datetime
#from fastai.tabular.core import add_datepart
import itertools
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from xgboost import XGBRegressor#, plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


class FBProph:
    def __init__(self):
        pass


    # import data
    def get_data(self):
        df = pd.read_csv('../../long_short_local/raw_data/cleaned_data_2y.csv')
        # convert 'Date' column to datetime values
        df['Date'] = pd.to_datetime(df['Date'].str[:10])

        return df


    # create SMA (10, 20, 60 days)
    def create_sma(self, df):
        sma_list = [10, 20, 60]

        for sma in sma_list:
            if sma == 10:
                sma_10_df_prep = pd.DataFrame(df['Date'])

                for ratio in df.columns:
                    test_df = df[['Date', ratio]]
                    sma10 = pd.DataFrame(ta.sma(test_df[ratio], length=10))
                    loop_df = pd.concat([test_df, sma10], axis=1, ignore_index=False)
                    loop_df.rename(columns={'SMA_10': f'{ratio}_SMA_10'}, inplace=True)
                    sma_10_df_prep = pd.concat([sma_10_df_prep, loop_df], axis=1, ignore_index=False)
                    sma_10_df = sma_10_df_prep.iloc[:, 3:]

            elif sma == 20:
                sma_20_df_prep = pd.DataFrame(df['Date'])

                for ratio in df.columns:
                    test_df = df[['Date', ratio]]
                    sma20 = pd.DataFrame(ta.sma(test_df[ratio], length=20))
                    loop_df = pd.concat([test_df, sma20], axis=1, ignore_index=False)
                    loop_df.rename(columns={'SMA_20': f'{ratio}_SMA_20'}, inplace=True)
                    sma_20_df_prep = pd.concat([sma_20_df_prep, loop_df], axis=1, ignore_index=False)
                    sma_20_df = sma_20_df_prep.iloc[:, 3:]

            else:
                sma_60_df_prep = pd.DataFrame(df['Date'])

                for ratio in df.columns:
                    test_df = df[['Date', ratio]]
                    sma60 = pd.DataFrame(ta.sma(test_df[ratio], length=60))
                    loop_df = pd.concat([test_df, sma60], axis=1, ignore_index=False)
                    loop_df.rename(columns={'SMA_60': f'{ratio}_SMA_60'}, inplace=True)
                    sma_60_df_prep = pd.concat([sma_60_df_prep, loop_df], axis=1, ignore_index=False)
                    sma_60_df = sma_60_df_prep.iloc[:, 3:]

        return sma_10_df, sma_20_df, sma_60_df


    # create RSI 14
    def create_rsi(self, df):
        rsi_14_df_prep = pd.DataFrame(df['Date'])

        for ratio in df.columns:
            test_df = df[['Date', ratio]]
            rsi14 = pd.DataFrame(ta.rsi(test_df[ratio], length=14))
            loop_df = pd.concat([test_df, rsi14], axis=1, ignore_index=False)
            loop_df.rename(columns={'RSI_14': f'{ratio}_RSI_14'}, inplace=True)
            rsi_14_df_prep = pd.concat([rsi_14_df_prep, loop_df], axis=1, ignore_index=False)
            rsi_14_df = rsi_14_df_prep.iloc[:, 3:]
        return rsi_14_df


    def concatenate_df(self, sma_10_df, sma_20_df, sma_60_df, rsi_14_df):
        # concatenate all 4 dataframse
        concat_10_20_df = pd.concat([sma_10_df, sma_20_df], axis=1)
        concat_10_20_60_df = pd.concat([concat_10_20_df, sma_60_df], axis=1)
        combined_df = pd.concat([concat_10_20_60_df, rsi_14_df], axis=1)
        # remove duplicates of 'Date' column
        dropped_date_df = combined_df.loc[:,~combined_df.columns.duplicated()]
        dropped_date_df.set_index('Date', inplace=True)
        # sort the ratios by column name
        sorted_df = dropped_date_df.reindex(sorted(dropped_date_df.columns, reverse=False), axis=1)
        # clean dataframe
        sorted_df = sorted_df.fillna(sorted_df.median())
        #sorted_df.reset_index(inplace=True)

        return sorted_df


    # extract features with prophet
    def prophet_features(self, df, horizon=30):
        temp_df = df.reset_index()
        ratio_name = df.columns[0]
        temp_df = temp_df[['Date', ratio_name]]
        temp_df.rename(columns={'Date': 'ds', ratio_name: 'y'}, inplace=True)

        # take last week of the dataset for validation
        train_set, test_set = temp_df.iloc[:-horizon,:], temp_df.iloc[-horizon:,:]

        # define prophet model
        m = Prophet(
                    growth='linear',
                    seasonality_mode='additive',
                    interval_width=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
        # train prophet model
        m.fit(train_set)

        # extract features from data using prophet to predict train set
        predictions_train = m.predict(train_set.drop('y', axis=1))
        # extract features from data using prophet to predict test set
        predictions_test = m.predict(test_set.drop('y', axis=1))
        # merge train and test predictions
        predictions = pd.concat([predictions_train, predictions_test], axis=0)

        return predictions_test, test_set


    def fb_mape(self, df, preds_df):
        # extract 'date' columns
        date_col = preds_df['ds'].iloc[:, 0]
        # select every 5th column to get the ratio name
        preds_cols = list(sorted_df.columns[::5])
        # PREDICTED SET
        # get all 'yhat' from predictions
        forecast_df = preds_df['yhat']
        # rename columns
        forecast_df.columns = preds_cols
        # TEST SET
        # get all 'y' from test set
        df = df['y']
        # rename columns
        df.columns = preds_cols

        # compute MAPE
        y_true, y_pred = np.array(df), np.array(forecast_df)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return mape


    def generate_predictions(self, sorted_df, preds):
        # select every 5th column to get the ratio name
        preds_cols = sorted_df.columns[::5]

        preds_df = pd.DataFrame()
        i = 0

        for pred in preds:
            preds_df[preds_cols[i]] = pred
            i += 1

        check_df = sorted_df.reset_index()
        actual_start_date = (check_df['Date'].iloc[-1] +
                    datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        actual_end_date = (check_df['Date'].iloc[-1] +
                   datetime.timedelta(days=30)).strftime("%Y-%m-%d")

        preds_df['Date'] = pd.date_range(actual_start_date, actual_end_date)
        first_column = preds_df.pop('Date')
        preds_df.insert(0, 'Date', first_column)

        return preds_df


    # create model for each ratio
    # get number of ratios
    def run_model(self, sorted_df):
        num_of_ratios = int(sorted_df.shape[1] / 5)

        # create column pointers
        col_l = 0
        col_r = 5

        # create dict for mape
        preds_df = pd.DataFrame()
        test_df = pd.DataFrame()

        for i in range(0, num_of_ratios, 1):
            ratio_df = sorted_df.iloc[:, col_l:col_r]
            predictions, test_set = self.prophet_features(ratio_df)
            preds_df = pd.concat([preds_df, predictions], axis=1)
            test_df = pd.concat([test_df, test_set], axis=1)
            col_l += 5
            col_r += 5

        return test_df, preds_df


    def fb_forecast(self, df, horizon=30):
        temp_df = df.reset_index()
        ratio_name = df.columns[0]
        temp_df = temp_df[['Date', ratio_name]]
        temp_df.rename(columns={'Date': 'ds', ratio_name: 'y'}, inplace=True)

        # define prophet model
        m = Prophet(
                    growth='linear',
                    seasonality_mode='additive',
                    interval_width=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False
                )
        # train prophet model
        m.fit(temp_df)

        future = m.make_future_dataframe(periods=80)
        forecast = m.predict(future)
        # summarize the forecast
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        return forecast


    def generate_forecast(self, sorted_df):
        # call forecast function
        num_of_ratios = int(sorted_df.shape[1] / 5)

        # create column pointers
        col_l = 0
        col_r = 5

        # create dict for mape
        forecast_df = pd.DataFrame()
        return_forecast_df = pd.DataFrame()

        for i in range(0, num_of_ratios, 1):
            ratio_df = sorted_df.iloc[:, col_l:col_r]
            return_forecast_df = self.fb_forecast(ratio_df)
            forecast_df = pd.concat([forecast_df, return_forecast_df], axis=1)
            col_l += 5
            col_r += 5

        return forecast_df


    def clean_df(self, sorted_df, forecast_df):
        final_forecast_df = forecast_df['yhat']
        # select every 5th column to get the ratio name
        preds_cols = sorted_df.columns[::5]
        # rename columns
        final_forecast_df.columns = preds_cols
        all_dates = forecast_df['ds'].iloc[:, 0]
        final_forecast_df = pd.concat([final_forecast_df, all_dates], axis=1)
        final_forecast_df.rename(columns={'ds': 'Date'}, inplace=True)
        check_df = sorted_df.reset_index()
        actual_start_date = (check_df['Date'].iloc[-1] + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        actual_end_date = (check_df['Date'].iloc[-1] + datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        really_final_df = final_forecast_df[final_forecast_df.Date.between(actual_start_date, actual_end_date)]
        #final_forecast_df['Date'] = pd.date_range(actual_start_date, actual_end_date)
        first_column = really_final_df.pop('Date')
        really_final_df.insert(0, 'Date', first_column)

        return really_final_df



if __name__ == '__main__':
    data = FBProph()
    df = data.get_data()
    sma_10_df, sma_20_df, sma_60_df = data.create_sma(df)
    print('sma computed')
    rsi_14_df = data.create_rsi(df)
    print('rsi computed')
    sorted_df = data.concatenate_df(sma_10_df, sma_20_df, sma_60_df, rsi_14_df)
    print('sorted_df created')
    print(sorted_df)
    test_df, preds_df = data.run_model(sorted_df)
    mape = data.fb_mape(test_df, preds_df)
    forecast_df = data.generate_forecast(sorted_df)
    final_df = data.clean_df(sorted_df, forecast_df)
    print('cleaned dataframe')
    print(final_df)
