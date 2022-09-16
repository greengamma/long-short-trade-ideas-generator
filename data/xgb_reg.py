from ctypes import get_errno
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import cross_val_score, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import pandas_ta as ta
import datetime
#from fastai.tabular.core import add_datepart
import itertools
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from xgboost import XGBRegressor  #, plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error


class ModelXGBoost:

    def __init__(self):
        pass

    # import data
    def get_data(self):
        df = pd.read_excel('raw_data/ratios.xlsx')
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
                    loop_df = pd.concat([test_df, sma10],
                                        axis=1,
                                        ignore_index=False)
                    loop_df.rename(columns={'SMA_10': f'{ratio}_SMA_10'},
                                   inplace=True)
                    sma_10_df_prep = pd.concat([sma_10_df_prep, loop_df],
                                               axis=1,
                                               ignore_index=False)
                    sma_10_df = sma_10_df_prep.iloc[:, 3:]

            elif sma == 20:
                sma_20_df_prep = pd.DataFrame(df['Date'])

                for ratio in df.columns:
                    test_df = df[['Date', ratio]]
                    sma20 = pd.DataFrame(ta.sma(test_df[ratio], length=20))
                    loop_df = pd.concat([test_df, sma20],
                                        axis=1,
                                        ignore_index=False)
                    loop_df.rename(columns={'SMA_20': f'{ratio}_SMA_20'},
                                   inplace=True)
                    sma_20_df_prep = pd.concat([sma_20_df_prep, loop_df],
                                               axis=1,
                                               ignore_index=False)
                    sma_20_df = sma_20_df_prep.iloc[:, 3:]

            else:
                sma_60_df_prep = pd.DataFrame(df['Date'])

                for ratio in df.columns:
                    test_df = df[['Date', ratio]]
                    sma60 = pd.DataFrame(ta.sma(test_df[ratio], length=60))
                    loop_df = pd.concat([test_df, sma60],
                                        axis=1,
                                        ignore_index=False)
                    loop_df.rename(columns={'SMA_60': f'{ratio}_SMA_60'},
                                   inplace=True)
                    sma_60_df_prep = pd.concat([sma_60_df_prep, loop_df],
                                               axis=1,
                                               ignore_index=False)
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
            rsi_14_df_prep = pd.concat([rsi_14_df_prep, loop_df],
                                       axis=1,
                                       ignore_index=False)
            rsi_14_df = rsi_14_df_prep.iloc[:, 3:]
        return rsi_14_df

    def concatenate_df(self, sma_10_df, sma_20_df, sma_60_df, rsi_14_df):
        # concatenate all 4 dataframse
        concat_10_20_df = pd.concat([sma_10_df, sma_20_df], axis=1)
        concat_10_20_60_df = pd.concat([concat_10_20_df, sma_60_df], axis=1)
        combined_df = pd.concat([concat_10_20_60_df, rsi_14_df], axis=1)
        # remove duplicates of 'Date' column
        dropped_date_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        dropped_date_df.set_index('Date', inplace=True)
        # sort the ratios by column name
        sorted_df = dropped_date_df.reindex(sorted(dropped_date_df.columns,
                                                   reverse=False),
                                            axis=1)
        # clean dataframe
        sorted_df = sorted_df.fillna(sorted_df.median())
        return sorted_df

    # extract features with prophet
    def prophet_features(self, df, horizon=30):
        temp_df = df.reset_index()
        ratio_name = df.columns[0]
        temp_df = temp_df[['Date', ratio_name]]
        temp_df.rename(columns={'Date': 'ds', ratio_name: 'y'}, inplace=True)

        # take last week of the dataset for validation
        train_set, test_set = temp_df.iloc[:-horizon, :], temp_df.iloc[
            -horizon:, :]

        # define prophet model
        m = Prophet(growth='linear',
                    seasonality_mode='additive',
                    interval_width=0.95,
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False)
        # train prophet model
        m.fit(train_set)

        # extract features from data using prophet to predict train set
        predictions_train = m.predict(train_set.drop('y', axis=1))
        # extract features from data using prophet to predict test set
        predictions_test = m.predict(test_set.drop('y', axis=1))
        # merge train and test predictions
        predictions = pd.concat([predictions_train, predictions_test], axis=0)

        return predictions

    # train XGBoost model
    def train_xgb_with_prophet_features(self,
                                        df,
                                        horizon=30,
                                        lags=[1, 2, 3, 4, 5]):
        # create a dataframe with all the new features created with Prophet
        new_prophet_features = self.prophet_features(df, horizon=horizon)
        df.reset_index(inplace=True)

        # merge the Prophet features df with the first df
        df = pd.merge(df,
                      new_prophet_features,
                      left_on=['Date'],
                      right_on=['ds'],
                      how='inner')
        df.drop('ds', axis=1, inplace=True)
        df.set_index('Date', inplace=True)

        # create some lag variables using Prophet predictions (yhat column)
        for lag in lags:
            df[f'yhat_lag_{lag}'] = df['yhat'].shift(lag)
        df.dropna(axis=0, how='any')

        ratio_name = df.columns[0]
        X = df.drop(ratio_name, axis=1)
        y = df[ratio_name]

        # take last week of the dataset for validation
        X_train, X_test = X.iloc[:-horizon, :], X.iloc[-horizon:, :]
        y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

        # define XGBoost model, train it and make predictions
        xgb_model = XGBRegressor(n_estimators=1000,
                                 learning_rate=0.01,
                                 max_depth=6,
                                 subsample=0.6,
                                 colsample_bytree=0.7,
                                 random_state=42,
                                 eval_metric='mape')
        #model = XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)

        #  compute MAE
        mae = np.round(mean_absolute_error(y_test, predictions), 3)
        # compute MAPE
        y_true, y_pred = np.array(y_test), np.array(predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # plot reality vs prediction for the last 95 days of the dataset
        #fig = plt.figure(figsize=(16,6))
        #plt.title(f'{ratio_name}: Actual vs Prediction - MAPE {round(mape, 2)} & MAE {mae}', fontsize=20)
        #plt.plot(y_test, color='mediumblue')
        #plt.plot(pd.Series(predictions, index=y_test.index), color='darkviolet')
        #plt.xlabel('Date', fontsize=16)
        #plt.ylabel('Ratio', fontsize=16)
        #plt.legend(labels=['Actual', 'Prediction'], fontsize=16)
        #plt.grid()
        #plt.show()

        mape_dict = {}
        mape_dict[ratio_name] = mape
        return [mape_dict, predictions]

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
        mape_list = []
        preds = []

        for i in range(0, num_of_ratios, 1):
            ratio_df = sorted_df.iloc[:, col_l:col_r]
            #mape_dict.update(self.train_xgb_with_prophet_features(ratio_df))
            returned_mape_dict, returned_preds = self.train_xgb_with_prophet_features(
                ratio_df)
            mape_list.append(returned_mape_dict)
            preds.append(returned_preds)
            col_l += 5
            col_r += 5
            # convert list of dicts to dict

        mape_dict = {k: v for element in mape_list for k, v in element.items()}
        mape_xgb_reg = pd.DataFrame(mape_dict.items(),
                                    columns=['ratio', 'MAPE'])
        return mape_xgb_reg, preds


if __name__ == '__main__':
    data = ModelXGBoost()
    df = data.get_data()
    sma_10_df, sma_20_df, sma_60_df = data.create_sma(df)
    print('sma computed')
    rsi_14_df = data.create_rsi(df)
    print('rsi computed')
    sorted_df = data.concatenate_df(sma_10_df, sma_20_df, sma_60_df, rsi_14_df)
    mape_xgb_reg, preds = data.run_model(sorted_df)
    print('received mape_xgb_reg')
    preds_df = data.generate_predictions(sorted_df, preds)
    print(mape_xgb_reg)
    print(preds_df)
