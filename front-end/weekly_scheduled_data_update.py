from data.refactored_data import Data
from models.CNN import CNN_model
from data.arima import Arima_model
from data.xgb_reg import ModelXGBoost
from data.prophet import FBProph
import pandas as pd
import numpy as np
import datetime

data = Data()

#Shuffle Data:

# Update Data
# tickers = data.get_tickers()
# print('tickers')
# prices = data.get_prices(tickers, '2y')
# print('prices')
# ratios = data.get_ratios(6)
# print('ratios')
# sma10 = data.create_SMA(10)
# sma20 = data.create_SMA(20)
# sma60 = data.create_SMA(60)
# print('moving averages')

# # update CNN model
CNN = CNN_model()
# # ##Get Data
df = CNN.get_data('raw_data/ratios.xlsx')

actual_start_date = (df['Date'].iloc[-1] +
                     datetime.timedelta(days=1)).strftime("%Y-%m-%d")
actual_end_date = (df['Date'].iloc[-1] +
                   datetime.timedelta(days=30)).strftime("%Y-%m-%d")

mape_predictions_cnn = pd.DataFrame()
actual_predictions_cnn = pd.DataFrame()
mape_predictions_cnn['Date'] = df['Date'].iloc[-31:-1]
mape_predictions_cnn.reset_index(inplace=True, drop=True)
actual_predictions_cnn['Date'] = pd.date_range(actual_start_date,
                                               actual_end_date)
df.drop('Date', inplace=True, axis=1)
##create dictionary of ratios
seperate_ratios = CNN.seperate_ratios(df)
#Create Model
model = CNN.create_model(200, 30)
for columns in df:
    X, y = CNN.split_sequence(seperate_ratios[columns], 200, 30)
    X_train, y_train, X_test, y_test = CNN.test_train_splits(X, y)
    X_test_reshaped, X_train_reshaped = CNN.reshape(X_test, X_train, 200, 1)
    CNN.fit_model(model, X_train_reshaped, y_train, 500)
    #Make prediction on known vals X_test
    mape_predictions = CNN.make_prediction(model, X_test)
    mape_predictions_cnn[columns] = mape_predictions[0]
    #Make prediction on unknown vals
    X_actual = np.array(seperate_ratios[columns][-200:]).reshape(1, 200, 1)
    actual_predictions = CNN.make_prediction(model, X_actual)
    actual_predictions_cnn[columns] = actual_predictions[0]

mape_predictions_cnn.to_csv('raw_data/CNN_mape_predictions.csv', index=False)
actual_predictions_cnn.to_csv('raw_data/CNN_actual_predictions.csv',
                              index=False)

CNN_df = pd.read_csv('raw_data/CNN_mape_predictions.csv')
ratios_df = pd.read_excel('raw_data/ratios.xlsx')

CNN_mapes = CNN.make_mape(CNN_df, ratios_df)
CNN_mapes.to_csv('raw_data/CNN_mapes.csv', index=False)

# Arima model
# arima = Arima_model()
# mape = arima.run_model(df, 30)
# actual = arima.run_model(df, 0)

# mape['Date'] = list(df['Date'].iloc[-30:])
# actual['Date'] = pd.date_range(actual_start_date, actual_end_date)
# ##Move date to the first column
# first_column = mape.pop('Date')
# mape.insert(0, 'Date', first_column)

# first_column = actual.pop('Date')
# actual.insert(0, 'Date', first_column)

# mape.to_csv('raw_data/Arima_mape_predictions.csv', index=False)
# actual.to_csv('raw_data/Arima_actual_predictions.csv', index=False)
# Arima_df = pd.read_csv('raw_data/Arima_mape_predictions.csv')
# Arima_mapes = CNN.make_mape(Arima_df, ratios_df)
# Arima_mapes.to_csv('raw_data/Arima_mapes.csv', index=False)

#LSTM
#run model
#make 30 day predictions on df
# save predictions to {model_name}.csv

#Prophet
# data = FBProph()
# df = data.get_data()
# sma_10_df, sma_20_df, sma_60_df = data.create_sma(df)
# print('sma computed')
# rsi_14_df = data.create_rsi(df)
# print('rsi computed')
# sorted_df = data.concatenate_df(sma_10_df, sma_20_df, sma_60_df, rsi_14_df)
# print('sorted_df created')
# test_df, preds_df = data.run_model(sorted_df)
# mape = data.fb_mape(test_df, preds_df, sorted_df)
# forecast_df = data.generate_forecast(sorted_df)
# final_df = data.clean_df(sorted_df, forecast_df)
# # mape_preds_df = data.clean_df(sorted_df, test_df)
# print('cleaned dataframe')

# forecast_df = forecast_df[[
#     'ds',
#     'yhat',
# ]]
# forecast_df = forecast_df.iloc[:, 9:]
# names = final_df.columns
# forecast_df.set_axis(names, axis=1, inplace=True)
# forecast_df = forecast_df.loc[474:504]

# mape_df = data.get_ratio_mapes(df, sorted_df)
# print('forecast_df')
# print(forecast_df)

# print('test_df')
# print(test_df)

# final_df.to_csv('raw_data/prophet_actual_predictions.csv', index=False)
# mape_df.to_csv('raw_data/prophet_mapes.csv', index=False)
# forecast_df.to_csv('raw_data/prophet_mape_predictions.csv', index=False)
