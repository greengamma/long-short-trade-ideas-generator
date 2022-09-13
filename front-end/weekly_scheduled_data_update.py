from data.refactored_data import Data
from models.CNN import CNN_model
from data.arima import Arima_model
import pandas as pd
import numpy as np
import datetime

data = Data()

#Shuffle Data:

# Update Data
# tickers = data.get_tickers()
# prices = data.get_prices(tickers)
# ratios = data.get_ratios(6)
# sma10 = data.create_SMA(10)
# sma20 = data.create_SMA(20)
# sma60 = data.create_SMA(60)

# update CNN model
CNN = CNN_model()
##Get Data
df = CNN.get_data('raw_data/ratios.xlsx')

actual_start_date = (df['Date'].iloc[-1] +
                     datetime.timedelta(days=1)).strftime("%Y-%m-%d")
actual_end_date = (df['Date'].iloc[-1] +
                   datetime.timedelta(days=30)).strftime("%Y-%m-%d")

# mape_predictions_cnn = pd.DataFrame()
# actual_predictions_cnn = pd.DataFrame()
# mape_predictions_cnn['Date'] = df['Date'].iloc[-31:-1]
# mape_predictions_cnn.reset_index(inplace=True, drop=True)
# actual_predictions_cnn['Date'] = pd.date_range(actual_start_date,
#                                                actual_end_date)
# df.drop('Date', inplace=True, axis=1)
# ##create dictionary of ratios
# seperate_ratios = CNN.seperate_ratios(df)
# #Create Model
# model = CNN.create_model(200, 30)
# for columns in df:
#     X, y = CNN.split_sequence(seperate_ratios[columns], 200, 30)
#     X_train, y_train, X_test, y_test = CNN.test_train_splits(X, y)
#     X_test_reshaped, X_train_reshaped = CNN.reshape(X_test, X_train, 200, 1)
#     CNN.fit_model(model, X_train_reshaped, y_train, 1000)
#     #Make prediction on known vals X_test
#     mape_predictions = CNN.make_prediction(model, X_test)
#     mape_predictions_cnn[columns] = mape_predictions[0]
#     #Make prediction on unknown vals
#     X_actual = np.array(seperate_ratios[columns][-200:]).reshape(1, 200, 1)
#     actual_predictions = CNN.make_prediction(model, X_actual)
#     actual_predictions_cnn[columns] = actual_predictions[0]

# mape_predictions_cnn.to_csv('raw_data/CNN_preds_mape.csv', index=False)
# actual_predictions_cnn.to_csv('raw_data/CNN_actual_prediction.csv',
#                           index=False)

# CNN_df = pd.read_csv('raw_data/CNN_preds_mape.csv')
ratios_df = pd.read_excel('raw_data/ratios.xlsx')

# CNN_mapes = CNN.make_mape(CNN_df, ratios_df)
# CNN_mapes.to_csv('raw_data/CNN_mapes.csv', index=False)

# Arima model
arima = Arima_model()
mape = arima.run_model(df, 30)
actual = arima.run_model(df, 0)

mape['Date'] = list(df['Date'].iloc[-30:])
actual['Date'] = pd.date_range(actual_start_date, actual_end_date)
##Move date to the first column
first_column = mape.pop('Date')
mape.insert(0, 'Date', first_column)

first_column = actual.pop('Date')
actual.insert(0, 'Date', first_column)

mape.to_csv('raw_data/Arima_preds_mape.csv', index=False)
actual.to_csv('raw_data/Arima_actual_predictions.csv', index=False)
Arima_df = pd.read_csv('raw_data/Arima_preds_mape.csv')
Arima_mapes = CNN.make_mape(Arima_df, ratios_df)
Arima_mapes.to_csv('raw_data/Arima_mapes.csv', index=False)

#LSTM
#run model
#make 30 day predictions on df
# save predictions to {model_name}.csv

#XGB_BOOST
#run model
#make 30 day predictions on df
# save predictions to {model_name}.csv

#RNN
#run model
#make 30 day predictions on df
# save predictions to {model_name}.csv

##MAPE Calculations
