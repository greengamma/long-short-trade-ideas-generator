import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
import yfinance as yf
import datetime


class ModelLstm:

    def __init__(self):
        pass

    # import data
    def get_data(self):
        df = pd.read_excel('raw_data/ratios.xlsx')
        return df

    # split data into train and validation
    def split_train_val(self, df):
        length_data = len(df)
        split_ratio = 0.7  # %70 train + %30 validation
        length_train = round(length_data * split_ratio)
        length_validation = length_data - length_train
        #print("Data length :", length_data)
        #print("Train data length:", length_train)
        #print("Validation data lenth:", length_validation)

        train_data = df[:length_train].iloc[:, :2]
        train_data['Date'] = pd.to_datetime(
            train_data['Date'])  # converting to date time object

        validation_data = df[length_train:].iloc[:, :2]
        validation_data['Date'] = pd.to_datetime(
            validation_data['Date'])  # converting to date time object

        return train_data, validation_data, length_train, length_validation

    # create train dataset from train split
    def train_split(self, train_data):
        dataset_train = train_data.iloc[:, 1].values
        # Change 1d array to 2d array
        # Changing shape from (1692,) to (1692,1)
        dataset_train = np.reshape(dataset_train, (-1, 1))
        #dataset_train.shape
        dataset_train_scaled = dataset_train

        return dataset_train_scaled

    # create X_train and y_train from train data
    def create_x_y_train(self, df, length_train):
        X_train = []
        y_train = []

        time_step = 100

        for i in range(time_step, length_train):
            X_train.append(df[i - time_step:i, 0:1])
            y_train.append(df[i, 0:1])

        # convert list to array
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        X_train = X_train[:int(X_train.shape[0] * 0.95)]
        X_val = X_train[int(X_train.shape[0] * 0.95):]
        y_train = y_train[:int(y_train.shape[0] * 0.95)]
        y_val = y_train[int(y_train.shape[0] * 0.95):]

        return X_train, X_val, y_train, y_val, time_step

    # create test dataset from validation data
    # R
    #Converting array and scaling
    def create_x_y_test(self, validation_data, length_validation, time_step):
        dataset_validation = validation_data.iloc[:,
                                                  1].values  # getting "Ratio" column and converting to array
        dataset_validation = np.reshape(dataset_validation,
                                        (-1, 1))  # converting 1D to 2D array
        #scaled_dataset_validation =  scaler.fit_transform(dataset_validation)  # scaling  values to between 0 and 1
        scaled_dataset_validation = dataset_validation

        X_test = []
        y_test = []

        for i in range(time_step, length_validation):
            X_test.append(scaled_dataset_validation[i - time_step:i, 0])
            y_test.append(scaled_dataset_validation[i, 0])

        # Converting to array
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(
            X_test,
            (X_test.shape[0], X_test.shape[1], 1))  # reshape to 3D array
        y_test = np.reshape(y_test, (-1, 1))  # reshape to 2D array

        return X_test, y_test

    # create LSTM model
    def generate_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        es = EarlyStopping(patience=20, restore_best_weights=True)
        model_lstm = Sequential()

        model_lstm.add(
            LSTM(20, return_sequences=False,
                 input_shape=(X_train.shape[1], 1)))  #64 lstm neuron block
        model_lstm.add(Dense(32))
        model_lstm.add(Dense(1))

        model_lstm.compile(loss="mape",
                           optimizer="rmsprop",
                           metrics=["mae", "mape"])
        history2 = model_lstm.fit(X_train,
                                  y_train,
                                  epochs=400,
                                  batch_size=64,
                                  validation_data=(X_val, y_val),
                                  callbacks=[es])
        mape = model_lstm.evaluate(X_test, y_test)

        return mape[2], model_lstm

    # implemenation refactored
    def run_model(self, df):

        mape_dict = {}

        for ratio in df:
            # split into train/test
            if ratio == 'Date':
                continue
            else:
                one_ratio_df = pd.DataFrame(df[['Date', ratio]])
                train_data, validation_data, length_train, length_validation = self.split_train_val(
                    one_ratio_df)
                # call train_split
                dataset_train_scaled = self.train_split(train_data)
                # create X_train, y_train
                X_train, X_val, y_train, y_val, time_step = self.create_x_y_train(
                    dataset_train_scaled, length_train)
                # create X_test, y_test
                X_test, y_test = self.create_x_y_test(validation_data,
                                                      length_validation,
                                                      time_step)
                # run LSTM model

                mape, model = self.generate_model(X_train, y_train, X_val,
                                                  y_val, X_test, y_test)

                mape_dict[ratio] = round(mape, 3)

        mape_lstm = pd.DataFrame(mape_dict.items(), columns=['ratio', 'MAPE'])

        return mape_lstm, model

    def prep_data(self, df):
        df.set_index('Date', inplace=True)
        tmp_list = []
        fut_list = []

        #Getting the last 100 days records
        for ratio in df.columns:
            fut_inp = df[len(df) - 100:][ratio]
            fut_inp = fut_inp.values.reshape(1, -1)
            fut_list.append(fut_inp)
            tmp_inp = list(fut_inp)
            #Creating list of the last 100 data
            tmp_inp = tmp_inp[0].tolist()
            tmp_list.append(tmp_inp)

        return tmp_list, fut_list

    def lstm_predict(self, model, tmp_list, fut_list):
        # predicting next 30 days price suing the current data
        # it will predict in sliding window manner (algorithm) with stride 1
        preds_list = []
        for x in range(0, len(tmp_list)):
            tmp_inp = tmp_list[x]
            fut_inp = fut_list[x]
            lstm_preds = []
            n_steps = 100
            i = 0
            while (i < 30):

                if (len(tmp_inp) > 100):
                    fut_inp = np.array(tmp_inp[1:])
                    fut_inp = fut_inp.reshape(1, -1)
                    fut_inp = fut_inp.reshape((1, n_steps, 1))
                    yhat = model.predict(fut_inp, verbose=0)
                    tmp_inp.extend(yhat[0].tolist())
                    tmp_inp = tmp_inp[1:]
                    lstm_preds.extend(yhat.tolist())
                    i = i + 1
                else:
                    fut_inp = fut_inp.reshape((1, n_steps, 1))
                    yhat = model.predict(fut_inp, verbose=0)
                    tmp_inp.extend(yhat[0].tolist())
                    lstm_preds.extend(yhat.tolist())
                    i = i + 1

            preds_list.append(lstm_preds)

        return preds_list

    def clean_df(self, lstm_preds, test_df):
        #final_forecast_df = pd.DataFrame()
        check_df = test_df.copy()
        df = pd.DataFrame()
        cols = list(test_df.columns)
        cols.pop(0)
        preds_array = np.array(lstm_preds)
        preds_resh = preds_array.reshape(10, 30)
        final_df = pd.DataFrame(preds_resh).T
        final_df.columns = cols

        # convert 'Date' column
        df['Date'] = pd.to_datetime(check_df['Date'])
        df['Date'].iloc[-1] + datetime.timedelta(days=1)

        actual_start_date = (df['Date'].iloc[-1] +
                             datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        actual_end_date = (df['Date'].iloc[-1] +
                           datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        final_df['Date'] = pd.date_range(actual_start_date, actual_end_date)
        first_column = final_df.pop('Date')
        final_df.insert(0, 'Date', first_column)

        return final_df


if __name__ == '__main__':
    data = ModelLstm()
    df = data.get_data()
    print('data received')
    mape_lstm, model = data.run_model(df)
    print('mape_lstm received')
    print(mape_lstm)

    test_df = df.copy()
    tmp_list, fut_list = data.prep_data(test_df)
    lstm_preds = data.lstm_predict(model, tmp_list, fut_list)
    test_df = df.copy()
    final_forecast_df = data.clean_df(lstm_preds, test_df)
    print(final_forecast_df)
    final_forecast_df.to_csv('raw_data/lstm_actual_predictions.csv',
                             index=False)
    mape_lstm.to_csv('raw_data/lstm_mapes.csv', index=False)
