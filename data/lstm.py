import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense
import yfinance as yf


class ModelLstm:

    def __init__(self):
        pass

    # import data
    def get_data(self):
        df = pd.read_csv('../../long_short_local/raw_data/cleaned_data_2y.csv')
        return df


    # split data into train and validation
    def split_train_val(self, df):
        length_data = len(df)
        split_ratio = 0.7           # %70 train + %30 validation
        length_train = round(length_data * split_ratio)
        length_validation = length_data - length_train
        #print("Data length :", length_data)
        #print("Train data length:", length_train)
        #print("Validation data lenth:", length_validation)

        train_data = df[:length_train].iloc[:,:2]
        train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object

        validation_data = df[length_train:].iloc[:,:2]
        validation_data['Date'] = pd.to_datetime(validation_data['Date'])  # converting to date time object

        return train_data, validation_data, length_train, length_validation


    # create train dataset from train split
    def train_split(self, train_data):
        dataset_train = train_data.iloc[:, 1].values
        # Change 1d array to 2d array
        # Changing shape from (1692,) to (1692,1)
        dataset_train = np.reshape(dataset_train, (-1,1))
        #dataset_train.shape
        dataset_train_scaled = dataset_train

        return dataset_train_scaled


    # create X_train and y_train from train data
    def create_x_y_train(self, df, length_train):
        X_train = []
        y_train = []

        time_step = 20 #change that?

        for i in range(time_step, length_train):
            X_train.append(df[i-time_step:i,0:1])
            y_train.append(df[i,0:1])

        # convert list to array
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
        y_train = np.reshape(y_train, (y_train.shape[0],1))

        X_train = X_train[:int(X_train.shape[0]*0.95)]
        X_val = X_train[int(X_train.shape[0]*0.95):]
        y_train = y_train[:int(y_train.shape[0]*0.95)]
        y_val = y_train[int(y_train.shape[0]*0.95):]

        return X_train, X_val, y_train, y_val, time_step


    # create test dataset from validation data
    # R
    #Converting array and scaling
    def create_x_y_test(self, validation_data, length_validation, time_step):
        dataset_validation = validation_data.iloc[:,1].values  # getting "Ratio" column and converting to array
        dataset_validation = np.reshape(dataset_validation, (-1,1))  # converting 1D to 2D array
        #scaled_dataset_validation =  scaler.fit_transform(dataset_validation)  # scaling  values to between 0 and 1
        scaled_dataset_validation = dataset_validation

        X_test = []
        y_test = []

        for i in range(time_step, length_validation):
            X_test.append(scaled_dataset_validation[i-time_step:i,0])
            y_test.append(scaled_dataset_validation[i,0])

        # Converting to array
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))  # reshape to 3D array
        y_test = np.reshape(y_test, (-1,1))  # reshape to 2D array

        return X_test, y_test


    # create LSTM model
    def generate_model(self, X_train, y_train, X_val, y_val, X_test, y_test):
        es = EarlyStopping(patience=20, restore_best_weights=True)
        model_lstm = Sequential()

        model_lstm.add(LSTM(20, return_sequences=False, input_shape=(X_train.shape[1],1))) #64 lstm neuron block
        model_lstm.add(Dense(32))
        model_lstm.add(Dense(1))

        model_lstm.compile(loss = "mape", optimizer = "rmsprop", metrics = ["mae", "mape"])
        history2 = model_lstm.fit(X_train, y_train, epochs = 400, batch_size = 64,validation_data = (X_val, y_val),callbacks=[es])
        mape = model_lstm.evaluate(X_test, y_test)

        return mape[2]


    # implemenation refactored
    def run_model(self, df):

        mape_dict = {}

        for ratio in df:
        # split into train/test
            if ratio == 'Date':
                continue
            else:
                one_ratio_df = pd.DataFrame(df[['Date', ratio]])
                train_data, validation_data, length_train, length_validation = self.split_train_val(one_ratio_df)
                # call train_split
                dataset_train_scaled = self.train_split(train_data)
                # create X_train, y_train
                X_train, X_val, y_train, y_val, time_step = self.create_x_y_train(dataset_train_scaled, length_train)
                # create X_test, y_test
                X_test, y_test = self.create_x_y_test(validation_data, length_validation, time_step)
                # run LSTM model
                mape = self.generate_model(X_train, y_train, X_val, y_val, X_test, y_test)
                mape_dict[ratio] = round(mape, 3)

        return mape_dict


if __name__ == '__main__':
    data = ModelLstm()
    df = data.get_data()
    print('data received')
    mape = data.run_model(df)
    print('mape received')
    print(mape)
