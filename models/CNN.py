from numpy import array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from datetime import timedelta, date


class CNN_model():

    def __init__(self):
        pass

    def get_data(self, file_location):
        ''' retrieves data from csv returns dataframe'''
        return pd.read_csv(file_location)

    def seperate_ratios(self, ratios_df):
        '''Seperates the ratios into lists and returns a library of the lists with column name as the key and list as value'''
        df = ratios_df.copy()
        df.set_index('Date', inplace=True)
        ratios_dict = {}
        for columns in ratios_df:
            ratios_dict[columns] = ratios_df[columns].to_list()

        return ratios_dict

    # split a univariate sequence into samples
    def split_sequence(self, sequence, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out
            # check if we are beyond the sequence
            if out_end_ix > len(sequence):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    def test_train_splits(self, X, y):
        X_train = X[0:X.shape[0] - 1]
        X_test = X[X.shape[0] - 1:]
        X_train.shape
        X_test.shape
        y_train = y[0:y.shape[0] - 1]
        y_test = y[y.shape[0] - 1:]

        return X_train, y_train, X_test, y_test

    def reshape(self, X_test, X_train, timesteps, features):
        X_test_reshaped = X_test.reshape(
            (X_test.shape[0], timesteps, features))
        X_train_reshaped = X_train.reshape(
            (X_train.shape[0], timesteps, features))
        return X_test_reshaped, X_train_reshaped

    def create_model(self, input_shape, output):
        model = Sequential()
        model.add(
            Conv1D(filters=64,
                   kernel_size=2,
                   activation='relu',
                   input_shape=(input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(
            Conv1D(filters=32,
                   kernel_size=2,
                   activation='relu',
                   input_shape=(input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(output))
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit_model(self, model, X_train, y_train, epochs):
        model.fit(X_train, y_train, epochs=epochs, verbose=0)

    def make_prediction(self, fitted_model, X_test):
        return fitted_model.predict(X_test, verbose=0)

    def save_predictions(self, predictions):
        ratios = pd.read_csv('raw_data/ratios.csv')
        ratios['Date'] = pd.to_datetime(ratios['Date'])

        df = pd.DataFrame()
        df['Preds'] = predictions[0]
        df['Date'] = 1
        for i in range(len(predictions[0])):
            df['Date'].iloc[i] = (ratios['Date'].iat[-1] +
                                  timedelta(days=i + 1)).strftime('%Y-%m-%d')
        df.to_csv(f'raw_data/CNN_preds.csv')


if __name__ == '__main__':
    cnn = CNN_model()
    df = cnn.get_data('raw_data/ratios.csv')
    ratios = cnn.seperate_ratios(df)
    X, y = cnn.split_sequence(ratios['ALB_ZBRA'], 200, 30)
    X_train, y_train, X_test, y_test = cnn.test_train_splits(X, y)
    X_train, X_test = cnn.reshape(X_train, X_test, 200, 1)
    model = cnn.create_model(200, 30)
    cnn.fit_model(model, X_train, y_train, 1000)
    predictions = cnn.make_prediction(model, X_test)
    cnn.save_predictions(predictions)
