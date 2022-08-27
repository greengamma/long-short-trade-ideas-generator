# 0. Imports
import pandas as pd
import yfinance as yf
import numpy as np
import xlrd
import time
import random
import numpy as np
import sys
import os

# 1. Data download
## read ticker data as Series and convert to list
tickers = pd.read_csv('raw_data/tickers.csv', squeeze=True)
tickers = tickers.tolist()

price_list = []

for ticker in tickers:
    stock = yf.Ticker(ticker).history(period="3mo").reset_index()
    stock = stock[['Date', 'Close']]
    stock.rename(columns={'Close': ticker}, inplace=True)
    price_list.append(stock)

# 2. Compute ratios
## concatenate all prices into one big dataframe
df = pd.concat(price_list, axis=1)
df = df.loc[:, ~df.columns.duplicated()].copy()

df.set_index('Date', inplace=True)
ratio_df = pd.concat([df[df.columns.difference([col])].div(df.where(df !=0, np.nan)[col], axis=0)\.add_suffix("_" + col) for col in df.columns], axis=1)

# 3. Compute daily price change in %
## compute the % change of the ratio between today and yesterday for the last X days
df_delta = ratio_df.pct_change(periods=1)
df_delta = df_delta.iloc[-4:]

## only select ratios with consistent positive price changes
df_newstocks = df_delta[df_delta.iloc[:] >= 0]
positiveRatiosPercent = df_newstocks.drop(df_newstocks.columns[
    df_newstocks.apply(lambda col: col.isnull().sum() > 0)],
                                          axis=1)
labelList = list(positiveRatiosPercent.columns.values)
df_positiveRatios = ratio_df[labelList]

# 4. Friday closing
df_positiveRatios.reset_index(inplace=True)

## assign weekdays to dataframe
df1 = df_positiveRatios.copy()
df1['Date'] = pd.to_datetime(df1['Date']).copy()
day_of_week_df = df1.copy()
day_of_week_df['day_of_week'] = day_of_week_df['Date'].dt.day_name()
## only keep the Fridays
friday_df = day_of_week_df.copy()
mask = day_of_week_df['day_of_week'] == "Friday"
friday_df = friday_df[mask]
## drop the other weekdays
friday_df.drop("day_of_week", axis=1, inplace=True)

## return only those ratios which had an increasing trend for n weeks, based on Fridays' closing prices

while 1:
    try:
        week_count = int(eval(input('For last n weeks: ')))
        break
    except:
        print('Please enter an integer')

new_df = friday_df.iloc[-week_count:, :]

column_index = 0
increasing_trend_df = []

increasing_trend_list = []

while 1:
    try:

        ## parsing by column
        temp_series = new_df.iloc[:, column_index]
        temp_series_list = temp_series.values.tolist()
        temp_series_list_sorted = sorted(temp_series_list,
                                         key=float,
                                         reverse=False)

        if temp_series_list == temp_series_list_sorted:
            increasing_trend_list.append(True)
        else:
            increasing_trend_list.append(False)
        column_index += 1

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        break

increasing_trend_df = new_df.iloc[:, increasing_trend_list]

## filters the dataframe accoring to last n weeks with a positive trend:
## increasing_trend_df: contains WEEKLY closing prices for 'n' Fridays
## complete_df: contains DAILY closing prices for the last 3 months
df_positiveRatios.reset_index(inplace=True)
complete_df = df_positiveRatios[increasing_trend_df.columns]

complete_df.to_excel('cleaned_data.xlsx')

len(complete_df.columns)
