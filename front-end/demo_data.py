import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta


class DemoData:

    def __init__(self):
        pass

    def getData(self):
        date1 = date.today()
        date_3 = date1 - relativedelta(months=3)
        df = pd.read_csv('raw_data/sp500_stocks.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df_last_year = df[df['Date'] < pd.to_datetime(date1)]
        df_last_year = df_last_year[
            df_last_year['Date'] > pd.to_datetime(date_3)]
        df_last_year = df_last_year[['Date', 'Symbol', 'Close']]
        return df_last_year

    # make dictionary of stock symbols and data tables for each.
    def makeDictionary(self, df_last_year):
        symbols = df_last_year['Symbol'].unique()
        variable_names = []
        for symbol in symbols:
            variable_names.append(f'{symbol}_df')

        for i, symbol in enumerate(symbols):
            variable_names[i] = df_last_year[df_last_year['Symbol'] == symbol]
        stock_dict = {}
        for i, symbol in enumerate(symbols):
            stock_dict[symbol] = variable_names[i]
        return symbols, stock_dict

    def merge(self, stockA, stockB):
        merged_stocks = pd.merge(stock_dict[stockA],
                                 stock_dict[stockB],
                                 on='Date')
        return merged_stocks


if __name__ == '__main__':
    demo = DemoData()
    data = demo.getData()
    symbols, stock_dict = demo.makeDictionary(data)
    merged = demo.merge('OKE', 'LYB')
    print(merged)
