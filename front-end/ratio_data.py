import pandas as pd
import numpy as np


class Ratio_data:

    def __init__(self):
        pass

    def getData(self):
        df = pd.read_excel('raw_data/cleaned_data.xlsx')
        # df['Date'] = pd.to_datetime(df['Date'])
        # df_last_year = df[df['Date'] < pd.to_datetime('2021-12-31')]
        # df_last_year = df_last_year[
        #     df_last_year['Date'] > pd.to_datetime('2020-12-31')]
        # df_last_year = df_last_year[['Date', 'Symbol', 'Close']]
        # return df_last_year
        return df

    def split_hedge_names(self, df):
        hedges = df.columns[1:]
        hedge_pairs = []
        for hedge in hedges:
            hedge_pairs.append(hedge.split('_'))
        return hedge_pairs


if __name__ == '__main__':
    ratios = Ratio_data()
    data = ratios.getData()
    hedge_pairs = ratios.split_hedge_names(data)
    print(hedge_pairs[0][1])
