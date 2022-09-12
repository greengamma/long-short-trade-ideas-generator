import pandas as pd

# import ratios
df = pd.read_excel('raw_data/cleaned_data.xlsx')
# convert 'Date' column to datetime values
# df['Date'] = pd.to_datetime(df['Date'].str[:10])
# convert ratios to list items
ratios = list(df.columns)
# remove 'Date' column
ratios.pop(0)
# import all S&P500 tickers, their names & industry affiliation
tickers = pd.read_excel('raw_data/ticks.xlsx')
# create a list of tuples which is converted from the tickers variable
tuple_list = list(tickers.itertuples(index=False))

# retrieve long/short names from the tuple_list variable


def tickers_to_name(df):
    long_name = {}
    short_name = {}
    ratios = list(df.columns)
    ratios.pop(0)
    for ratio in ratios:
        long = ratio.split('_')[0]
        short = ratio.split('_')[1]
        for item in tuple_list:
            if long in item:
                long_name[long] = item[1]
        for item in tuple_list:
            if short in item:
                short_name[short] = item[1]
    return long_name, short_name
