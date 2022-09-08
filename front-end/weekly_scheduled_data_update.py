from data.refactored_data import Data

data = Data()

# Update streamlit vals
tickers = data.get_tickers()
prices = data.get_prices(tickers)
ratios = data.get_ratios(10)
sma10 = data.create_SMA(10)
sma20 = data.create_SMA(20)
sma60 = data.create_SMA(60)

# update models
