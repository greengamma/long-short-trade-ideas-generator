import pandas as pd

# URL of the Wikipedia page with S&P 500 companies
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Read the HTML table into a DataFrame
tables = pd.read_html(url)
df = tables[0]

# Extract the 'Symbol' column which contains the tickers
tickers = df['Symbol']

# Save to CSV
tickers.to_csv('tickers_new.csv', index=False, header=True)
