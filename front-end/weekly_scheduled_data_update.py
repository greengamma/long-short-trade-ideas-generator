from data.refactored_data import Data
from models.CNN import CNN_model
import pandas as pd
import datetime

data = Data()

# Update Data
# tickers = data.get_tickers()
# prices = data.get_prices(tickers)
# ratios = data.get_ratios(10)
# sma10 = data.create_SMA(10)
# sma20 = data.create_SMA(20)
# sma60 = data.create_SMA(60)

# update CNN model
CNN = CNN_model()
df = CNN.get_data('raw_data/cleaned_data.xlsx').dropna(axis=1)
## takes first 10 ratios
df_short = df.iloc[:, :6].set_index('Date')
seperate_ratios = CNN.seperate_ratios(df)
all_predictions_cnn = pd.DataFrame()
model = CNN.create_model(200, 30)
df['Date'] = pd.to_datetime(df['Date'])
start_date = (df['Date'].iloc[-1] +
              datetime.timedelta(days=1)).strftime("%Y-%m-%d")
end_date = (df['Date'].iloc[-1] +
            datetime.timedelta(days=30)).strftime("%Y-%m-%d")
all_predictions_cnn['Date'] = pd.date_range(start_date, end_date)
all_predictions_cnn.to_csv('raw_data/CNN_preds.csv', index=False)
for columns in df_short:
    X, y = CNN.split_sequence(seperate_ratios[columns], 200, 30)
    X_train, y_train, X_test, y_test = CNN.test_train_splits(X, y)
    X_train, X_test = CNN.reshape(X_train, X_test, 200, 1)
    CNN.fit_model(model, X_train, y_train, 1000)
    predictions = CNN.make_prediction(model, X_test)
    all_predictions_cnn[columns] = predictions[0]
all_predictions_cnn.to_csv('raw_data/CNN_preds.csv', index=False)
#Add Date Column

# lstm model
