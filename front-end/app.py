from pickletools import read_stringnl_noescape
from dateutil.relativedelta import relativedelta, FR

import streamlit as st
import streamlit_authenticator as stauth
import yaml
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
from data.ticker_name_industry import tickers_to_name
from data.refactored_data import Data
from data.performance import create_pls
# from data.ticker_name_industry import ticker_names
# Uses entire screen width
st.set_page_config(layout="wide")

with st.spinner('Loading Data...'):

    @st.cache
    def init():
        #-------------------------------------------------------------
        # Data Retrieval
        #-------------------------------------------------------------
        data_retrieval = Data()
        model_name = 'prophet'
        ratios = pd.read_excel('raw_data/ratios.xlsx').dropna(
            axis=1).iloc[:, :11]
        tickers = pd.read_excel('raw_data/ticks.xlsx')
        prices = pd.read_excel('raw_data/weekly_prices.xlsx')
        sma10 = pd.read_excel('raw_data/sma_10_days.xlsx')
        sma20 = pd.read_excel('raw_data/sma_20_days.xlsx')
        sma60 = pd.read_excel('raw_data/sma_60_days.xlsx')

        #mapes
        mapes = pd.read_csv(f'raw_data/{model_name}_mapes.csv')

        prediction_actual = pd.read_csv(
            f'raw_data/{model_name}_actual_predictions.csv')
        prediction_mape = pd.read_csv(
            f'raw_data/{model_name}_mape_predictions.csv')
        prediction_actual['Date'] = pd.to_datetime(prediction_actual['Date'])
        prediction_mape['Date'] = pd.to_datetime(prediction_mape['Date'])
        ratios['Date'] = pd.to_datetime(ratios['Date'])
        ##reorder column names to match ratios
        column_names = ratios.columns
        prediction_actual = prediction_actual.reindex(columns=column_names)
        prediction_mape = prediction_mape.reindex(columns=column_names)

        #Function to merge stock data for plotting
        def merge(prices, stockA, stockB):
            merged_stocks = prices[['Date', stockA, stockB]]
            return merged_stocks

        ##Splits column names and returns list of this weeks hedge pairs
        hedge_pairs = data_retrieval.split_hedge_names(ratios)

        #merges stock data for each hedge into one for plotting
        merged = []
        for pairs in hedge_pairs:
            merged.append(merge(prices, pairs[0], pairs[1]))
        merged_length = len(merged)

        ## Makes Dictionary of tickers(keys) to names(vals)
        long_names, short_names = tickers_to_name(ratios)

        # Makes a dictionary of profit and loss for current ratios for last 1 and three months
        one_month_pl, three_month_pl = create_pls(10_000, ratios)

        return ratios, tickers, prices, merged, merged_length, hedge_pairs, sma10, sma20, sma60, prediction_actual, prediction_mape, long_names, short_names, one_month_pl, three_month_pl, mapes

    data, symbols, stock_dict, merged, merged_length, hedge_pairs, sma10, sma20, sma60, predictions_actual, prediction_mape, long_names, short_names, one_month_pl, three_month_pl, mapes = init(
    )


#########################################################################################
#defines local style sheet
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


local_css("front-end/style.css")

#--------------------------
#Auth Requirements
#--------------------------

with open('front-end/config.yaml') as file:
    config = yaml.load(file, Loader=stauth.SafeLoader)

authenticator = stauth.Authenticate(config['credentials'],
                                    config['cookie']['name'],
                                    config['cookie']['key'],
                                    config['cookie']['expiry_days'],
                                    config['preauthorized'])

#----------------------------------------------------------------------------
#SIDEBAR
#----------------------------------------------------------------------------

with st.sidebar:

    name, authentication_status, username = authenticator.login(
        'Login', 'main')

    # if authentication_status:
    #     try:
    #         if authenticator.reset_password(username, 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)
#-----------------------------------------------------------------------------
#Title
#-----------------------------------------------------------------------------
st.title('HedgeInvest')
#-----------------------------------------------------------------------------
#Tabs
#-----------------------------------------------------------------------------
if authentication_status:
    authenticator.logout('Logout', 'main')

    hedges = []
    for i in range(merged_length):
        hedges.append(
            f'Long: {hedge_pairs[i][0]} / Short: {hedge_pairs[i][1]} ')

    tabs = st.tabs(hedges)
    #runs through all stocks pairs and adds a tab for each with relevant graphs
    for i, tab in enumerate(tabs):
        with tabs[i]:
            header_col_1, header_col_2 = st.columns(2, gap='small')
            with header_col_1:
                st.header(
                    f'{long_names[hedge_pairs[i][0]]} ({hedge_pairs[i][0]})')
                st.subheader(
                    f'in {symbols.loc[symbols["ticker"] == hedge_pairs[i][0],"industry" ].iloc[0]}'
                )

            with header_col_2:
                st.header(
                    f'{short_names[hedge_pairs[i][1]]} ({hedge_pairs[i][1]})')
                st.subheader(
                    f'in {symbols.loc[symbols["ticker"] == hedge_pairs[i][1],"industry" ].iloc[0]}'
                )

            col1, col2 = st.columns(2, gap="small")
            with col1:
                st.header('Stock Prices')
                # plot both stocks last 3 months data
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                stocks_data = merged[i].drop('Date', axis=1)
                stock1, = ax1.plot(merged[i]['Date'],
                                   stocks_data[stocks_data.columns[0]],
                                   color='green')

                ##Axis info for date config
                fig1.autofmt_xdate()
                ax1.grid(True)
                plt.gca().xaxis.set_major_formatter(
                    mdates.DateFormatter('%d/%m/%Y'))
                plt.gca().xaxis.set_major_locator(
                    mdates.WeekdayLocator(byweekday=(FR)))
                ax1.set_xlim([data['Date'].iat[-60], data['Date'].iat[-1]])
                ##twin axes to plot different y axis due to large differences in stock prices
                ax1b = ax1.twinx()
                stock2, = ax1b.plot(merged[i]['Date'],
                                    stocks_data[stocks_data.columns[1]],
                                    color='red')

                ##Axis info
                ax1.set_xlabel('Date')
                ax1.set_ylabel(f'Price for {hedge_pairs[i][0]}')
                ax1.set_ylabel(f'Price for {hedge_pairs[i][0]}',
                               color="black",
                               fontsize=10)
                ax1b.set_ylabel(f'Price for {hedge_pairs[i][1]}',
                                color="purple",
                                fontsize=10)
                ##Legend Info

                ax1.legend(
                    handles=[stock1, stock2],
                    labels=[f'{hedge_pairs[i][0]}', f'{hedge_pairs[i][1]}'],
                    loc='upper left',
                    fontsize=10)
                #display as graph option to discuss
                # st.pyplot(fig)

                #Converts graph into png for display due to constraints around figsize in stramlit
                buf = BytesIO()
                fig1.savefig(buf, format="png")
                st.image(buf)
            with col2:
                st.header('Current Ratio and prediction')
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                fig2.autofmt_xdate()
                ratio, = ax2.plot(data['Date'], data[data.columns[i + 1]])
                avg10, = ax2.plot(data['Date'], sma10[sma10.columns[i]])
                avg20, = ax2.plot(data['Date'], sma20[sma20.columns[i]])
                avg60, = ax2.plot(data['Date'], sma60[sma60.columns[i]])
                preds, = ax2.plot(
                    pd.to_datetime(predictions_actual['Date']),
                    predictions_actual[predictions_actual.columns[i + 1]])
                ax2.legend(handles=[ratio, avg10, avg20, avg60, preds],
                           labels=[
                               'Ratio', '10 day rolling average',
                               '20 day rolling average',
                               '60 day rolling average', 'Prediction'
                           ],
                           loc='upper left',
                           fontsize=10)
                ##Sets X Limit to last 60 days up to including predictions
                ax2.set_xlim([
                    data['Date'].iat[-60],
                    pd.to_datetime(predictions_actual['Date'].iat[29])
                ])
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Ratio')
                ax2.grid(True)
                plt.gca().xaxis.set_major_formatter(
                    mdates.DateFormatter('%d/%m/%Y'))
                plt.gca().xaxis.set_major_locator(
                    mdates.WeekdayLocator(byweekday=(FR)))
                #display as graph option to discuss
                # st.pyplot(fig2)

                # Converts graph into png for display due to constraints around figsize in streamlit
                buf = BytesIO()
                fig2.savefig(buf, format="png")
                st.image(buf)

            ##########################################################################
            # Section2 Predictions from last 30 days
            ##########################################################################
            st.markdown(
                """<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)
            st.header('Model prediction on last 30 days')
            col_1_past, col_2_past = st.columns(2, gap="small")
            with col_2_past:

                st.text(
                    f'Model Accuracy for last 30 days: ={round(mapes["MAPE"][i],2)}'
                )
            with col_1_past:
                st.header(
                    'Current Ratio and model prediction for last 30 days')
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                fig2.autofmt_xdate()
                ratio, = ax2.plot(data['Date'], data[data.columns[i + 1]])
                # avg10, = ax2.plot(data['Date'], sma10[sma10.columns[i]])
                # avg20, = ax2.plot(data['Date'], sma20[sma20.columns[i]])
                # avg60, = ax2.plot(data['Date'], sma60[sma60.columns[i]])
                preds, = ax2.plot(
                    pd.to_datetime(prediction_mape['Date']),
                    prediction_mape[prediction_mape.columns[i + 1]])
                ax2.legend(
                    handles=[
                        ratio,
                        # avg10,
                        # avg20,
                        # avg60,
                        preds
                    ],
                    labels=[
                        'Ratio',
                        # '10 day average',
                        # '20 day average',
                        # '60 day average',
                        'Prediction'
                    ],
                    loc='upper left',
                    fontsize=10)
                ##Sets X Limit to last 60 days up to including predictions
                ax2.set_xlim([
                    data['Date'].iat[-60],
                    pd.to_datetime(prediction_mape['Date'].iat[29])
                ])
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Ratio')
                ax2.grid(True)
                plt.gca().xaxis.set_major_formatter(
                    mdates.DateFormatter('%d/%m/%Y'))
                plt.gca().xaxis.set_major_locator(
                    mdates.WeekdayLocator(byweekday=(FR)))
                #display as graph option to discuss
                # st.pyplot(fig2)

                # Converts graph into png for display due to constraints around figsize in streamlit
                buf = BytesIO()
                fig2.savefig(buf, format="png")
                st.image(buf)
            ##########################################################################
            #Section 3 ROI on this weeks hedges for last 6 weeks ($10k investment)
            ##########################################################################

            st.markdown(
                """<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)
            st.header('Expected ROI in next 30 days ($10k investment) ')
            total = round(
                10000 *
                (predictions_actual[predictions_actual.columns[i + 1]].loc[29]
                 /
                 predictions_actual[predictions_actual.columns[i + 1]].loc[0]),
                2)
            st.text(
                f'Model predicts a $10,000 investment wil be worth ${total} ')

            if total > 10_000:
                reccomendation = f'Short :{hedge_pairs[i][1]} and Long {hedge_pairs[i][0]}'
                explanation = f'The model predicts a continuation of the current trend '
            else:
                reccomendation = f'Short:{hedge_pairs[i][0]} and Long {hedge_pairs[i][1]}'
                explanation = 'The model predicts this ratio will start to fall but profit \ncan still be achieved by reversing the hedge'
            st.text('Reccomendation:')
            st.text(reccomendation)
            st.text(explanation)
            col3, col4 = st.columns(2, gap="small")

            with col3:
                pass

            with col4:
                pass
##########################################################################
#Section 3 previous weeks suggestions current ROI
##########################################################################

    lastFriday = datetime.datetime.now() + relativedelta(weekday=FR(-1))
    two_weeks = lastFriday - datetime.timedelta(days=7)
    three_weeks = two_weeks - datetime.timedelta(days=7)
    four_weeks = three_weeks - datetime.timedelta(days=7)
    date = f'{lastFriday.day}-{lastFriday.month}-{lastFriday.year}'
    date_two_weeks = f'{two_weeks.day}-{two_weeks.month}-{two_weeks.year}'
    date_three_weeks = f'{three_weeks.day}-{three_weeks.month}-{three_weeks.year}'
    date_four_weeks = f'{four_weeks.day}-{four_weeks.month}-{four_weeks.year}'

    st.markdown(
        """<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)

    st.header(f'Hedges for {date} current ROI ')

    st.header(f'Hedges for {date_two_weeks} current ROI ')

    st.header(f'Hedges for {date_three_weeks} current ROI ')

    st.header(f'Hedges for {date_four_weeks} current ROI ')

elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
