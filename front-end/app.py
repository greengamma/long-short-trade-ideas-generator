import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from data.refactored_data import Data
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
# Uses entire screen width
st.set_page_config(layout="wide")


@st.cache
def init():
    #-------------------------------------------------------------
    # Data Retrieval
    #-------------------------------------------------------------
    data_retrieval = Data()

    ratios = pd.read_excel('raw_data/cleaned_data.xlsx').drop(['Unnamed: 0'],
                                                              axis=1)
    tickers = pd.read_csv('raw_data/tickers.csv')
    prices = pd.read_excel('raw_data/weekly_prices.xlsx')
    sma10 = pd.read_excel('raw_data/sma_10_days.xlsx').drop(['Unnamed: 0'],
                                                            axis=1)
    sma20 = pd.read_excel('raw_data/sma_20_days.xlsx').drop(['Unnamed: 0'],
                                                            axis=1)
    sma60 = pd.read_excel('raw_data/sma_60_days.xlsx').drop(['Unnamed: 0'],
                                                            axis=1)

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

    ## Stock Dictionary currently not upto date will need to update with current stock data.

    return ratios, tickers, prices, merged, merged_length, hedge_pairs, sma10, sma20, sma60


data, symbols, stock_dict, merged, merged_length, hedge_pairs, sma10, sma20, sma60 = init(
)


#defines local style sheet
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


local_css("style.css")

#--------------------------
#Callback functions
#--------------------------

#----------------------------------------------------------------------------
#SIDEBAR
#----------------------------------------------------------------------------
st.title('HedgeFinder')

add_sidebar_title = st.sidebar.title('Filter Options')

add_selectbox = st.sidebar.selectbox("Industry", ("A", "B", "C"),
                                     key='option1')

add_sidebar_button = st.sidebar.button('Filter')

#-----------------------------------------------------------------------------
#Tabs
#-----------------------------------------------------------------------------
hedges = []
for i in range(merged_length):
    hedges.append(f'Long: {hedge_pairs[i][0]} / Short: {hedge_pairs[i][1]} ')

tabs = st.tabs(hedges)
#runs through all stocks pairs and adds a tab for each with relevant graphs
for i, tab in enumerate(tabs):
    with tabs[i]:
        st.header(f'{hedge_pairs[i][0]} and {hedge_pairs[i][1]}')
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.header('Stock Price')
            # plot both stocks last 3 months data
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            stocks_data = merged[i].drop('Date', axis=1)
            ax1.plot(merged[i]['Date'], stocks_data)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            fig1.autofmt_xdate()
            ax1.grid(True)
            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter('%m/%d/%Y'))
            plt.gca().xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=(FR)))
            #display as graph option to discuss
            # st.pyplot(fig)

            #Converts graph into png for display due to constraints around figsize in stramlit
            buf = BytesIO()
            fig1.savefig(buf, format="png")
            st.image(buf)
        with col2:
            st.header('Ratio')
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.autofmt_xdate()
            ratio, = ax2.plot(data['Date'], data[data.columns[i + 1]])
            avg10, = ax2.plot(data['Date'], sma10[sma10.columns[i]])
            avg20, = ax2.plot(data['Date'], sma20[sma20.columns[i]])
            avg60, = ax2.plot(data['Date'], sma60[sma60.columns[i]])
            ax2.legend(handles=[ratio, avg10, avg20, avg60],
                       labels=[
                           'daily ratio', '10 day average', '20 day average',
                           '60 day average'
                       ],
                       loc='upper left',
                       fontsize=10)
            # ax2.set_ylim([0, 1])
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Ratio')
            ax2.grid(True)
            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter('%m/%d/%Y'))
            plt.gca().xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=(FR)))
            #display as graph option to discuss
            # st.pyplot(fig)

            #Converts graph into png for display due to constraints around figsize in streamlit
            buf = BytesIO()
            fig2.savefig(buf, format="png")
            st.image(buf)
