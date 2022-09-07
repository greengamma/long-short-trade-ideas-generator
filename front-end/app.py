import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from demo_data import DemoData
from ratio_data import Ratio_data

# Uses entire screen width
st.set_page_config(layout="wide")


#Kaggle Data to be replaced with modelled data
@st.cache
def init():
    ratios = Ratio_data()
    sp500 = DemoData()

    #Function to merge stock data for plotting
    def merge(stockA, stockB):
        merged_stocks = pd.merge(stock_dict[stockA],
                                 stock_dict[stockB],
                                 on='Date')
        return merged_stocks

    sp500_data = sp500.getData()
    symbols, stock_dict = sp500.makeDictionary(sp500_data)
    data = ratios.getData()

    ##Splits column names and returns list of this weeks hedge pairs
    hedge_pairs = ratios.split_hedge_names(data)

    #merges stock data for each hedge into one for plotting
    merged = []
    for pairs in hedge_pairs:
        merged.append(merge(pairs[0], pairs[1]))
    merged_length = len(merged)

    ## Stock Dictionary currently not upto date will need to update with current stock data.

    return data, symbols, stock_dict, merged, merged_length, hedge_pairs


data, symbols, stock_dict, merged, merged_length, hedge_pairs = init()


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
            ax1.plot(merged[i]['Date'], merged[i][['Close_x', 'Close_y']])
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            #display as graph option to discuss
            # st.pyplot(fig)

            #Converts graph into png for display due to constraints around figsize in stramlit
            buf = BytesIO()
            fig1.savefig(buf, format="png")
            st.image(buf)
        with col2:
            st.header('Ratio')
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(data['Date'], data[data.columns[i + 1]])
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Ratio')
            #display as graph option to discuss
            # st.pyplot(fig)

            #Converts graph into png for display due to constraints around figsize in streamlit
            buf = BytesIO()
            fig2.savefig(buf, format="png")
            st.image(buf)