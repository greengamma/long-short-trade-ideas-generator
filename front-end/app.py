import streamlit as st
import numpy as np
import pandas as pd
from demo_data import DemoData

st.set_page_config(layout="wide")


#Kaggle Data to be replaced with modelled data
@st.cache
def init():
    demo = DemoData()
    data = demo.getData()
    symbols, stock_dict = demo.makeDictionary(data)
    return symbols, stock_dict


symbols, stock_dict = init()

# Uses entire screen width


#defines local style sheet
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()),
                    unsafe_allow_html=True)


local_css("style.css")

#--------------------------
#Callback functions
#--------------------------
stockA = 'MMM'
stockB = 'ZTS'


def merge(stockA, stockB):
    merged_stocks = pd.merge(stock_dict[stockA], stock_dict[stockB], on='Date')
    return merged_stocks


merged = merge(stockA, stockB)

#----------------------------------------------------------------------------
#SIDEBAR
#----------------------------------------------------------------------------
st.title('HedgeFinder')

add_sidebar_title = st.sidebar.title('Filter Options')

add_selectbox = st.sidebar.selectbox("Industry", ("A", "B", "C"),
                                     key='option1')
stockA = st.sidebar.selectbox(
    "Stock A",
    (symbols),
    key='option2',
)
stockB = st.sidebar.selectbox(
    "Stock B",
    (symbols),
    key='option3',
)
add_sidebar_button = st.sidebar.button('Filter')

merged = merge(stockA, stockB)

#-----------------------------------------------------------------------------
#Tabs
#-----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    'Hedge1',
    'Hedge2',
    'Hedge3',
    'Hedge4',
    'Hedge5',
])

with tab1:
    st.header(f'{stockA} and {stockB}')
    st.line_chart(data=merged[['Close_x', 'Close_y']],
                  x=None,
                  y=None,
                  width=0,
                  height=0,
                  use_container_width=True)

with tab2:
    stock_name2 = 'ZION'
    st.header(stock_name2)
    st.line_chart(data=stock_dict[stock_name2]['Close'],
                  x=None,
                  y=None,
                  width=0,
                  height=0,
                  use_container_width=True)
with tab3:
    stock_name3 = 'GD'
    st.header(stock_name3)
    st.line_chart(data=stock_dict[stock_name3]['Close'],
                  x=None,
                  y=None,
                  width=0,
                  height=0,
                  use_container_width=True)
with tab4:
    stock_name4 = 'FANG'
    st.header(stock_name4)
    st.line_chart(data=stock_dict[stock_name4]['Close'],
                  x=None,
                  y=None,
                  width=0,
                  height=0,
                  use_container_width=True)
with tab5:
    stock_name5 = 'BXP'
    st.header(stock_name5)
    st.line_chart(data=stock_dict[stock_name5]['Close'],
                  x=None,
                  y=None,
                  width=0,
                  height=0,
                  use_container_width=True)
