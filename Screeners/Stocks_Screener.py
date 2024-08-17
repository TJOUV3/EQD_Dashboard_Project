import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

#region Get Data yfinance
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    return table.set_index('Symbol')['Security']

def get_stock_data(tickers, period="1mo"):
    data = yf.download(tickers, period=period)
    return data['Adj Close']

@st.cache_data(ttl=3600)  # Cache the result for 1 hour
def fetch_company_names():
    return get_sp500_tickers()


def should_update_data(last_update, refresh_interval):
    return datetime.now() - last_update > refresh_interval

#endregion

#region Compute returns and vol
def calculate_returns(data):
    return data.pct_change().iloc[-1]  # Get the last day's return
def calculate_volatility(data):
    return data.pct_change().std() * np.sqrt(252)

def create_stock_table(data, returns, volatility):
    table = pd.DataFrame({
        'Ticker': data.columns,
        'Price': data.iloc[-1],
        '% Change': returns * 100,
        'Volatility': volatility * 100
    })
    table = table.round(2)
    table.dropna(inplace=True)
    table['Price'] = table['Price'].map('${:,.2f}'.format)
    table['% Change'] = table['% Change'].map('{:+.2f}%'.format)
    table['Volatility'] = table['Volatility'].map('{:.2f}%'.format)
    return table.set_index('Ticker')

#endregion

st.title("S&P 500 Stocks Screener")
if st.button('Refresh Data'):
    st.session_state.last_update = datetime.min
    st.rerun()

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.min

# Fetch company names (this is cached)
company_names = fetch_company_names()
tickers = company_names.index.tolist()

# Check if we need to update the data
refresh_interval = timedelta(minutes=60)  # Adjust this as needed
if should_update_data(st.session_state.last_update, refresh_interval):
    with st.spinner('Fetching latest stock data...'):
        st.session_state.stock_data = get_stock_data(tickers)
        st.session_state.last_update = datetime.now()
    st.success('Data updated successfully!')
else:
    st.info(f'Using cached data. Last update: {st.session_state.last_update}')

# Calculate returns
returns = calculate_returns(st.session_state.stock_data)
volatility = calculate_volatility(st.session_state.stock_data)

st.subheader("S&P 500 Stock Details")
stock_table = create_stock_table(st.session_state.stock_data, returns, volatility)
st.dataframe(stock_table, use_container_width=True)

