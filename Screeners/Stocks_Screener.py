import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

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
def get_historical_data(ticker, period='1mo'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data
def create_stock_charts(selected_stocks):
    if not selected_stocks:
        return None

    num_charts = len(selected_stocks)
    fig = make_subplots(rows=num_charts, cols=1, subplot_titles=selected_stocks)

    for i, ticker in enumerate(selected_stocks, start=1):
        hist_data = get_historical_data(ticker)
        trace = go.Scatter(
            x=hist_data.index,
            y=hist_data['Close'],
            mode='lines',
            name=ticker,
            line=dict(color='#00BFFF', width=2)
        )
        fig.add_trace(trace, row=i, col=1)

        fig.update_yaxes(title_text="Price", row=i, col=1)
        
        current_price = hist_data['Close'].iloc[-1]

    fig.update_layout(height=300*num_charts)
    return fig
#endregion

st.title("S&P 500 Stocks Screener")
if st.button('Refresh Data'):
    st.session_state.last_update = datetime.min
    st.rerun()

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.min
def color_percent_change(val):
    color = 'green' if val.startswith('+') else 'red'
    return f'color: {color}'

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

# Calculate returns
returns = calculate_returns(st.session_state.stock_data)
volatility = calculate_volatility(st.session_state.stock_data)

# Initialize session state for selected stocks if it doesn't exist
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []

# Create placeholder for charts
chart_placeholder = st.empty()

# Create stock table
stock_table = create_stock_table(st.session_state.stock_data, returns, volatility)
# Add a checkbox column to the dataframe 
stock_table['Select'] = [ticker in st.session_state.selected_stocks for ticker in stock_table.index]

def color_percent_change(val):
    color = 'green' if val.startswith('+') else 'red'
    return f'color: {color}'

# Apply the styling to the DataFrame
styled_df = stock_table.style.applymap(color_percent_change, subset=['% Change'])

# Create an editable dataframe with conditional formatting
edited_df = st.data_editor(
    styled_df,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select to view chart",
            default=False,
        ),
        "Ticker": st.column_config.TextColumn(
            "Ticker",
            width="small",
        ),
        "Price": st.column_config.NumberColumn(
            "Price",
            format="$%.2f",
            width="small",
        ),
        "% Change": st.column_config.TextColumn(
            "% Change",
            width="small",
        ),
        "Volatility": st.column_config.TextColumn(
            "Volatility",
            width="small",
        ),
    },
    disabled=["Ticker", "Price", "% Change", "Volatility"],
    use_container_width=True,
    column_order=[ "Ticker", "Price", "% Change", "Volatility","Select",],
    hide_index=True,
    key="stock_table",
)

# Update selected stocks based on checkbox state
st.session_state.selected_stocks = edited_df[edited_df['Select']].index.tolist()

# Create and display charts for selected stocks
with chart_placeholder.container():
    if st.session_state.selected_stocks:
        with st.spinner('Loading charts...'):
            fig = create_stock_charts(st.session_state.selected_stocks)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select one or more stocks to view their charts.")
