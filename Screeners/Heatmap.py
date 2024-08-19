import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=7200)  # Cache for 1 hour
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    
    sp500_data = table[['Symbol', 'Security', 'GICS Sector']]
    tickers = sp500_data['Symbol'].tolist()
    market_caps = []
    
    for ticker in tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            market_cap = stock_info.get('marketCap', None)
            market_caps.append(market_cap)
        except Exception as e:
            market_caps.append(None)
            print(f"Erreur lors de la récupération des données pour {ticker}: {e}")
    
    sp500_data['Market Cap'] = market_caps
    return sp500_data.set_index('Symbol')

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def get_last_day_return(stock_prices):
    return stock_prices.pct_change().iloc[-1] * 100

def update_sp500_data():
    sp500_data = get_sp500_tickers()
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=7)
    tickers = sp500_data.index.tolist()
    stock_prices = get_stock_data(tickers, start_date, end_date)
    last_day_return = get_last_day_return(stock_prices)
    sp500_data['Last Day Return (%)'] = last_day_return
    sector_performance = sp500_data.groupby('GICS Sector')['Last Day Return (%)'].mean().reset_index()
    sector_performance.rename(columns={'Last Day Return (%)': 'Sector Average Return (%)'}, inplace=True)
    sp500_data = sp500_data.reset_index().merge(sector_performance, on='GICS Sector', how='left')
    return sp500_data

# Initialize session state
if 'sp500_data' not in st.session_state:
    st.session_state.sp500_data = update_sp500_data()
    st.session_state.last_update = datetime.now()

# Add a refresh button
if st.button('Refresh Data'):
    st.session_state.sp500_data = update_sp500_data()
    st.session_state.last_update = datetime.now()
    st.success('Data refreshed successfully!')

# Display last update time
st.write(f"Last updated: {st.session_state.last_update}")

# Use the data from session state
sp500_data = st.session_state.sp500_data

color_scale = [
    (0, "rgb(255, 0, 0)"),        # Bright red for lowest negative return
    (0.25, "rgb(180, 0, 0)"),     # Darker red
    (0.4, "rgb(90, 0, 0)"),       # Even darker red
    (0.49, "rgb(20, 0, 0)"),      # Almost black (for values just below 0)
    (0.5, "rgb(0, 0, 0)"),        # Black (for values very close to 0)
    (0.51, "rgb(0, 20, 0)"),      # Almost black (for values just above 0)
    (0.6, "rgb(0, 90, 0)"),       # Dark green
    (0.75, "rgb(0, 180, 0)"),     # Brighter green
    (1, "rgb(0, 255, 0)")         # Bright green for highest positive return
]
min_return = sp500_data['Last Day Return (%)'].min()
max_return = sp500_data['Last Day Return (%)'].max()

fig = px.treemap(
    sp500_data,
    path=['GICS Sector', 'Security'],
    values='Market Cap',
    color='Last Day Return (%)',
    color_continuous_scale=color_scale,
    color_continuous_midpoint=0,
    range_color=[min_return, max_return],
    hover_data={'Market Cap': ':,.2f', 'Last Day Return (%)': ':.2f'},
    title="S&P 500 Daily Performance",
    labels={'Last Day Return (%)': 'Last Day Return (%)'}
)

fig.update_layout(
    height=800,
    title_font_size=24,
    title_x=0.4,
    coloraxis_showscale=False 
) 
fig.data[0].texttemplate = "%{label}<br>%{customdata[1]:.2f}%"

st.plotly_chart(fig, use_container_width=True)