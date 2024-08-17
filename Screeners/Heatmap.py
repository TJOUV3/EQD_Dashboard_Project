import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from datetime import datetime, timedelta

#region Get Data
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    return table.set_index('Symbol')['Security']

def get_stock_data(tickers, period="1mo"):
    data = yf.download(tickers, period=period)
    return data['Adj Close']
def should_update_data(last_update, refresh_interval):
    return datetime.now() - last_update > refresh_interval

@st.cache_data(ttl=3600)  # Cache the result for 1 hour
def fetch_company_names():
    return get_sp500_tickers()

#endregion

#region Computation and plot
def calculate_returns(data):
    return data.pct_change().iloc[-1]  # Get the last day's return
def get_color(value, color_scale):
    return f"rgb{tuple(int(x*255) for x in color_scale(value)[:3])}"
def create_custom_grid(returns, company_names):
    n = len(returns)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    fig = go.Figure()

    # Create a custom colormap
    cmap = cm.get_cmap('RdYlGn')  # Red -> Yellow -> Green
    norm = Normalize(vmin=-0.015, vmax=0.015)  # Adjust these values to change the color scaling

    for i, (ticker, ret) in enumerate(returns.items()):
        row = i // cols
        col = i % cols
        
        color = get_color(norm(ret), cmap)
        
        fig.add_trace(go.Scatter(
            x=[col, col+1, col+1, col, col],
            y=[row, row, row+1, row+1, row],
            fill="toself",
            fillcolor=color,
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
            hoverinfo='text',
            text=f"{company_names[ticker]}<br>Return: {ret:.2%}"
        ))
        
        fig.add_annotation(
            x=col+0.5, y=row+0.5,
            text=f"{ticker}<br>{ret:.2%}",
            showarrow=False,
            font=dict(size=8, color='black')
        )

    fig.update_layout(
        title='S&P 500 Grid',
        height=800,
        width=1000,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
#endregion


st.title("S&P 500 Heatmap")
if st.button('Refresh Data'):
    st.session_state.last_update = datetime.min
    st.rerun()
# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = get_stock_data(get_sp500_tickers())
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.min

# Fetch company names (this is cached)
company_names = fetch_company_names()
# Plot only last 50 (trop lourd sinon)
tickers = company_names.index[:50].tolist()

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
# Create custom grid
fig = create_custom_grid(returns, company_names)
st.plotly_chart(fig, use_container_width=True)

