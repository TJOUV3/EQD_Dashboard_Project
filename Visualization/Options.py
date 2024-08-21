import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)[0]
    return table.set_index('Symbol')['Security']
def get_option_data(ticker):
    api_key = 'VQRJ3T8STJ972AZW'
    url = f'https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data['data']
# Function to create Implied Volatility Smile plot
def plot_iv_smile(df):
    fig = go.Figure()
    for option_type in ['call', 'put']:
        df_filtered = df[df['type'] == option_type]
        fig.add_trace(go.Scatter(x=df_filtered['strike'], y=df_filtered['implied_volatility'],
                                 mode='markers+lines', name=option_type.capitalize()))
    fig.update_layout(title='Implied Volatility Smile',
                      xaxis_title='Strike Price',
                      yaxis_title='Implied Volatility')
    return fig

def plot_option_chain_volume(df):
    fig = px.bar(df, x='strike', y='volume', color='type', barmode='group',
                 title='Option Chain Volume')
    fig.update_xaxes(title='Strike Price')
    fig.update_yaxes(title='Volume')
    return fig

def plot_open_interest(df):
    fig = px.line(df, x='strike', y='open_interest', color='type',
                  title='Open Interest vs Strike Price')
    fig.update_xaxes(title='Strike Price')
    fig.update_yaxes(title='Open Interest')
    return fig

def plot_greeks(df):
    fig = go.Figure()
    greeks = ['delta', 'gamma', 'theta', 'vega']
    for greek in greeks:
        fig.add_trace(go.Scatter(x=df['strike'], y=df[greek], mode='lines', name=greek.capitalize()))
    fig.update_layout(title='Option Greeks vs Strike Price',
                      xaxis_title='Strike Price',
                      yaxis_title='Greek Value')
    return fig

def plot_bid_ask_spread(df):
    df['spread'] = df['ask'].apply(lambda x: float(x)) - df['bid'].apply(lambda x: float(x))
    fig = px.bar(df, x='strike', y='spread', color='type',
                 title='Bid-Ask Spread by Strike Price')
    fig.update_xaxes(title='Strike Price')
    fig.update_yaxes(title='Spread')
    return fig

def plot_put_call_ratio(df):
    put_call_ratio = df.groupby('strike').apply(lambda x: x[x['type'] == 'put']['volume'].sum() / 
                                                          x[x['type'] == 'call']['volume'].sum())
    fig = px.bar(x=put_call_ratio.index, y=put_call_ratio.values,
                 title='Put-Call Ratio by Strike Price')
    fig.update_xaxes(title='Strike Price')
    fig.update_yaxes(title='Put-Call Ratio')
    return fig

tickers = get_sp500_tickers().index.tolist()
option = st.selectbox('Chose a stock',tickers)
df = pd.DataFrame(get_option_data(option))

# Add a feature to filter by option type
st.subheader('Filter by Option Type')
option_type = st.radio('Select option type:', ('All', 'Call', 'Put'))
if option_type != 'All':
    filtered_df = df[df['type'] == option_type.lower()]
    st.dataframe(filtered_df)
else:
    filtered_df = df
    st.dataframe(filtered_df)

st.subheader('Implied Volatility Smile')
st.plotly_chart(plot_iv_smile(filtered_df))

st.subheader('Option Chain Volume')
st.plotly_chart(plot_option_chain_volume(filtered_df))

st.subheader('Open Interest vs Strike Price')
st.plotly_chart(plot_open_interest(filtered_df))

st.subheader('Option Greeks Comparison')
st.plotly_chart(plot_greeks(filtered_df))

st.subheader('Bid-Ask Spread')
st.plotly_chart(plot_bid_ask_spread(filtered_df))

st.subheader('Put-Call Ratio')
st.plotly_chart(plot_put_call_ratio(filtered_df))