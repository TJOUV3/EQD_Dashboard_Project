#region import

import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
import numpy as np
import yfinance as yf
import QuantLib as ql
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime,timedelta
import scipy
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D

#endregion

#region yahoo finance

def get_stock_price_and_volatility(ticker, period='1y'):

    # Retrieve stock data from Yahoo Finance
    stock = yf.Ticker(ticker)
    
    # Get recent historical data
    data = stock.history(period=period)
    
    # Extract the latest closing price
    latest_price = data['Close'].iloc[-1]
    first_price = data['Close'].iloc[0]
    N = len(data['Close']) - 2
    
    # Compute daily log returns
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Drop NaN values that result from the shift operation
    data = data.dropna()
    
    # Calculate the standard deviation of log returns
    daily_volatility = data['Log Return'].std()
    var = data['Log Return'].var()
    
    # Annualize the vol (assuming 252 trading days in a year)
    annual_volatility = daily_volatility * np.sqrt(252)
    mu = data['Log Return'].mean()
    sigma = np.std(np.diff(data['Log Return'])) * np.sqrt(252)
    rho = np.corrcoef(data['Log Return'][:-1], np.diff(data['Log Return']))[0, 1]
    
    return first_price, var, mu, annual_volatility, N, data['Close'], latest_price, sigma, rho

def get_all_options(ticker, option_params):
    # Créer un objet Ticker avec yfinance
    stock = yf.Ticker(ticker)
    option_type = option_params[0]  # 'call' ou 'put'
    K = option_params[1]  # Strike price
    r = option_params[2]  # Taux sans risque
    t = option_params[3]  # Temps jusqu'à l'échéance (en années)

    # Obtenir les dates d'expiration disponibles pour les options
    expirations = stock.options
    all_options_data = []

    # Parcourir chaque date d'expiration
    for expiration_date in expirations:
        # Obtenir la chaîne d'options pour cette date d'expiration
        options_chain = stock.option_chain(expiration_date)

        # Sélectionner les options selon le type (call ou put)
        if option_type == 'call':
            options = options_chain.calls
        elif option_type == 'put':
            options = options_chain.puts
        else:
            return "Invalid option type. Choose 'call' or 'put'."

        # Ajouter une colonne 'expiration' pour la date d'échéance
        options['expiration'] = expiration_date

        # Ajouter les options à la liste
        all_options_data.append(options)

    # Combiner les DataFrames pour chaque date d'expiration en un seul DataFrame
    options_df = pd.concat(all_options_data, ignore_index=True)
    
    # Calculer les limites de strikes basées sur K
    lower_limit = K * 0.70  # 70% de K
    upper_limit = K * 1.30  # 130% de K

    # Filtrer les options pour ne garder que celles dont les strikes sont dans la plage de K ± 30%
    options_df = options_df[(options_df['strike'] >= lower_limit) & (options_df['strike'] <= upper_limit)]

    # Ne conserver que les colonnes nécessaires
    return options_df[['strike', 'expiration', 'lastPrice', 'bid', 'ask', 'impliedVolatility']]

#endregion

def create_volatility_table(options_df):
    # Convertir la colonne 'expiration' en datetime
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])

    # Créer une table pivot avec les strikes en lignes et les expirations en colonnes
    volatility_table = options_df.pivot_table(
        index='strike',
        columns='expiration',
        values='impliedVolatility',
        aggfunc='mean'  # Utiliser 'mean' au cas où il y aurait des doublons
    )

    # Interpolation linéaire sur les lignes (strikes)
    volatility_table = volatility_table.interpolate(method='linear', axis=0)  # Interpoler sur les lignes

    # Interpolation linéaire sur les colonnes (expirations)
    volatility_table = volatility_table.interpolate(method='linear', axis=1)  # Interpoler sur les colonnes

    return volatility_table

def plot_volatility_surface(volatility_table):
    # Convert the index (strike prices) to numeric type if it's not already
    volatility_table.index = pd.to_numeric(volatility_table.index)

    # Convert column names (expiration dates) to datetime
    volatility_table.columns = pd.to_datetime(volatility_table.columns)

    # Calculate days until expiration
    today = datetime.now()
    days_to_expiration = [(exp - today).days for exp in volatility_table.columns]

    # Create the surface plot
    surface = go.Surface(
        x=volatility_table.index,
        y=days_to_expiration,
        z=volatility_table.values.T,
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility')
    )

    # Create the layout
    layout = go.Layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility',
            xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True),
            yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True),
            zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True)
        ),
        width=900,
        height=700,
        margin=dict(r=10, l=10, b=10, t=40)
    )

    # Create the figure
    fig = go.Figure(data=[surface], layout=layout)

    # Display the plot in Streamlit
    st.plotly_chart(fig,use_container_width=True)



st.title("Option Volatility Surface Visualization")

# Add user inputs
ticker = st.text_input("Enter stock ticker", value="NVDA")
option_type = st.selectbox("Select option type", ['call', 'put'])
strike = st.number_input("Enter reference strike price", value=100.0, step=1.0)
risk_free_rate = st.number_input("Enter risk-free rate", value=0.05, step=0.01)
time_to_expiry = st.number_input("Enter time to expiry (years)", value=0.1, step=0.1)

if st.button("Generate Volatility Surface"):
    try:
        # Fetch options data
        options_data = get_all_options(ticker, [option_type, strike, risk_free_rate, time_to_expiry])
        
        volatility_table = create_volatility_table(options_data)

        with st.expander("Raw Volatility Data"):
            st.dataframe(volatility_table)

        st.subheader("Volatility Surface")
        plot_volatility_surface(volatility_table)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
