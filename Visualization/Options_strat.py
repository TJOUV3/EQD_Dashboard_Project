#region import

import pandas as pd
import matplotlib.pyplot as plt
#import openai
import os
#import huggingface_hub
import io
import base64
#from huggingface_hub import InferenceClient
import numpy as np
import yfinance as yf
import QuantLib as ql
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime,timedelta

#endregion

#region functions payoffs, option_description, derivated_products
def payoff(call_put, St,K,Prenium,type,Maturity, buy_sell, nb_of_options):
    if call_put == 'call':
        if buy_sell == 'buy':
            return nb_of_options*(max(0, St-K) - Prenium)
        else:
            return nb_of_options*(-max(0, St-K) + Prenium)
    elif call_put == 'put':
        if buy_sell == 'buy':
            return nb_of_options*(max(K-St, 0) - Prenium)
        else:
            return nb_of_options*(-max(K-St, 0) + Prenium)
    elif call_put == 'digital call':
        if buy_sell == 'buy':
            if St>K:
                return nb_of_options*(100 - Prenium)
            else:
                return nb_of_options*(0 - Prenium)
        else:
            if St>K:
                return nb_of_options*(-100 + Prenium)
            else:
                return nb_of_options*(0 + Prenium)
    else:
        return 0

def option_description(call_put, Strike, type_option, Maturity, buy_sell, nb_of_options):
    if buy_sell == 'buy':
        buy_sell = 'Long'
    else:
        buy_sell = 'Short'
    return str(buy_sell) + ' ' + str(nb_of_options) + ' ' + str(type_option) + ' ' + str(call_put) + ' ' +  str(Maturity) + 'Y' +  ', Strike = ' + str(Strike)

def derivated_products(St, list_options):
    resulting_payoff = 0
    test = False
    if list_options:
        for sublist in list_options:
            if sublist:
                resulting_payoff = resulting_payoff + payoff(sublist[0], St, sublist[1], sublist[2], sublist[3], sublist[4], sublist[5], sublist[6])

    return resulting_payoff

#endregion

#region Stock Prce & Option Price

def get_stock_price_and_volatility(ticker, period='1y'):

    # Retrieve stock data from Yahoo Finance
    stock = yf.Ticker(ticker)
    
    # Get recent historical data
    data = stock.history(period=period)
    
    # Extract the latest closing price
    latest_price = data['Close'].iloc[-1]
    
    # Compute daily log returns
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Drop NaN values that result from the shift operation
    data = data.dropna()
    
    # Calculate the standard deviation of log returns
    daily_volatility = data['Log Return'].std()
    
    # Annualize the volatility (assuming 252 trading days in a year)
    annual_volatility = daily_volatility * np.sqrt(252)
    
    return latest_price, annual_volatility


def get_historical_data(ticker, period='1mo'):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=360)
    data = stock.history(start=start_date, end=end_date)
    return data

def get_all_options(ticker, option_params):
    stock = yf.Ticker(ticker)
    option_type = option_params[0]  # 'call' ou 'put'
    K = option_params[1]  # Strike price
    r = option_params[2]  # Taux sans risque
    t = option_params[3]  # Temps jusqu'à l'échéance (en années)
    expirations = stock.options
    all_options_data = []

    for expiration_date in expirations:
        options_chain = stock.option_chain(expiration_date)
        
        if option_type == 'call':
            options = options_chain.calls
        elif option_type == 'put':
            options = options_chain.puts
        else:
            return "Invalid option type. Choose 'call' or 'put'."
        options['expiration'] = expiration_date
        all_options_data.append(options)

    options_df = pd.concat(all_options_data, ignore_index=True)
    
    lower_limit = K * 0.85  # 85% de K
    upper_limit = K * 1.15  # 115% de K

    options_df = options_df[(options_df['strike'] >= lower_limit) & (options_df['strike'] <= upper_limit)]

    unique_strikes = options_df['strike'].unique()
    unique_times = options_df['expiration'].unique()

    days = int(t * 365)
    today = datetime.now()
    expiration_date = today + timedelta(days=days)

    if K not in unique_strikes:
        new_row_K = options_df.iloc[0].copy()
        new_row_K['strike'] = K
        new_row_K['expiration'] = expiration_date.strftime('%Y-%m-%d')
        new_row_K['impliedVolatility'] = np.NaN 
        options_df = options_df.append(new_row_K, ignore_index=True)

    if t not in unique_times:
        new_row_t = options_df.iloc[0].copy()
        new_row_t['strike'] = K
        new_row_t['expiration'] = expiration_date.strftime('%Y-%m-%d')
        new_row_t['impliedVolatility'] = 0
        options_df = options_df.append(new_row_t, ignore_index=True)

    options_df = options_df.sort_values(by=['strike', 'expiration'], ascending=[True, True]).reset_index(drop=True)

    return options_df[['strike', 'expiration', 'lastPrice', 'bid', 'ask', 'impliedVolatility']]

def create_volatility_table(options_df):
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    volatility_table = options_df.pivot_table(
        index='strike',
        columns='expiration',
        values='impliedVolatility',
        #aggfunc='mean'  # Utiliser 'mean' au cas où il y aurait des doublons
    )
    volatility_table.replace(0, np.nan, inplace=True)

    volatility_table = volatility_table.interpolate(method='quadratic', axis=0)
    volatility_table = volatility_table.interpolate(method='quadratic', axis=1)
    

    return volatility_table

def plot_volatility_surface(volatility_table, strike, expiration_date):
    volatility_table.index = pd.to_numeric(volatility_table.index)
    volatility_table.columns = pd.to_datetime(volatility_table.columns)
    today = datetime.now()
    days_to_expiration = [(exp - today).days for exp in volatility_table.columns]

    days = int(expiration_date * 365)
    today = datetime.now()
    expiration_date = today + timedelta(days=days)
    expiration_date = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')

    # Retrieve the volatility value at the user-specified point if it exists in the table
    try:
        implied_vol = volatility_table.loc[strike, expiration_date]
        user_point = True
    except KeyError:
        implied_vol = None
        user_point = False

    # Create the surface plot
    surface = go.Surface(
        x=volatility_table.index,
        y=days_to_expiration,
        z=volatility_table.values.T,
        colorscale='Viridis',
        colorbar=dict(title='Implied Volatility')
    )

    # Create a trace for the user-specified point in red, if it's within the surface data
    if user_point:
        #st.success("We found the point")
        point_trace = go.Scatter3d(
            x=[strike],
            y=[days],
            z=[implied_vol],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='User Point'
        )
        data = [surface, point_trace]
    else:
        st.warning("The specified strike or expiration date is not within the range of the volatility table.")
        data = [surface]

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
    fig = go.Figure(data= data, layout=layout)

    # Display the plot in Streamlit
    st.title("Option Volatility Surface Visualization")
    st.plotly_chart(fig,use_container_width=True)


def create_stock_chart(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Closing Price',
        line=dict(color='#00BFFF', width=2)
    ))

    fig.update_layout(
        title=f'{ticker} Closing Price - Last Month',
        yaxis_title='Price',
        xaxis_title='Date',
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.2)'
        )
    )

    return fig

def calculate_option_price(option_params, St):
    option_type = option_params[0] 
    K = option_params[1]  # Strike price
    r = option_params[2]  # Risk-free rate
    t = option_params[3]  # Time to maturity (in years)
    vol = option_params[4]  # Volatility
    buy_sell = option_params[5]  # Buy or sell
    nb_options = option_params[6]   # Nb of options wanted

    # Set up the dates
    calendar = ql.NullCalendar()
    today = ql.Date.todaysDate()
    maturity_date = today + int(t * 365)
    
    # Set up the option payoff and exercise
    if option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    elif option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type == 'digital call':
        payoff = ql.CashOrNothingPayoff(ql.Option.Call, K, 100)
    else:
        raise ValueError("Unrecognized option type. Choose 'call' or 'put'.")

    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)
    
    # Set up the market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(St))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    flat_vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, vol, ql.Actual365Fixed()))
    
    # Set up the Black-Scholes-Merton process
    bsm_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)
    
    # Calculate the option price
    option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
    price = option.NPV()

    return price

#endregion

#region Simulate functions
    
def Simulate_data_call_put(limit_inf, limit_sup, call_put, strike, prenium, type_option,maturity,buy_sell, nb_of_options):
    result = []
    descr = option_description(call_put, strike, type_option, maturity, buy_sell, nb_of_options)
    for price in range(limit_inf,limit_sup):
        result.append(payoff(call_put,price,strike,prenium,type_option,maturity,buy_sell, nb_of_options))
    return result, descr

def Simulate_data_dervative(list_options):
    result = [sum(elements) for elements in zip(*list_options)]
    return result

#endregion

#region Plot

def plot_function(def_range, res, list_of_options, descr_options, legend, color_line):
    # Convert the range to a list to be compatible with Plotly
    def_range = list(def_range)

    # Create a Plotly figure
    fig = go.Figure()
    
    # Add a horizontal line at y=0 (zero line) for reference
    fig.add_trace(go.Scatter(x=def_range, y=[0]*len(def_range), mode='lines', 
                             line=dict(color='rgba(255, 255, 255, 0.2)'), 
                             name='Zero Line'))

    # Plot the derivative product as a solid red line
    fig.add_trace(go.Scatter(x=def_range, y=res, mode='lines', 
                             line=dict(color='red', width=3), 
                             name="Derivative Product"))

    # Plot additional option derivatives as dashed lines with reduced opacity
    for i in range(len(list_of_options)):
        # Convert descr_options[i] to a string if it's not already a string
        option_name = str(descr_options[i]) if isinstance(descr_options[i], list) else descr_options[i]
        
        fig.add_trace(go.Scatter(x=def_range, y=list_of_options[i], mode='lines', 
                                 line=dict(color=color_line[i],dash='dash', width=2), 
                                 opacity=0.5, 
                                 name=option_name))

    # Set the title, axis labels, and legend title
    fig.update_layout(title=legend[0],
                        xaxis_title=legend[1],
                        yaxis_title=legend[2],
                        legend_title="Legend",
                        paper_bgcolor='rgba(255, 255, 255, 0)',
                        autosize=True,  # Enable autosize
                        plot_bgcolor='rgba(255, 255, 255, 0.1)')

    # Add gridlines to the x and y axes with specific styling
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.25)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,0,0,0.25)')

    return fig

#endregion

#region Greeks

def calculate_greek(option_params, greek, lim_inf, lim_sup, actual_underlying_price):
    # Initialize a list to store the Greek values
    res_greek = []
    descriptions = []

    # Extract option parameters
    option_type, K, r, t, vol, buy_sell, nb_options = option_params

    # Set up the QuantLib environment
    calendar = ql.NullCalendar()
    settlement_date = ql.Date.todaysDate()
    maturity_date = settlement_date + int(t * 365)  # Time to maturity in days
    day_count = ql.Actual365Fixed()

    # Create the option payoff and exercise
    if option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    elif option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
    elif option_type == 'digital call':
        payoff = ql.CashOrNothingPayoff(ql.Option.Call, K, 100)

    exercise = ql.EuropeanExercise(maturity_date)

    # Generate the option description
    description = f"{option_type.capitalize()} option, Strike: {K}, Maturity: {t} years, {buy_sell.capitalize()}"
    descriptions.append(description)

    # Initialize a variable to store the Greek value at the actual underlying price
    actual_greek_value = None

    # Set up the market data for the actual underlying price
    print(f"actual_underlying_price: {actual_underlying_price}")
    spot_handle_actual = ql.QuoteHandle(ql.SimpleQuote(actual_underlying_price))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, ql.QuoteHandle(ql.SimpleQuote(r)), day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, ql.QuoteHandle(ql.SimpleQuote(vol)), day_count))
    process_actual = ql.BlackScholesProcess(spot_handle_actual, flat_ts, vol_ts)

    # Create the option for the actual underlying price
    european_option_actual = ql.VanillaOption(payoff, exercise)
    european_option_actual.setPricingEngine(ql.AnalyticEuropeanEngine(process_actual))

    # Calculate the Greek value for the actual underlying price
    multiplier = 1 * nb_options if buy_sell == 'buy' else -1 * nb_options

    if greek == 'option_Price':
        actual_greek_value = multiplier * european_option_actual.NPV()
    elif greek == 'delta':
        actual_greek_value = multiplier * european_option_actual.delta()
    elif greek == 'theta':
        actual_greek_value = multiplier * european_option_actual.theta()
    elif greek == 'gamma':
        actual_greek_value = multiplier * european_option_actual.gamma()
    elif greek == 'vega':
        actual_greek_value = multiplier * european_option_actual.vega()
    elif greek == 'rho':
        actual_greek_value = multiplier * european_option_actual.rho()
    else:
        raise ValueError("Unrecognized Greek. Choose from: 'option_Price', 'delta', 'theta', 'gamma', 'vega', 'rho'.")

    # Iterate over each underlying price in the given range
    for S in range(lim_inf, lim_sup):
        # Set up the market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, ql.QuoteHandle(ql.SimpleQuote(r)), day_count))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, ql.QuoteHandle(ql.SimpleQuote(vol)), day_count))

        # Set up the Black-Scholes process
        process = ql.BlackScholesProcess(spot_handle, flat_ts, vol_ts)

        # Create the option
        european_option = ql.VanillaOption(payoff, exercise)

        # Set the pricing engine
        european_option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        # Calculate the Greek value
        multiplier = 1 * nb_options if buy_sell == 'buy' else -1 * nb_options

        if greek == 'option_Price':
            res_greek.append(multiplier * european_option.NPV())
        elif greek == 'delta':
            res_greek.append(multiplier * european_option.delta())
        elif greek == 'theta':
            res_greek.append(multiplier * european_option.theta())
        elif greek == 'gamma':
            res_greek.append(multiplier * european_option.gamma())
        elif greek == 'vega':
            res_greek.append(multiplier * european_option.vega())
        elif greek == 'rho':
            res_greek.append(multiplier * european_option.rho())
        else:
            raise ValueError("Unrecognized Greek. Choose from: 'option_Price', 'delta', 'theta', 'gamma', 'vega', 'rho'.")

    # Return the list of Greek values and the corresponding option descriptions
    return res_greek, descriptions, actual_greek_value

def execute_functions(list_options, greek, lim_inf, lim_sup, actual_ul_price):
    list_values = []
    list_descr = []
    list_actual_greek = []
    for options in list_options:
        result_greek, descr_option_greek, actual_greek = calculate_greek(options, greek, lim_inf, lim_sup, actual_ul_price)
        list_values.append(result_greek)
        list_descr.append(descr_option_greek)
        list_actual_greek.append(actual_greek)

    greek_value = sum(list_actual_greek)  
    return list_values, list_descr, greek_value

#endregion


#region Hedging

def delta_hedging_ptf(actual_delta_value, ticker):
    stock_to_buy_or_sell = 0
    nb_of_stocks_per_options = 100
    stock_to_buy_or_sell = nb_of_stocks_per_options * actual_delta_value
    stock_price = round(get_stock_price_and_volatility(ticker, period='1y')[0],2)
    if actual_delta_value > 0:
        cost_hedging = stock_to_buy_or_sell * stock_price
        type_trade = 'Sell'
    elif actual_delta_value < 0:
        cost_hedging = (-1) * stock_to_buy_or_sell * stock_price
        type_trade = 'Buy'
    else :
        cost_hedging = 0
        type_trade = ""
    data = {
    'Stock': [ticker],
    'Trade': [type_trade],
    'Quantity': [stock_to_buy_or_sell],
    'Asset Price': [stock_price],
    'Total Cost': [cost_hedging]
}
    tab = pd.DataFrame(data).reset_index(drop=True)
    return tab


#endregion

#region Parameters

# Example for NVIDIA
def Get_parameters(ticker="NVDA"):

    price_stocks, volatility_stocks = get_stock_price_and_volatility(ticker)

    K_Put = round(price_stocks) + 5       
    K_Call = round(price_stocks) - 5
    T1 = 1
    r1 = 0.035
    type1 = "EU"    #Option's Type (EU ou US)
    M1 = 1          #Maturity
    nb_o1 = 1
    nb_o2 = 1
    option1_params = ['call', K_Call, r1, T1, volatility_stocks, 'sell', nb_o1] # Buy Call option parameters
    option2_params = ['call', K_Put, r1, T1, volatility_stocks, 'buy', nb_o2] # Sell Put option parameters
    options_params = [
        option1_params,  # Buy Call option parameters
        option2_params   # Sell Put option parameters
    ]

    P_Call = calculate_option_price(option1_params, price_stocks)          #Prenium
    P_Put = calculate_option_price(option2_params, price_stocks)

    print("Prenium of the call : ", P_Call)
    print("Prenium of the put : ", P_Put)

    lim_inf = 1
    lim_sup = round(round(price_stocks) * 2)

    return {'price_stocks':price_stocks,
            'volatility_stocks':volatility_stocks,
            'lim_inf':lim_inf,
            'lim_sup':lim_sup
            }
def update_selected_greek():
    st.session_state.selected_greek = st.session_state.greek_radio
#endregion

#region Streamlit

if "symbols_list" not in st.session_state:
    st.session_state.symbols_list = None

st.set_page_config(layout='wide', page_title='Equity Derivatives Dashboard')

st.markdown("""
    <style>
        footer {display: none}
        [data-testid="stHeader"] {display: none}
    </style>
    """, unsafe_allow_html=True)

try:
    with open('/Users/thomasjouve/Documents/Python/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    with open('styles.css') as f:
         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<p class="params_text">OPTIONS PARAMETERS</p>', unsafe_allow_html=True)
    st.divider()
    
    with st.form(key='params_form'):
        stock_ticker = st.text_input("Stock", "NVDA")
        type_trades = st.selectbox("Trade type", ['buy', 'sell'])
        quantity = st.number_input("Number of Options", value=1, step=1)
        type_option_cp = st.selectbox("Option's type", ['call', 'put', 'digital call'])
        type_eu_us = st.selectbox("Option's type", ['EU', 'US'])
        strike = st.number_input("Strike", value=0.0, step=0.1)
        maturity = st.number_input("Time to Maturity (in Y)", value=0.0, step=0.1)
        r = st.number_input("Risk free rate", value=0.0, step=0.01)
        color_wanted = st.color_picker("Pick A Color", "#00f900")
        
        submitted = st.form_submit_button(label='Submit', use_container_width=True)
        clear = st.form_submit_button(label='🗑️', use_container_width=True)

# Main layout
# title_col, emp_col, equity_col, vol_col, date_col, price_chart_col = st.columns([1,0.2,1,1,1,2])
title_col, emp_col, equity_col, vol_col, delta_col, gamma_col, vega_col = st.columns([1,0.2,1,1,1,1,1])

# Initialize session state variables
if 'L_options' not in st.session_state:
    st.session_state.L_options = []
if 'L_options_2' not in st.session_state:
    st.session_state.L_options_2 = []
if 'L_descr_options' not in st.session_state:
    st.session_state.L_descr_options = []
if 'L_color' not in st.session_state:
    st.session_state.L_color = []
if 'selected_greek' not in st.session_state:
    st.session_state.selected_greek = 'payoff'
if 'plots' not in st.session_state:
    st.session_state.plots = {}

# Title column
with title_col:
    st.markdown('<p class="dashboard_title">Equity Derivatives<br>Dashboard</p>', unsafe_allow_html=True)

# Info columns
with equity_col:
    with st.container():
        nvidia_price = f"{Get_parameters(stock_ticker)['price_stocks']:.2f}"
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .nvidia_text {
            margin-bottom: 0;
        }
        .price_details {
            margin-top: 15px;
        }
        </style>
        """
        
        # Création du contenu HTML avec le style appliqué
        content = f"""
        {style}
        <div class="custom-container">
            <p class="nvidia_text">{stock_ticker} Price</p>
            <p class="price_details">{nvidia_price}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

with vol_col:
    with st.container():
        nvidia_vol = f"{Get_parameters(stock_ticker)['volatility_stocks'] * 100:.2f}%"
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .vol_text {
            margin-bottom: 0;
        }
        .price_details {
            margin-top: 15px;
        }
        </style>
        """
        
        # Création du contenu HTML avec le style appliqué
        content = f"""
        {style}
        <div class="custom-container">
            <p class="vol_text">{stock_ticker} Annual Vol</p>
            <p class="price_details">{nvidia_vol}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

def update_delta_value():
    delta_value = execute_functions(
        st.session_state.L_options_2, 'delta',
        1, 250,float(nvidia_price))[2]
    return round(delta_value, 2)

def update_gamma_value():
    gamma_value = execute_functions(
        st.session_state.L_options_2, 'gamma',
        1, 250,float(nvidia_price))[2]
    return round(gamma_value, 3)

def update_vega_value():
    vega_value = execute_functions(
        st.session_state.L_options_2, 'vega',
        1, 250,float(nvidia_price))[2]
    return round(vega_value, 2)

# Main content area
chart_col, data_col = st.columns([3,1])

# Form submission logic
if submitted:
        st.session_state.stock_data = get_historical_data(stock_ticker)
        days = int(maturity * 365)
        today = datetime.now()
        expiration_date = today + timedelta(days=days)
        price_stock, vol_stock = get_stock_price_and_volatility(stock_ticker, period='1y')
        try:
            # Fetch options data
            options_data = get_all_options(stock_ticker, [type_option_cp, strike, r, maturity])       
            volatility_table = create_volatility_table(options_data)
            expiration_date = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')
            implied_vol = volatility_table.loc[strike, expiration_date]
            print(implied_vol)
            vol_graph = plot_volatility_surface(volatility_table, strike, maturity)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            implied_vol = vol_stock

        option1_params = [type_option_cp, strike, r, maturity, implied_vol, type_trades, quantity]
        st.session_state.L_options_2.append(option1_params)
        option_prenium = calculate_option_price(option1_params, price_stock)
        result_option1, descr_option1 = Simulate_data_call_put(
            Get_parameters(stock_ticker)['lim_inf'],
            Get_parameters(stock_ticker)['lim_sup'],
            type_option_cp, strike, option_prenium, type_eu_us, maturity, type_trades, quantity
        )
        st.session_state.L_options.append(result_option1)
        st.session_state.L_descr_options.append(descr_option1)
        st.session_state.L_color.append(color_wanted)
        
        result_derivative_product = Simulate_data_dervative(st.session_state.L_options)
        
        # Generate all the plots
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho', 'option_Price']
        for greek in greeks:
            result, descr, actual_greek = execute_functions(
                st.session_state.L_options_2, greek,
                Get_parameters(stock_ticker)['lim_inf'],
                Get_parameters(stock_ticker)['lim_sup'],
                price_stock
            )
            der = Simulate_data_dervative(result)
            st.session_state.plots[greek] = plot_function(
                range(Get_parameters(stock_ticker)['lim_inf'], Get_parameters(stock_ticker)['lim_sup']),
                der, result, descr,
                [f'{greek.capitalize()} of the Options', 'Stock Price (St)', greek.capitalize()],
                st.session_state.L_color
            )
        
        # Payoff plot
        st.session_state.plots['payoff'] = plot_function(
            range(Get_parameters(stock_ticker)['lim_inf'], Get_parameters(stock_ticker)['lim_sup']),
            result_derivative_product, st.session_state.L_options, st.session_state.L_descr_options,
            ['Payoff of the Option', 'Stock Price (St)', 'Payoff'], st.session_state.L_color
        )


with delta_col:
    with st.container():
        delta_value = update_delta_value()
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .delta_text {
            margin-bottom: 0;
        }
        .price_details {
            margin-top: 15px;
        }
        </style>
        """
       
        # Création du contenu HTML avec le style appliqué
        content = f"""
        {style}
        <div class="custom-container">
            <p class="delta_text">Delta Value</p>
            <p class="price_details">{delta_value}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)  

with gamma_col:
    with st.container():
        gamma_value = update_gamma_value()
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .gamma_text {
            margin-bottom: 0;
        }
        .price_details {
            margin-top: 15px;
        }
        </style>
        """
       
        # Création du contenu HTML avec le style appliqué
        content = f"""
        {style}
        <div class="custom-container">
            <p class="gamma_text">Gamma Value</p>
            <p class="price_details">{gamma_value}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)     

with vega_col:
    with st.container():
        vega_value = update_vega_value()
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .vega_text {
            margin-bottom: 0;
        }
        .price_details {
            margin-top: 15px;
        }
        </style>
        """
       
        # Création du contenu HTML avec le style appliqué
        content = f"""
        {style}
        <div class="custom-container">
            <p class="vega_text">Vega Value</p>
            <p class="price_details">{vega_value}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True) 


if st.session_state.plots:
    st.markdown(f"<p class='price_details'>{st.session_state.L_descr_options}</p>", unsafe_allow_html=True)
    
tab_payoff, tab_delta, tab_gamma, tab_theta, tab_vega, tab_rho, tab_option_price = st.tabs(['Payoff','Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Option Price'])
with tab_payoff:
    if 'payoff' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['payoff'], use_container_width=True,theme = None)
with tab_delta:
    if 'delta' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['delta'], use_container_width=True)

with tab_gamma:
    if 'gamma' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['gamma'], use_container_width=True)

with tab_theta:
    if 'theta' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['theta'], use_container_width=True)

with tab_vega:
    if 'vega' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['vega'], use_container_width=True)

with tab_rho:
    if 'rho' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['rho'], use_container_width=True)

with tab_option_price:
    if 'option_Price' in st.session_state.plots:
        st.plotly_chart(st.session_state.plots['option_Price'], use_container_width=True)
    # greeks = ['delta', 'gamma', 'theta', 'vega', 'rho', 'option_Price']
    # st.radio("Select Greek to display:", 
    #             ['payoff'] + greeks, 
    #             key='greek_radio',
    #             on_change=update_selected_greek,
    #             format_func=lambda x: x.capitalize() if x != 'option_Price' else 'Option Price',
    #             horizontal=True)
    
    # if st.session_state.selected_greek in st.session_state.plots:
    #     st.plotly_chart(st.session_state.plots[st.session_state.selected_greek], use_container_width=True)



# with data_col:
#     with st.container():
#         delta_value = update_delta_value()
#         tab = delta_hedging_ptf(delta_value, stock_ticker)
#         st.table(tab)

# with chart_col:
if st.session_state.plots:
    with st.expander("Raw Volatility Data"):
        st.dataframe(volatility_table)
        vol_graph
        

if clear:
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []
    st.session_state.L_options_2 = []
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []
    if 'stock_data' in st.session_state:
        del st.session_state.stock_data

#endregion