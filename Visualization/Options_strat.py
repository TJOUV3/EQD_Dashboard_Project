#region import

import pandas as pd
import matplotlib.pyplot as plt
#import openai
import os
import huggingface_hub
import io
import base64
from huggingface_hub import InferenceClient
import numpy as np
import yfinance as yf
import QuantLib as ql
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime

#endregion

#region functions payoffs, option_description, derivated_products
def payoff(call_put, St,K,Prenium,type,Maturity, buy_sell):
    if call_put == 'call':
        if buy_sell == 'buy':
            return max(0, St-K) - Prenium
        else:
            return -max(0, St-K) + Prenium
    elif call_put == 'put':
        if buy_sell == 'buy':
            return max(K-St, 0) - Prenium
        else:
            return -max(K-St, 0) + Prenium
    else:
        return 0

def option_description(call_put, Strike, type_option, Maturity, buy_sell):
    return str(buy_sell) + ' ' + str(type_option) + ' ' + str(call_put) + ' ' +  str(Maturity) + 'Y' +  ', Strike = ' + str(Strike)

def derivated_products(St, list_options):
    resulting_payoff = 0
    test = False
    if list_options:
        for sublist in list_options:
            if sublist:
                resulting_payoff = resulting_payoff + payoff(sublist[0], St, sublist[1], sublist[2], sublist[3], sublist[4], sublist[5])

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

def calculate_option_price(option_params, St):
    option_type = option_params[0] 
    K = option_params[1]  # Strike price
    r = option_params[2]  # Risk-free rate
    t = option_params[3]  # Time to maturity (in years)
    vol = option_params[4]  # Volatility
    buy_sell = option_params[5]  # Buy or sell

    # Set up the dates
    calendar = ql.NullCalendar()
    today = ql.Date.todaysDate()
    maturity_date = today + int(t * 365)
    
    # Set up the option payoff and exercise
    if option_type == 'call':
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    elif option_type == 'put':
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, K)
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
    
def Simulate_data_call_put(limit_inf, limit_sup, call_put, strike, prenium, type_option,maturity,buy_sell):
    result = []
    descr = option_description(call_put, strike, type_option, maturity, buy_sell)
    for price in range(limit_inf,limit_sup):
        result.append(payoff(call_put,price,strike,prenium,type_option,maturity,buy_sell))
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
    option_type, K, r, t, vol, buy_sell = option_params

    # Set up the QuantLib environment
    calendar = ql.NullCalendar()
    settlement_date = ql.Date.todaysDate()
    maturity_date = settlement_date + int(t * 365)  # Time to maturity in days
    day_count = ql.Actual365Fixed()

    # Create the option payoff and exercise
    payoff = ql.PlainVanillaPayoff(ql.Option.Call if option_type == 'call' else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(maturity_date)

    # Generate the option description
    description = f"{option_type.capitalize()} option, Strike: {K}, Maturity: {t} years, {buy_sell.capitalize()}"
    descriptions.append(description)

    # Initialize a variable to store the Greek value at the actual underlying price
    actual_greek_value = None

    # Set up the market data for the actual underlying price
    spot_handle_actual = ql.QuoteHandle(ql.SimpleQuote(actual_underlying_price))
    flat_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, ql.QuoteHandle(ql.SimpleQuote(r)), day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, ql.QuoteHandle(ql.SimpleQuote(vol)), day_count))
    process_actual = ql.BlackScholesProcess(spot_handle_actual, flat_ts, vol_ts)

    # Create the option for the actual underlying price
    european_option_actual = ql.VanillaOption(payoff, exercise)
    european_option_actual.setPricingEngine(ql.AnalyticEuropeanEngine(process_actual))

    # Calculate the Greek value for the actual underlying price
    multiplier = 1 if buy_sell == 'buy' else -1

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
        multiplier = 1 if buy_sell == 'buy' else -1

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
    
    return list_values, list_descr, list_actual_greek

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
    option1_params = ['call', K_Call, r1, T1, volatility_stocks, 'sell'] # Buy Call option parameters
    option2_params = ['call', K_Put, r1, T1, volatility_stocks, 'buy'] # Sell Put option parameters
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
        type_option_cp = st.selectbox("Option's type", ['call', 'put'])
        type_eu_us = st.selectbox("Option's type", ['EU', 'US'])
        strike = st.number_input("Strike", value=0.0, step=0.1)
        maturity = st.number_input("Time to Maturity (in Y)", value=0.0, step=0.1)
        r = st.number_input("Risk free rate", value=0.0, step=0.01)
        color_wanted = st.color_picker("Pick A Color", "#00f900")
        
        submitted = st.form_submit_button(label='Submit', use_container_width=True)
        clear = st.form_submit_button(label='🗑️', use_container_width=True)

# Main layout
title_col, emp_col, equity_col, vol_col, date_col, x_col, y_col = st.columns([1,0.2,1,1,1,1,1])

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

with date_col:
    with st.container():

        nvidia_vol = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .date_text {
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
            <p class="date_text">Last Update</p>
            <p class="price_details">{nvidia_vol}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

with x_col:
    with st.container():
        delta_value = execute_functions(st.session_state.L_options_2, 'delta',
                Get_parameters(stock_ticker)['lim_inf'],
                Get_parameters(stock_ticker)['lim_sup'],
                nvidia_price)
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .x_text {
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
            <p class="x_text">Delta Value</p>
            <p class="price_details">{delta_value}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

with y_col:
    with st.container():
        nvidia_vol = 'Y'
        
        # Définition du style CSS inline
        style = """
        <style>
        .custom-container {
        }
        .y_text {
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
            <p class="y_text">Y</p>
            <p class="price_details">{nvidia_vol}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)


# Main content area
chart_col, data_col = st.columns([3,1])

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

# Form submission logic
if submitted:
        price_stock, vol_stock = get_stock_price_and_volatility(stock_ticker, period='1y')
        option1_params = [type_option_cp, strike, r, maturity, vol_stock, type_trades]
        st.session_state.L_options_2.append(option1_params)
        option_prenium = calculate_option_price(option1_params, price_stock)
        result_option1, descr_option1 = Simulate_data_call_put(
            Get_parameters(stock_ticker)['lim_inf'],
            Get_parameters(stock_ticker)['lim_sup'],
            type_option_cp, strike, option_prenium, type_eu_us, maturity, type_trades
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

with chart_col:
    if st.session_state.plots:
        st.markdown(f"<p class='price_details'>{st.session_state.L_descr_options}</p>", unsafe_allow_html=True)
        
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho', 'option_Price']
        st.radio("Select Greek to display:", 
                    ['payoff'] + greeks, 
                    key='greek_radio',
                    on_change=update_selected_greek,
                    format_func=lambda x: x.capitalize() if x != 'option_Price' else 'Option Price',
                    horizontal=True)
        
        if st.session_state.selected_greek in st.session_state.plots:
            st.plotly_chart(st.session_state.plots[st.session_state.selected_greek], use_container_width=True)

if clear:
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []
    st.session_state.L_options_2 = []
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []

#endregion