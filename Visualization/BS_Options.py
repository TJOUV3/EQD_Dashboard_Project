import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.interpolate import griddata
from pandas.tseries.offsets import BDay
from itertools import product, combinations


import streamlit as st

def get_stock_price_and_volatility(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    latest_price = data['Close'].iloc[-1]
    first_price = data['Close'].iloc[0]
    N = len(data['Close']) - 2
    
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    daily_volatility = data['Log Return'].std()
    annual_volatility = daily_volatility * np.sqrt(252)
    mu = data['Log Return'].mean()
    sigma = np.std(np.diff(data['Log Return'])) * np.sqrt(252)
    rho = np.corrcoef(data['Log Return'][:-1], np.diff(data['Log Return']))[0, 1]
    
    return first_price, annual_volatility, mu, latest_price, sigma, rho

def get_all_options(ticker, option_params):
    stock = yf.Ticker(ticker)
    option_type = option_params[0]
    K = option_params[1]
    r = option_params[2]
    t = option_params[3]
    expirations = stock.options
    all_options_data = []

    # Date d'hier
    yesterday = datetime.now() - BDay(1)

    for expiration_date in expirations:
        options_chain = stock.option_chain(expiration_date)
        
        options = options_chain.calls if option_type == 'call' else options_chain.puts
        options['expiration'] = expiration_date

        # Convertir lastTradeDate en datetime pour pouvoir la comparer
        options['lastTradeDate'] = pd.to_datetime(options['lastTradeDate'])
        
        # Filtrer les options qui ont été échangées hier
        options_yesterday = options[options['lastTradeDate'].dt.date == yesterday.date()]

        all_options_data.append(options_yesterday)
    
    if all_options_data:
        options_df = pd.concat(all_options_data, ignore_index=True)

        # Filtrer par la plage de prix autour du strike (K)
        lower_limit = K * 0.5
        upper_limit = K * 1.5
        options_df = options_df[(options_df['strike'] >= lower_limit) & (options_df['strike'] <= upper_limit)]

        # Ajouter une option avec strike = K si ce strike n'existe pas
        if K not in options_df['strike'].unique():
            new_row_K = options_df.iloc[0].copy()
            new_row_K['strike'] = K
            # Vérifie et convertit expiration_date en datetime si nécessaire
            if isinstance(expiration_date, str):
                expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')

            new_row_K['expiration'] = expiration_date.strftime('%Y-%m-%d')
            new_row_K['impliedVolatility'] = np.NaN
            options_df = pd.concat([options_df, pd.DataFrame([new_row_K])], ignore_index=True)

        options_df = options_df.sort_values(by=['strike', 'expiration']).reset_index(drop=True)
        
        return options_df
    else:
        print("Aucune option trouvée pour la date de négociation d'hier.")
        return None

def sabr_implied_vol(alpha, beta, rho, nu, F, K, T):
    if F == K:
        FK_beta = F ** (1 - beta)
        vol = alpha / FK_beta * (1 + ((1 - beta)**2 / 24) * (alpha**2 / FK_beta**2) * T +
                                 ((1 - beta)**4 / 1920) * (alpha**4 / FK_beta**4) * T)
    else:
        log_FK = np.log(F / K)
        FK_beta = (F * K) ** ((1 - beta) / 2)
        z = (nu / alpha) * FK_beta * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
        vol = (alpha / FK_beta) * (z / x_z) * (1 + ((1 - beta)**2 / 24) * (log_FK ** 2) +
                                              ((1 - beta)**4 / 1920) * (log_FK ** 4))
    return vol

def sabr_loss(params, strikes, market_vols, F, T):
    alpha, beta, rho, nu = params
    model_vols = np.array([sabr_implied_vol(alpha, beta, rho, nu, F, K, T) for K in strikes])
    return np.sum((market_vols - model_vols) ** 2)

def calibrate_sabr(strikes, market_vols, F, T, beta=0.5):
    initial_guess = [0.2, beta, 0.0, 0.2]
    bounds = [(0.01, 2.0), (0.0, 1.0), (-0.999, 0.999), (0.01, 2.0)]
    result = minimize(sabr_loss, initial_guess, args=(strikes, market_vols, F, T), bounds=bounds, method='L-BFGS-B')
    return result.x

def construct_volatility_table(options_df):
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    volatility_table = options_df.pivot(index='strike', columns='expiration', values='impliedVolatility')
    return volatility_table

def fit_volatility_surface(volatility_table, spot_price, risk_free_rate):
    filled_vol_table = volatility_table.copy()
    for expiry in volatility_table.columns:
        T = (expiry - datetime.today()).days / 365.0
        F = spot_price * np.exp(risk_free_rate * T)
        strikes = volatility_table.index.values
        market_vols = volatility_table[expiry].values
        missing_idx = np.isnan(market_vols)
        if missing_idx.any():
            sabr_params = calibrate_sabr(strikes[~missing_idx], market_vols[~missing_idx], F, T)
            filled_vol_table.loc[missing_idx, expiry] = [sabr_implied_vol(*sabr_params, F, K, T) for K in strikes[missing_idx]]
    return filled_vol_table

def plot_volatility_surface(data):
    # Extraire les valeurs X, Y, Z
    strikes = data['strike'].values
    expirations = data['days_to_expiration'].values
    volatilities = data['impliedVolatility'].values

    # Créer une grille pour interpolation
    strike_grid, expiration_grid = np.meshgrid(np.unique(strikes), np.unique(expirations))
    vol_grid = griddata((strikes, expirations), volatilities, (strike_grid, expiration_grid), method='cubic')
    vol_grid = np.nan_to_num(vol_grid, nan=0.0)
    # Création du graphique 3D avec Plotly
    fig = go.Figure(data=[go.Surface(
        z=vol_grid,
        x=strike_grid,
        y=expiration_grid,
        colorscale='Viridis',  # Choix de la palette de couleurs
        colorbar=dict(title='Implied Volatility'),  # Titre de la barre de couleurs
        opacity=0.7  # Opacité de la surface
    )])

    # Ajout des labels et du titre
    fig.update_layout(
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility'
        ),
        title='Implied Volatility Surface'
    )

    # Afficher la figure
    return fig

def get_nearest_vol(options_df, K, T):
    """
    Finds the nearest implied volatility in the volatility table based on strike and time to maturity.
    
    Parameters:
    - options_df: Pandas DataFrame containing implied volatilities with expiration dates as columns
    - K: Strike price
    - T: Time to maturity (years)
    
    Returns:
    - Closest implied volatility
    """
    # Ensure column names are treated as expiration dates
    try:
        expirations = pd.to_datetime(options_df.columns)
    except Exception as e:
        raise ValueError("Error parsing expiration dates from DataFrame columns.") from e

    if len(expirations) == 0:
        raise ValueError("No valid expiration dates found in DataFrame.")

    # Convert maturities into time-to-expiry in years
    today = pd.to_datetime("today")
    time_to_maturities = np.array([(exp - today).days / 365.0 for exp in expirations])

    if len(time_to_maturities) == 0:
        raise ValueError("No valid time-to-maturity values found.")

    # Find the nearest expiration
    nearest_T_idx = np.argmin(np.abs(time_to_maturities - T))
    nearest_expiration = expirations[nearest_T_idx]

    # Find the nearest strike price
    available_strikes = options_df.index.to_numpy()
    if len(available_strikes) == 0:
        raise ValueError("No valid strike prices found in DataFrame.")

    nearest_K_idx = np.argmin(np.abs(available_strikes - K))
    nearest_strike = available_strikes[nearest_K_idx]

    # Retrieve the closest implied volatility
    nearest_vol = options_df.loc[nearest_strike, nearest_expiration]

    return nearest_vol

def Stock(S0, S, weight):
    '''
    Function to compute the price and Greeks of a stock position.
    
    Parameters:
    S0 : float : Initial stock price
    S : float : Current stock price
    weight : float : Position in the stock (-1 for short, 1 for long)
    
    Returns:
    dict : A dictionary containing the price and Greeks of the stock position
    '''
    price = weight * (S - S0)  # Stock price scaled by position
    payoff = weight * (S - S0)
    delta = weight  # Delta of a stock is always 1 for long, -1 for short
    gamma = 0  # No convexity for a stock position
    theta = 0  # No time decay for a stock
    vega = 0  # No volatility sensitivity
    
    return {
        'payoff':payoff,
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

def BlackScholes(S, K, r, T, option_type, weight):
    """
    Compute the Black-Scholes option price and Greeks (adjusted for position weight).
    
    Parameters:
    - S (float): Spot price of the underlying asset.
    - K (float): Strike price.
    - sigma (float): Volatility.
    - r (float): Risk-free interest rate.
    - T (float): Time to maturity.
    - option_type (str): "call" or "put".
    - weight (float): Position weight (+1 for long, -1 for short).
    
    Returns:
    - Dictionary containing option price and Greeks, all adjusted by weight.
    """
    
    if option_type == "call":
        vol_table = st.session_state.fitted_volatility_table_call
    elif option_type == "put":
        vol_table = st.session_state.fitted_volatility_table_put
    
    sigma = get_nearest_vol(vol_table, K, T)
    
    d1 = (np.log(S / K) + (r + 0.5 * (sigma ** 2)) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * T)
        delta = norm.cdf(d1)
        payoff = max(0,S-K)
    elif option_type == "put":
        price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S
        delta = norm.cdf(d1) - 1  # Ensure correct delta for puts
        payoff = max(0,K-S)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Adjust all outputs by weight
    return {
        "payoff": weight * payoff,
        "price": weight * price,
        "delta": weight * delta,
        "gamma": weight * gamma,
        "theta": weight * theta,
        "vega": weight * vega
    }

def generate_greeks_table(S0, option_type, risk_free_rate):
    """
    Generates gamma and vega tables for a range of strikes and maturities.
    
    Parameters:
    - S0: Current spot price.
    - risk_free_rate: Risk-free interest rate.
    - options_df: DataFrame containing implied volatilities.
    
    Returns:
    - gamma_table: List of dictionaries with gamma values.
    - vega_table: List of dictionaries with vega values.
    """
    maturities = [round(m, 2) for m in [1/52, 2/52, 1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12, 2]]
    strikes = np.round(np.linspace(0.5 * S0, 1.5 * S0, 30)).astype(int)
    delta_table = []
    gamma_table = []
    vega_table = []

    for T in maturities:
        for K in strikes:
            option_data = BlackScholes(S0, K, risk_free_rate, T, option_type,1)
            delta_table.append({"strike": K, "maturity": T, "delta": option_data["delta"]})
            gamma_table.append({"strike": K, "maturity": T, "gamma": option_data["gamma"]})
            vega_table.append({"strike": K, "maturity": T, "vega": option_data["vega"]})
    
    return delta_table, gamma_table, vega_table

def structured_product(S0, options, stocks):
    """
    Compute the structured product price and Greeks by summing individual options.
    
    Parameters:
    - S0: Current spot price.
    - options: List of option dictionaries.
    - stocks: List of stock dictionaries

    Returns:
    - Dictionary containing the structured product's total price and Greeks.
    """
    structured_results = {
        "payoff":{},
        "price": {},
        "delta": {},
        "gamma": {},
        "theta": {},
        "vega": {}
    }

    spot_prices = np.linspace(0.2 * S0, 1.8 * S0, min(200, int(1.8 * S0) - int(0.2 * S0) + 1)).astype(int)
    spot_prices = np.unique(spot_prices)
    progress_bar_structured_product = st.progress(0)
    i=0
    for S in spot_prices:
        progress = int((i + 1) / len(spot_prices) * 100)
        progress_bar_structured_product.progress(progress)
        i+=1
        total_payoff = 0
        total_price = 0
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0

        for opt in options:
            option_type = opt['type']
            K = opt['strike']
            weight = opt['weight']
            T = opt['T']
            r = opt['r']


            # Call BlackScholes for this option at current S
            bs_results = BlackScholes(S, K, r, T, option_type, weight)

            total_payoff += bs_results["payoff"]
            total_price += bs_results["price"]
            total_delta += bs_results["delta"]
            total_gamma += bs_results["gamma"]
            total_theta += bs_results["theta"]
            total_vega += bs_results["vega"]
        
        if stocks:
            for stock in stocks:
                weight = stock['weight']

                # Call Stock at current S
                stock_results = Stock(S0, S, weight)

                total_payoff += stock_results["payoff"]
                total_price += stock_results["price"]
                total_delta += stock_results["delta"]
                total_gamma += stock_results["gamma"]
                total_theta += stock_results["theta"]
                total_vega += stock_results["vega"]

        # Store aggregated results
        structured_results["payoff"][S] = total_payoff
        structured_results["price"][S] = total_price
        structured_results["delta"][S] = total_delta
        structured_results["gamma"][S] = total_gamma
        structured_results["theta"][S] = total_theta
        structured_results["vega"][S] = total_vega

    return structured_results

def plot_greeks(results, options, S0, stocks):
    """
    Function to plot the Greeks and option price with markers for S0 and a vertical line for K using Plotly.
    
    Parameters:
    - results: Dictionary containing option prices and Greeks.
    - S0: The current spot price of the stock (highlighted with a marker).
    - options: List of dictionaries, each containing option parameters.
    - stocks: List of dictionaries, each containing stock parameters.
    """
    # Create subplots
    fig = make_subplots(rows=2, cols=3, subplot_titles=("Payoff", "Option Price", "Delta", "Gamma", "Theta", "Vega"))

    # Extract Spot Prices (x-axis)
    spot_prices = list(results["price"].keys())

    # Define Greeks and labels
    greek_labels = {
        "payoff": ("Payoff", "Payoff", "black"),
        "price": ("Option Price", "Price", "blue"),
        "delta": ("Delta", "Delta", "red"),
        "gamma": ("Gamma", "Gamma", "green"),
        "theta": ("Theta", "Theta", "orange"),
        "vega": ("Vega", "Vega", "purple"),
    }

    structured_results = structured_product(S0, options, stocks)

    # Find the closest spot price to S0
    spot_price_array = np.array(spot_prices)
    closest_S = spot_price_array[np.abs(spot_price_array - S0).argmin()]

    # Compute the stock Greeks across all spot prices
    stock_values = {greek: [] for greek in greek_labels.keys()}
    
    for S in spot_prices:
        total_stock_results = {"payoff": 0, "price": 0, "delta": 0, "gamma": 0, "theta": 0, "vega": 0}
        
        for stock in stocks:
            stock_res = Stock(S0, S, stock['weight'])
            for greek in stock_values.keys():
                total_stock_results[greek] += stock_res[greek]
        
        # Store stock values for each spot price
        for greek in stock_values.keys():
            stock_values[greek].append(total_stock_results[greek])

    # Plot each Greek in its respective subplot
    for i, (greek, (title, ylabel, color)) in enumerate(greek_labels.items()):
        row = (i // 3) + 1  # Row index (1 or 2)
        col = (i % 3) + 1   # Column index (1, 2, or 3)

        # Plot Structured Product (Main Line)
        fig.add_trace(
            go.Scatter(
                x=spot_prices,
                y=list(structured_results[greek].values()),
                mode="lines",
                name="Structured Product",
                line=dict(color=color, width=2.5),
            ),
            row=row,
            col=col,
        )

        # Plot Stock Position (Dashed Line)
        if stocks:
            fig.add_trace(
                go.Scatter(
                    x=spot_prices,
                    y=stock_values[greek],
                    mode="lines",
                    name=f"{pos} {abs(stock['weight'])} Stocks",
                    line=dict(color="black", width=1.5, dash="dashdot"),
                ),
                row=row,
                col=col,
            )

        # Add marker at **closest S0** on the structured product line
        if closest_S in structured_results[greek]:
            fig.add_trace(
                go.Scatter(
                    x=[closest_S],
                    y=[structured_results[greek][closest_S]],
                    mode="markers",
                    name="Current Value",
                    marker=dict(color="black", size=10),
                ),
                row=row,
                col=col,
            )

        # Plot Each Individual Option as Dotted Lines + Strike Lines
        for j, opt in enumerate(options):
            option_values = []  # Store computed values for each spot price
            if opt['weight'] < 0:
                pos = "Short"
            else:
                pos = "Long"
            for S in spot_prices:
                bs_results = BlackScholes(S, opt['strike'], opt['r'], opt['T'], opt['type'], opt['weight'])
                option_values.append(bs_results[greek])  # Append the scalar value

            fig.add_trace(
                go.Scatter(
                    x=spot_prices,
                    y=option_values,
                    mode="lines",
                    name=f"{pos} {abs(opt['weight'])} {opt['type']} {opt['strike']} {round(opt['T'],2)}Y",
                    line=dict(color="gray", width=1.2, dash="dot"),
                ),
                row=row,
                col=col,
            )

            # Add **dotted vertical line at strike price (K)**
            fig.add_trace(
                go.Scatter(
                    x=[opt['strike'], opt['strike']],
                    mode="lines",
                    line=dict(color="gray", width=1, dash="dash"),
                    showlegend=False
                ),
                row=row,
                col=col,
            )

        # Formatting
        fig.update_xaxes(title_text="Spot Price", row=row, col=col)
        fig.update_yaxes(title_text=ylabel, row=row, col=col)

    # Hide any empty subplots (if present)
    if len(greek_labels) < 6:
        fig.update_traces(visible=False, row=2, col=3)

    # Adjust layout and show plot
    fig.update_layout(
        title_text="Greeks and Option Price",
        showlegend=True,
        height=800,
        width=1200,
    )
    st.plotly_chart(fig)

def daily_pnl(S0, S_new, days, options, stocks):
    """
    Compute the daily PnL of the structured product given a new stock price and time decay.

    Parameters:
    - S0: Initial stock price.
    - S_new: New stock price after 'days' days.
    - days: Number of days elapsed.
    - options: List of option dictionaries.
    - stocks: List of stock dictionaries.

    Returns:
    - PnL value of the structured product.
    """
    # Update T for each option
    updated_options = []
    for opt in options:
        new_T = max(opt['T'] - days / 252, 0)  # Ensure T is non-negative
        updated_opt = opt.copy()
        updated_opt['T'] = new_T
        updated_options.append(updated_opt)

    # Compute Greeks at initial S0
    initial_results = structured_product(S0, options, stocks)

    # Find closest available spot price in the structured_product dictionary
    closest_S0 = min(initial_results["price"].keys(), key=lambda x: abs(x - S0))

    P0 = initial_results["price"][closest_S0]
    delta_0 = initial_results["delta"][closest_S0]
    gamma_0 = initial_results["gamma"][closest_S0]
    theta_0 = initial_results["theta"][closest_S0]

    # Compute Greeks at new price S_new and reduced time
    new_results = structured_product(S_new, updated_options, stocks)
    
    closest_S_new = min(new_results["price"].keys(), key=lambda x: abs(x - S_new))

    P_new = new_results["price"][closest_S_new]

    # Calculate PnL components
    delta_pnl = delta_0 * (S_new - S0)
    gamma_pnl = 0.5 * gamma_0 * ((S_new - S0) ** 2)
    theta_pnl = (theta_0 * days)/252

    # Total PnL
    total_pnl = (P_new - P0) + delta_pnl + gamma_pnl + theta_pnl

    return total_pnl


def plot_pnl(S0,S_new, days, options, stocks):
    """
    Plot the PnL curve of the structured product based on a range of spot prices using Plotly.

    Parameters:
    - S0: Initial stock price.
    - days: Number of days elapsed.
    - options: List of option dictionaries.
    - stocks: List of stock dictionaries.

    Returns:
    - None (displays an interactive Plotly figure).
    """
    # Generate a range of spot prices from 20% to 180% of S0
    spot_prices = np.linspace(S0 * 0.2, S0 * 1.8, 200)
    pnl_values = []
    progress_bar = st.progress(0)
    status_text=st.empty()
    i=0
    for S_temp in spot_prices:
        i+=1
        pnl = daily_pnl(S0, S_temp, days, options, stocks)
        pnl_values.append(pnl)
        # Update progress bar and status text
        progress = int((i + 1) / len(spot_prices) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Calculating PnL... {progress}% complete (In {days} days)")

    # Ensure pnl_values is not empty
    if not pnl_values:
        st.error("No PnL values calculated. Check the inputs and the `daily_pnl` function.")
        return

    # Create the Plotly figure
    fig_pnl = go.Figure()

    # Add the PnL curve
    fig_pnl.add_trace(
        go.Scatter(
            x=spot_prices,
            y=pnl_values,
            mode="lines",
            name="PnL Curve",
            line=dict(color="blue", width=2),
        )
    )

    # Add a horizontal line for zero PnL
    fig_pnl.add_trace(
        go.Scatter(
            x=[min(spot_prices), max(spot_prices)],
            y=[0, 0],
            mode="lines",
            name="Zero PnL",
            line=dict(color="black", dash="dash"),
        )
    )

    # Add a vertical line for the initial price (S0)
    fig_pnl.add_trace(
        go.Scatter(
            x=[S_new, S_new],
            y=[min(pnl_values), max(pnl_values)],
            mode="lines",
            name=f"New Price (S_new = {S_new})",
            line=dict(color="red", dash="dash"),
        )
    )

    # Update layout
    fig_pnl.update_layout(
        title=f"PnL vs Spot Price (after {days} days)",
        xaxis_title="Spot Price (S)",
        yaxis_title="PnL",
        legend_title="Legend",
        showlegend=True,
        template="plotly_white",
        height=600,
        width=1000,
    )

    # Add grid
    fig_pnl.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig_pnl.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Display the figure in Streamlit
    st.plotly_chart(fig_pnl)

def find_optimal_combination(gamma_table, vega_table, GAMMA_WANTED, VEGA_WANTED):
    best_solution = None
    min_gamma_error = float('inf')
    min_vega_error = float('inf')
    # Convert "N/A" to None for easier checks
    gamma_target = None if GAMMA_WANTED == "N/A" else GAMMA_WANTED
    vega_target = None if VEGA_WANTED == "N/A" else VEGA_WANTED
    # Creating a combined dictionary with (strike, maturity) as the key and (gamma, vega) as the value
    combined_dict = {
        (entry["strike"], entry["maturity"]): (entry["gamma"], next((v["vega"] for v in vega_table if v["strike"] == entry["strike"] and v["maturity"] == entry["maturity"]), None))
        for entry in gamma_table
    }
    # Extract all unique (K, T) pairs
    option_pairs = list(combined_dict.keys())

    print("Option pairs:", option_pairs)

    # Iterate through all valid option pairs
    for (K1, T1), (K2, T2) in itertools.combinations_with_replacement(option_pairs, 2):
        
        # Skip if (K1, T1) is the same as (K2, T2)
        if (K1, T1) == (K2, T2):
            continue
        
        gamma1, vega1 = combined_dict[(K1, T1)]
        gamma2, vega2 = combined_dict[(K2, T2)]

        # Iterate through all possible integer weights in [-2, -1, 1, 2] (excluding 0)
        for w1, w2 in itertools.product([-2, -1, 1, 2], repeat=2):
            gamma_hedge = w1 * gamma1 + w2 * gamma2
            vega_hedge = w1 * vega1 + w2 * vega2

            gamma_error = abs(gamma_hedge - gamma_target) if gamma_target is not None else 0
            vega_error = abs(vega_hedge - vega_target) if vega_target is not None else 0

            # Conditions based on what's specified:
            gamma_ok = (gamma_target is None) or (gamma_error <= 0.001)
            vega_ok = (vega_target is None) or (vega_error <= 0.5)

            # Only update best solution if conditions are met
            if gamma_ok and vega_ok:
                if (gamma_error < min_gamma_error and gamma_target is not None) or \
                   (vega_error < min_vega_error and vega_target is not None):
                    print('new best solution')
                    best_solution = [(K1, T1, w1), (K2, T2, w2)]
                    min_gamma_error, min_vega_error = gamma_error, vega_error
                    print("error gamma : ",min_gamma_error)

    return best_solution


def find_closest_key(dict_data, S0):
    # Find the closest key to S0 in the dictionary
    closest_key = min(dict_data.keys(), key=lambda k: abs(k - S0))
    return closest_key


def hedging(S0,structured_results, options, stocks, delta_target, gamma_target, vega_target, gamma_table, vega_table):
    closest_key = find_closest_key(structured_results["delta"], S0)

    delta_current = structured_results["delta"].get(closest_key, 0)
    gamma_current = structured_results["gamma"].get(closest_key, 0)
    print("current gamma : ",  gamma_current)
    vega_current = structured_results["vega"].get(closest_key, 0)

    # Find the optimal option combination to hedge gamma and vega
    res_gamma_target = None if gamma_target == "N/A" else (gamma_target - gamma_current)
    print("gamma wanted to be hedged: ",res_gamma_target)
    res_vega_target = None if vega_target == "N/A" else (vega_target - vega_current)
    print("vega wanted to be hedged: ",res_vega_target)

    optimal_pair = find_optimal_combination(gamma_table, vega_table, res_gamma_target, res_vega_target)

    if not optimal_pair or optimal_pair[0] is None:
        print("No valid gamma/vega hedging solution found. Keeping original portfolio.")
        return options, stocks  # Return original portfolio if no valid solution

    # Print selected options for hedging with their respective gamma values
    print("Selected options for hedging:", optimal_pair)
    for (K, T, weight) in optimal_pair:
        if weight != 0:
            gamma_value = next((entry["gamma"] for entry in gamma_table if entry["strike"] == K and entry["maturity"] == T), None)
            print(f"Option (K={K}, T={T}, weight={weight}) has gamma: {gamma_value}")

    # Determine whether we need to buy a put or short a call to reduce delta exposure
    option_type = "put" if delta_current > delta_target else "call"

    # Add the selected options to the portfolio
    for (K, T, weight) in optimal_pair:
        if weight != 0:
            options.append({"type": option_type, "strike": K, "weight": weight, "T": T, 'r': 0.02})

    # Recalculate structured product Greeks after adding options
    structured_results = structured_product(S0, options, stocks)

    # Delta hedging: adjust stock holdings to match delta target
    delta_adjustment = delta_target - structured_results["delta"].get(closest_key, 0)
    if delta_adjustment != 0:
        print("Performing delta hedging...")
        stocks.append({"type": "stock", "weight": round(delta_adjustment, 2)})

    return options, stocks




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
# Initialize session state variables
 


if 'stock_ticker' not in st.session_state:
    st.session_state.stock_ticker = 'NVDA'
if 'delta' not in st.session_state:
    st.session_state.delta = 0.0
if 'gamma' not in st.session_state:
    st.session_state.gamma = 0.0
if 'vega' not in st.session_state:
    st.session_state.vega = 0.0
if 'theta' not in st.session_state:
    st.session_state.theta = 0.0



if 'S0' not in st.session_state:
    st.session_state.S0 = ""
if 'annual_volatility' not in st.session_state:
    st.session_state.annual_volatility = ""
if 'mu' not in st.session_state:
    st.session_state.mu = ""
if 'latest_price' not in st.session_state:
    st.session_state.latest_price = 200
if 'sigma' not in st.session_state:
    st.session_state.sigma = ""
if 'rho' not in st.session_state:
    st.session_state.rho = ""


if 'options_data_call' not in st.session_state:
    st.session_state.options_data_call = []
if 'options_data_put' not in st.session_state:
    st.session_state.options_data_put = []
if 'volatility_table_call' not in st.session_state:
    st.session_state.volatility_table_call = []
if 'volatility_table_put' not in st.session_state:
    st.session_state.volatility_table_put = []
if 'fitted_volatility_table_call' not in st.session_state:
    st.session_state.fitted_volatility_table_call = []
if 'fitted_volatility_table_put' not in st.session_state:
    st.session_state.fitted_volatility_table_put = []
if 'options_df_long_call' not in st.session_state:
    st.session_state.options_df_long_call = []
if 'options_df_long_put' not in st.session_state:
    st.session_state.options_df_long_put = []
if 'L_options' not in st.session_state:
    st.session_state.L_options = []
if 'type_trades' not in st.session_state:
    st.session_state.type_trades = 'buy'  # Default trade type
if 'quantity' not in st.session_state:
    st.session_state.quantity = 1  # Default number of options
if 'type_option_cp' not in st.session_state:
    st.session_state.type_option_cp = 'call'  # Default option type
if 'type_eu_us' not in st.session_state:
    st.session_state.type_eu_us = 'EU'  # Default option style (European or American)
if 'strike' not in st.session_state:
    st.session_state.strike = 0.0  # Default strike price
if 'maturity' not in st.session_state:
    st.session_state.maturity = 0.0  # Default time to maturity (in years)
if 'r' not in st.session_state:
    st.session_state.r = 0.0  # Default risk-free rate
if 'tol' not in st.session_state:
    st.session_state.tol = 1e-2  # Default tolerance
if 'color_wanted' not in st.session_state:
    st.session_state.color_wanted = "#00f900"  # Default color
if 'L_stocks' not in st.session_state:
    st.session_state.L_stocks = []
if 'output' not in st.session_state:
    st.session_state.output = {}
if 'PNL_in_days' not in st.session_state:
    st.session_state.PNL_in_days = 1
if 'S_new' not in st.session_state:
    st.session_state.S_new = 0

if 'delta_wanted' not in st.session_state:
    st.session_state.delta_wanted = 0
if 'gamma_wanted' not in st.session_state:
    st.session_state.gamma_wanted = 0
if 'vega_wanted' not in st.session_state:
    st.session_state.vega_wanted = 0

if 'delta_table' not in st.session_state:
    st.session_state.delta_table = []
if 'gamma_table' not in st.session_state:
    st.session_state.gamma_table = []
if 'vega_table' not in st.session_state:
    st.session_state.vega_table = []

if 'L_stocks_hedging' not in st.session_state:
    st.session_state.L_stocks_hedging = []
if 'L_option_hedging' not in st.session_state:
    st.session_state.L_option_hedging = []
if 'output_hedging' not in st.session_state:
    st.session_state.output_hedging = []

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
if 'volatility_tables' not in st.session_state:
    st.session_state.volatility_tables = []
if 'volatility_surfaces' not in st.session_state:
    st.session_state.volatility_surfaces = []
if 'price_tables' not in st.session_state:
    st.session_state.price_tables = []
if 'price_surfaces' not in st.session_state:
    st.session_state.price_surfaces = []

with st.sidebar:
    st.markdown('<p class="params_text">OPTIONS PARAMETERS</p>', unsafe_allow_html=True)
    st.divider()
    
    with st.form(key='params_form'):
        st.session_state.stock_ticker = st.text_input("Stock", "NVDA")
        #st.selectbox(label="Tickers",options=sp500_tickers)
        st.session_state.type_trades = st.selectbox("Trade type", ['buy', 'sell'])
        st.session_state.quantity = st.number_input("Number of Options", value=1, step=1)
        st.session_state.type_option_cp = st.selectbox("Option's type", ['call', 'put', 'digital call','stock'])
        st.session_state.type_eu_us = st.selectbox("Option's type", ['EU', 'US'])
        st.session_state.strike = st.number_input("Strike", value=0.0, step=0.1)
        st.session_state.maturity = st.number_input("Time to Maturity (in Y)", value=0.0, step=0.1)
        st.session_state.r = st.number_input("Risk free rate", value=0.0, step=0.01)
        
        submitted = st.form_submit_button(label='Submit', use_container_width=True)
        clear = st.form_submit_button(label='🗑️', use_container_width=True)

if clear:
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

if submitted:
    st.session_state.S0, st.session_state.annual_volatility, st.session_state.mu, st.session_state.latest_price, st.session_state.sigma, st.session_state.rho = get_stock_price_and_volatility(st.session_state.stock_ticker)

    st.session_state.options_data_call = get_all_options(st.session_state.stock_ticker, ['call', round(st.session_state.latest_price), 0.05, 0.1])
    st.session_state.options_data_put = get_all_options(st.session_state.stock_ticker, ['put', round(st.session_state.latest_price), 0.05, 0.1])

    st.session_state.volatility_table_call = construct_volatility_table(st.session_state.options_data_call)
    st.session_state.volatility_table_put = construct_volatility_table(st.session_state.options_data_put)
    st.session_state.fitted_volatility_table_call = fit_volatility_surface(st.session_state.volatility_table_call, st.session_state.latest_price, 0.03)
    st.session_state.fitted_volatility_table_put = fit_volatility_surface(st.session_state.volatility_table_put,st.session_state. latest_price, 0.03)

    st.session_state.options_df_long_call = st.session_state.fitted_volatility_table_call.reset_index().melt(id_vars=['strike'], var_name='expiration', value_name='impliedVolatility')
    st.session_state.options_df_long_put = st.session_state.fitted_volatility_table_put.reset_index().melt(id_vars=['strike'], var_name='expiration', value_name='impliedVolatility')

    st.session_state.options_df_long_call['expiration'] = pd.to_datetime(st.session_state.options_df_long_call['expiration'])
    st.session_state.options_df_long_put['expiration'] = pd.to_datetime(st.session_state.options_df_long_put['expiration'])

    # Supprimer les valeurs NaN si nécessaire
    st.session_state.options_df_long_call = st.session_state.options_df_long_call.dropna(subset=['impliedVolatility'])
    st.session_state.options_df_long_put = st.session_state.options_df_long_put.dropna(subset=['impliedVolatility'])

    # Convertir expiration en jours avant expiration
    today = pd.to_datetime("today")
    st.session_state.options_df_long_call['days_to_expiration'] = (st.session_state.options_df_long_call['expiration'] - today).dt.days
    st.session_state.options_df_long_put['days_to_expiration'] = (st.session_state.options_df_long_put['expiration'] - today).dt.days


    if st.session_state.type_option_cp == 'stock':
        st.session_state.L_stocks.append({
            'weight':st.session_state.quantity * -1 if st.session_state.type_trades == 'sell' else st.session_state.quantity,
        })
    else:
        st.session_state.L_options.append({
            'type':st.session_state.type_option_cp,
            'strike':st.session_state.strike,
            'weight':st.session_state.quantity * -1 if st.session_state.type_trades == 'sell' else st.session_state.quantity,
            'T':st.session_state.maturity,
            'r':st.session_state.r,
        })

    st.session_state.delta_table,st.session_state.gamma_table,st.session_state.vega_table = generate_greeks_table(st.session_state.latest_price,'call',st.session_state.r)
    st.session_state.output = structured_product(st.session_state.latest_price,st.session_state.L_options,st.session_state.L_stocks)

    st.session_state.delta = st.session_state.output["delta"][int(st.session_state.latest_price)]
    st.session_state.gamma = st.session_state.output["gamma"][int(st.session_state.latest_price)]
    st.session_state.vega = st.session_state.output["vega"][int(st.session_state.latest_price)]
    st.session_state.theta = st.session_state.output["theta"][int(st.session_state.latest_price)]



# Main layout
# title_col, emp_col, equity_col, vol_col, date_col, price_chart_col = st.columns([1,0.2,1,1,1,2])
title_col, emp_col, equity_col, vol_col, delta_col, gamma_col, vega_col,theta_col = st.columns([1,0.2,1,1,1,1,1,1])
tab_greeks,tab_PNL,tab_hedging = st.tabs(['Greeks','PNL','Hedging'])
# Title column
with title_col:
    st.markdown('<p class="dashboard_title">Equity Derivatives<br>Dashboard</p>', unsafe_allow_html=True)



with tab_greeks:
    if len(st.session_state.output) != 0:
        plot_greeks(st.session_state.output,st.session_state.L_options,st.session_state.latest_price,st.session_state.L_stocks)
with tab_PNL:
    col_PNL_in_days,col_S_new = st.columns(2)
    with st.form(key="PNL"):
        with col_PNL_in_days:
            st.session_state.PNL_in_days = st.number_input('PNL in X days',min_value=0,value=1)
        with col_S_new:
            st.session_state.S_new = st.number_input(f'{st.session_state.stock_ticker} value',min_value=0.00,value=float(get_stock_price_and_volatility(st.session_state.stock_ticker)[3]),step=0.01)
        submit_pnl = st.form_submit_button(label='Compute PNL ', use_container_width=True)
    if submit_pnl:
        plot_pnl(st.session_state.latest_price,st.session_state.S_new,st.session_state.PNL_in_days,st.session_state.L_options,st.session_state.L_stocks)

with tab_hedging:
    col_delta,col_gamma,col_vega = st.columns(3)
    with st.form(key="Hedging"):
        with col_delta:
            st.session_state.delta_wanted = st.number_input("Delta wanted")
        with col_gamma:
            st.session_state.gamma_wanted = st.number_input("Gamma wanted")
        with col_vega:
            st.session_state.vega_wanted = st.text_input("Vega wanted",value='N/A')
            
        submit_hedging = st.form_submit_button(label='Compute Hedging ', use_container_width=True)
    if submit_hedging:
        if not st.session_state.vega_wanted == 'N/A':
           st.session_state.vega_wanted = float(st.session_state.vega_wanted) 
        st.session_state.L_option_hedging,st.session_state.L_stocks_hedging = hedging(int(st.session_state.latest_price),st.session_state.output,st.session_state.L_options,st.session_state.L_stocks,st.session_state.delta_wanted,st.session_state.gamma_wanted,st.session_state.vega_wanted,st.session_state.gamma_table,st.session_state.vega_table)
        st.session_state.output_hedging = structured_product(st.session_state.latest_price,st.session_state.L_option_hedging,st.session_state.L_stocks_hedging)
        plot_greeks(st.session_state.output_hedging,st.session_state.L_option_hedging,st.session_state.latest_price,st.session_state.L_stocks_hedging)
        st.session_state.delta = st.session_state.output_hedging["delta"][int(st.session_state.latest_price)]
        st.session_state.gamma = st.session_state.output_hedging["gamma"][int(st.session_state.latest_price)]
        st.session_state.vega = st.session_state.output_hedging["vega"][int(st.session_state.latest_price)]
        st.session_state.theta = st.session_state.output_hedging["theta"][int(st.session_state.latest_price)]

# Info columns
with equity_col:
    with st.container():
        nvidia_price = f"{get_stock_price_and_volatility(st.session_state.stock_ticker)[3]:.2f}"
        
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
            <p class="nvidia_text">{st.session_state.stock_ticker} Price</p>
            <p class="price_details">{nvidia_price}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

with vol_col:
    with st.container():
        nvidia_vol = f"{get_stock_price_and_volatility(st.session_state.stock_ticker)[1] * 100:.2f}%"
        
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
            <p class="vol_text">{st.session_state.stock_ticker} Annual Vol</p>
            <p class="price_details">{nvidia_vol}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)

with delta_col:
    with st.container():        
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
            <p class="delta_text">∆ Delta</p>
            <p class="price_details">{round(st.session_state.delta,2)}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)
with gamma_col:
    with st.container():        
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
            <p class="gamma_text">Γ Gamma</p>
            <p class="price_details">{round(st.session_state.gamma,2)}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)
with vega_col:
    with st.container():        
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
            <p class="vega_text">ν Vega</p>
            <p class="price_details">{round(st.session_state.vega,2) }</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)
with theta_col:
    with st.container():        
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
            <p class="delta_text">Θ Theta</p>
            <p class="price_details">{round(st.session_state.theta,2)}</p>
        </div>
        """
        
        # Affichage du contenu
        st.markdown(content, unsafe_allow_html=True)
# with st.expander('Volatility surfaces'):
#     col_call,col_put = st.columns(2)
#     with col_call:
#         if len(st.session_state.options_df_lonG_call) != 0:
#             st.dataframe(st.session_state.options_df_long_call)
#             st.plotly_chart(plot_volatility_surface(st.session_state.options_df_long_call),key='vol_surface_call')
#     with col_put:
#         if len(st.session_state.options_df_long_put) != 0:
#             st.dataframe(st.session_state.options_df_long_put)
#             st.plotly_chart(plot_volatility_surface(st.session_state.options_df_long_put),key='vol_surface_put')

