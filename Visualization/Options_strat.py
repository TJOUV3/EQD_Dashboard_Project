#region import

import pandas as pd
import matplotlib.pyplot as plt
#import openai
import os
import io
import base64
import numpy as np
import yfinance as yf
import QuantLib as ql
import streamlit as st
import plotly.graph_objs as go
from plotly.graph_objs import Surface
from datetime import datetime,timedelta
from scipy.interpolate import griddata

from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

from scipy.integrate import quad
from scipy.optimize import minimize

import mibian

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

def option_description(call_put, Strike, type_option, Maturity, buy_sell, nb_of_options, prenium):
    if buy_sell == 'buy':
        buy_sell = 'Long'
    else:
        buy_sell = 'Short'
    return str(buy_sell) + ' ' + str(nb_of_options) + ' ' + str(type_option) + ' ' + str(call_put) + ' ' +  str(Maturity) + 'Y' +  ', K = ' + str(Strike) + ' @ ' +str(round(prenium,2)) +'$'

def derivated_products(St, list_options):
    resulting_payoff = 0
    test = False
    if list_options:
        for sublist in list_options:
            if sublist:
                resulting_payoff = resulting_payoff + payoff(sublist[0], St, sublist[1], sublist[2], sublist[3], sublist[4], sublist[5], sublist[6])

    return resulting_payoff

#endregion

#region Stock Price, Option Price & Yield

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

    vol_table = data['Log Return'].rolling(window=252).std()*np.sqrt(252)

    sigma = (np.log(vol_table/vol_table.shift(1)).dropna()).std()*np.sqrt(252)
    
    return latest_price, annual_volatility, sigma

def risk_free_curve():
    yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    yeilds = np.array([4.70,4.69,4.63,4.42,4.32,4.26,4.18,4.20,4.25,4.30,4.58,4.47]).astype(float)/100
    #NSS model calibrate
    curve_fit, status = calibrate_nss_ols(yield_maturities,yeilds)

    return curve_fit

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
        #options_df = options_df.append(new_row_K, ignore_index=True)
        options_df = pd.concat([options_df, pd.DataFrame([new_row_K])], ignore_index=True)

    if t not in unique_times:
        new_row_t = options_df.iloc[0].copy()
        new_row_t['strike'] = K
        new_row_t['expiration'] = expiration_date.strftime('%Y-%m-%d')
        new_row_t['impliedVolatility'] = 0
        #options_df = options_df.append(new_row_t, ignore_index=True)
        options_df = pd.concat([options_df, pd.DataFrame([new_row_t])], ignore_index=True)

    options_df = options_df.sort_values(by=['strike', 'expiration'], ascending=[True, True]).reset_index(drop=True)

    return options_df[['strike', 'expiration', 'lastPrice', 'bid', 'ask', 'impliedVolatility']]

def create_volatility_table(options_df):
    # Conversion de la colonne 'expiration' en type datetime
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    
    # Création d'un tableau de volatilité avec strikes en index et expirations en colonnes
    volatility_table = options_df.pivot_table(
        index='strike',
        columns='expiration',
        values='impliedVolatility'
    )
    
    # Remplacer les zéros par NaN pour ne pas interférer avec l'interpolation
    volatility_table.replace(0, np.nan, inplace=True)

    # Extraire les indices (strikes et expirations) et les valeurs
    strikes = volatility_table.index.values
    expirations = volatility_table.columns.values

    # Préparer les points connus (non-NaN) pour l'interpolation
    points = []
    values = []

    for i, strike in enumerate(strikes):
        for j, expiration in enumerate(expirations):
            if not np.isnan(volatility_table.iat[i, j]):  # Ignorer les NaN
                points.append((strike, expiration))
                values.append(volatility_table.iat[i, j])

    # Vérifier si suffisamment de points sont disponibles pour l'interpolation
    if len(points) < 3:
        raise ValueError("Pas assez de points non-NaN pour l'interpolation.")
    
    # Créer une grille complète pour tous les points (strikes, expirations)
    grid_x, grid_y = np.meshgrid(strikes, expirations, indexing='ij')

    # Appliquer l'interpolation bilinéaire (avec méthode 'nearest' ou 'cubic' pour tester)
    try:
        interpolated_values = griddata(points, values, (grid_x, grid_y), method='linear')
    except Exception as e:
        #print(f"Erreur avec interpolation bilinéaire: {e}")
        #print("Essai avec méthode 'nearest'.")
        interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')

    # Créer un DataFrame avec les valeurs interpolées
    interpolated_table = pd.DataFrame(interpolated_values, index=strikes, columns=expirations)

    return interpolated_table

def create_price_table(options_df):
    # Conversion de la colonne 'expiration' en type datetime
    options_df['expiration'] = pd.to_datetime(options_df['expiration'])
    
    # Création d'un tableau de volatilité avec strikes en index et expirations en colonnes
    price_table = options_df.pivot_table(
        index='strike',
        columns='expiration',
        values='lastPrice'
    )
    
    # Remplacer les zéros par NaN pour ne pas interférer avec l'interpolation
    price_table.replace(0, np.nan, inplace=True)

    # Extraire les indices (strikes et expirations) et les valeurs
    strikes = price_table.index.values
    expirations = price_table.columns.values

    # Préparer les points connus (non-NaN) pour l'interpolation
    points = []
    values = []

    for i, strike in enumerate(strikes):
        for j, expiration in enumerate(expirations):
            if not np.isnan(price_table.iat[i, j]):  # Ignorer les NaN
                points.append((strike, expiration))
                values.append(price_table.iat[i, j])

    # Vérifier si suffisamment de points sont disponibles pour l'interpolation
    if len(points) < 3:
        raise ValueError("Pas assez de points non-NaN pour l'interpolation.")
    
    # Créer une grille complète pour tous les points (strikes, expirations)
    grid_x, grid_y = np.meshgrid(strikes, expirations, indexing='ij')

    # Appliquer l'interpolation bilinéaire (avec méthode 'nearest' ou 'cubic' pour tester)
    try:
        interpolated_values = griddata(points, values, (grid_x, grid_y), method='linear')
    except Exception as e:
        #print(f"Erreur avec interpolation bilinéaire: {e}")
        #print("Essai avec méthode 'nearest'.")
        interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')

    # Créer un DataFrame avec les valeurs interpolées
    interpolated_table = pd.DataFrame(interpolated_values, index=strikes, columns=expirations)

    return interpolated_table

def table_vol_to_dataframe(volatility_table):
    volSurfaceLong = volatility_table.melt(ignore_index=False).reset_index()
    volSurfaceLong.columns = ['strike','maturity', 'price']
    today = datetime.today()
    volSurfaceLong['years_to_maturity'] = volSurfaceLong['maturity'].apply(lambda x: (x - today).days / 365)

    # Calculate the risk free rate for each maturity using the fitted yield curve
    volSurfaceLong['rate'] = volSurfaceLong['years_to_maturity'].apply(risk_free_curve())

    return volSurfaceLong

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
    )

    # Create the figure
    fig = go.Figure(data= data, layout=layout)

    return fig


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

#region Heston

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):

    # constants
    a = kappa*theta
    b = kappa+lambd

    # common terms w.r.t phi
    rspi = rho*sigma*phi*1j

    # define d parameter given phi and b
    d = np.sqrt( (rho*sigma*phi*1j - b)**2 + (phi*1j+phi**2)*sigma**2 )

    # define g parameter given phi, b and d
    g = (b-rspi+d)/(b-rspi-d)

    # calculate characteristic function by components
    exp1 = np.exp(r*phi*1j*tau)
    term2 = S0**(phi*1j) * ( (1-g*np.exp(d*tau))/(1-g) )**(-2*a/sigma**2)
    exp2 = np.exp(a*tau*(b-rspi+d)/sigma**2 + v0*(b-rspi+d)*( (1-np.exp(d*tau))/(1-g*np.exp(d*tau)) )/sigma**2)

    return exp1*term2*exp2

def integrand(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K*heston_charfunc(phi,*args)
    denominator = 1j*phi*K**(1j*phi)
    return numerator/denominator

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi=umax/N #dphi is width

    for i in range(1,N):
        # rectangular integration
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_charfunc(phi-1j,*args) - K * heston_charfunc(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator

    return np.real((S0 - K*np.exp(-r*tau))/2 + P/np.pi)

def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r, K)

    real_integral, err = np.real(quad(integrand, 0, 100, args=args) )

    return (S0 - K*np.exp(-r*tau))/2 + real_integral/np.pi

def calibration_params(volSurfaceLong, latest_price, vol_histo, sigma):
    S0 = latest_price
    r = volSurfaceLong['rate'].to_numpy('float')
    K = volSurfaceLong['strike'].to_numpy('float')
    tau = volSurfaceLong['years_to_maturity'].to_numpy('float')
    P = volSurfaceLong['price'].to_numpy('float')

    
    params = {"v0": {"x0": np.sqrt(vol_histo), "lbub": [1e-3,1]},
          "kappa": {"x0": 3, "lbub": [1e-3,5]},
          "theta": {"x0": 0.05, "lbub": [1e-3,0.1]},
          "sigma": {"x0": 0.5, "lbub": [1e-2,1]},
          "rho": {"x0": -0.8, "lbub": [-1,0]},
          "lambd": {"x0": 0.03, "lbub": [-1,1]},
          }

    x0 = [param["x0"] for key, param in params.items()]
    bnds = [param["lbub"] for key, param in params.items()]

    return S0, r, K, tau, P, params, x0, bnds

def SqErr(x, S0, P, K, tau, r, x0):

    v0, kappa, theta, sigma, rho, lambd = [param for param in x]

    # Decided to use rectangular integration function in the end
    err = np.sum( (P-heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))**2 /len(P) )

    # Zero penalty term - no good guesses for parameters
    pen = np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] )

    return err + pen

@st.cache_data
def f_minimize(S0, P, K, tau, r, x0, tol, bnds):
    result = minimize(SqErr, x0, args=(S0, P, K, tau, r, x0), tol=1, method='SLSQP', options={'maxiter': 1000}, bounds=bnds)
    v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
    return v0, kappa, theta, sigma, rho, lambd

def add_result(volSurfaceLong, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    volSurfaceLong['heston_price'] = heston_prices
    return volSurfaceLong


def plot_mesh(data_volSurfaceLong, color_wanted):
    fig = go.Figure(data=[go.Mesh3d(x=data_volSurfaceLong.maturity, y=data_volSurfaceLong.strike, z=data_volSurfaceLong.price, color=color_wanted, opacity=0.9)])

    fig.add_scatter3d(x=data_volSurfaceLong.maturity, y=data_volSurfaceLong.strike, z=data_volSurfaceLong.heston_price, mode='markers', marker=dict(size=5, color='black'))

    fig.update_layout(
        title_text='Market Prices (Mesh) vs Calibrated Heston Prices (Markers)',
        scene = dict(xaxis_title='TIME (Years)',
                    yaxis_title='STRIKES (Pts)',
                    zaxis_title='INDEX OPTION PRICE (Pts)',
                    xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True),
                    yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True),
                    zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True)),
        height=800,
        width=800,
    )

    return fig

#endregion

#region Simulate functions
    
def Simulate_data_call_put(limit_inf, limit_sup, call_put, strike, prenium, type_option,maturity,buy_sell, nb_of_options):
    result = []
    descr = option_description(call_put, strike, type_option, maturity, buy_sell, nb_of_options, prenium)
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

def plot_simulation(df_simulation):
    fig = go.Figure()
    for col in df_simulation.columns:
        fig.add_trace(
            go.Scatter(
                x=df_simulation.index,
                y=df_simulation[col],
                mode='lines',
                name=f'Sim {col}',
                showlegend=False
            )
        )

    # Update layout
    fig.update_layout(
        title="Monte Carlo Stock Price Simulations",
        xaxis_title="Week",
        yaxis_title="Price",
        showlegend=False,
    )

    return fig

def plot_histogram(data, title, position):
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=data,
        histnorm='probability',
        nbinsx=30,
        name='Distribution'
    ))
    
    # Calculate statistics
    mean_val = data.mean()
    p5_val = np.percentile(data, 5)
    p95_val = np.percentile(data, 95)
    
    # Add vertical lines
    fig.add_vline(x=mean_val, line_dash="dash", line_color="black", line_width=2)
    fig.add_vline(x=p5_val, line_dash="dash", line_color="red", line_width=2)
    fig.add_vline(x=p95_val, line_dash="dash", line_color="blue", line_width=2)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="P&L",
        yaxis_title="Probability",
        showlegend=False,
        height=400,
        annotations=[
            # Add statistics text
            dict(
                x=np.percentile(data, 0 if position == 's' else 99),
                y=0.9,
                xref="x",
                yref="paper",
                text=f'P95: {p95_val:.2f}',
                showarrow=False,
                font=dict(color="blue")
            ),
            dict(
                x=np.percentile(data, 0 if position == 's' else 99),
                y=0.8,
                xref="x",
                yref="paper",
                text=f'Mean: {mean_val:.2f}',
                showarrow=False
            ),
            dict(
                x=np.percentile(data, 0 if position == 's' else 99),
                y=0.7,
                xref="x",
                yref="paper",
                text=f'P5: {p5_val:.2f}',
                showarrow=False,
                font=dict(color="red")
            )
        ]
    )
    
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
def delta(flag, S0, K, T, r, sigma):
    """Calculate option Greeks
        Returns: delta
    """
    # Convert T from years to days if necessary
    T_days = T * 365 if T < 10 else T
    
    # Convert rate and sigma to percentage
    r_percent = r * 100 if r < 1 else r
    sigma_percent = sigma * 100
    
    bs = mibian.BS([S0, K, r_percent, T_days], sigma_percent)
    
    greeks = {
        'delta': bs.callDelta if flag.lower() == 'c' else bs.putDelta,
    }
    
    return greeks['delta']


def calc_delta(flag, price, K, time, r, sigma, position='s'):
    if time == 0:
        return np.nan
    else:
        if position=='l':
            return int(delta(flag, price, K, time, r, sigma)*100)
        else:
            return -int(delta(flag, price, K, time, r, sigma)*100)

def adjust(delta, total):
    if delta < 0:
        return 'Buy {0}'.format(abs(delta))
    elif delta > 0:
        return 'Sell {0}'.format(abs(delta))
    elif delta == 0:
        return 'None'
    else:
        if total < 0:
            return 'Sell {0}'.format(abs(total))
        elif total > 0:
            return 'Buy {0}'.format(abs(total))
        else:
            return np.nan

def totalAdj(counter,time):
    if time > 0:
        if counter < 0:
            return 'Long {0}'.format(abs(counter))
        elif counter > 0:
            return 'Short {0}'.format(abs(counter))
        else:
            return np.nan
    else:
            return np.nan

def cashAdj(delta, price, time, total):
    if time > 0:
        return delta*price
    else:
        return -total*price


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

    price_stocks, volatility_stocks, sigma_stock = get_stock_price_and_volatility(ticker)

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
        #st.selectbox(label="Tickers",options=sp500_tickers)
        type_trades = st.selectbox("Trade type", ['buy', 'sell'])
        quantity = st.number_input("Number of Options", value=1, step=1)
        type_option_cp = st.selectbox("Option's type", ['call', 'put', 'digital call'])
        type_eu_us = st.selectbox("Option's type", ['EU', 'US'])
        strike = st.number_input("Strike", value=0.0, step=0.1)
        maturity = st.number_input("Time to Maturity (in Y)", value=0.0, step=0.1)
        r = st.number_input("Risk free rate", value=0.0, step=0.01)

        tol = st.select_slider("Select a tolerance",options=[1e-2,1e-1,1])

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
if 'volatility_tables' not in st.session_state:
    st.session_state.volatility_tables = []
if 'volatility_surfaces' not in st.session_state:
    st.session_state.volatility_surfaces = []
if 'price_tables' not in st.session_state:
    st.session_state.price_tables = []
if 'price_surfaces' not in st.session_state:
    st.session_state.price_surfaces = []

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

# Form submission logic
if submitted:
        st.session_state.stock_data = get_historical_data(stock_ticker)
        days = int(maturity * 365)
        today = datetime.now()
        expiration_date = today + timedelta(days=days)
        price_stock, vol_stock, sigma_stock = get_stock_price_and_volatility(stock_ticker, period='1y')
        try:
            # Fetch options data
            options_data = get_all_options(stock_ticker, [type_option_cp, strike, r, maturity])
            volatility_table = create_volatility_table(options_data)
            price_table = create_price_table(options_data)
            expiration_date = pd.to_datetime(expiration_date).strftime('%Y-%m-%d')
            implied_vol = volatility_table.loc[strike, expiration_date]
            volSurfaceLong = table_vol_to_dataframe(price_table)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            implied_vol = vol_stock

        option1_params = [type_option_cp, strike, r, maturity, implied_vol, type_trades, quantity]
        st.session_state.L_options_2.append(option1_params)

        S0, r, K, tau, P, params, x0, bnds = calibration_params(volSurfaceLong, price_stock, vol_stock, sigma_stock)
        v0, kappa, theta, sigma, rho, lambd = f_minimize(S0, P, K, tau, r, x0, tol, bnds)
        volSurfaceLong2 = add_result(volSurfaceLong, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)

        option_prenium = calculate_option_price(option1_params, price_stock)
        result_option1, descr_option1 = Simulate_data_call_put(
            Get_parameters(stock_ticker)['lim_inf'],
            Get_parameters(stock_ticker)['lim_sup'],
            type_option_cp, strike, option_prenium, type_eu_us, maturity, type_trades, quantity
        )
        option_id = option_description(type_option_cp, strike, type_eu_us, maturity, type_trades, quantity, option_prenium)
        st.session_state.volatility_tables.append(volatility_table)
        fig_vol_surface = plot_volatility_surface(volatility_table, strike, maturity)
        st.session_state.volatility_surfaces.append(fig_vol_surface)

        st.session_state.price_tables.append(price_table)
        fig_price_surface = plot_mesh(volSurfaceLong2, color_wanted)
        st.session_state.price_surfaces.append(fig_price_surface)

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
            ['Profit of the Option', 'Stock Price (St)', 'Profit'], st.session_state.L_color
        )


    # Display current cart
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

with st.container(border=True):
    st.subheader('Portfolio')

    if ('L_descr_options' in st.session_state and 
        'L_color' in st.session_state and 
        st.session_state.L_descr_options and 
        len(st.session_state.L_descr_options) == len(st.session_state.L_color)):
        
        for i, (opt, color) in enumerate(zip(st.session_state.L_descr_options, st.session_state.L_color)):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                option_content = f"""
                <div class="option-container" style="
                    border: 2px solid {color};
                    margin-bottom: 15px;
                    padding: 15px;
                    border-radius: 10px;
                    box-shadow: -6px 8px 20px 1px #00000052;
                ">
                    <p style="margin: 0;">{opt}</p>
                </div>
                """       
                st.markdown(option_content, unsafe_allow_html=True)
            
            with col2:
                if st.button("Delete", key=f"del_{i}"):
                    st.session_state.L_descr_options.pop(i)
                    st.session_state.L_options_2.pop(i)
                    st.session_state.L_color.pop(i)
                    st.rerun()
    else:
        st.warning("No options")

    st.write('<span class="custom-frame"/>', unsafe_allow_html=True)
    
tab_payoff, tab_delta, tab_gamma, tab_theta, tab_vega, tab_rho, tab_option_price, tab_delta_hedging = st.tabs(['Payoff','Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'Option Price','Delta Hedging'])

with tab_payoff:
    if 'payoff' in st.session_state.plots:
            fig_payoff = st.session_state.plots['payoff']
            fig_payoff.update_layout(height=500)
            st.plotly_chart(st.session_state.plots['payoff'], use_container_width=True)
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
with tab_delta_hedging:
    st.write("Delta hedging")
    Simulate_delta_hedging = st.button("Simulate Delta Hedging")
    M = st.slider("Number of simulation",500,2000,1000)

    if Simulate_delta_hedging:


        Dynamic_Hedging_Results = pd.DataFrame(data=[], columns=[], index=['Original Option P&L','Original Stock P&L','Adjustment P&L', \
                                                                                    'Carry (interest) on options', 'Carry (interest) on stock', \
                                                                                    'Interest on Adjustments'])
        Dynamic_Hedging_Results.index.name = 'Dynamic hedging results'
        
        """
            chopper le bon prix de l'action et faire un spread fantome
        
            call_bid,call_ask,put_bid,put_ask = 16.4,16.9,15.8,16.1

            to the right value in yfinance table
        """

        call_bid,call_ask,put_bid,put_ask = 14.0,16.28,15,15.1
        # Parameters  

        N = int(round(st.session_state.L_options_2[0][3] * 52,0))
        sigma = 0.3
        S0 = float(nvidia_price)
        DTE = 50
        T = DTE/365
        r = st.session_state.L_options_2[0][2]
        DT = T/N
        TTE = [DT*N-DT*i for i in range(0,N+1)]


        # Realized Volatility
        sigma = st.session_state.L_options_2[0][4]

        # Position in Option contract
        k = st.session_state.L_options_2[0][1]
        K = k
        position_map = {
            "buy":"b",
            "sell":"s"
        }

        position = position_map[st.session_state.L_options_2[0][5]]
        flag_map = {
            "call":"c",
            "put":"p"
        }
        
        flag = flag_map[st.session_state.L_options_2[0][0]]

        nudt = (r - 0.5*sigma**2) * DT
        sigmasdt = sigma*np.sqrt(DT)

        no_hedge = []
        static_hedge = []

        # number of sims

        St = S0
        St_series = [np.array([St for m in range(M)])]
        for i in range(N):
            St = St_series[-1]
            Stn = np.round( St * np.exp(nudt + sigmasdt*np.random.normal(0,1,M)) , 2)
            St_series.append(Stn)

        St_series = np.array(St_series)

        df_simulation = pd.DataFrame(St_series, columns = [i for i in range(M)])
        df_simulation.index.name = 'Week'
        
        with st.expander("Show simulations results"):
            st.plotly_chart(plot_simulation(df_simulation), use_container_width=True)
            st.dataframe(df_simulation)
        df_simulation.insert(0, "Time", np.round(TTE,2))

        progress_text = "Simulation in progress. Please wait."
        progress_delta_hedging = st.progress(0, text=progress_text)
        for sim in range(M):
            percent_complete = (sim + 1) / M
            
            # Update progress bar with more detailed text
            progress_delta_hedging.progress(
                percent_complete, 
                text=f"Running simulation {sim+1} of {M} ({(percent_complete*100):.0f}%)"
            )
            hedgeSim = df_simulation.loc[:,['Time',sim]]
            hedgeSim.columns = ['Time', 'Price']

            # hedge calcs
            hedgeSim['delta'] = hedgeSim.apply(lambda x: calc_delta(flag, x['Price'], K, x['Time'], r, sigma, position), axis=1)
            hedgeSim['Total Delta Position'] = (hedgeSim.delta - hedgeSim.delta.shift(1))
            totaladjust_c = [hedgeSim['Total Delta Position'][:i].sum() for i in range(1,N+1)]
            hedgeSim['totaladjust_c'] = [hedgeSim['Total Delta Position'][:i].sum() for i in range(1,N+2)]
            hedgeSim['Adjustment Contracts'] = hedgeSim.apply(lambda x: adjust(x['Total Delta Position'], x['totaladjust_c']), axis=1)
            hedgeSim['Total Adjustment'] = hedgeSim.apply(lambda x: totalAdj(x['totaladjust_c'],x['Time']), axis=1)
            hedgeSim['totaladjust_c'] = [hedgeSim['Total Delta Position'][:i].sum() for i in range(1,N+2)]
            hedgeSim['Adjustment Cashflow'] = hedgeSim.apply(lambda x: cashAdj(x['Total Delta Position'],x['Price'],x['Time'], x['totaladjust_c']), axis=1)
            hedgeSim['Interest on Adjustments'] = hedgeSim.apply(lambda x: round(x['Adjustment Cashflow']*r*x['Time'],2), axis=1)
            hedgeSim = hedgeSim.drop(columns=['totaladjust_c'])

            # calculate payoffs
            if flag == 'c':
                if position == 's':
                    optprice = call_bid
                    option_pnl = 100*(optprice - np.maximum(hedgeSim.loc[11,'Price']-K,0))
                    # delta will be negative if short
                    stock_pnl = hedgeSim.loc[0,'delta']*(S0 - hedgeSim.loc[11,'Price'])
                    adj_pnl = hedgeSim['Adjustment Cashflow'].sum()
                    option_carry = 100*optprice*r*T
                    # delta will be negative if short
                    stock_carry = hedgeSim.loc[0,'delta']*S0*r*T
                    int_adj_pnl = hedgeSim['Interest on Adjustments'].sum()
                else:
                    optprice = call_ask
                    option_pnl = 100*(np.maximum(hedgeSim.loc[11,'Price']-K,0) - optprice)
                    # delta will be positive if long
                    stock_pnl = hedgeSim.loc[0,'delta']*(S0 - hedgeSim.loc[11,'Price'])
                    adj_pnl = hedgeSim['Adjustment Cashflow'].sum()
                    option_carry = -100*optprice*r*T
                    # delta will be positive if long
                    stock_carry = hedgeSim.loc[0,'delta']*S0*r*T
                    int_adj_pnl = hedgeSim['Interest on Adjustments'].sum()

            elif flag == 'p':
                if position == 's':
                    optprice = put_bid
                    option_pnl = 100*(optprice - np.maximum(K-hedgeSim.loc[11,'Price'],0))
                    # delta will be positive if short
                    stock_pnl = hedgeSim.loc[0,'delta']*(S0 - hedgeSim.loc[11,'Price'])
                    adj_pnl = hedgeSim['Adjustment Cashflow'].sum()
                    option_carry = 100*optprice*r*T
                    # delta will be positive if short
                    stock_carry = hedgeSim.loc[0,'delta']*S0*r*T
                    int_adj_pnl = hedgeSim['Interest on Adjustments'].sum()
                else:
                    optprice = put_ask
                    option_pnl = 100*(np.maximum(K-hedgeSim.loc[11,'Price'],0) - optprice)
                    # delta will be negative if long
                    stock_pnl = hedgeSim.loc[0,'delta']*(S0 - hedgeSim.loc[11,'Price'])
                    adj_pnl = hedgeSim['Adjustment Cashflow'].sum()
                    option_carry = -100*optprice*r*T
                    # delta will be negative if long
                    stock_carry = hedgeSim.loc[0,'delta']*S0*r*T
                    int_adj_pnl = hedgeSim['Interest on Adjustments'].sum()

            data=[option_pnl,stock_pnl,adj_pnl,option_carry,stock_carry,int_adj_pnl]

            #add to dataframe
            Dynamic_sim = pd.DataFrame(data=data, columns=[sim], index=['Original Option P&L','Original Stock P&L','Adjustment P&L', \
                                                                                    'Carry (interest) on options', 'Carry (interest) on stock', \
                                                                                    'Interest on Adjustments'])
            Dynamic_Hedging_Results[sim] = Dynamic_sim[sim]
            no_hedge.append(option_pnl+option_carry)
            static_hedge.append(option_pnl+option_carry+stock_pnl+stock_carry)

        progress_delta_hedging.empty()
        st.success(f"Simulation completed! {M} iterations processed.")
        Dynamic_Hedging_Results = pd.concat([
            Dynamic_Hedging_Results, 
            pd.DataFrame(Dynamic_Hedging_Results.sum(axis=0)).T.rename(index={0: 'TOTAL CASHFLOW'})
        ])            
        st.dataframe(Dynamic_Hedging_Results)

        col_dynamic,col_static,col_no_hedging = st.columns([1,1,1])
        with col_dynamic:
            x = Dynamic_Hedging_Results.loc['TOTAL CASHFLOW',]
            fig1 = plot_histogram(x, 'Dynamic Delta Hedging', position)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("**Dynamic Hedging**")
            dynamic_x = Dynamic_Hedging_Results.loc['TOTAL CASHFLOW',]
            st.write(f"Mean: {dynamic_x.mean():.2f}")
            st.write(f"P5: {np.percentile(dynamic_x, 5):.2f}")
            st.write(f"P95: {np.percentile(dynamic_x, 95):.2f}")

        # Static Hedging Plot
        with col_static:
            x = np.array(static_hedge)
            fig2 = plot_histogram(x, 'Static Delta Hedging', position)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("**Static Hedging**")
            static_x = np.array(static_hedge)
            st.write(f"Mean: {static_x.mean():.2f}")
            st.write(f"P5: {np.percentile(static_x, 5):.2f}")
            st.write(f"P95: {np.percentile(static_x, 95):.2f}")
        # No Hedging Plot
        with col_no_hedging:
            x = np.array(no_hedge)
            fig3 = plot_histogram(x, 'No Delta Hedging', position)
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("**No Hedging**")
            no_hedge_x = np.array(no_hedge)
            st.write(f"Mean: {no_hedge_x.mean():.2f}")
            st.write(f"P5: {np.percentile(no_hedge_x, 5):.2f}")
            st.write(f"P95: {np.percentile(no_hedge_x, 95):.2f}")

    
if st.session_state.L_descr_options:
    with st.expander("Raw Volatility Data", expanded=True):

        # Création des onglets pour chaque description d'option
        vol_tabs = st.tabs(st.session_state.L_descr_options)

        # Affichage de chaque table et surface de volatilité dans l'onglet correspondant
        for i, descr in enumerate(st.session_state.L_descr_options):
            with vol_tabs[i]:
                # Vérification que la table et la surface existent pour cette option
                if i < len(st.session_state.volatility_tables) and i < len(st.session_state.volatility_surfaces):
                    # Affichage de la table de volatilité
                    st.write(f"Volatility Table for {descr}")
                    st.dataframe(st.session_state.volatility_tables[i])

                    # Affichage de la surface de volatilité
                    st.write(f"Volatility Surface for {descr}")
                    st.plotly_chart(st.session_state.volatility_surfaces[i], use_container_width=True)
                else:
                    st.warning(f"No volatility data available for {descr}")
            with vol_tabs[i]:
                # Vérification que la table et la surface existent pour cette option
                if i < len(st.session_state.price_tables) and i < len(st.session_state.price_surfaces):
                    # Affichage de la table de volatilité
                    st.write(f"Price Table for {descr}")
                    st.dataframe(st.session_state.price_tables[i])

                    # Affichage de la surface de volatilité
                    st.write(f"Price Surface for {descr}")
                    st.plotly_chart(st.session_state.price_surfaces[i], use_container_width=True)
                else:
                    st.warning(f"No price data available for {descr}")

if clear:
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []
    st.session_state.L_options_2 = []
    st.session_state.L_options = []
    st.session_state.L_descr_options = []
    st.session_state.L_color = []
    st.session_state.plots = {}
    if 'stock_data' in st.session_state:
        del st.session_state.stock_data

#endregion