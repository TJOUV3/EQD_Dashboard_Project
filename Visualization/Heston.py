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
from openbb import obb


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

# Spécifiez le ticker pour lequel vous voulez obtenir les options
ticker = "NVDA"

def get_options_data(ticker, option_params):

    stock = yf.Ticker(ticker)
    option_type = option_params[0] 
    K = option_params[1]  # Strike price
    r = option_params[2]  # Risk-free rate
    t = option_params[3]  # Time to maturity (in years)

    # Obtenez les dates d'expiration disponibles pour les options
    expirations = stock.options
    print("Dates d'expiration disponibles :", expirations)

    # Choisissez une date d'expiration spécifique
    expiration_date = expirations[-1]  # Sélection de la dernière date disponible
    if option_type == 'call':
        res_tab = options_chain.calls(expiration_date)
    elif option_type == 'put':
        res_tab = options_chain.puts(expiration_date)
    else :
        return 'no data available for this type of option'


    # Obtenez la chaîne d'options pour cette date d'expiration
    options_chain = stock.option_chain(expiration_date)

    return res_tab

# Affichez les données des options d'achat (calls) et de vente (puts)
print("Options Call:")
print(options_chain.calls.head(20))
print("\nOptions Put:")
print(options_chain.puts.head(20))

exit()
#endregion


#region Heston


def simulate_heston_trajectory(params):
    # Initialisation des trajectoires pour le prix et la variance
    S = np.zeros(params['N'] + 1)
    v = np.zeros(params['N'] + 1)
    
    # Conditions initiales
    S[0] = params['S0']
    v[0] = params['v0']
    
    # Simulation des trajectoires
    for t in range(1, params['N'] + 1):
        # Génération de deux variables gaussiennes corrélées pour W_t^S et W_t^v
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        W_S = Z1
        W_v = params['rho'] * Z1 + np.sqrt(1 - params['rho'] ** 2) * Z2
        
        # Mise à jour de la variance (méthode d'Euler)
        v[t] = (v[t - 1] + params['kappa'] * (params['theta'] - v[t - 1]) * params['dt'] +
                params['sigma'] * np.sqrt(max(v[t - 1], 0)) * np.sqrt(params['dt']) * W_v)
        
        # Assurer que la variance reste positive
        v[t] = max(v[t], 0)
        print(v[t])
        
        # Mise à jour du prix de l'actif (méthode d'Euler)
        S[t] = S[t - 1] * np.exp((params['mu'] - 0.5 * v[t - 1]) * params['dt'] +
                                  np.sqrt(v[t - 1] * params['dt']) * W_S)
    
    return S, v

# Fonction de coût pour la calibration
def heston_calibration_cost(params, S0, v0, mu, observed_volatility, T=1, N=252):
    kappa, theta, sigma, rho = params
    heston_params = {
        'S0': S0, 'v0': v0, 'mu': mu, 'kappa': kappa, 'theta': theta, 'sigma': sigma, 
        'rho': rho, 'T': T, 'N': N, 'dt': T / N
    }
    
    # Simulation des trajectoires avec les paramètres actuels
    S_sim, v_sim = simulate_heston_trajectory(heston_params)
    
    # Calcul de la volatilité simulée
    simulated_volatility = np.std(np.diff(np.log(S_sim))) * np.sqrt(252)
    
    # Calcul de l'erreur entre la volatilité simulée et observée
    error = (simulated_volatility - observed_volatility) ** 2
    return error

# Paramètres initiaux et optimisation
ticker = 'NVDA'
T = 1
S0, v0, mu, observed_volatility, N, prices_observed, latest_price, sigma, rho = get_stock_price_and_volatility(ticker, period='1y')
print('vol obs :', observed_volatility)

# Valeurs initiales des paramètres pour l'optimisation
initial_params = [2.0, v0, sigma, rho]

cost_history = []

def wrapper_cost_function(params):
    cost = heston_calibration_cost(params, S0, v0, mu, observed_volatility, T, N)
    cost_history.append(cost)  # Enregistrer le coût pour chaque itération
    return cost

# Exécution de l'optimisation
result = minimize(wrapper_cost_function, initial_params, method='Nelder-Mead', options={
    'disp': True,            # Affiche les informations de convergence
    'maxiter': 1000,        # Nombre maximum d'itérations
    'gtol': 1e-4            # Tolérance pour le gradient
})

if result.success:
    print("Optimisation réussie avec Nelder-Mead !")
else:
    print("Optimisation échouée avec Nelder-Mead:", result.message)

print("Paramètres optimisés :", result.x)
print("Valeur de la fonction de coût à l'optimum :", result.fun)
'''
result = minimize(wrapper_cost_function, initial_params, method='BFGS', options={
    'disp': True,            # Affiche les informations de convergence
    'maxiter': 10000,        # Nombre maximum d'itérations
    'gtol': 1e-4            # Tolérance pour le gradient
})

if result.success:
    print("Optimisation réussie avec BFSG !")
else:
    print("Optimisation échouée avec BFSG:", result.message)

print("Paramètres optimisés :", result.x)
print("Valeur de la fonction de coût à l'optimum :", result.fun)
'''
# Affichage des résultats
optimized_params = result.x

plt.plot(cost_history)  # cost_history est une liste des valeurs de la fonction de coût
plt.xlabel('Itérations')
plt.ylabel('Fonction de Coût')
plt.title('Évolution de la Fonction de Coût pendant la Minimisation')
plt.show()

# Résultats optimisés
kappa_opt, theta_opt, sigma_opt, rho_opt = optimized_params
print("Paramètres optimisés :", optimized_params)
# [ 1.02562661e+00  1.01548188e-03  7.65983185e-01 -7.65939699e-01] --> opti : 2.7998892407084842e-08

heston_params = {
    'S0': S0, 
    'v0': v0,  
    'mu': mu,  
    'kappa': kappa_opt,  
    'theta': theta_opt,  
    'sigma': sigma_opt,  
    'rho': rho_opt,  
    'T': T,
    'N': N,
    'dt': T / N
}

# Simulation des prix avec les paramètres optimisés
S_sim, _ = simulate_heston_trajectory(heston_params)
# Calcul de l'erreur quadratique moyenne (RMSE) entre les prix observés et simulés
rmse = np.sqrt(mean_squared_error(prices_observed, S_sim))
print(f"Erreur quadratique moyenne (RMSE) entre les prix observés et simulés : {rmse:.2f}")

print('len N : ', N)
print('len S_sim : ', len(S_sim))

# Affichage du graphique
plt.figure(figsize=(12, 6))
plt.plot(prices_observed.index, prices_observed, label='Prix Observés', color='blue')
plt.plot(prices_observed.index, S_sim, label='Prix Simulés (Heston)', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Prix de l\'Actif')
plt.title(f'Comparaison des Prix Observés et Simulés avec le Modèle de Heston ({ticker})')
plt.legend()
plt.show()
