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

# Spécifiez le ticker pour lequel vous voulez obtenir les options
ticker = "NVDA"

import yfinance as yf
import pandas as pd

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

# Example usage
options_data = get_all_options("NVDA", ['call', 130, 0.05, 0.1])
volatility_table = create_volatility_table(options_data)
print(volatility_table)
print(volatility_table.head())
print(volatility_table.index)  # Should be strike prices
print('col :', volatility_table.columns)  # Should be expiration dates
print(volatility_table.isnull().sum())  # Check for any remaining NaN values

'''
def get_options_data(ticker, option_params):
    stock = yf.Ticker(ticker)
    option_type = option_params[0]  # 'call' ou 'put'
    K = option_params[1]  # Strike price
    r = option_params[2]  # Taux sans risque
    t = option_params[3]  # Temps jusqu'à l'échéance (en années)

    # Convertir le temps en jours
    days_until_maturity = int(t * 365)
    print(f"Jours jusqu'à l'échéance: {days_until_maturity}")

    # Calculer la date d'échéance souhaitée
    expiration_wanted = datetime.today() + timedelta(days=days_until_maturity)
    expiration_wanted_str = expiration_wanted.strftime('%Y-%m-%d')
    print('Date d’échéance désirée:', expiration_wanted_str)

    # Obtenir les dates d'expiration disponibles pour les options
    expirations = stock.options
    print("Dates d'expiration disponibles :", expirations)

    # Sélectionner la date d'expiration la plus proche
    closest_expiration = min(expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - expiration_wanted))
    print(f"Date d'expiration la plus proche: {closest_expiration}")

    # Obtenir la chaîne d'options pour cette date d'expiration
    options_chain = stock.option_chain(closest_expiration)

    # Sélectionner les options en fonction du type (call ou put)
    if option_type == 'call':
        res_tab = options_chain.calls
    elif option_type == 'put':
        res_tab = options_chain.puts
    else:
        return 'No data available for this type of option'

    # Filtrer les options selon le strike price souhaité
    filtered_options = res_tab[res_tab['strike'] == K]
    
    if filtered_options.empty:
        return f"No options found for strike price {K} on {closest_expiration}"
    
    return filtered_options

# Exemple d'utilisation
print(get_options_data("NVDA", ['call', 130, 0.05, 0.1]))
'''
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
