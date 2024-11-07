
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

from Screeners.Stocks_Screener import get_sp500_tickers

st.title("Main")

sp500_tickers = get_sp500_tickers().index.tolist()
default_ix = sp500_tickers.index("NVDA")

value = st.selectbox(label="Tickers",options=sp500_tickers,index=default_ix)
st.success(value)
