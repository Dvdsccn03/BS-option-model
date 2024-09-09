import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap


# Defining functions
# Call and Put functions
def call(S, K, T, r, sigma, q=0):
    ''' Calcola il prezzo di un call con il metodo di Black-Scholes '''

    # calcolo dei parametri d1 e d2
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # calcolo del prezzo di un call con il metodo di Black-Scholes
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def put(S, K, T, r, sigma, q=0):
    ''' Calcola il prezzo di una put con il metodo di Black-Scholes '''

    # calcolo dei parametri d1 e d2
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    # calcolo del prezzo di una put con il metodo di Black-Scholes
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)



# IV call and put functions
def impliedVol_call(p, S, K, T, r, q=0, max_iter=1000, tol=1e-6):
    """ Calcola la volatilità implicita di un'opzione call utilizzando il metodo di Newton-Raphson """
    
    # Funzione per calcolare il prezzo dell'opzione call
    def PriceDifference(sigma):
        return call(S, K, T, r, sigma, q) - p
    
    # Derivata della funzione di Black-Scholes rispetto a sigma
    def vega(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    # Implementazione del metodo di Newton-Raphson
    sigma_guess = 0.3  # Valore iniziale per la volatilità
    for i in range(max_iter):
        price_diff = PriceDifference(sigma_guess)
        if abs(price_diff) < tol:
            return sigma_guess
        else:
            # Calcolo della nuova stima di volatilità
            sigma_guess -= price_diff / vega(sigma_guess)
    
    # Se non converge, restituisci None
    return None


def impliedVol_put(p, S, K, T, r, q=0, max_iter=1000, tol=1e-6):
    """ Calcola la volatilità implicita di un'opzione put utilizzando il metodo di Newton-Raphson """
    
    # Funzione per calcolare il prezzo dell'opzione put
    def PriceDifference(sigma):
        return put(S, K, T, r, sigma, q) - p
    
    # Derivata della funzione di Black-Scholes rispetto a sigma
    def vega(sigma):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    
    # Implementazione del metodo di Newton-Raphson
    sigma_guess = 0.3  # Valore iniziale per la volatilità
    for i in range(max_iter):
        price_diff = PriceDifference(sigma_guess)
        if abs(price_diff) < tol:
            return sigma_guess
        else:
            # Calcolo della nuova stima di volatilità
            sigma_guess -= price_diff / vega(sigma_guess)
    
    # Se non converge, restituisci None
    return None



# Greeks functions
# delta: sensitivity of the option's price to changes in the price of the underlying asset
def delta_call(S, K, T, r, sigma, q=0):
    ''' Calcola il delta di un call '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.cdf(d1)

def delta_put(S, K, T, r, sigma, q=0):
    ''' Calcola il delta di una put '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * -(norm.cdf(-d1))

# gamma: rate of change of Delta with respect to changes in the underlying asset's price
def gamma_calc(S, K, T, r, sigma, q=0):
    ''' Calcola il gamma di un call o una put '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return np.exp(-q*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

# vega: sensitivity of the option's price to changes in the volatility of the underlying asset
def vega_calc(S, K, T, r, sigma, q=0):
    ''' Calcola il vega di un call o una put '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return (S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T))/100

# theta: sensitivity of the option's price to the passage of time, also known as time decay
def theta_call(S, K, T, r, sigma, q=0):
    ''' Calcola il theta di un call '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma * np.exp(-q*T)) / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q*T) * norm.cdf(d1)
    term3 = r * K * np.exp(-r*T) * norm.cdf(d2)
    return (term1 - term2 - term3)/365

def theta_put(S, K, T, r, sigma, q=0):
    ''' Calcola il theta di una put '''
    d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    term1 = - (S * norm.pdf(d1) * sigma * np.exp(-q*T)) / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q*T) * norm.cdf(-d1)
    term3 = r * K * np.exp(-r*T) * norm.cdf(-d2)
    return (term1 + term2 + term3)/365

# rho: sensitivity of the option's price to changes in the risk-free interest rate
def rho_call(S, K, T, r, sigma, q=0):
    ''' Calcola il rho di un call '''
    d2 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T)) - sigma*np.sqrt(T)
    return (K * T * np.exp(-r*T) * norm.cdf(d2))/100

def rho_put(S, K, T, r, sigma, q=0):
    ''' Calcola il rho di una put '''
    d2 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T)) - sigma*np.sqrt(T)
    return (-K * T * np.exp(-r*T) * norm.cdf(-d2))/100








# Title and LinkedIn
col_i, col_t = st.columns([3,1])
with col_i: st.header('Black-Scholes Option Pricing Model')
with col_t: st.markdown("""Created by 
    <a href="https://www.linkedin.com/in/davide-saccone/" target="_blank">
        <button style="background-color: #262730; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
            Davide Saccone
        </button>
    </a>
    """, unsafe_allow_html=True)



# Sidebar parameters
st.sidebar.header('Input BS parameters')
S = st.sidebar.number_input('Stock Price (S)', min_value=0.0, value=100.0, step=0.01)
K = st.sidebar.number_input('Strike Price (k)', min_value=0.0, value=120.0, step=0.01)
exp = st.sidebar.date_input('Expiry Date', value=dt.datetime(2025,9,19))
exp = dt.datetime.combine(exp, dt.datetime.min.time())
T = (exp - dt.datetime.today()).days / 365
r = st.sidebar.number_input('Risk Free Rate (r) in decimal', min_value=0.0, value=0.02, step=0.01)
sigma = st.sidebar.number_input('Volatility (sigma) in decimal', min_value=0.0, value=0.2, step=0.01)
q = st.sidebar.number_input('Annual dividend yield (q)', min_value=0.0, value=0.0, step=0.01)


# Get functions results
callPrice = call(S, K, T, r, sigma, q)
putPrice = put(S, K, T, r, sigma, q)

deltaCall = delta_call(S, K, T, r, sigma)
deltaPut = delta_put(S, K, T, r, sigma)
gamma = gamma_calc(S, K, T, r, sigma)
vega = vega_calc(S, K, T, r, sigma)
thetaCall = theta_call(S, K, T, r, sigma)
thetaPut = theta_put(S, K, T, r, sigma)
rhoCall = rho_call(S, K, T, r, sigma)
rhoPut = rho_put(S, K, T, r, sigma)


# Display results
st.write("")
col1, col2 = st.columns(2)
with col1: st.metric(label='BS Call option price', value=f"${callPrice:.2f}")
with col2: st.metric(label='BS Put option price', value=f"${putPrice:.2f}")

with col1:
    with st.expander("Call greeks"):
        st.write(f"**Delta:** {deltaCall:.3f}")
        st.write(f"**Gamma:** {gamma:.3f}")
        st.write(f"**Vega:** {vega:.3f}")
        st.write(f"**Theta:** {thetaCall:.3f}")
        st.write(f"**Rho:** {rhoCall:.3f}")

with col2:
    with st.expander("Put greeks"):
        st.write(f"**Delta:** {deltaPut:.3f}")
        st.write(f"**Gamma:** {gamma:.3f}")
        st.write(f"**Vega:** {vega:.3f}")
        st.write(f"**Theta:** {thetaPut:.3f}")
        st.write(f"**Rho:** {rhoPut:.3f}")



# Create heatmaps for call and put option prices
S_values = np.linspace(S * 1.5, S * 0.5, 9)
sigma_values = np.linspace(0.1, 0.5, 9)
call_prices = np.zeros((len(S_values), len(sigma_values)))
put_prices = np.zeros((len(S_values), len(sigma_values)))

for i, S_val in enumerate(S_values):
    for j, sigma_val in enumerate(sigma_values):
        call_prices[i, j] = call(S_val, K, T, r, sigma_val, q)
        put_prices[i, j] = put(S_val, K, T, r, sigma_val, q)

call_fig = go.Figure(data=go.Heatmap(
    z=call_prices,
    x=np.round(sigma_values, 2),
    y=np.round(S_values, 2),
    colorscale='blues'))
call_fig.update_layout(
    title='Call Option Prices',
    xaxis_title='Volatility (sigma)',
    yaxis_title='Stock Price (S)')

put_fig = go.Figure(data=go.Heatmap(
    z=put_prices,
    x=np.round(sigma_values, 2),
    y=np.round(S_values, 2),
    colorscale='blues'))
put_fig.update_layout(
    title='Put Option Prices',
    xaxis_title='Volatility (sigma)',
    yaxis_title='Stock Price (S)')

with col1: st.plotly_chart(call_fig)
with col2: st.plotly_chart(put_fig)




# Plot Greeks over time until expiration
total_days = (exp - dt.datetime.today()).days
exp_dates = pd.date_range(start=dt.datetime.today(), end=exp, periods=total_days)
T_values = [(total_days - (date - dt.datetime.today()).days) / 365 for date in exp_dates]

greeks_over_time = pd.DataFrame({
    'Date': exp_dates,
    'Delta Call': [delta_call(S, K, T, r, sigma) for T in T_values],
    'Delta Put': [delta_put(S, K, T, r, sigma) for T in T_values],
    'Gamma': [gamma_calc(S, K, T, r, sigma) for T in T_values],
    'Vega': [vega_calc(S, K, T, r, sigma) for T in T_values],
    'Theta Call': [theta_call(S, K, T, r, sigma) for T in T_values],
    'Theta Put': [theta_put(S, K, T, r, sigma) for T in T_values],
    'Rho Call': [rho_call(S, K, T, r, sigma) for T in T_values],
    'Rho Put': [rho_put(S, K, T, r, sigma) for T in T_values],
})


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Delta Call'], mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig1.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Gamma'], mode='lines', name='Gamma', line=dict(color='#E6E6FA')))
fig1.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Vega'], mode='lines', name='Vega', line=dict(color='#B57EDC')))
fig1.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Theta Call'], mode='lines', name='Theta', line=dict(color='#87CEFA')))
fig1.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Rho Call'], mode='lines', name='Rho', line=dict(color='#DDA0DD')))
fig1.update_layout(title='Greeks Over Time Until Expiration', xaxis_title='Date', yaxis_title='Value', legend_title='Greeks')
with col1:
    with st.expander("Call Greeks over time"):
        st.plotly_chart(fig1)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Delta Put'], mode='lines', name='Delta', line=dict(color='#ADD8E6')))
fig2.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Gamma'], mode='lines', name='Gamma', line=dict(color='#E6E6FA')))
fig2.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Vega'], mode='lines', name='Vega', line=dict(color='#B57EDC')))
fig2.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Theta Put'], mode='lines', name='Theta', line=dict(color='#87CEFA')))
fig2.add_trace(go.Scatter(x=greeks_over_time['Date'], y=greeks_over_time['Rho Put'], mode='lines', name='Rho', line=dict(color='#DDA0DD')))
fig2.update_layout(title='Greeks Over Time Until Expiration', xaxis_title='Date', yaxis_title='Value', legend_title='Greeks')
with col2:
    with st.expander("Put Greeks over time"):
        st.plotly_chart(fig2)

st.write("")
st.write("")
st.write("")




# Implied volatility calculation
st.header('Implied Volatility')
col3, col4 = st.columns(2)
with col3: pcall = st.number_input('Market call option price', min_value=0.0, value=callPrice, step=0.01)
with col4: pput = st.number_input('Market put option price', min_value=0.0, value=putPrice, step=0.01)


IVcall = impliedVol_call(pcall, S, K, T, r, q)
IVput = impliedVol_put(pput, S, K, T, r, q)

st.write("")
with col3: st.metric('Implied volatility call', value=f"{IVcall:.2f}" if IVcall is not None else 'Calculation Error')
with col4: st.metric('Implied volatility put', value=f"{IVput:.2f}" if IVput is not None else 'Calculation Error')




# Compare with historical volatility
st.write("")
st.header("Compare with historical volatility")
ticker = st.text_input('Yahoo Stock Ticker', value='AAPL')
col5, col6 = st.columns(2)
with col5: start = st.date_input('Start Date', value=dt.datetime(2021, 1, 1))
with col6: end = st.date_input('End Date', value=dt.datetime.today())

stockData = yf.download(ticker, start, end)
stockData['dReturns'] = stockData['Close'].pct_change()
stockData['HVol'] = stockData['dReturns'].rolling(window=30).std() * np.sqrt(252)


# Plot comparizon
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=stockData.index,
    y=stockData['HVol'],
    mode='lines',
    name='Historical Volatility (30-day Annualized)',
    line=dict(color='lightblue')
))

# Implied Volatility Call
if IVcall is not None:
    fig3.add_trace(go.Scatter(
        x=[stockData.index.min(), stockData.index.max()],
        y=[IVcall, IVcall],
        mode='lines',
        name='Implied Volatility Call',
        line=dict(color='blue')
    ))

# Implied Volatility Put
if IVput is not None:
    fig3.add_trace(go.Scatter(
        x=[stockData.index.min(), stockData.index.max()],
        y=[IVput, IVput],
        mode='lines',
        name='Implied Volatility Put',
        line=dict(color='purple')
    ))

# Aggiorna layout
fig3.update_layout(
    title='Historical and Implied Volatility',
    xaxis_title='Date',
    yaxis_title='Volatility',
    margin=dict(t=50, b=20),
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=0.72)
)

st.plotly_chart(fig3, use_container_width=True)
