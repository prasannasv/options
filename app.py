import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import datetime
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="SPY/XSP Cash-Secured Put Analyzer", layout="wide")
st.title("📈 Cash-Secured Put Analyzer (SPY Proxy)")
st.markdown("""
*Visualizes historical data and helps you pick the optimal Put Option to sell based on your Delta and Income/Collateral goals.*
*(Note: We use SPY by default as it provides the most robust free options data via yfinance, and perfectly mimics XSP).*
""")

TICKER = "SPY"
RISK_FREE_RATE = 0.045 # Assumed 4.5% Risk Free Rate for Black-Scholes

# --- DATA FETCHING (Price & Indicators) ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    # Fetch 1 year to ensure we have enough data to calculate the 200 SMA
    df = yf.download(ticker, period="1y")
    
    # Calculate Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands (20 SMA +/- 2 StdDev)
    df['StdDev_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['StdDev_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['StdDev_20'] * 2)
    
    return df

df_price = get_historical_data(TICKER)
current_price = float(df_price['Close'].iloc[-1].iloc[0]) if isinstance(df_price['Close'].iloc[-1], pd.Series) else float(df_price['Close'].iloc[-1])
default_strike = float(df_price['SMA_50'].iloc[-1].iloc[0]) if isinstance(df_price['SMA_50'].iloc[-1], pd.Series) else float(df_price['SMA_50'].iloc[-1])
if np.isnan(default_strike): default_strike = current_price

# --- PLOT 1: HISTORICAL CHART ---
st.subheader("1. Price History & Technical Indicators (Last 6 Months)")

# Truncate view to last 6 months for the chart
df_6m = df_price[df_price.index >= (pd.Timestamp.today() - pd.DateOffset(months=6))]

fig_price = go.Figure()
# Candlesticks
fig_price.add_trace(go.Candlestick(x=df_6m.index, open=df_6m['Open'].squeeze(), high=df_6m['High'].squeeze(), 
                                   low=df_6m['Low'].squeeze(), close=df_6m['Close'].squeeze(), name='Price'))
# Moving Averages
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_20'].squeeze(), line=dict(color='blue', width=1), name='20 SMA'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_50'].squeeze(), line=dict(color='orange', width=1.5), name='50 SMA'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_200'].squeeze(), line=dict(color='red', width=2), name='200 SMA'))
# Bollinger Bands
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['BB_Upper'].squeeze(), line=dict(color='gray', width=1, dash='dot'), name='BB Upper'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['BB_Lower'].squeeze(), fill='tonexty', line=dict(color='gray', width=1, dash='dot'), name='BB Lower'))

fig_price.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
st.plotly_chart(fig_price, use_container_width=True)


# --- USER INPUTS ---
st.subheader("2. Setup Your Strategy")
col1, col2, col3, col4 = st.columns(4)

with col1:
    target_delta = st.slider("Target Delta (Absolute)", min_value=0, max_value=100, value=10, step=1)
with col2:
    user_strike = st.number_input("Target Strike Price", value=int(default_strike), step=1)
with col3:
    goal_type = st.radio("What is your goal constraint?", ["Target Monthly Income", "Capital to Invest"])
with col4:
    if goal_type == "Target Monthly Income":
        goal_amount = st.number_input("Target Amount per Month ($)", value=1000, step=100)
    else:
        goal_amount = st.number_input("Available Capital to Invest ($)", value=50000, step=1000)

# --- BLACK-SCHOLES DELTA CALCULATION ---
def calc_put_delta(S, K, t_years, r, sigma):
    if t_years <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*t_years) / (sigma * np.sqrt(t_years))
    put_delta = norm.cdf(d1) - 1
    return abs(put_delta) # Return absolute delta (0.0 to 1.0)

# --- FETCH OPTIONS DATA ---
@st.cache_data(ttl=600)
def fetch_options_chain(ticker, current_price):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    
    today = datetime.date.today()
    valid_expirations = []
    
    # Filter to expirations <= 60 days
    for exp in expirations:
        exp_date = datetime.datetime.strptime(exp, '%Y-%m-%d').date()
        dte = (exp_date - today).days
        if 0 < dte <= 60:
            valid_expirations.append((exp, dte))
            
    all_puts = []
    for exp, dte in valid_expirations:
        try:
            chain = tk.option_chain(exp)
            puts = chain.puts
            # Estimate mid price. If bid/ask are 0, use lastPrice
            puts['MidPrice'] = np.where((puts['bid'] > 0) & (puts['ask'] > 0), 
                                        (puts['bid'] + puts['ask']) / 2, 
                                        puts['lastPrice'])
            
            # Filter out illiquid garbage
            puts = puts[puts['MidPrice'] > 0.05]
            
            # Calculate Delta
            t_years = dte / 365.0
            puts['Calculated_Delta'] = puts.apply(
                lambda row: calc_put_delta(current_price, row['strike'], t_years, RISK_FREE_RATE, row['impliedVolatility']), axis=1
            )
            puts['Calculated_Delta_100'] = puts['Calculated_Delta'] * 100
            
            # Calculate Monthly Multiplier (how many times you can repeat this trade per month)
            multiplier = 30 / dte 
            puts['Monthly_Multiplier'] = multiplier
            puts['DTE'] = dte
            puts['Expiry'] = exp
            
            all_puts.append(puts)
        except Exception:
            continue
            
    if all_puts:
        return pd.concat(all_puts, ignore_index=True)
    return pd.DataFrame()

with st.spinner(f"Fetching option chains for {TICKER} (0-60 Days)..."):
    df_options = fetch_options_chain(TICKER, current_price)

if df_options.empty:
    st.error("No options data available at the moment. Yahoo Finance might be rate limiting.")
    st.stop()


# --- CALCULATE METRICS BASED ON GOALS ---
df_options['Premium_Contract'] = df_options['MidPrice'] * 100
df_options['Collateral_Contract'] = df_options['strike'] * 100

if goal_type == "Target Monthly Income":
    # How many contracts to hit the target monthly income?
    # Target / (Premium per contract * how many times a month you can sell it)
    df_options['Contracts_Needed'] = goal_amount / (df_options['Premium_Contract'] * df_options['Monthly_Multiplier'])
    df_options['Secondary_Metric'] = df_options['Contracts_Needed'] * df_options['Collateral_Contract']
    secondary_y_title = "Collateral Needed ($)"
else:
    # How much income can you generate with your capital?
    # Floor division because you can't sell a fractional contract
    df_options['Contracts_Affordable'] = np.floor(goal_amount / df_options['Collateral_Contract'])
    df_options['Secondary_Metric'] = df_options['Contracts_Affordable'] * df_options['Premium_Contract'] * df_options['Monthly_Multiplier']
    secondary_y_title = "Projected Monthly Income ($)"

# --- PLOT 2 & 3: OPTIONS CHARTS ---
st.subheader("3. Option Chains Analysis")
colA, colB = st.columns(2)

# --- CHART A: FIXED DELTA ---
with colA:
    st.markdown(f"**Best matches for Target Delta: {target_delta}**")
    
    # For each DTE, find the strike with the delta closest to the target
    best_delta_options = df_options.iloc[
        (df_options['Calculated_Delta_100'] - target_delta).abs().groupby(df_options['DTE']).idxmin()
    ].sort_values('DTE')
    
    fig_delta = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Premium Trace
    fig_delta.add_trace(
        go.Bar(x=best_delta_options['DTE'], y=best_delta_options['MidPrice'], name="Premium ($)", marker_color='rgba(50, 171, 96, 0.6)'),
        secondary_y=False,
    )
    # Secondary Metric Trace
    fig_delta.add_trace(
        go.Scatter(x=best_delta_options['DTE'], y=best_delta_options['Secondary_Metric'], name=secondary_y_title, 
                   mode='lines+markers', line=dict(color='purple', width=3)),
        secondary_y=True,
    )
    
    fig_delta.update_layout(title="Premiums vs Expiry (Constant Delta)", xaxis_title="Days to Expiration (DTE)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_delta.update_yaxes(title_text="Option Premium ($)", secondary_y=False)
    fig_delta.update_yaxes(title_text=secondary_y_title, secondary_y=True)
    
    st.plotly_chart(fig_delta, use_container_width=True)

# --- CHART B: FIXED STRIKE ---
with colB:
    st.markdown(f"**Premiums for Fixed Strike: ${user_strike}**")
    
    # Filter for the specific strike chosen by the user
    fixed_strike_options = df_options[df_options['strike'] == user_strike].sort_values('DTE')
    
    if fixed_strike_options.empty:
        st.warning(f"No active option volume found exactly at Strike ${user_strike}. Try adjusting the strike.")
    else:
        fig_strike = make_subplots(specs=[[{"secondary_y": True}]])
        
        # We will use a Scatter plot with color gradient to represent the Delta of the option
        fig_strike.add_trace(
            go.Scatter(
                x=fixed_strike_options['DTE'], 
                y=fixed_strike_options['MidPrice'], 
                name="Premium ($)", 
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=fixed_strike_options['Calculated_Delta_100'],
                    colorscale='RdYlGn_r', # Red = High Delta (Risk), Green = Low Delta (Safe)
                    showscale=True,
                    colorbar=dict(title="Delta")
                ),
                line=dict(color='rgba(150, 150, 150, 0.4)', width=1)
            ),
            secondary_y=False,
        )
        
        # Secondary Metric Trace
        fig_strike.add_trace(
            go.Scatter(x=fixed_strike_options['DTE'], y=fixed_strike_options['Secondary_Metric'], name=secondary_y_title, 
                       mode='lines+markers', line=dict(color='purple', width=3)),
            secondary_y=True,
        )
        
        fig_strike.update_layout(title=f"Premiums vs Expiry (Strike = ${user_strike})", xaxis_title="Days to Expiration (DTE)",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_strike.update_yaxes(title_text="Option Premium ($)", secondary_y=False)
        fig_strike.update_yaxes(title_text=secondary_y_title, secondary_y=True)
        
        st.plotly_chart(fig_strike, use_container_width=True)

st.markdown("""
---
**Important Mathematical Notes:** 
1. **Target Income Scaling:** To achieve a monthly goal with options that expire in less (or more) than 30 days, the calculation scales the premium by `(30 / DTE)`. For example, a 15 DTE option assumes you can roll/sell it 2 times per month.
2. **Delta Calculation:** Free options APIs don't reliably stream "Greeks", so Delta is calculated dynamically here using the Black-Scholes formula, mapping the option's Implied Volatility back to a Delta.
""")
