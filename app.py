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
st.set_page_config(page_title="SPY Put Options Analyzer", layout="wide")
st.title("📈 Put Options Strategy Analyzer (SPY Proxy)")
st.markdown("""
*Compare **Cash-Secured Puts (CSP)** and **Put Credit Spreads (PCS)**. Optimize your strategy based on Delta, Income Targets, and Capital constraints.*
""")

TICKER = "SPY"
RISK_FREE_RATE = 0.045 # Assumed 4.5% Risk Free Rate for Black-Scholes

# --- DATA FETCHING (Price & Indicators) ---
@st.cache_data(ttl=3600)
def get_historical_data(ticker):
    # Fetch 1 year to ensure we have enough data to calculate the 200 SMA
    df = yf.download(ticker, period="1y", progress=False)
    
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
fig_price.add_trace(go.Candlestick(x=df_6m.index, open=df_6m['Open'].squeeze(), high=df_6m['High'].squeeze(), 
                                   low=df_6m['Low'].squeeze(), close=df_6m['Close'].squeeze(), name='Price'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_20'].squeeze(), line=dict(color='blue', width=1), name='20 SMA'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_50'].squeeze(), line=dict(color='orange', width=1.5), name='50 SMA'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['SMA_200'].squeeze(), line=dict(color='red', width=2), name='200 SMA'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['BB_Upper'].squeeze(), line=dict(color='gray', width=1, dash='dot'), name='BB Upper'))
fig_price.add_trace(go.Scatter(x=df_6m.index, y=df_6m['BB_Lower'].squeeze(), fill='tonexty', line=dict(color='gray', width=1, dash='dot'), name='BB Lower'))

fig_price.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
st.plotly_chart(fig_price, use_container_width=True)


# --- USER INPUTS ---
st.subheader("2. Setup Your Strategy")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    target_delta = st.slider("Target Delta (Abs)", min_value=0, max_value=100, value=10, step=1)
with col2:
    user_strike = st.number_input("Target Strike ($)", value=int(default_strike), step=1)
with col3:
    spread_width = st.number_input("Spread Width ($)", value=5, step=1, help="Difference between short put and long put for credit spreads.")
with col4:
    goal_type = st.radio("Goal Constraint", ["Target Monthly Income", "Capital to Invest"])
with col5:
    if goal_type == "Target Monthly Income":
        goal_amount = st.number_input("Target Amount / Month ($)", value=1000, step=100)
    else:
        goal_amount = st.number_input("Available Capital ($)", value=50000, step=1000)

# --- BLACK-SCHOLES DELTA CALCULATION ---
def calc_put_delta(S, K, t_years, r, sigma):
    if t_years <= 0 or sigma <= 0: return 0.0
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2)*t_years) / (sigma * np.sqrt(t_years))
    put_delta = norm.cdf(d1) - 1
    return abs(put_delta)

# --- FETCH OPTIONS DATA ---
@st.cache_data(ttl=600)
def fetch_options_chain(ticker, current_price):
    tk = yf.Ticker(ticker)
    expirations = tk.options
    today = datetime.date.today()
    valid_expirations = [(exp, (datetime.datetime.strptime(exp, '%Y-%m-%d').date() - today).days) for exp in expirations if 0 < (datetime.datetime.strptime(exp, '%Y-%m-%d').date() - today).days <= 60]
            
    all_puts = []
    for exp, dte in valid_expirations:
        try:
            chain = tk.option_chain(exp)
            puts = chain.puts
            puts['MidPrice'] = np.where((puts['bid'] > 0) & (puts['ask'] > 0), (puts['bid'] + puts['ask']) / 2, puts['lastPrice'])
            puts = puts[puts['MidPrice'] > 0.05]
            
            t_years = dte / 365.0
            puts['Calculated_Delta'] = puts.apply(lambda row: calc_put_delta(current_price, row['strike'], t_years, RISK_FREE_RATE, row['impliedVolatility']), axis=1)
            puts['Calculated_Delta_100'] = puts['Calculated_Delta'] * 100
            puts['Monthly_Multiplier'] = 30 / dte 
            puts['DTE'] = dte
            puts['Expiry'] = exp
            
            all_puts.append(puts)
        except Exception:
            continue
            
    return pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()

with st.spinner(f"Fetching option chains for {TICKER} (0-60 Days)..."):
    df_options = fetch_options_chain(TICKER, current_price)

if df_options.empty:
    st.error("No options data available at the moment.")
    st.stop()

# --- CALCULATE SPREADS ---
def calculate_spreads(df, width):
    spread_data = []
    for dte, group in df.groupby('DTE'):
        for _, short_leg in group.iterrows():
            short_strike = short_leg['strike']
            target_long_strike = short_strike - width
            
            # Find the closest long strike that is <= target_long_strike
            long_legs = group[group['strike'] <= target_long_strike]
            if not long_legs.empty:
                long_leg = long_legs.loc[long_legs['strike'].idxmax()]
                net_premium = short_leg['MidPrice'] - long_leg['MidPrice']
                
                # Only keep rational pricing (Credit Spreads)
                if net_premium > 0:
                    actual_width = short_leg['strike'] - long_leg['strike']
                    spread_data.append({
                        'DTE': dte,
                        'Short_Strike': short_strike,
                        'Long_Strike': long_leg['strike'],
                        'Short_Delta_100': short_leg['Calculated_Delta_100'],
                        'Net_Premium': net_premium,
                        'Net_Premium_Contract': net_premium * 100,
                        'Collateral_Contract': actual_width * 100,
                        'Monthly_Multiplier': short_leg['Monthly_Multiplier']
                    })
    return pd.DataFrame(spread_data)

df_spreads = calculate_spreads(df_options, spread_width)

# --- CALCULATE GOAL METRICS ---
df_options['Premium_Contract'] = df_options['MidPrice'] * 100
df_options['Collateral_Contract'] = df_options['strike'] * 100

if goal_type == "Target Monthly Income":
    # CSP Metrics
    df_options['Contracts_Needed'] = goal_amount / (df_options['Premium_Contract'] * df_options['Monthly_Multiplier'])
    df_options['Secondary_Metric'] = df_options['Contracts_Needed'] * df_options['Collateral_Contract']
    
    # Spread Metrics
    if not df_spreads.empty:
        df_spreads['Contracts_Needed'] = goal_amount / (df_spreads['Net_Premium_Contract'] * df_spreads['Monthly_Multiplier'])
        df_spreads['Secondary_Metric'] = df_spreads['Contracts_Needed'] * df_spreads['Collateral_Contract']
    
    secondary_y_title = "Collateral Needed ($)"
else:
    # CSP Metrics
    df_options['Contracts_Affordable'] = np.floor(goal_amount / df_options['Collateral_Contract'])
    df_options['Secondary_Metric'] = df_options['Contracts_Affordable'] * df_options['Premium_Contract'] * df_options['Monthly_Multiplier']
    
    # Spread Metrics
    if not df_spreads.empty:
        df_spreads['Contracts_Affordable'] = np.floor(goal_amount / df_spreads['Collateral_Contract'])
        df_spreads['Secondary_Metric'] = df_spreads['Contracts_Affordable'] * df_spreads['Net_Premium_Contract'] * df_spreads['Monthly_Multiplier']
    
    secondary_y_title = "Projected Monthly Income ($)"

# --- HELPER FUNCTION TO PLOT ---
def plot_options_chart(df, is_spread, x_col, premium_col, delta_col, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Primary Y: Premium
    marker_config = dict(size=10, color=df[delta_col], colorscale='RdYlGn_r', showscale=True, colorbar=dict(title="Delta")) if delta_col else dict(color='rgba(50, 171, 96, 0.6)')
    mode_config = 'markers+lines' if delta_col else 'bars'
    
    if delta_col:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[premium_col], name="Net Premium ($)" if is_spread else "Premium ($)",
                                 mode=mode_config, marker=marker_config, line=dict(color='rgba(150, 150, 150, 0.4)', width=1)), secondary_y=False)
    else:
        fig.add_trace(go.Bar(x=df[x_col], y=df[premium_col], name="Net Premium ($)" if is_spread else "Premium ($)", marker=marker_config), secondary_y=False)
        
    # Secondary Y: Metric
    fig.add_trace(go.Scatter(x=df[x_col], y=df['Secondary_Metric'], name=secondary_y_title, 
                             mode='lines+markers', line=dict(color='purple', width=3)), secondary_y=True)
    
    fig.update_layout(title=title, xaxis_title="Days to Expiration (DTE)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Premium Received ($)" if not is_spread else "Net Spread Premium ($)", secondary_y=False)
    fig.update_yaxes(title_text=secondary_y_title, secondary_y=True)
    return fig

# --- PLOT 2: CASH SECURED PUTS ---
st.subheader("3. Cash-Secured Puts (CSP)")
colA, colB = st.columns(2)

with colA:
    # Match Target Delta
    csp_delta = df_options.iloc[(df_options['Calculated_Delta_100'] - target_delta).abs().groupby(df_options['DTE']).idxmin()].sort_values('DTE')
    st.plotly_chart(plot_options_chart(csp_delta, False, 'DTE', 'MidPrice', None, f"CSP Premiums (Target Delta: {target_delta})"), use_container_width=True)

with colB:
    # Fixed Strike
    csp_strike = df_options[df_options['strike'] == user_strike].sort_values('DTE')
    if csp_strike.empty: st.warning(f"No CSP volume found at Strike ${user_strike}.")
    else: st.plotly_chart(plot_options_chart(csp_strike, False, 'DTE', 'MidPrice', 'Calculated_Delta_100', f"CSP Premiums (Strike = ${user_strike})"), use_container_width=True)

# --- PLOT 3: PUT CREDIT SPREADS ---
st.subheader("4. Put Credit Spreads (PCS)")

if df_spreads.empty:
    st.warning("Could not calculate spreads. The chosen spread width might be too narrow/wide for the available options chain data.")
else:
    colC, colD = st.columns(2)
    
    with colC:
        # Match Target Delta for the SHORT leg
        pcs_delta = df_spreads.iloc[(df_spreads['Short_Delta_100'] - target_delta).abs().groupby(df_spreads['DTE']).idxmin()].sort_values('DTE')
        st.plotly_chart(plot_options_chart(pcs_delta, True, 'DTE', 'Net_Premium', None, f"PCS Net Premiums (Short Leg Delta: {target_delta})"), use_container_width=True)
        
    with colD:
        # Fixed Short Strike
        pcs_strike = df_spreads[df_spreads['Short_Strike'] == user_strike].sort_values('DTE')
        if pcs_strike.empty: 
            st.warning(f"No Put Credit Spread volume found where short strike is ${user_strike}.")
        else:
            st.plotly_chart(plot_options_chart(pcs_strike, True, 'DTE', 'Net_Premium', 'Short_Delta_100', f"PCS Net Premiums (Short Strike = ${user_strike})"), use_container_width=True)

st.markdown("""
---
**Important Technical Notes:** 
1. **Spread Collateral Definition:** For Put Credit Spreads, collateral is calculated strictly as `Spread Width × 100` per contract. Some brokers subtract the premium received from your gross collateral requirement, but this uses gross collateral to be conservative.
2. **Spread Construction:** The program iterates through every put, assumes it is the "Short" leg, subtracts your requested spread width, and "Buys" the closest valid strike below that threshold to calculate the Net Premium.
""")
