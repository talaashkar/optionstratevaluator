import streamlit as st
import yfinance as yf
import numpy as np
import math
from scipy.stats import norm
from datetime import datetime, date
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set page configuration - only call once at the very top
st.set_page_config(
    page_title="Options Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Pricing Functions ---
def black_scholes_call(S, K, T, r, sigma):
    # Ensure T is not zero or negative for log and sqrt operations
    if T <= 0:
        return max(S - K, 0) # Intrinsic value at expiry
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    # Ensure T is not zero or negative for log and sqrt operations
    if T <= 0:
        return max(K - S, 0) # Intrinsic value at expiry
    d1 = (math.log(S/K) + (r + 0.5 * sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, sigma):
    # Ensure T is not zero or negative
    if T <= 0 or sigma <= 0:
        # Handle division by zero or log of non-positive
        return {
            'Delta': {'Call': 0, 'Put': 0},
            'Gamma': 0,
            'Theta': {'Call': 0, 'Put': 0},
            'Vega': 0,
            'Rho': {'Call': 0, 'Put': 0}
        }

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put = norm.cdf(d1) - 1 # Corrected: or norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    # Theta is typically annualized. Divide by 365 for daily Theta if needed for display.
    theta_call = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2))
    theta_put = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2))
    
    vega = S * norm.pdf(d1) * math.sqrt(T) / 100 # Vega is per 1% change in sigma
    rho_call = K * T * math.exp(-r * T) * norm.cdf(d2) / 100 # Rho is per 1% change in r
    rho_put = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'Delta': {'Call': delta_call, 'Put': delta_put},
        'Gamma': gamma,
        'Theta': {'Call': theta_call, 'Put': theta_put}, # Annualized Theta
        'Vega': vega,
        'Rho': {'Call': rho_call, 'Put': rho_put}
    }

def calculate_time_to_expiry(expiry_date_str):
    expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date() # Convert to date object
    today = date.today() # Get today's date
    delta = expiry_date - today
    return max(delta.days / 365, 0.0001) # Ensure time to expiry is never zero

@st.cache_data # Cache yfinance data to avoid repeated calls
def get_options_chain(ticker, expiry):
    try:
        stock = yf.Ticker(ticker)
        chain = stock.option_chain(expiry)
        return chain.calls, chain.puts
    except Exception as e:
        st.error(f"Error fetching options chain for {ticker} on {expiry}: {e}. Please check the ticker and expiry date.")
        return None, None

def plot_greeks(S, K, T, r, sigma):
    st.markdown("### Greek Sensitivities across Stock Prices")
    show_calls = st.checkbox("Show Call Greeks", value=True, key="show_calls_greeks")
    show_puts = st.checkbox("Show Put Greeks", value=True, key="show_puts_greeks")
    selected_greeks = st.multiselect(
        "Select Greeks to Display:",
        ['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
        default=['Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
        key="selected_greeks_plot"
    )

    stock_prices_range = np.linspace(S * 0.7, S * 1.3, 100) # Adjusted range for better visualization
    
    # Initialize lists for all greeks
    greeks_data = {
        'Delta_Call': [], 'Delta_Put': [],
        'Gamma': [],
        'Theta_Call': [], 'Theta_Put': [],
        'Vega': [],
        'Rho_Call': [], 'Rho_Put': []
    }

    for price in stock_prices_range:
        greeks = calculate_greeks(price, K, T, r, sigma)
        greeks_data['Delta_Call'].append(greeks['Delta']['Call'])
        greeks_data['Delta_Put'].append(greeks['Delta']['Put'])
        greeks_data['Gamma'].append(greeks['Gamma'])
        greeks_data['Theta_Call'].append(greeks['Theta']['Call'] / 365) # Display daily Theta
        greeks_data['Theta_Put'].append(greeks['Theta']['Put'] / 365)   # Display daily Theta
        greeks_data['Vega'].append(greeks['Vega'])
        greeks_data['Rho_Call'].append(greeks['Rho']['Call'])
        greeks_data['Rho_Put'].append(greeks['Rho']['Put'])

    fig = go.Figure()
    
    if 'Delta' in selected_greeks:
        if show_calls:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Delta_Call'], mode='lines', name='Delta Call', line=dict(color='blue')))
        if show_puts:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Delta_Put'], mode='lines', name='Delta Put', line=dict(color='blue', dash='dash')))
    
    if 'Gamma' in selected_greeks:
        fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Gamma'], mode='lines', name='Gamma', line=dict(color='orange')))

    if 'Theta' in selected_greeks:
        if show_calls:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Theta_Call'], mode='lines', name='Theta Call (Daily)', line=dict(color='green')))
        if show_puts:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Theta_Put'], mode='lines', name='Theta Put (Daily)', line=dict(color='green', dash='dash')))
    
    if 'Vega' in selected_greeks:
        fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Vega'], mode='lines', name='Vega', line=dict(color='purple')))
    
    if 'Rho' in selected_greeks:
        if show_calls:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Rho_Call'], mode='lines', name='Rho Call', line=dict(color='red')))
        if show_puts:
            fig.add_trace(go.Scatter(x=stock_prices_range, y=greeks_data['Rho_Put'], mode='lines', name='Rho Put', line=dict(color='red', dash='dash')))

    fig.update_layout(
        title='Option Greeks Across Stock Prices',
        xaxis_title='Stock Price',
        yaxis_title='Greek Value',
        template='plotly_dark',
        hovermode='x unified',
        width=900,
        height=600
    )
    st.plotly_chart(fig)

def render_trade_evaluation(eval_results, education_text, breakeven_label="Breakeven Price"):
    st.markdown("### 📈 Trade Evaluation")

    def format_breakeven_prices(prices):
        prices_str = [f"${b:.2f}" for b in prices]
        if len(prices_str) == 1:
            return prices_str[0]
        elif len(prices_str) == 2:
            return " and ".join(prices_str)
        else:
            return ", ".join(prices_str[:-1]) + f", and {prices_str[-1]}"

    if isinstance(eval_results['Breakeven'], (list, tuple)):
        breakeven_str = format_breakeven_prices(eval_results['Breakeven'])
        st.write(f"**{breakeven_label}s:**") # Bold label on one line
        st.text(breakeven_str) # Formatted prices on the next line
    else:
        # --- NEW AGGRESSIVE HTML LINE (for single breakeven) ---
        st.write(f"**{breakeven_label}:**") # Bold label on one line
        st.text(f"${eval_results['Breakeven']:.2f}") # Formatted price on the next line

    try:
        # Attempt to convert to float and format
        formatted_max_loss = f"${float(eval_results['Max Loss']):.2f}"
    except (ValueError, TypeError):
        # If conversion fails, keep the original string
        formatted_max_loss = str(eval_results['Max Loss'])
    st.write(f"**Max Loss:** {formatted_max_loss}")
    
    try:
        # Attempt to convert to float and format
        formatted_max_profit = f"${float(eval_results['Max Profit']):.2f}"
    except (ValueError, TypeError):
        # If conversion fails, keep the original string
        formatted_max_profit = str(eval_results['Max Profit'])
    st.write(f"**Max Profit:** {formatted_max_profit}")
    
    pop_value = eval_results.get('POP', None)
    if pop_value is not None and isinstance(pop_value, (int, float)):
        st.write(f"**Probability of Profit (POP):** {pop_value:.2%}")
    else:
        st.write(f"**Probability of Profit (POP):** {pop_value if pop_value is not None else 'N/A'}")

    st.write(f"**Expected Value (Approx.):** ${eval_results['Expected Value']:.2f}")
    st.write(f"**Risk Rating:** {eval_results['Risk Rating']}")

    st.markdown("### 🎓 Educational Insights")
    st.markdown(education_text)


# --- Streamlit App Layout ---
st.markdown("""
# **Options Analytics Dashboard**
Welcome to your pro trading terminal. Analyze options, Greeks, and market comparisons — all in one place.
""")

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker", 'AAPL', help="Enter the stock ticker you want to analyze, like 'AAPL' for Apple.").upper()

# --- Fetch and Initialize Data ---
options_data = yf.Ticker(ticker)
expirations = options_data.options

# Initialize variables to avoid NameError
S = None
T = None
sigma = None
calls = None
puts = None
strikes = []
K = None
market_call_price = 0.0
market_put_price = 0.0
selected_strike = None # Initialize selected_strike

if not expirations:
    st.sidebar.warning(f"No options expirations found for ticker: **{ticker}**. Please try another ticker or check Yahoo Finance.")
    st.stop() # Stop execution if no expirations are found

expiry = st.sidebar.selectbox("Select Expiration Date", expirations, help="Select an expiration date for the option chain you want to analyze.")

# Fetch historical data for current price and volatility
try:
    data = options_data.history(period="1d")
    S = data['Close'].iloc[-1] if not data.empty else 100.0
except Exception as e:
    st.error(f"Could not fetch current stock price for {ticker}. Using default $100. Error: {e}")
    S = 100.0

try:
    hist_data = options_data.history(period="60d") # Use a longer period for more stable sigma
    returns = hist_data['Close'].pct_change().dropna()
    sigma = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.25 # Ensure enough data for std dev
except Exception as e:
    st.error(f"Could not estimate volatility for {ticker}. Using default 25%. Error: {e}")
    sigma = 0.25

T = calculate_time_to_expiry(expiry)
r = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0, help="Theoretical return of a risk-free investment (default ~3%). Affects option pricing.") / 100

st.write(f"**Current Stock Price:** ${S:.2f}")
st.write(f"**Time to Expiry:** {T:.4f} years")
st.write(f"**Estimated Volatility (sigma):** {sigma:.2%}")

# Get options chain after S, T, r, sigma are defined
calls, puts = get_options_chain(ticker, expiry)

if calls is None or puts is None or calls.empty or puts.empty:
    st.error("Could not retrieve options chain data. Please try another ticker or expiration.")
    st.stop() # Stop if options chain is not available

option_type = st.selectbox("Option Type (for general pricing/greeks tab)", ['Call', 'Put'], key="main_option_type")

if option_type == 'Call':
    strikes = calls['strike'].tolist()
else:
    strikes = puts['strike'].tolist()

if not strikes:
    st.error("No strike prices found for the selected option type and expiration.")
    st.stop()

# Set a default selected strike, e.g., closest to ATM
default_strike_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - S))
selected_strike = st.selectbox("Select Strike Price (for general pricing/greeks tab)", options=strikes, index=default_strike_idx, key="main_selected_strike")
K = selected_strike

# Fetch market prices for the initially selected strike, used as defaults later
if not calls.empty and selected_strike in calls['strike'].values:
    market_call_price = calls[calls['strike'] == selected_strike].iloc[0]['lastPrice']
else:
    market_call_price = black_scholes_call(S, selected_strike, T, r, sigma) # Fallback to model price

if not puts.empty and selected_strike in puts['strike'].values:
    market_put_price = puts[puts['strike'] == selected_strike].iloc[0]['lastPrice']
else:
    market_put_price = black_scholes_put(S, selected_strike, T, r, sigma) # Fallback to model price


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Market Comparison", "Greeks Visualization", "Options Strategies Simulator", "Option Pricing Dynamics"])

# --- Main Content ---
with tab1:
    st.write("## 🧩 Market vs Black-Scholes Model")

    st.subheader("Calls")
    # Use selected_strike as default for the comparison strikes
    call_strike_comp = st.selectbox("Call Strike Price for Comparison", options=strikes, index=strikes.index(selected_strike) if selected_strike in strikes else 0, key="call_strike_comp")
    
    current_market_call_price = calls[calls['strike'] == call_strike_comp].iloc[0]['lastPrice'] if not calls[calls['strike'] == call_strike_comp].empty else 0.0
    model_call_price_comp = black_scholes_call(S, call_strike_comp, T, r, sigma)

    st.write(f"Market Call Price: **${current_market_call_price:.2f}**")
    st.write(f"Model Call Price: **${model_call_price_comp:.2f}**")

    st.subheader("Puts")
    put_strike_comp = st.selectbox("Put Strike Price for Comparison", options=strikes, index=strikes.index(selected_strike) if selected_strike in strikes else 0, key="put_strike_comp")
    
    current_market_put_price = puts[puts['strike'] == put_strike_comp].iloc[0]['lastPrice'] if not puts[puts['strike'] == put_strike_comp].empty else 0.0
    model_put_price_comp = black_scholes_put(S, put_strike_comp, T, r, sigma)

    st.write(f"Market Put Price: **${current_market_put_price:.2f}**")
    st.write(f"Model Put Price: **${model_put_price_comp:.2f}**")

    st.markdown("---")

    st.subheader("Calls Chain")
    st.dataframe(calls[['strike', 'lastPrice', 'impliedVolatility', 'bid', 'ask', 'volume', 'openInterest']]) # Added more columns for completeness

    st.subheader("Puts Chain")
    st.dataframe(puts[['strike', 'lastPrice', 'impliedVolatility', 'bid', 'ask', 'volume', 'openInterest']]) # Added more columns for completeness

with tab2:
    st.markdown("## Greeks Visualization")
    # Use the strike selected in the main sidebar for Greek visualization
    plot_greeks(S, selected_strike, T, r, sigma)

with tab3:
    st.markdown("## 🛠️ Options Strategies Simulator")

    strategy = st.selectbox(
        "Select an Options Strategy:",
        ["Covered Call", "Protective Put", "Long Straddle", "Long Call", "Long Put"],
        key="strategy_select"
    )

    stock_price_at_entry = st.number_input("Current Stock Price for Entry:", min_value=0.01, value=S, key="strategy_stock_price_entry")
    contracts = st.number_input("Number of Contracts (100 shares per contract):", min_value=1, value=1, help="Adjusts total PnL based on number of contracts traded.", key="num_contracts_strategy")

    # Define a range of stock prices for PnL plotting
    stock_prices_for_pnl = np.linspace(stock_price_at_entry * 0.5, stock_price_at_entry * 1.5, 200) # More points for smoother graph

    if strategy == "Covered Call":
        st.subheader("Covered Call Parameters")

        # Set default values for strike and premium based on selected_strike and market prices
        call_strike_strat = st.number_input("Call Option Strike Price:", min_value=0.01, value=float(selected_strike), key="cc_strike")
        call_premium_strat = st.number_input("Call Option Premium Received per Share:", min_value=0.01, value=market_call_price, key="cc_premium")

        def covered_call_pnl(current_stock_prices, initial_stock_price, call_strike, call_premium, contracts):
            shares = contracts * 100
            pnl = []
            for price in current_stock_prices:
                # PnL from stock: (current price - initial purchase price) * shares
                # PnL from call: premium received - max(0, current price - strike) * shares
                total_pnl = ((price - initial_stock_price) + call_premium - max(0, price - call_strike)) * shares
                pnl.append(total_pnl)
            return pnl

        pnl = covered_call_pnl(stock_prices_for_pnl, stock_price_at_entry, call_strike_strat, call_premium_strat, contracts)

        def covered_call_evaluator(initial_stock_price, call_strike, T, r, sigma, call_premium):
            # Breakeven = Initial Stock Price - Premium received
            breakeven = initial_stock_price - call_premium

            # Max Profit = (Call Strike - Initial Stock Price + Premium Received) * 100
            # If initial_stock_price > call_strike, capped profit is just the premium received.
            # If initial_stock_price < call_strike, capped profit is (call_strike - initial_stock_price + call_premium) * 100
            max_profit = ((call_strike - initial_stock_price) + call_premium) * 100
            if max_profit < 0: # Capped at premium if initial stock price is too high or call is ITM
                max_profit = call_premium * 100
            
            # Max Loss = (Initial Stock Price - Premium Received) * 100 (if stock goes to zero)
            max_loss = (initial_stock_price - call_premium) * 100

            # Probability of Profit (POP): Probability that stock price at expiry > breakeven
            # Or, P(Stock > (Initial_Stock_Price - Premium_Received))
            # d2 for call expiring worthless (stock price < strike)
            # A more common interpretation for covered call POP is P(stock price at expiry < strike price)
            # This means the call expires worthless, and you keep the premium and stock.
            # d1 and d2 here are calculated for the call option at strike K
            d1_pop = (np.log(initial_stock_price / call_strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            pop = norm.cdf(-d1_pop) # Probability the call expires out-of-the-money (worthless)

            # Expected Value is complex for capped profit/loss. A rough approximation:
            # P(Profit) * Avg_Profit_Scenario - P(Loss) * Avg_Loss_Scenario
            # For simplicity, we can use (Max Profit * POP) - (Max Loss * (1-POP)) but this is highly approximate
            ev = max_profit * pop - max_loss * (1 - pop)


            # Risk rating based on POP (higher pop means safer if it means call expires worthless)
            if pop > 0.6: # Higher threshold for covered call POP
                risk = "Low Risk (Call likely expires worthless)"
            elif pop > 0.3:
                risk = "Moderate Risk"
            else:
                risk = "Higher Risk (Call likely in-the-money)"

            return {
                'Breakeven': breakeven,
                'Max Profit': max_profit,
                'Max Loss': max_loss,
                'POP': pop,
                'Expected Value': ev,
                'Risk Rating': risk
            }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices_for_pnl, y=pnl, mode='lines', name='Covered Call PnL', line=dict(color='green')))
        
        # Add breakeven line
        eval_results_temp = covered_call_evaluator(stock_price_at_entry, call_strike_strat, T, r, sigma, call_premium_strat)
        breakeven_price = eval_results_temp['Breakeven']
        fig.add_vline(x=breakeven_price, line=dict(color='yellow', dash='dash'), annotation_text=f"Breakeven: ${breakeven_price:.2f}", annotation_position="top right")

        fig.update_layout(
            title="Covered Call PnL at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig)

        eval_results = covered_call_evaluator(stock_price_at_entry, call_strike_strat, T, r, sigma, call_premium_strat)

        education_text = """
        - **When to Use:** When you own the underlying stock and want to generate income, especially in neutral to mildly bullish or bearish markets.
        - **Breakeven:** Initial stock purchase price minus premium received.
        - **Risk:** Limited downside protection (premium received cushions losses). If stock goes to zero, max loss is (Stock Price - Premium Received).
        - **Reward:** Capped at the strike price plus premium (if stock rises above strike, your stock is called away).
        - **Pro Tip:** Ideal in neutral to mildly bullish markets where you expect the stock to stay below the strike or rise only slightly.
        """
        render_trade_evaluation(eval_results, education_text)

    elif strategy == "Protective Put":
        st.subheader("Protective Put Parameters")

        put_strike_strat = st.number_input("Put Option Strike Price:", min_value=0.01, value=float(selected_strike), key="pp_strike")
        put_premium_strat = st.number_input("Put Option Premium Paid per Share:", min_value=0.01, value=market_put_price, key="pp_premium")

        def protective_put_pnl(current_stock_prices, initial_stock_price, put_strike, put_premium, contracts):
            shares = contracts * 100
            pnl = []
            for price in current_stock_prices:
                # PnL from stock: (current price - initial purchase price) * shares
                # PnL from put: max(0, strike - current price) - premium paid
                total_pnl = ((price - initial_stock_price) + (max(0, put_strike - price) - put_premium)) * shares
                pnl.append(total_pnl)
            return pnl

        pnl = protective_put_pnl(stock_prices_for_pnl, stock_price_at_entry, put_strike_strat, put_premium_strat, contracts)

        def protective_put_evaluator(initial_stock_price, put_strike, T, r, sigma, put_premium):
            # Breakeven = Initial Stock Price + Premium paid
            breakeven = initial_stock_price + put_premium

            # Max Loss = (Initial Stock Price - Put Strike + Premium Paid) * 100, if stock falls below put strike
            # If Initial Stock Price is already below put strike, max loss is (Premium Paid) * 100
            # If the stock falls to 0, max loss is (Initial_Stock_Price - 0 + put_premium) - K = (initial_stock_price - K + put_premium)*100
            max_loss = (initial_stock_price - put_strike + put_premium) * 100 # Capped at this value when stock drops below K
            if initial_stock_price < put_strike: # If put is already ITM at entry, the cost is just premium
                max_loss = put_premium * 100 # This scenario might not be typical for 'protective'
            
            # Max Profit = Unlimited (stock price can rise infinitely, offset by premium)
            max_profit = "Unlimited"

            # Probability of Profit (POP): Probability that stock price at expiry > breakeven.
            # This means the put expires worthless, and the cost is only the premium.
            # d2 for put expiring worthless (stock price > strike)
            d2_pop = (np.log(initial_stock_price / put_strike) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            pop = norm.cdf(d2_pop) # Probability the put expires out-of-the-money (worthless)

            # Expected Value approx: Limited loss, unlimited profit.
            # Simplistic EV: (Profit if stock rises significantly) * POP - (Max Loss) * (1 - POP)
            # This is hard to quantify with unlimited profit. A qualitative description is often better.
            ev = (initial_stock_price * 0.1 * pop * 100) - (max_loss * (1 - pop)) # Example: 10% stock rise profit

            # Risk rating based on POP (higher pop means put likely expires worthless, so less "costly" as insurance)
            if pop > 0.7:
                risk = "Low Risk (Put likely expires worthless)"
            elif pop > 0.4:
                risk = "Moderate Risk"
            else:
                risk = "High Risk (Put likely in the money)"

            return {
                'Breakeven': breakeven,
                'Max Loss': max_loss,
                'Max Profit': max_profit,
                'POP': pop,
                'Expected Value': ev,
                'Risk Rating': risk
            }   

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices_for_pnl, y=pnl, mode='lines', name='Protective Put PnL', line=dict(color='red')))
        
        # Add breakeven line
        eval_results_temp = protective_put_evaluator(stock_price_at_entry, put_strike_strat, T, r, sigma, put_premium_strat)
        breakeven_price = eval_results_temp['Breakeven']
        fig.add_vline(x=breakeven_price, line=dict(color='yellow', dash='dash'), annotation_text=f"Breakeven: ${breakeven_price:.2f}", annotation_position="top right")

        fig.update_layout(
            title="Protective Put PnL at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig)

        eval_results = protective_put_evaluator(stock_price_at_entry, put_strike_strat, T, r, sigma, put_premium_strat)

        education_text = """
        - **When to Use:** When you want downside protection on stock you own, often in uncertain or potentially bearish markets.
        - **Breakeven:** Initial stock purchase price plus premium paid.
        - **Risk:** Limited to (initial stock price - put strike + premium paid). If stock goes to zero, max loss is capped at this value.
        - **Reward:** Unlimited upside potential from the stock, minus the premium paid.
        - **Pro Tip:** Great in volatile markets or uncertain times as portfolio insurance.
        """
        render_trade_evaluation(eval_results, education_text)

    elif strategy == "Long Straddle":
        st.subheader("Long Straddle Parameters")

        straddle_strike_strat = st.number_input("Straddle Strike Price (Call & Put):", min_value=0.01, value=float(selected_strike), key="straddle_strike")
        call_premium_strat = st.number_input("Call Option Premium Paid per Share:", min_value=0.01, value=market_call_price, key="straddle_call_premium")
        put_premium_strat = st.number_input("Put Option Premium Paid per Share:", min_value=0.01, value=market_put_price, key="straddle_put_premium")

        def longstraddle_pnl(current_stock_prices, strike_price, call_premium, put_premium, contracts):
            pnl = []
            total_premium = call_premium + put_premium
            for price in current_stock_prices:
                call_payoff = max(price - strike_price, 0)
                put_payoff = max(strike_price - price, 0)
                total_pnl = (call_payoff + put_payoff - total_premium) * contracts * 100
                pnl.append(total_pnl)
            return pnl

        pnl = longstraddle_pnl(stock_prices_for_pnl, straddle_strike_strat, call_premium_strat, put_premium_strat, contracts)

        def long_straddle_evaluator(S_current, K, T, r, sigma, call_premium, put_premium):
            total_premium = call_premium + put_premium
            
            # Breakeven points (two): 
            lower_breakeven = K - total_premium
            upper_breakeven = K + total_premium
            breakeven = [lower_breakeven, upper_breakeven]

            # Max Loss = Total premiums paid
            max_loss = total_premium * 100

            # Max Profit = Unlimited (big moves in either direction)
            max_profit = "Unlimited"

            # Probability of Profit (POP): Probability that stock price is OUTSIDE the breakeven range.
            # P(S_T < lower_breakeven) + P(S_T > upper_breakeven)
            d2_lower_be = (np.log(S_current / lower_breakeven) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2_upper_be = (np.log(S_current / upper_breakeven) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            # POP = P(stock price < lower breakeven) + P(stock price > upper breakeven)
            pop = norm.cdf(-d2_lower_be) + (1 - norm.cdf(-d2_upper_be)) # Check this for accuracy based on standard POP definition

            # The probability of profit for a straddle is the probability that the price ends *outside* the breakeven points.
            # P(S_T < Lower Breakeven) + P(S_T > Upper Breakeven)
            d1_lower_be_pop = (np.log(S_current / lower_breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d1_upper_be_pop = (np.log(S_current / upper_breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            pop_actual = norm.cdf(-d1_lower_be_pop) + (1 - norm.cdf(d1_upper_be_pop)) # Probability of being below lower BE or above upper BE
            pop_actual = max(0, min(1, pop_actual)) # Ensure POP is between 0 and 1

            # Expected Value is hard due to unlimited profit. It's usually negative due to time decay.
            # Simplified: assuming it goes to breakeven, the loss is max_loss.
            ev = -max_loss * (1 - pop_actual) # This is a very rough approximation

            # Risk rating based on POP
            if pop_actual > 0.6:
                risk = "Moderate Risk (Requires significant move for profit)"
            elif pop_actual > 0.3:
                risk = "Higher Risk"
            else:
                risk = "High Risk (Low probability of significant move)"

            return {
                'Breakeven': breakeven,
                'Max Loss': max_loss,
                'Max Profit': max_profit,
                'POP': pop_actual,
                'Expected Value': ev,
                'Risk Rating': risk
            }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices_for_pnl, y=pnl, mode='lines', name='Long Straddle PnL', line=dict(color='purple')))
        
        # Add breakeven lines
        eval_results_temp = long_straddle_evaluator(stock_price_at_entry, straddle_strike_strat, T, r, sigma, call_premium_strat, put_premium_strat)
        lower_breakeven, upper_breakeven = eval_results_temp['Breakeven']
        fig.add_vline(x=lower_breakeven, line=dict(color='yellow', dash='dash'), annotation_text=f"Lower BE: ${lower_breakeven:.2f}", annotation_position="top left")
        fig.add_vline(x=upper_breakeven, line=dict(color='yellow', dash='dash'), annotation_text=f"Upper BE: ${upper_breakeven:.2f}", annotation_position="top right")

        fig.update_layout(
            title="Long Straddle PnL at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig)

        eval_results = long_straddle_evaluator(stock_price_at_entry, straddle_strike_strat, T, r, sigma, call_premium_strat, put_premium_strat)

        education_text = """
            - **When to Use:** Expecting a big move in either direction (up or down) from the underlying stock, but unsure of the direction. Requires significant volatility.
            - **Breakeven:** Two points — lower (Strike - Total Premium) and upper (Strike + Total Premium) — around the strike price.
            - **Risk:** Limited to the total premium paid (sum of call and put premiums).
            - **Reward:** Unlimited profit potential on big moves (either up or down).
            - **Pro Tip:** Best when implied volatility is low and expected to rise sharply, or before major news events (e.g., earnings).
        """
        render_trade_evaluation(eval_results, education_text)

    elif strategy == "Long Call":
        st.subheader("Long Call Parameters")

        call_strike_strat = st.number_input("Call Option Strike Price:", min_value=0.01, value=float(selected_strike), key="lc_strike")
        call_premium_strat = st.number_input("Call Option Premium Paid per Share:", min_value=0.01, value=market_call_price, key="lc_premium")

        def long_call_pnl(current_stock_prices, call_strike, call_premium, contracts):
            pnl = []
            for price in current_stock_prices:
                call_payoff = max(price - call_strike, 0)
                total_pnl = (call_payoff - call_premium) * contracts * 100
                pnl.append(total_pnl)
            return pnl

        pnl = long_call_pnl(stock_prices_for_pnl, call_strike_strat, call_premium_strat, contracts)

        def long_call_evaluator(S_current, K, T, r, sigma, premium):
            # Breakeven = Strike + Premium paid
            breakeven = K + premium

            # Max Loss is premium paid
            max_loss = premium * 100  # per contract (100 shares)

            # Max profit = theoretically unlimited, but we can say "unlimited" for display
            max_profit = "Unlimited"

            # Probability of Profit (POP) = P(Stock Price at Expiry > Breakeven Price)
            # Use d1 for probability of option being ITM for call, but for breakeven, it's P(S_T > Breakeven)
            d1_breakeven = (np.log(S_current / breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            pop_breakeven = 1 - norm.cdf(d1_breakeven) # P(S_T > Breakeven) for a call is norm.cdf(d2_breakeven)
            pop_breakeven = norm.cdf((np.log(S_current/breakeven) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T)))
            pop_breakeven = max(0, min(1, pop_breakeven)) # Ensure POP is between 0 and 1

            # Expected Value (Approx.) = (Probability of Profit * Avg Profit) - (Probability of Loss * Max Loss)
            # This is highly simplified due to unlimited profit.
            ev = (pop_breakeven * (S_current * 0.1 * 100)) - (max_loss * (1 - pop_breakeven)) # Assuming a 10% gain for profit scenario
            
            # Risk rating based on POP
            if pop_breakeven > 0.7:
                risk = "Moderate Risk (Needs significant upward move)"
            elif pop_breakeven > 0.4:
                risk = "Higher Risk"
            else:
                risk = "High Risk (Low probability of profit)"

            return {
                'Breakeven': breakeven,
                'Max Loss': max_loss,
                'Max Profit': max_profit,
                'POP': pop_breakeven,
                'Expected Value': ev,
                'Risk Rating': risk
            }

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices_for_pnl, y=pnl, mode='lines', name='Long Call PnL', line=dict(color='blue')))
        
        breakeven_price = call_strike_strat + call_premium_strat # Re-calculate breakeven for plot line
        fig.add_vline(x=breakeven_price, line=dict(color='yellow', dash='dash'), annotation_text=f"Breakeven: ${breakeven_price:.2f}", annotation_position="top right")
        
        fig.update_layout(
            title="Long Call PnL at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig)

        eval_results = long_call_evaluator(stock_price_at_entry, call_strike_strat, T, r, sigma, call_premium_strat)

        education_text = """
            - **When to Use:** When you expect the stock to rise significantly above the strike + premium paid (bullish outlook).
            - **Breakeven:** Stock must rise to **strike + premium** for you to profit.
            - **Risk:** Capped to the premium you pay, making it defined risk.
            - **Reward:** Unlimited upside potential.
            - **Pro Tip:** Ideal in **high volatility or strongly bullish** environments.
        """
        render_trade_evaluation(eval_results, education_text) 
    
    elif strategy == "Long Put":
        st.subheader("Long Put Parameters")

        put_strike_strat = st.number_input("Put Option Strike Price:", min_value=0.01, value=float(selected_strike), key="lp_strike")
        put_premium_strat = st.number_input("Put Option Premium Paid per Share:", min_value=0.01, value=market_put_price, key="lp_premium")

        def long_put_pnl(current_stock_prices, put_strike, put_premium, contracts):
            pnl = []
            for price in current_stock_prices:
                put_payoff = max(put_strike - price, 0)
                total_pnl = (put_payoff - put_premium) * contracts * 100
                pnl.append(total_pnl)
            return pnl

        pnl = long_put_pnl(stock_prices_for_pnl, put_strike_strat, put_premium_strat, contracts)
        
        def long_put_evaluator(S_current, K, T, r, sigma, premium):
            # Breakeven = Strike - Premium paid
            breakeven = K - premium

            # Max Loss is premium paid
            max_loss = premium * 100  # per contract (100 shares)

            # Max profit limited to strike price * 100 shares (if stock goes to 0) MINUS premium
            max_profit = (K * 100) - max_loss
            if max_profit < 0: # If premium is higher than strike, max profit is theoretical, effectively capped at -premium
                max_profit = 0 # Or could be stated as "Limited (e.g., $K * 100 - premium)"

            # Probability of Profit (POP) = P(Stock Price at Expiry < Breakeven Price)
            # Use d1 for probability of put being ITM, but for breakeven, it's P(S_T < Breakeven)
            d1_breakeven = (np.log(S_current / breakeven) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            pop_breakeven = norm.cdf(-d1_breakeven) # P(S_T < Breakeven)
            pop_breakeven = max(0, min(1, pop_breakeven)) # Ensure POP is between 0 and 1

            # Expected Value (Approx.)
            ev = (max_profit * pop_breakeven) - (max_loss * (1 - pop_breakeven))

            # Risk rating based on POP
            if pop_breakeven > 0.7:
                risk = "Moderate Risk (Needs significant downward move)"
            elif pop_breakeven > 0.4:
                risk = "Higher Risk"
            else:
                risk = "High Risk (Low probability of profit)"

            return {
                'Breakeven': breakeven,
                'Max Loss': max_loss,
                'Max Profit': max_profit,
                'POP': pop_breakeven,
                'Expected Value': ev,
                'Risk Rating': risk
            }
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_prices_for_pnl, y=pnl, mode='lines', name='Long Put PnL', line=dict(color='orange')))
        
        breakeven_price = put_strike_strat - put_premium_strat # Recalculate breakeven for plot line
        fig.add_vline(x=breakeven_price, line=dict(color='yellow', dash='dash'), annotation_text=f"Breakeven: ${breakeven_price:.2f}", annotation_position="top left")

        fig.update_layout(
            title="Long Put PnL at Expiration",
            xaxis_title="Stock Price at Expiration",
            yaxis_title="Profit / Loss ($)",
            template='plotly_dark',
            hovermode='x unified'
        )
        st.plotly_chart(fig)

        eval_results = long_put_evaluator(stock_price_at_entry, put_strike_strat, T, r, sigma, put_premium_strat)

        education_text = """
            - **When to Use:** When you expect a sharp decline in the stock price (bearish outlook).
            - **Breakeven:** Stock must fall to strike price minus premium.
            - **Risk:** Limited to the premium paid.
            - **Reward:** Max profit if stock goes to zero (limited, not unlimited).
            - **Pro Tip:** Best when implied volatility is low but expected to rise.
            """
        render_trade_evaluation(eval_results, education_text) 

# --- Option Pricing Dynamics Tab ---
with tab4:
    st.markdown("## Option Pricing Dynamics - 3D Surface")
    st.write(f"**Strike Price (K):** ${selected_strike:.2f}")
    st.write(f"**Volatility (σ):** {sigma:.2%}")
    st.write(f"**Risk-Free Rate (r):** {r:.2%}")

    stock_prices_3d = np.linspace(S * 0.5, S * 1.5, 50)
    times_to_expiration_3d = np.linspace(0.01, T, 50) # Ensure time doesn't start at 0

    if selected_strike < S * 0.5 or selected_strike > S * 1.5:
        st.warning(f"⚠️ The selected strike price (${selected_strike:.2f}) is quite far from the current stock price range shown (${S * 0.5:.2f} - ${S * 1.5:.2f}). The surface plot may be less informative or unstable.")

    option_prices_3d = np.zeros((len(times_to_expiration_3d), len(stock_prices_3d)))

    for i, T_val in enumerate(times_to_expiration_3d):
        for j, stock_price_val in enumerate(stock_prices_3d):
            try:
                if option_type == 'Call':
                    price = black_scholes_call(stock_price_val, selected_strike, T_val, r, sigma)
                else:
                    price = black_scholes_put(stock_price_val, selected_strike, T_val, r, sigma)
            except Exception as e:
                # Handle cases where B-S might fail (e.g., T=0, sigma=0)
                price = 0
            option_prices_3d[i, j] = price

    fig = go.Figure(data=[go.Surface(
        z=option_prices_3d,
        x=stock_prices_3d,
        y=times_to_expiration_3d,
        colorscale='Viridis'
    )])

    fig.update_layout(
        title=f'{option_type} Option Price Surface',
        scene=dict(
            xaxis_title='Stock Price',
            yaxis_title='Time to Expiration (Years)',
            zaxis_title='Option Price'
        ),
        width=900,
        height=700,
        template='plotly_dark'
    )
    st.plotly_chart(fig)