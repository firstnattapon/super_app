import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from .models import TickerData, SystemConfig
from .calculators import calculate_shannon_benchmark, black_scholes

def render_dashboard_summary(tickers: list[TickerData], config: SystemConfig):
    st.title("📊 Pool Cash Flow (Professional)")
    
    # Metrics Row
    total_val = sum(t.net_liquidity for t in tickers)
    total_cash_locked = sum(t.cash_locked for t in tickers)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Net Liquidity", f"${total_val:,.2f}", "Portfolio Value")
    col2.metric("Cash Pool (Locked)", f"${total_cash_locked:,.2f}", "Available for Rebalance")
    col3.metric("VIX Level", f"{config.vix_level:.2f}", "Market Fear")
    col4.metric("Risk-Free Rate", f"{config.risk_free_rate*100:.2f}%", "Yield Base")

    st.markdown("---")

def render_dragon_portfolio():
    st.header("🐲 Flywheel 0: Dragon Portfolio")
    
    # Simple Pie Chart
    labels = ['Equity', 'Fixed Income', 'Gold', 'Long Volatility', 'Commodity']
    values = [20, 20, 20, 20, 20]
    fig = px.pie(values=values, names=labels, hole=0.4, title="Asset Allocation (Target: All-Weather)")
    st.plotly_chart(fig, use_container_width=True)

def render_shannon_journal(tickers: list[TickerData]):
    st.header("📈 Flywheel 1 & 2: Shannon's Journal")
    
    selected_ticker = st.selectbox("Select Ticker", [t.symbol for t in tickers])
    ticker = next((t for t in tickers if t.symbol == selected_ticker), None)
    
    if ticker:
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"${ticker.current_price:.2f}")
        col2.metric("Fix Capital", f"${ticker.fix_capital:.2f}")

        # History Table
        if ticker.logs:
            st.subheader("Transaction History")
            data = [{"Action": l.note[:50]+"...", "Value": l.amount} for l in ticker.logs]
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.info("No transaction history available yet.")
        
        # Theoretical Benchmark Chart
        prices = [ticker.current_price * x for x in [0.8, 0.9, 1.0, 1.1, 1.2]]
        benchmarks = [calculate_shannon_benchmark(ticker.fix_capital, ticker.current_price, p) for p in prices]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=benchmarks, name="Shannon Benchmark"))
        fig.update_layout(title="Theoretical Payoff", xaxis_title="Price", yaxis_title="Portfolio Value")
        st.plotly_chart(fig, use_container_width=True)

def render_options_inventory(tickers: list[TickerData]):
    st.header("🛡️ Flywheel 3, 4, 6: Options Inventory")
    
    tabs = st.tabs(["Buying Power (LEAPS)", "Shields (Puts)", "Income (Shorts)"])
    
    with tabs[0]:
        st.subheader("Deep ITM LEAPS (Stock Replacement)")
        # Placeholder for Inventory Table
        st.dataframe(pd.DataFrame(columns=["Ticker", "Strike", "Exp", "Delta", "PnL"]), hide_index=True)
        if st.button("Add New LEAPS"):
            st.toast("Add LEAPS feature coming soon")

    with tabs[1]:
        st.subheader("Black Swan Shields (Long Put)")
        st.info("Zero-Cost Hedging Strategy Active")

    with tabs[2]:
        st.subheader("Synthetic Dividend (Short Volatility)")
        st.warning("Ensure all Short calls are covered by LEAPS!")

def render_collateral_yield(config: SystemConfig):
    st.header("💰 Flywheel 7: Collateral Magic")
    
    capital = st.number_input("Idle Capital ($)", value=100000)
    
    # Calculate Yields
    tbill_yield = capital * config.risk_free_rate
    vol_yield_est = capital * config.collateral_yield
    
    total_yield = tbill_yield + vol_yield_est
    
    st.success(f"Projected Annual Yield: ${total_yield:,.2f} ({(total_yield/capital)*100:.1f}%)")
    
    df = pd.DataFrame({
        "Source": ["Risk-Free (T-Bills)", "Volatility Harvesting"],
        "Amount": [tbill_yield, vol_yield_est]
    })
    st.bar_chart(df.set_index("Source"))
