import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

# --- Utility Functions ---

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price."""
    if T <= 0:
        return np.maximum(S - K, 0) if option_type == 'call' else np.maximum(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def generate_gbm(S0, mu, sigma, T, dt, n_sims=1):
    """Generate Geometric Brownian Motion path."""
    n_steps = int(T / dt)
    time_series = np.linspace(0, T, n_steps + 1)
    W = np.random.standard_normal(size=(n_sims, n_steps)) 
    W = np.cumsum(W, axis=1) * np.sqrt(dt)
    W = np.insert(W, 0, 0, axis=1)
    
    S = S0 * np.exp((mu - 0.5 * sigma ** 2) * time_series + sigma * W)
    return time_series, S.flatten()


# --- Chapter 0: Introduction (Flywheel 0) ---
def chapter_0_introduction():
    st.header("บทนำ: Flywheel 0 (Dragon Portfolio)")
    st.markdown("""
    **Flywheel 0** คือรากฐานที่สำคัญที่สุด (All-Weather Foundation) 
    ออกแบบมาเพื่อให้พอร์ตอยู่รอดได้ในทุกสภาวะเศรษฐกิจ (เงินเฟ้อ, เงินฝืด, เติบโต, ถดถอย)
    
    > "มังกรไม่ได้พยายามทำนายสภาพอากาศ แต่มันถูกสร้างมาให้บินได้ทั้งในพายุและแดดจ้า"
    """)

    # Dragon Portfolio Allocation
    labels = ['Equity (Growth)', 'Fixed Income (Deflation)', 'Gold (Inflation)', 'Long Volatility (Crisis)', 'Commodity Trend (Inflation)']
    values = [20, 20, 20, 20, 20]
    colors = ['#1f77b4', '#aec7e8', '#ffbb78', '#d62728', '#2ca02c']

    fig = px.pie(names=labels, values=values, title='Dragon Portfolio Allocation', 
                 color_discrete_sequence=colors, hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

    st.info("โครงสร้างนี้กระจายความเสี่ยงไปในสินทรัพย์ที่ 'ไม่สัมพันธ์กัน' (Uncorrelated) เพื่อให้พอร์ตเติบโตได้อย่างยั่งยืนในระยะยาว")


# --- Chapter 1: The Baseline ---
def chapter_1_baseline():
    st.header("บทที่ 1: The Baseline (เส้นมาตรวัด)")
    st.markdown("""
    **Concept:** เส้น Benchmark คือกระจกเงาที่สะท้อนว่า หากเรารักษามูลค่าพอร์ตให้คงที่ (Constant Value) ผลตอบแทนเชิงทฤษฎี (Theoretical PnL) 
    ควรจะเป็นเท่าไหร่ โดยอิงจากสมการ:
    """)
    st.latex(r"B_{theory} = k + fix\_c \times (\ln(\frac{P_t}{P_0}) + 0.5\sigma^2)")
    st.latex(r"B_{simple (No\ k)} = fix\_c + fix\_c \times \ln(\frac{P_t}{P_0})")
    st.markdown("""
    โดยที่:
    - $B_t$: มูลค่าพอร์ต Benchmark ณ ราคา $P_t$
    - $k$: ค่า Expected Value (EV) ที่เสียไปให้กับ LEAPS (Cost of Convexity) หรือ Intercept
    - $fix\_c$: ทุนที่ต้องการรักษาให้คงที่ (Fix Capital)
    """)

    col1, col2 = st.columns(2)
    with col1:
        fix_c = st.number_input("Fix Capital (fix_c)", value=100000, step=1000)
        k_val = st.number_input("ค่า EV ที่เสียไป (k)", value=2000, step=100, help="ต้นทุนค่าเสียโอกาสหรือ Time Decay ของ LEAPS")
    with col2:
        P0 = st.number_input("ราคาเริ่มต้น (P0)", value=100.0, step=1.0)
        sigma = st.slider("ความผันผวน (Sigma)", 0.1, 1.0, 0.5, key="ch1_sigma", help="ใช้คำนวณ Volatility Premium ที่จะได้รับ")

    # Simulation parameters: Price Range 20 to 180 (if P0=100)
    # Scaled to be 0.2 * P0 to 1.8 * P0
    price_min = P0 * 0.2
    price_max = P0 * 1.8
    Pt = np.linspace(price_min, price_max, 200)
    
    # Calculate PnL
    # Buy & Hold (Linear): Start at fix_c (assuming fully invested for comparison)
    # Note: Usually Buy & Hold starts at Capital. Here Capital = fix_c.
    buy_hold_value = (fix_c / P0) * Pt
    
    # Formula 1: B_t = (Capital - k) + fix_c * (ln(Pt/P0) + 0.5 * sigma^2 * T)
    T = 1.0 # Assume 1 Year Horizon for Theoretical view
    vol_premium = 0.5 * (sigma ** 2) * T
    intercept = fix_c - k_val
    benchmark_value_with_vol = intercept + (fix_c * (np.log(Pt / P0) + vol_premium))
    
    # Formula 2 (Requested): Simple Shannon (No k, No Vol Premium)
    # Just the Logarithmic Utility starting at fix_c
    benchmark_value_simple = fix_c + (fix_c * np.log(Pt / P0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Pt, y=buy_hold_value, name="Buy & Hold (Linear)", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=Pt, y=benchmark_value_simple, name=f"Shannon Simple (No k)", line=dict(dash='dot', color='gray')))
    fig.add_trace(go.Scatter(x=Pt, y=benchmark_value_with_vol, name=f"Shannon + Vol Premium - k", line=dict(dash='dash', color='orange')))
    
    # Add annotation for Volatility Premium Difference
    fig.add_annotation(
        x=P0, y=benchmark_value_with_vol[len(Pt)//2],
        text=f"Diff due to Vol: +{vol_premium*100:.1f}%",
        showarrow=True, arrowhead=1, ax=40, ay=-40, font=dict(color="orange")
    )
    
    # Intersection Point
    # Where does Log return beats Linear return?
    # At P0: Linear = fix_c, Log = fix_c - k. (Log is lower due to cost)
    # As P increases, Linear (convex?) No Linear is linear. Log is Concave.
    # Wait, Log grows purely slower than Linear for P > P0.
    # Log beats Linear only if Linear drops significantly?
    # At P < P0: Linear drops fast. Log drops slower (concave).
    # Let's see the graph.
    
    fig.update_layout(
        title="Theoretical PnL: Linear vs Logarithmic (Price vs Value)", 
        xaxis_title="Stock Price ($)", 
        yaxis_title="Portfolio Value ($)"
    )
    fig.add_vline(x=P0, line_dash="dot", annotation_text="Start Price")
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **การวิเคราะห์กราฟ:**
    1. **แกน X คือราคา (Price):** ราคาวิ่งจากซ้าย (ต่ำ) ไปขวา (สูง)
    2. **เส้นสีส้ม (Benchmark):** เริ่มต้นต่ำกว่าเส้นสีฟ้าเพราะเสียค่า k (EV Lost)
    3. **ความโค้ง (Concave):** สังเกตว่าเมื่อราคาตกต่ำ (ไปทางซ้าย) เส้นสีส้มจะอยู่ *สูงกว่า* เส้นสีฟ้า (ขาดทุนน้อยกว่า) 
       แต่มื่อราคาพุ่งขึ้น (ไปทางขวา) เส้นสีส้มจะขึ้นช้ากว่า (กำไรน้อยกว่า)
    4. **เป้าหมาย:** เราต้องใช้ **Flywheel 2 (Rebalancing)** เพื่อดึงเส้นสีส้มให้ยกตัวขึ้นชนะเส้นสีฟ้าให้ได้!
    """)


# --- Chapter 2: Volatility Harvest ---
def chapter_2_volatility_harvest():
    st.header("บทที่ 2: Volatility Harvest (การเก็บเกี่ยวความผันผวน)")
    st.markdown("""
    **Concept:** การทำ Continuous Rebalancing "ซื้อถูก-ขายแพง" ช่วยให้เราเก็บ Cashflow สะสมได้เรื่อยๆ 
    ยิ่งตลาดผันผวนและไม่มีเทรนด์ (Mean Reversion) ยิ่งได้กำไรส่วนเกิน (Volatility Premium) สูงสุด
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fix_c = st.number_input("Fix Capital (fix_c)", value=100000.0, step=1000.0, key="ch2_fixc")
    with col2:
        sigma = st.slider("ความผันผวน (Sigma)", 0.1, 1.0, 0.5, key="ch2_sigma")
    with col3:
        drift = st.slider("แนวโน้มตลาด (Drift)", -0.5, 0.5, 0.0, step=0.01, key="ch2_drift", help="0 = Sideways, >0 = Bull, <0 = Bear")

    P0 = 100.0

    if st.button("Start Rebalancing", key="ch2_run"):
        # Generate Price Path
        t, Pt = generate_gbm(P0, drift, sigma, 1.0, 1/252)
        
        # Rebalancing Simulation
        cash = 0
        shares = fix_c / P0 
        
        portfolio_values = []
        cash_accumulated = []
        shares_held = []
        
        current_shares = shares
        current_cash = 0
        
        for price in Pt:
            current_value = current_shares * price
            diff = current_value - fix_c
            
            # Rebalance logic: Reset to fix_c
            if diff > 0: # Sell
                shares_to_sell = diff / price
                current_shares -= shares_to_sell
                current_cash += diff
            elif diff < 0: # Buy
                shares_to_buy = abs(diff) / price
                current_shares += shares_to_buy
                current_cash -= abs(diff)
            
            shares_held.append(current_shares)
            portfolio_values.append(current_shares * price + current_cash)
            cash_accumulated.append(current_cash)

        # Plotting
        fig = go.Figure()
        
        # Plot 1: Cash Generated
        fig.add_trace(go.Scatter(y=cash_accumulated, name="Cumulative Cash Generated (Vol Premium)", 
                                 fill='tozeroy', line=dict(color='#2ca02c')))
        
        # Plot 2: Stock Price (Secondary Y-axis)
        fig.add_trace(go.Scatter(y=Pt, name="Underlying Asset Price", yaxis="y2", 
                                 line=dict(color='gray', width=1, dash='dot')))
        
        fig.update_layout(
            title="Volatility Premium Harvesting",
            yaxis=dict(title="Cash Generated ($)"),
            yaxis2=dict(title="Price ($)", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        total_cash = cash_accumulated[-1]
        roi = (total_cash / fix_c) * 100
        
        st.success(f"""
        **ผลลัพธ์:**
        - Total Cash Generated: ${total_cash:,.2f}
        - Yield on Capital: {roi:.2f}%
        """)
        
        # New Chart: Theoretical Payoff at T=1 (Price vs Value)
        # Reusing logic from Chapter 1 for consistency
        price_min = P0 * 0.2
        price_max = P0 * 1.8
        Pt_theory = np.linspace(price_min, price_max, 100)
        
        # Theoretical Value of Rebalanced Portfolio (Shannon's Demon)
        # Formula: W_T = fix_c * (1 + ln(P_T/P_0) + 0.5 * sigma^2 * T)
        # The last term (0.5 * sigma^2 * T) is the "Volatility Premium" harvested
        T_theory = 1.0
        vol_yield_theoretical = 0.5 * (sigma ** 2) * T_theory
        rebalance_value = fix_c * (1 + np.log(Pt_theory / P0) + vol_yield_theoretical)
        
        buy_hold_value = (fix_c / P0) * Pt_theory
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=Pt_theory, y=buy_hold_value, name="Buy & Hold (Linear)", line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=Pt_theory, y=rebalance_value, name=f"Rebalancing (Concave + Vol Premium {vol_yield_theoretical*100:.1f}%)", line=dict(dash='dash', color='orange')))
        
        # Add annotation for Volatility Premium
        fig2.add_annotation(
            x=P0, y=rebalance_value[len(Pt_theory)//2],
            text=f"+ Vol Premium: {vol_yield_theoretical*100:.1f}%",
            showarrow=True, arrowhead=1, ax=40, ay=-40
        )
        
        fig2.update_layout(
            title=f"Theoretical Payoff at T={T_theory} Year (Includes Volatility Premium)",
            xaxis_title="Stock Price at Year End ($)",
            yaxis_title="Total Portfolio Value ($)"
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        > **Insight:** 
        > เส้นสีส้ม (Rebalancing) จะลอยตัวสูงขึ้นกว่าเส้น Buy & Hold ในระยะยาว 
        > เพราะเราเก็บเกี่ยว **Volatility Premium** (ส่วนต่าง $\\frac{{1}}{{2}}\\sigma^2$) 
        > ยิ่งผันผวนมาก (High Sigma) เส้นสีส้มยิ่งลอยขึ้นสูง
        """)


# --- Chapter 3: Convexity Engine ---
def chapter_3_convexity_engine():
    st.header("บทที่ 3: Convexity Engine (Stock Replacement 80/20)")
    st.markdown("""
    **Concept:** ใช้กลยุทธ์ **Stock Replacement** โดยพลิกสัดส่วนเป็น **80% LEAPS Call Option** และ **20% Liquidity (Cash)**
    
    - **80% LEAPS**: ใช้เงินส่วนใหญ่ซื้อ Option Deep ITM (Delta ~0.9) เพื่อจำลองการถือหุ้นจำนวนมหาศาล (Super Leverage)
    - **20% Liquidity**: เก็บเงินสดไว้สำหรับ **Volatility Harvest** (รอช้อนซื้อ/Rebalance)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        stock_price = st.number_input("ราคาหุ้นปัจจุบัน (Spot Price)", value=100.0)
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0) / 100
    with col2:
        strike_price = st.number_input("LEAPS Strike Price (Deep ITM)", value=80.0)
        volatility = st.number_input("Implied Volatility (%)", value=50.0) / 100

    time_to_expiry_months = st.slider("Time to Expiry (Months)", 1, 24, 12)
    T = time_to_expiry_months / 12.0

    # Calculate Premium
    premium = black_scholes(stock_price, strike_price, T, risk_free_rate, volatility, 'call')
    delta = norm.cdf((np.log(stock_price/strike_price) + (risk_free_rate + 0.5*volatility**2)*T) / (volatility*np.sqrt(T)))
    
    st.metric(label="LEAPS Premium (Cost per share)", value=f"${premium:.2f}", delta=f"Delta: {delta:.2f}")

    # Payoff Diagram Simulation
    prices = np.linspace(stock_price * 0.2, stock_price * 1.8, 100)
    
    # Calculate Theoretical Value at Expiry (Intrinsic only) vs Current (Black-Scholes)
    # Portfolio Value = Cash Portion + LEAPS Portion
    # Strategy: 80% LEAPS, 20% Cash (Aggressive / Leveraged)
    # Initial Capital = 100 (Standardized)
    initial_capital = 100
    
    # Allocation
    alloc_leaps = 0.80 * initial_capital
    alloc_cash = 0.20 * initial_capital
    
    # Quantities
    # Number of LEAPS contracts purchasable with $80
    qty_leaps = alloc_leaps / premium
    # Cash held in T-Bills/Money Market
    cash_value_expiry = alloc_cash * np.exp(risk_free_rate * T)
    
    # Portfolio Value at Expiry (Static)
    # Value = (Cash_T) + (Qty * Max(StockPrice_T - Strike, 0))
    leaps_component_expiry = qty_leaps * np.maximum(prices - strike_price, 0)
    portfolio_value_expiry = cash_value_expiry + leaps_component_expiry
    
    # Stock Only Comparison (100% Capital in Stock)
    qty_stock_only = initial_capital / stock_price
    stock_only_value = qty_stock_only * prices
    
    # Volatility Premium (Theoretical Gain from Rebalancing)
    # Note: With high leverage (80% LEAPS), the volatility harvest is significant.
    # We add standard Vol Premium on Capital basis.
    vol_gain = initial_capital * (0.5 * volatility**2 * T)
    portfolio_value_dynamic = portfolio_value_expiry + vol_gain 
    
    # Shannon Simple (Theoretical Reference, No Cost 'k')
    # B_t = Capital * (1 + ln(Pt/P0))
    shannon_simple_value = initial_capital * (1 + np.log(prices / stock_price))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=stock_only_value, name="Stock Only (100% Stock)", line=dict(dash='dot', color='gray')))
    fig.add_trace(go.Scatter(x=prices, y=portfolio_value_expiry, name="80% LEAPS + 20% Cash (Static)", line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=prices, y=shannon_simple_value, name="Shannon Simple (Reference)", line=dict(dash='dot', color='purple')))
    fig.add_trace(go.Scatter(x=prices, y=portfolio_value_dynamic, name=f"Dynamic (Rebalanced + Vol Premium)", line=dict(width=3, color='orange')))
    
    fig.update_layout(
        title=f"Portfolio Value: 80% LEAPS + 20% Cash (Strike {strike_price})",
        xaxis_title="Stock Price at Future Date",
        yaxis_title="Total Portfolio Value ($)",
        hovermode="x unified"
    )
    fig.add_vline(x=stock_price, line_width=1, line_dash="dash", line_color="green", annotation_text="Current Price")
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("""
    **Aggressive Convexity (80/20):**
    1. **เส้นสีน้ำเงิน (Static):** พอร์ตนี้เน้น Leverage สูง (ถือ LEAPS 80%) 
       - ขาขึ้น: กำไรพุ่งแรงกว่าหุ้นปกติ (High Delta)
       - ขาลง: มีเงินสด 20% ช่วยพยุง (Floor) แต่ยังมีความเสี่ยงสูงจากมูลค่า Option ที่ลดลง
    2. **เส้นสีส้ม (Dynamic):** การมีเงินสด 20% ช่วยให้เราสามารถ Rebalance (ช้อนซื้อ) เมื่อตลาดผันผวน
       ทำให้เกิด **Volatility Harvest** ช่วยยกผลตอบแทนรวมให้สูงขึ้น
    """)


# --- Chapter 4: The Black Swan Shield ---
def chapter_4_black_swan_shield():
    st.header("บทที่ 4: The Black Swan Shield (โล่กันภัยวิกฤต)")
    st.markdown("""
    **Concept:** การป้องกันพอร์ตด้วย **Put Option (Protective Put)** 
    เพื่อสร้าง **Convexity ขาลง** (กำไรพุ่งเมื่อตลาดพัง) และอาศัยช่วงวิกฤต Rebalance เพื่อสร้างผลตอบแทนมหาศาล
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        stock_price = st.number_input("ราคาหุ้น (Price)", value=100.0, key="ch4_price")
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, key="ch4_r") / 100
        hedge_ratio = st.slider("Hedge Ratio (Put Amount)", 0.5, 5.0, 2.0, step=0.1, key="ch4_mid", help="1.0 = 1 Put per 100 Shares. 2.0 = Over-hedge")
    with col2:
        # LEAPS Parameters (Base)
        leaps_strike = st.number_input("Base LEAPS Strike (Deep ITM)", value=80.0, key="ch4_leaps_k")
        # Put Parameters (Hedge)
        put_strike = st.number_input("Hedge Put Strike (OTM Protection)", value=90.0, key="ch4_put_k")
        volatility = st.number_input("Implied Volatility (%)", value=50.0, key="ch4_vol") / 100

    time_to_expiry_months = st.slider("Time to Expiry (Months)", 1, 24, 12, key="ch4_time")
    T = time_to_expiry_months / 12.0

    # 1. Base Strategy: 80% LEAPS + 20% Cash
    initial_capital = 100
    alloc_leaps = 0.80 * initial_capital
    alloc_cash = 0.20 * initial_capital
    
    # Calculate LEAPS Premium (Cost per share)
    leaps_premium = black_scholes(stock_price, leaps_strike, T, risk_free_rate, volatility, 'call')
    qty_leaps = alloc_leaps / leaps_premium
    
    # Calculate Cash Value at Expiry
    cash_value_expiry = alloc_cash * np.exp(risk_free_rate * T)

    # 2. Hedge Strategy: Buy Puts
    # Put Cost
    put_premium = black_scholes(stock_price, put_strike, T, risk_free_rate, volatility, 'put')
    # Hedge Ratio acts on the "Equivalent Stock Exposure" or just "qty_leaps"?
    # Usually Hedge Ratio 1.0 means 1 Put per 1 Share.
    # Our "Share Equivalent" is qty_leaps (since Delta ~ 1). 
    # Or is it per $100 capital? User said "Hedge Ratio... Put Amount".
    # Let's assume Hedge Ratio is relative to the *Number of LEAPS Contracts* (Delta Matched).
    # Buying 2.0 Puts per 1 LEAPS.
    qty_puts = qty_leaps * hedge_ratio
    total_put_cost = qty_puts * put_premium
    
    # Adjusted Cash (We pay for Puts from the Cash portion? Or is it Extra capital?)
    # "Cost of Insurance". Let's deduct from Cash to see net performance.
    # If Cash < PutCost, we borrow (negative cash).
    remaining_cash_expiry = (alloc_cash - total_put_cost) * np.exp(risk_free_rate * T)

    st.metric(label="Structure Setup", 
              value=f"LEAPS: {qty_leaps:.2f} units", 
              delta=f"Puts: {qty_puts:.2f} units (Cost ${total_put_cost:.2f})")

    # Simulation Range: 20 to 180
    prices = np.linspace(stock_price * 0.2, stock_price * 1.8, 100)
    
    # Calculate Values at Expiry
    
    # A. Stock Only (Reference)
    qty_stock_only = initial_capital / stock_price
    stock_only_value = qty_stock_only * prices
    
    # Put Payoff (used by Shielded & Dynamic)
    put_payoff = qty_puts * np.maximum(put_strike - prices, 0)
    
    # Vol Premium term
    vol_premium = 0.5 * volatility**2 * T
    
    # B. Base 80/20 (Unhedged) — Baseline Theoretical PnL (No k, No Vol)
    # B_base = fix_c + fix_c * ln(Pt/P0)
    base_80_20_value = initial_capital + initial_capital * np.log(prices / stock_price)
    
    # C. Shielded Portfolio (80/20 + Puts) — Baseline - Put Cost + Put Payoff
    # B_shielded = (fix_c - put_cost) + fix_c * ln(Pt/P0) + put_payoff
    shielded_value = (initial_capital - total_put_cost) + initial_capital * np.log(prices / stock_price) + put_payoff
    
    # D. Dynamic Shield (+Vol Premium) — Baseline + Vol Premium - Put Cost + Put Payoff
    # B_dynamic = (fix_c - put_cost) + fix_c * (ln(Pt/P0) + 0.5σ²T) + put_payoff
    shielded_dynamic_value = (initial_capital - total_put_cost) + initial_capital * (np.log(prices / stock_price) + vol_premium) + put_payoff
    
    # E. Shannon Simple (Theoretical Reference, No Cost 'k')
    # B_t = Capital * (1 + ln(Pt/P0))
    shannon_simple_value = initial_capital * (1 + np.log(prices / stock_price))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=stock_only_value, name="Stock Only (Reference)", line=dict(dash='dot', color='gray')))
    fig.add_trace(go.Scatter(x=prices, y=base_80_20_value, name="Base 80/20 (Unhedged)", line=dict(dash='dash', color='blue')))
    fig.add_trace(go.Scatter(x=prices, y=shielded_value, name=f"Shielded 80/20 (+{hedge_ratio}x Puts)", line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=prices, y=shannon_simple_value, name="Shannon Simple (Reference)", line=dict(dash='dot', color='purple')))
    fig.add_trace(go.Scatter(x=prices, y=shielded_dynamic_value, name=f"Dynamic Shield (+Vol Premium)", line=dict(color='orange', width=3)))
    
    fig.update_layout(
        title=f"Black Swan Shield: 80/20 Base + {hedge_ratio}x Put Hedge",
        xaxis_title="Stock Price at Future Date",
        yaxis_title="Total Portfolio Value ($)",
        hovermode="x unified"
    )
    # Add vertical line at current price
    fig.add_vline(x=stock_price, line_width=1, line_dash="dash", line_color="green", annotation_text="Current Price")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Analysis:**
    1. **เส้นสีน้ำเงิน (Base 80/20):** พอร์ตพื้นฐานที่มี Leverage สูง (กำไรดีขาขึ้น แต่ขาลงเจ็บหนักถ้าหลุด Break-even)
    2. **เส้นสีเขียว (Shielded):** เมื่อเติม Put Option (ประกันภัย) เข้าไป
       - **ขาลง (Left):** กราฟจะดีดตัวขึ้นทำกำไร (Anti-Fragile) สวนทางกับตลาด
       - **ขาขึ้น (Right):** กำไรลดลงเล็กน้อยจากค่าเบี้ยประกัน
    3. **เส้นสีส้ม (Dynamic):** ผลลัพธ์คาดหวังเมื่อรวมการ Rebalance (Volatility Harvest)
    """)


# --- Chapter 5: Dynamic Scaling ---
def chapter_5_dynamic_scaling():
    st.header("บทที่ 5: Dynamic Scaling (สลับเกียร์ตาม VIX)")
    st.markdown("""
    **Concept:** ปรับเปลี่ยนกลยุทธ์ (Regime Switching) ตามระดับความกลัวในตลาด (VIX)
    เพื่อบริหารต้นทุนค่าเบี้ยประกัน (Cost of Insurance)
    """)
    
    vix = st.slider("ระดับดัชนี VIX (Volatility Index)", 10, 80, 20)
    
    status = ""
    action = ""
    color = ""
    
    if vix < 20:
        status = "Low Volatility (ตลาดสงบ)"
        action = "Scale Down: ถือ LEAPS 80/20, ลด Put Hedge, เก็บ Cash, เน้นทำ Synthetic Dividend"
        color = "green"
    elif vix < 40:
        status = "Medium Volatility (เริ่มผันผวน)"
        action = "Neutral: เริ่มสะสม Put, เตรียม Rebalance ถี่ขึ้น"
        color = "orange"
    else:
        status = "High Volatility (ตลาดแตก/Panic)"
        action = "Scale Up: อัด Put x2 เต็มสูบ, ขยาย fix_c, รอสวนตลาด (Anti-Fragile Mode)"
        color = "red"
        
    st.markdown(f"### Market Status: :{color}[{status}]")
    st.info(f"**Action Required:** {action}")
    
    # Visual Gauge
    st.progress(min(vix / 80, 1.0))


# --- Chapter 6: Synthetic Dividend ---
def chapter_6_synthetic_dividend():
    st.header("บทที่ 6: Synthetic Dividend (สร้างปันผลเทียม)")
    st.markdown("""
    **Concept:** สร้างกระแสเงินสด (Cashflow) เพิ่มเติมด้วยการขาย Option (Short)
    - **Short Call**: เก็บค่าเช่าจากขาขึ้น (Covered Call บน LEAPS)
    - **Short Put**: เก็บค่าเช่าจากขาลง (ใช้ Cash ค้ำ)
    
    เป้าหมายคือนำเงินนี้ไปจ่ายค่า Put Hedge ในบทที่ 4 (Zero-Cost Insurance)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        current_price = 100.0
        st.markdown("**Call Option (ฝั่งต้าน)**")
        call_strike = st.number_input("Short Call Strike", value=105.0, step=1.0)
        call_premium = st.number_input("Call Premium Received", value=1.5, step=0.1)
    with col2:
        st.markdown("**Put Option (ฝั่งรับ)**")
        put_strike = st.number_input("Short Put Strike", value=90.0, step=1.0)
        put_premium = st.number_input("Put Premium Received", value=1.0, step=0.1)
    
    total_credit = call_premium + put_premium
    st.metric(label="Total Credit Received (Synthetic Dividend)", value=f"${total_credit:.2f}")

    prices = np.linspace(80, 120, 200)
    
    # PnL Calculations
    short_call_pnl = -(np.maximum(prices - call_strike, 0)) + call_premium
    short_put_pnl = -(np.maximum(put_strike - prices, 0)) + put_premium
    total_yield = short_call_pnl + short_put_pnl
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=short_call_pnl, name="Short Call PnL", line=dict(dash='dot', color='red')))
    fig.add_trace(go.Scatter(x=prices, y=short_put_pnl, name="Short Put PnL", line=dict(dash='dot', color='orange')))
    fig.add_trace(go.Scatter(x=prices, y=total_yield, name="Total Credit (Income)", line=dict(width=3, color='gold')))
    
    # Profit Zone
    profit_mask = total_yield > 0
    # Add a filled area for profit
    fig.add_trace(go.Scatter(x=prices[profit_mask], y=total_yield[profit_mask], 
                             fill='tozeroy', mode='none', name='Profit Zone', fillcolor='rgba(255, 215, 0, 0.2)'))
    
    fig.add_hline(y=0, line_dash="solid", line_color="white")
    fig.update_layout(title="Cashflow Generation (PnL vs Price)", xaxis_title="Stock Price at Expiry", yaxis_title="PnL / Income ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("พื้นที่สีทอง (Profit Zone) คือช่วงราคาที่ตลาดวิ่งอยู่ในกรอบ แล้วเรากินเปล่าค่า Premium (Theta Decay)")


# --- Chapter 7: Collateral Magic ---
def chapter_7_collateral_magic():
    st.header("บทที่ 7: Collateral Magic (พลังทวีหลักประกัน)")
    st.markdown("""
    **Concept:** ใช้ T-Bills เป็นหลักประกัน (Collateral) แทนเงินสด เพื่อรับผลตอบแทนซ้อนทับ (Yield Stacking)
    """)
    
    col1, col2, col3 = st.columns(3)
    capital = st.number_input("Portfolio Capital ($)", value=100000, step=10000)
    
    with col1:
        tbill_rate = st.number_input("T-Bill Rate (%)", value=5.0) / 100
    with col2:
        # Calculate theoretical vol yield from user input sigma (interactive helper)
        est_sigma = st.slider("Typical Volatility (Sigma)", 0.1, 1.0, 0.5, key="ch7_sigma", help="ใช้คำนวณ Volatility Yield")
        theoretical_vol_yield = 0.5 * (est_sigma ** 2)
        st.caption(f"Theoretical Yield (0.5 * σ²): **{theoretical_vol_yield*100:.1f}%**")
        
        vol_yield = st.number_input("Est. Volatility Yield (%)", value=float(theoretical_vol_yield*100), step=1.0) / 100
    with col3:
        syn_div_yield = st.number_input("Est. Synthetic Div (%)", value=8.0) / 100
    
    yield_tbill = capital * tbill_rate
    yield_vol = capital * vol_yield
    yield_syn = capital * syn_div_yield
    
    total = yield_tbill + yield_vol + yield_syn
    
    df = pd.DataFrame({
        "Source": ["1. Risk-Free (T-Bills)", "2. Volatility Harvest", "3. Synthetic Dividend"],
        "Amount": [yield_tbill, yield_vol, yield_syn],
        "Color": ['#1f77b4', '#ff7f0e', '#2ca02c']
    })
    
    fig = go.Figure(data=[go.Bar(
        x=df['Source'], 
        y=df['Amount'], 
        text=df['Amount'].apply(lambda x: f"${x:,.0f}"),
        textposition='auto',
        marker_color=df['Color']
    )])
    
    fig.update_layout(
        title=f"Total Annual Yield Projection: ${total:,.2f} ({total/capital*100:.1f}%)",
        yaxis_title="Annual Income ($)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success(f"""
    **Yield Stacking Effect:**
    แทนที่จะได้แค่ 5% จากพันธบัตร หรือเสี่ยง 100% ในหุ้น
    เราซ้อนผลตอบแทน 3 ชั้น: ${yield_tbill:,.0f} + ${yield_vol:,.0f} + ${yield_syn:,.0f} = **${total:,.0f}**
    """)


# --- Chapter 8: Chain System (ระบบลูกโซ่) ---

def rollover_baseline(b, c_old, c_new, t_old, t_new, P):
    """
    Rollover Equation: b += c · ln(P/t) - c' · ln(P/t')
    แล้วตั้ง c = c', t = t'
    ทำให้เส้น Baseline ต่อเนื่องแม้เปลี่ยน fix_c
    """
    term_old = c_old * np.log(P / t_old)
    term_new = c_new * np.log(P / t_new)
    b_new = b + term_old - term_new
    return b_new

def chapter_chain_system():
    st.header("บทที่ 8: Chain System (ระบบลูกโซ่)")
    st.markdown("""
    **Concept:** เชื่อมกำไรจากทุก Flywheel เข้าเป็น **ลูกโซ่** (Chain) — 
    กำไรจากขั้นหนึ่งไหลไปเป็น "เชื้อเพลิง" ให้ขั้นถัดไป วนเป็นวงจร **ทั้งขาขึ้น + ขาลง**
    
    > **ขาขึ้น:** กำไร Shannon + Harvest → จ่ายค่า Put Hedge → Surplus → Scale Up fix_c = **Free Risk**
    > 
    > **ขาลง:** Put ระเบิดกำไร → เข้า **Pool CF** → Deploy (เมื่อ Regime กลับ) + Reserve (สำรอง)
    """)
    
    # --- Rollover Equation Explanation ---
    st.subheader("📐 Rollover Equation (สมการเส้นต่อเนื่อง)")
    st.markdown("""
    เมื่อ Scale Up เปลี่ยน fix_c → ใช้สมการ Rollover เพื่อให้เส้น Baseline ไม่กระโดด:
    """)
    st.latex(r"b \mathrel{+}= c \cdot \ln\!\left(\frac{P}{t}\right) - c' \cdot \ln\!\left(\frac{P}{t'}\right)")
    st.markdown("แล้วตั้ง: $c = c'$, $t = t'$")
    
    # --- Method Selection ---
    rollover_method = st.radio(
        "🔀 เลือกวิธี Rollover",
        ["แบบ 1: ล็อคกำไร + เริ่มใหม่ (t' = P ปัจจุบัน) ✅ แนะนำ",
         "แบบ 2: วัดจากจุดเดิม (t' = t เดิม)"],
        index=0,
        key="ch8_rollover_method",
        help="ทั้งสองแบบให้เส้นต่อเนื่อง ณ จุด Rollover เท่ากัน"
    )
    use_method_1 = "แบบ 1" in rollover_method
    
    # --- Show example for selected method ---
    if use_method_1:
        st.success("""
        **📗 แบบ 1: เปิดสมุดใหม่ (Reset t' = P)**
        - กำไรเก่าถูก "ล็อค" ไว้ใน b
        - t' = ราคาปัจจุบัน → `ln(P/P) = 0` → เทอมใหม่หายไป
        - **ใช้งานจริง** ง่าย ชัดเจน
        
        **ตัวอย่าง:** c=1500, t=12.6, P=24, c'=2500, **t'=24**
        
        $b = 0 + 1500 × \\ln(24/12.6) - 2500 × \\ln(24/24) = 966.54 - 0 = \\mathbf{966.54}$
        """)
    else:
        st.info("""
        **📘 แบบ 2: สมุดเล่มเดิม (t' = t เดิม)**
        - จุดอ้างอิงเดิมไม่เปลี่ยน → วัดทุกอย่างจากจุดเริ่มต้นเดียวกัน
        - เหมาะสำหรับ **วิเคราะห์ย้อนหลัง (Backtest)**
        
        **ตัวอย่าง:** c=1500, t=12.6, P=24, c'=2500, **t'=12.6**
        
        $b = 0 + 1500 × \\ln(24/12.6) - 2500 × \\ln(24/12.6) = (1500-2500) × 0.6445 = \\mathbf{-644.36}$
        """)
    
    st.markdown("---")
    
    # --- Input Panel ---
    st.subheader("⚙️ ตั้งค่า Chain Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fix_c = st.number_input("Fix Capital เริ่มต้น (c)", value=1500.0, step=100.0, key="ch8_fixc")
        P0 = st.number_input("ราคาอ้างอิงเริ่มต้น (t₀)", value=12.60, step=0.1, key="ch8_p0", format="%.2f")
    with col2:
        Pt = st.number_input("ราคาปัจจุบัน (P)", value=24.0, step=0.1, key="ch8_pt", format="%.2f")
        sigma = st.slider("ความผันผวน (σ)", 0.1, 1.0, 0.5, key="ch8_sigma")
    with col3:
        put_strike_pct = st.slider("Put Strike (% of P)", 50, 100, 90, key="ch8_putstrike", help="เช่น 90% ของราคาปัจจุบัน")
        hedge_ratio = st.slider("Hedge Ratio", 0.5, 5.0, 2.0, step=0.1, key="ch8_hedge")
    
    # --- Pool CF Settings ---
    st.markdown("#### 🏦 Pool CF Settings (Stage 4)")
    col_cf1, col_cf2 = st.columns(2)
    with col_cf1:
        deploy_ratio = st.slider("Deploy Ratio (จ่ายออกจาก Pool CF)", 0.0, 1.0, 0.70, step=0.05, key="ch8_deploy",
                                 help="สัดส่วนที่จะดึงออกไป Scale Up เมื่อ Regime กลับ (ที่เหลือเป็น Reserve)")
    with col_cf2:
        crash_price_pct = st.slider("Crash Scenario (% ของราคาปัจจุบัน)", 30, 100, 70, step=5, key="ch8_crash",
                                    help="จำลองราคาตกกี่ % เพื่อดู Put Payoff")
    
    r = 0.05  # Risk-free rate
    T = 1.0   # 1 Year horizon
    put_strike = Pt * (put_strike_pct / 100.0)
    P_crash = Pt * (crash_price_pct / 100.0)  # Crash scenario price
    
    # === STAGE 1: Shannon Simple + Harvest ===
    st.subheader("🔗 Stage 1: Shannon Simple + Volatility Harvest")
    
    shannon_profit = fix_c * np.log(Pt / P0)
    harvest_profit = fix_c * 0.5 * (sigma ** 2) * T
    total_stage1 = shannon_profit + harvest_profit
    
    col_s1a, col_s1b, col_s1c = st.columns(3)
    col_s1a.metric("Shannon Profit", f"${shannon_profit:,.2f}", 
                   delta=f"fix_c × ln({Pt:.1f}/{P0:.1f})")
    col_s1b.metric("Harvest Profit (½σ²)", f"${harvest_profit:,.2f}",
                   delta=f"fix_c × ½ × {sigma:.2f}²")
    col_s1c.metric("Total Stage 1", f"${total_stage1:,.2f}",
                   delta="Shannon + Harvest")
    
    # === STAGE 2: Hedge Put ===
    st.subheader("🛡️ Stage 2: Hedge Put (จ่ายค่าประกันจากกำไร)")
    
    put_premium = black_scholes(Pt, put_strike, T, r, sigma, 'put')
    equiv_shares = fix_c / Pt
    qty_puts = equiv_shares * hedge_ratio
    total_hedge_cost = qty_puts * put_premium
    surplus = total_stage1 - total_hedge_cost
    
    col_s2a, col_s2b, col_s2c = st.columns(3)
    col_s2a.metric("Put Premium (ต่อหุ้น)", f"${put_premium:,.2f}",
                   delta=f"Strike ${put_strike:.1f}")
    col_s2b.metric("Hedge Cost รวม", f"${total_hedge_cost:,.2f}",
                   delta=f"{qty_puts:.1f} Puts × ${put_premium:.2f}")
    col_s2c.metric("Surplus (ส่วนเหลือ)", f"${surplus:,.2f}",
                   delta="กำไร - ค่าประกัน", delta_color="normal" if surplus >= 0 else "inverse")
    
    # === STAGE 3: Scale Up + Rollover ===
    st.subheader("📈 Stage 3: Scale Up (Constant Value → Free Risk)")
    
    new_fix_c = fix_c + max(surplus, 0)
    free_risk = max(surplus, 0)
    
    # Choose reference price based on selected method
    if use_method_1:
        new_ref_price = Pt  # Method 1: Reset to current price
    else:
        new_ref_price = P0  # Method 2: Keep original reference
    
    # Calculate Rollover
    b_accumulated = rollover_baseline(0.0, fix_c, new_fix_c, P0, new_ref_price, Pt)
    
    method_label = "แบบ 1 (Reset)" if use_method_1 else "แบบ 2 (เดิม)"
    
    col_s3a, col_s3b, col_s3c = st.columns(3)
    col_s3a.metric("fix_c เดิม → ใหม่", f"${new_fix_c:,.2f}",
                   delta=f"+${free_risk:,.2f} (Free Risk)")
    col_s3b.metric("Rollover Baseline (b)", f"${b_accumulated:,.2f}",
                   delta=f"เส้นต่อเนื่อง ({method_label})")
    col_s3c.metric("ราคาอ้างอิงใหม่ (t')", f"${new_ref_price:,.2f}",
                   delta=f"เดิม: ${P0:.2f}")
    
    if surplus >= 0:
        st.success(f"""
        ✅ **Free Risk!** ทุนเพิ่มขึ้น ${free_risk:,.2f} โดยไม่ต้องใส่เงินใหม่
        - fix_c: ${fix_c:,.2f} → ${new_fix_c:,.2f}
        - ราคาอ้างอิง: ${P0:.2f} → ${new_ref_price:.2f}
        - Accumulated Baseline (b): ${b_accumulated:,.2f}
        """)
    else:
        st.warning(f"""
        ⚠️ กำไรไม่พอจ่ายค่า Hedge! ขาดอีก ${abs(surplus):,.2f}
        - ลด Hedge Ratio หรือรอให้ราคาวิ่งต่อ
        """)
    
    st.markdown("---")
    
    # === STAGE 4: Put Payoff → Pool CF (ราคาลง) ===
    st.subheader("💥 Stage 4: Put Payoff → Pool CF (เมื่อราคาลง)")
    st.markdown(f"""
    **Scenario:** ราคาตกจาก **${Pt:.2f}** → **${P_crash:.2f}** ({crash_price_pct}%)  
    Put Strike = **${put_strike:.2f}** | Hedge Ratio = **{hedge_ratio}x**
    """)
    
    # Calculate Put Payoff at crash price
    put_payoff_crash = max(put_strike - P_crash, 0)
    total_put_payoff = qty_puts * put_payoff_crash
    
    # Shannon loss at crash (for comparison)
    shannon_loss = fix_c * np.log(P_crash / P0) if P_crash > 0 else 0
    
    # Roll-down cost (optional: cost to re-hedge at lower strike)
    new_put_strike_crash = P_crash * (put_strike_pct / 100.0)
    rolldown_premium = black_scholes(P_crash, new_put_strike_crash, T, r, sigma, 'put')
    rolldown_cost = qty_puts * rolldown_premium
    
    # Net into Pool CF
    pool_cf_gross = total_put_payoff
    pool_cf_net = pool_cf_gross - rolldown_cost
    
    col_s4a, col_s4b, col_s4c = st.columns(3)
    col_s4a.metric("Put Payoff (ต่อสัญญา)", f"${put_payoff_crash:,.2f}",
                   delta=f"Strike {put_strike:.1f} - Price {P_crash:.1f}")
    col_s4b.metric("Total Put Payoff", f"${total_put_payoff:,.2f}",
                   delta=f"{qty_puts:.1f} Puts × ${put_payoff_crash:.2f}")
    col_s4c.metric("Shannon Loss (ref)", f"${shannon_loss:,.2f}",
                   delta="ขาดทุนจาก Baseline", delta_color="inverse")
    
    # --- Pool CF Dashboard ---
    st.subheader("🏦 Pool CF Dashboard")
    
    # Pool CF allocation
    deploy_amount = pool_cf_net * deploy_ratio
    reserve_amount = pool_cf_net * (1 - deploy_ratio)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Pool CF (Gross)", f"${pool_cf_gross:,.2f}",
                  delta="Put Payoff ทั้งหมด")
    col_p2.metric("Re-Hedge Cost", f"${rolldown_cost:,.2f}",
                  delta=f"Roll-Down Put @ ${new_put_strike_crash:.1f}", delta_color="inverse")
    col_p3.metric("Pool CF (Net)", f"${pool_cf_net:,.2f}",
                  delta="Gross - Re-Hedge", delta_color="normal" if pool_cf_net >= 0 else "inverse")
    
    # Deploy vs Reserve
    col_d1, col_d2 = st.columns(2)
    col_d1.metric(f"🚀 Deploy ({deploy_ratio*100:.0f}%)", f"${deploy_amount:,.2f}",
                  delta="→ Scale Up fix_c (เมื่อ Regime กลับ)")
    col_d2.metric(f"🛡️ Reserve ({(1-deploy_ratio)*100:.0f}%)", f"${reserve_amount:,.2f}",
                  delta="→ สำรองฉุกเฉิน")
    
    # Pool CF Rule Box
    if pool_cf_net > 0:
        st.success(f"""
        ✅ **Pool CF พร้อมใช้งาน!**  
        - กำไร Put: **${total_put_payoff:,.2f}** | หักค่า Re-Hedge: **${rolldown_cost:,.2f}**
        - **Deploy** ${deploy_amount:,.2f} → รอ VIX กลับต่ำ → Scale Up fix_c  
        - **Reserve** ${reserve_amount:,.2f} → สำรองไว้กันเหตุไม่คาด
        
        > **Rule:** Put กำไร → เข้า Pool CF ก่อนเสมอ → **ไม่ Scale Up ทันที** ขณะตลาดยัง Crash
        """)
    else:
        st.warning(f"""
        ⚠️ **Put Payoff ไม่พอจ่าย Re-Hedge!** ราคายังไม่ลงพอให้ Put ทำกำไร
        - Put Payoff: ${total_put_payoff:,.2f} | Re-Hedge Cost: ${rolldown_cost:,.2f}
        """)
    
    st.markdown("---")
    
    # === ROLLOVER HISTORY TABLE ===
    st.subheader("📋 Rollover History (ประวัติการ Roll)")
    
    history_data = {
        "Round": ["Start", "→ Round 1"],
        "c (fix_c)": [f"${fix_c:,.2f}", f"${new_fix_c:,.2f}"],
        "t (ref price)": [f"${P0:.2f}", f"${new_ref_price:.2f}"],
        "P (current)": [f"${P0:.2f}", f"${Pt:.2f}"],
        "b (accumulated)": [f"$0.00", f"${b_accumulated:,.2f}"],
        "Equation": [
            "—",
            f"0 + {fix_c:.0f}×ln({Pt:.1f}/{P0:.1f}) − {new_fix_c:.0f}×ln({Pt:.1f}/{new_ref_price:.1f})"
        ]
    }
    
    # Multi-round simulation: if user wants to simulate one more round
    if st.checkbox("🔄 จำลอง Round 2 (ราคาวิ่งต่อ)", key="ch8_round2"):
        P_round2 = st.number_input("ราคาที่ Round 2 (P₂)", value=Pt * 1.3, step=0.1, key="ch8_p2", format="%.2f")
        
        # Round 2 calculations
        shannon_r2 = new_fix_c * np.log(P_round2 / new_ref_price)
        harvest_r2 = new_fix_c * 0.5 * (sigma ** 2) * T
        total_r2 = shannon_r2 + harvest_r2
        
        put_strike_r2 = P_round2 * (put_strike_pct / 100.0)
        put_prem_r2 = black_scholes(P_round2, put_strike_r2, T, r, sigma, 'put')
        equiv_shares_r2 = new_fix_c / P_round2
        qty_puts_r2 = equiv_shares_r2 * hedge_ratio
        hedge_cost_r2 = qty_puts_r2 * put_prem_r2
        surplus_r2 = total_r2 - hedge_cost_r2
        
        fix_c_r2 = new_fix_c + max(surplus_r2, 0)
        # Apply same rollover method for Round 2
        if use_method_1:
            ref_r2 = P_round2       # Method 1: Reset to current price
        else:
            ref_r2 = new_ref_price  # Method 2: Keep previous reference
        b_r2 = rollover_baseline(b_accumulated, new_fix_c, fix_c_r2, new_ref_price, ref_r2, P_round2)
        
        history_data["Round"].append("→ Round 2")
        history_data["c (fix_c)"].append(f"${fix_c_r2:,.2f}")
        history_data["t (ref price)"].append(f"${ref_r2:.2f}")
        history_data["P (current)"].append(f"${P_round2:.2f}")
        history_data["b (accumulated)"].append(f"${b_r2:,.2f}")
        history_data["Equation"].append(
            f"{b_accumulated:.2f} + {new_fix_c:.0f}×ln({P_round2:.1f}/{new_ref_price:.1f}) − {fix_c_r2:.0f}×ln({P_round2:.1f}/{ref_r2:.1f})"
        )
        
        st.info(f"""
        **Round 2 Results ({method_label}):**
        - Shannon: ${shannon_r2:,.2f} | Harvest: ${harvest_r2:,.2f} | Total: ${total_r2:,.2f}
        - Hedge Cost: ${hedge_cost_r2:,.2f} | Surplus: ${surplus_r2:,.2f}
        - fix_c: ${new_fix_c:,.2f} → **${fix_c_r2:,.2f}** | b = **${b_r2:,.2f}**
        """)
    
    df_history = pd.DataFrame(history_data)
    st.table(df_history)
    
    st.markdown("---")
    
    # === SANKEY DIAGRAM (Full Cycle: ขาขึ้น + ขาลง) ===
    st.subheader("🌊 Profit Flow — Full Cycle (Sankey Diagram)")
    
    # Nodes: 0-Shannon, 1-Harvest, 2-Total, 3-HedgeCost, 4-Surplus, 5-ScaleUp,
    #         6-PutPayoff, 7-PoolCF, 8-Deploy, 9-Reserve
    sankey_labels = [
        f"Shannon Profit\n${shannon_profit:,.0f}",       # 0
        f"Harvest Profit\n${harvest_profit:,.0f}",        # 1
        f"Total Stage 1\n${total_stage1:,.0f}",           # 2
        f"Hedge Cost\n${total_hedge_cost:,.0f}",          # 3
        f"Surplus\n${max(surplus, 0):,.0f}",              # 4
        f"Scale Up fix_c\n${new_fix_c:,.0f}",             # 5
        f"Put Payoff\n${total_put_payoff:,.0f}",          # 6
        f"Pool CF\n${max(pool_cf_net, 0):,.0f}",          # 7
        f"Deploy ({deploy_ratio*100:.0f}%)\n${max(deploy_amount,0):,.0f}",  # 8
        f"Reserve ({(1-deploy_ratio)*100:.0f}%)\n${max(reserve_amount,0):,.0f}",  # 9
    ]
    
    # Links: ขาขึ้น (Stage 1→2→3,4→5) + ขาลง (6→7→8,9)
    sankey_source = [0, 1, 2, 2]
    sankey_target = [2, 2, 3, 4]
    sankey_value = [
        max(shannon_profit, 0.01), 
        max(harvest_profit, 0.01), 
        max(total_hedge_cost, 0.01), 
        max(surplus, 0.01)
    ]
    sankey_colors = [
        'rgba(99, 110, 250, 0.6)',   # Shannon → Total
        'rgba(0, 204, 150, 0.6)',    # Harvest → Total
        'rgba(239, 85, 59, 0.6)',    # Total → Hedge
        'rgba(255, 161, 90, 0.6)',   # Total → Surplus
    ]
    
    if surplus > 0:
        sankey_source.append(4)
        sankey_target.append(5)
        sankey_value.append(surplus)
        sankey_colors.append('rgba(171, 99, 250, 0.6)')
    
    # Stage 4: Put Payoff → Pool CF → Deploy + Reserve
    if total_put_payoff > 0:
        # Put Payoff → Pool CF
        sankey_source.append(6)
        sankey_target.append(7)
        sankey_value.append(max(pool_cf_net, 0.01))
        sankey_colors.append('rgba(255, 65, 54, 0.6)')
        
        if pool_cf_net > 0:
            # Pool CF → Deploy
            sankey_source.append(7)
            sankey_target.append(8)
            sankey_value.append(max(deploy_amount, 0.01))
            sankey_colors.append('rgba(25, 211, 243, 0.6)')
            # Pool CF → Reserve
            sankey_source.append(7)
            sankey_target.append(9)
            sankey_value.append(max(reserve_amount, 0.01))
            sankey_colors.append('rgba(180, 180, 180, 0.5)')
    
    node_colors = ['#636EFA', '#00CC96', '#FFA15A', '#EF553B', '#AB63FA', '#19D3F3',
                   '#FF4136', '#FF851B', '#2ECC40', '#AAAAAA']
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=1),
            label=sankey_labels,
            color=node_colors
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=sankey_value,
            color=sankey_colors
        )
    )])
    fig_sankey.update_layout(
        title="Chain System: Full Cycle Profit Flow (ขาขึ้น + ขาลง)",
        font_size=13,
        height=500
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # === WATERFALL CHART (Full Cycle) ===
    st.subheader("📊 Waterfall Chart — Full Cycle (ขาขึ้น + ขาลง)")
    
    waterfall_labels = [
        "Shannon Profit",
        "Harvest Profit", 
        "Total Stage 1",
        "Hedge Cost",
        "Surplus (Free Risk)",
        "New fix_c",
        "───── ขาลง ─────",
        "Put Payoff (Crash)",
        "Re-Hedge Cost",
        "Pool CF Net"
    ]
    waterfall_measures = [
        "relative", "relative", "total", "relative", "total", "total",
        "relative", "relative", "relative", "total"
    ]
    waterfall_values = [
        shannon_profit, 
        harvest_profit, 
        total_stage1,
        -total_hedge_cost, 
        surplus,
        new_fix_c,
        0,
        total_put_payoff,
        -rolldown_cost,
        pool_cf_net
    ]
    waterfall_text = [f"${v:,.2f}" if v != 0 else "▼" for v in waterfall_values]
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Chain", 
        orientation="v",
        measure=waterfall_measures,
        x=waterfall_labels,
        y=waterfall_values,
        text=waterfall_text,
        textposition="outside",
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
        increasing=dict(marker_color='#00CC96'),
        decreasing=dict(marker_color='#EF553B'),
        totals=dict(marker_color='#636EFA')
    ))
    fig_waterfall.update_layout(
        title="Chain System: Full Cycle Waterfall (ขาขึ้น + ขาลง)",
        yaxis_title="Value ($)",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # === PAYOFF PROFILE: Chained vs Stand-alone ===
    st.subheader("📈 Payoff Profile: Chained System")
    
    price_range = np.linspace(P0 * 0.3, Pt * 2.5, 200)
    
    # A. Shannon Simple (old fix_c, old ref price)
    baseline_old = fix_c * (1 + np.log(price_range / P0))
    
    # B. Shannon Simple with Rollover (new fix_c, new ref price, continuous)
    # After rollover: value = b + c' * (1 + ln(P / t'))
    baseline_rolled = b_accumulated + new_fix_c * (1 + np.log(price_range / new_ref_price))
    
    # C. Chained + Put protection payoff at current params
    put_payoff_chain = qty_puts * np.maximum(put_strike - price_range, 0)
    chained_shielded = baseline_rolled + put_payoff_chain - total_hedge_cost
    
    # D. Buy & Hold equivalent
    buy_hold = (fix_c / P0) * price_range
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=price_range, y=buy_hold, name="Buy & Hold",
        line=dict(dash='dot', color='gray', width=1)
    ))
    fig_payoff.add_trace(go.Scatter(
        x=price_range, y=baseline_old, name=f"Shannon (c={fix_c:.0f}, t={P0:.1f})",
        line=dict(dash='dash', color='purple', width=1)
    ))
    fig_payoff.add_trace(go.Scatter(
        x=price_range, y=baseline_rolled, name=f"After Rollover (c'={new_fix_c:.0f}, t'={new_ref_price:.1f})",
        line=dict(color='#636EFA', width=2)
    ))
    fig_payoff.add_trace(go.Scatter(
        x=price_range, y=chained_shielded, name=f"Chained + Shield ({hedge_ratio}x Puts)",
        line=dict(color='#00CC96', width=3)
    ))
    
    # Mark rollover point
    fig_payoff.add_vline(x=Pt, line_width=2, line_dash="dash", line_color="orange",
                         annotation_text=f"Rollover Point (P={Pt:.1f})")
    fig_payoff.add_vline(x=P0, line_width=1, line_dash="dot", line_color="gray",
                         annotation_text=f"Start (P₀={P0:.1f})")
    # Mark crash scenario price
    fig_payoff.add_vline(x=P_crash, line_width=2, line_dash="dash", line_color="red",
                         annotation_text=f"Crash ({crash_price_pct}%) P={P_crash:.1f}")
    
    fig_payoff.update_layout(
        title="Chain System: Continuous Baseline + Shield + Crash Scenario",
        xaxis_title="Stock Price ($)",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig_payoff, use_container_width=True)
    
    st.info(f"""
    **Chain System — Full Cycle Analysis:**
    
    **ขาขึ้น (Stage 1→3):**
    1. **เส้นสีม่วง (Shannon เดิม):** Baseline เดิมก่อน Scale Up
    2. **เส้นสีน้ำเงิน (After Rollover):** Baseline ใหม่หลัง Rollover — **ต่อเนื่อง** ไม่กระโดด
    3. **เส้นสีเขียว (Chained + Shield):** เมื่อเติม Put Hedge → ขาลงมี Anti-Fragile
    
    **ขาลง (Stage 4):**
    4. **Put ระเบิดกำไร** ${total_put_payoff:,.2f} → เข้า **Pool CF** ${pool_cf_net:,.2f}
    5. **Deploy** {deploy_ratio*100:.0f}% (${deploy_amount:,.2f}) เมื่อ Regime กลับ → **Scale Up**
    6. **Reserve** {(1-deploy_ratio)*100:.0f}% (${reserve_amount:,.2f}) → สำรองฉุกเฉิน
    
    > **หลักการ:** กำไร Put → Pool CF → **ห้าม Scale Up ทันทีตอนตลาด Crash** → รอ VIX กลับ → Deploy
    """)


# --- Master Study Guide and Quiz ---
def master_study_guide_quiz():
    st.header("📝 แบบทดสอบความเข้าใจ: Master Study Guide")
    st.markdown("แบบทดสอบสั้น 10 ข้อ (Short-Answer Quiz) เพื่อทบทวนความเข้าใจในกลยุทธ์ Flywheel 1-7")
    
    questions = [
        ("1. สมการ Baseline B_t = k + ... มีประโยชน์หลักอย่างไรในระบบนี้?", 
         "ใช้เป็นเส้นมาตรวัดทางคณิตศาสตร์เพื่อสะท้อนผลตอบแทนเชิงโครงสร้างที่แท้จริง หากระบบ Rebalance ทำงานได้ดี เส้นกำไรจริงจะต้องลอยตัวสูงกว่าเส้น Baseline เสมอ"),
        
        ("2. กระแสเงินสดที่เรียกว่า Volatility Premium เกิดขึ้นจากกระบวนการใด?", 
         "เกิดจากการทำ Daily Rebalance หรือการทยอยช้อนซื้อเมื่อราคาหุ้นลดลง และทยอยขายทำกำไรเมื่อราคาหุ้นสูงขึ้นอย่างต่อเนื่อง (Buy Low, Sell High)"),
        
        ("3. ทำไมเราจึงต้องใช้สัดส่วน 80% LEAPS + 20% Liquidity แทนการใช้เงิน 100% ซื้อ LEAPS ทั้งหมด?", 
         "80% ใช้สร้างอำนาจครอบครองหุ้น (Convexity) ส่วน 20% เป็นกองเงินสดสำรองที่ 'จำเป็น' ต้องมีไว้เพื่อเป็นสภาพคล่องในการทำ Rebalance"),
        
        ("4. คุณสมบัติ 'Flatline' ของ LEAPS ช่วยแก้ปัญหาการรับมีดตก (Falling Knife) ได้อย่างไร?", 
         "LEAPS จำกัดการขาดทุนสูงสุดไว้เท่ากับค่า Premium ต่อให้ราคาหุ้นเหลือศูนย์ กราฟขาดทุนของพอร์ตก็จะราบขนานไปกับพื้น ไม่ทะลุลงเหวลึกเหมือนหุ้นจริง"),
        
        ("5. การซื้อ Put Option ในสัดส่วน 2 เท่า (Over-hedging) ส่งผลต่อพอร์ตอย่างไรเมื่อเกิด Market Crash?", 
         "เปลี่ยนสถานะพอร์ตให้เป็น Anti-Fragile เมื่อตลาดพัง Put x2 จะระเบิดมูลค่าขึ้นมหาศาล ชดเชยความเสียหายและทำกำไรสุทธิสวนทางตลาด"),
        
        ("6. เรานำเงินทุนจากแหล่งใดมาใช้ซื้อ Put Option เพื่อป้องกันความเสี่ยง?", 
         "ใช้เงิน 'กำไรส่วนเกิน' จาก Volatility Premium (Flyweel 2) และ Synthetic Dividend (Flywheel 6) มาเป็นค่าเบี้ยประกันแบบ Zero-Cost Hedge โดยไม่ควักเนื้อ"),
        
        ("7. ปัจจัยใดที่ใช้เป็นตัวชี้วัดในการตัดสินใจ 'Scale Up' หรือ 'Scale Down' ระบบ?", 
         "สภาวะความผันผวนของตลาด (Volatility Regime) โดยมักจะใช้ดัชนี VIX Index เป็นเกณฑ์"),
        
        ("8. กลไกใดในระบบที่ทำหน้าที่สร้าง 'ปันผลเทียม (Synthetic Dividend)'?", 
         "การขาย Short Call หรือ Short Put ระยะสั้น เพื่อกินเปล่าค่าเสื่อมเวลา (Theta Decay) ตราบใดที่ราคาหุ้นไม่ทะลุกรอบ"),
        
        ("9. ค่าเสื่อมเวลา (Theta Decay) เป็นศัตรูกับ Flywheel ใด และเป็นมิตรกับ Flywheel ใด?", 
         "เป็นศัตรูกับฝั่ง Long Options (LEAPS, Put Hedge) แต่เป็นมิตรกับฝั่ง Short Options (Synthetic Dividend)"),
        
        ("10. การใช้บัญชี Portfolio Margin ร่วมกับ T-Bills สร้างข้อได้เปรียบอย่างไร?", 
         "ช่วยให้เงินต้นปลอดภัยใน T-Bills (กินดอกเบี้ย) ขณะที่ใช้ Margin Unlock อำนาจซื้อมาทำกำไรซ้อนทับอีกชั้นหนึ่ง (Yield Stacking)")
    ]
    
    score = 0
    for i, (q, a) in enumerate(questions):
        st.subheader(q)
        user_answer = st.text_area(f"คำตอบของคุณ (ข้อ {i+1})", key=f"q{i}")
        if st.checkbox(f"ดูเฉลย (ข้อ {i+1})", key=f"chk{i}"):
            st.info(f"**เฉลย:** {a}")
            st.write("---")

# --- Paper Trading Workshop ---
def paper_trading_workshop():
    st.header("🛠️ Paper Trading Workshop: Real Portfolio Calculation")
    st.markdown("""
    **ภารกิจ:** ลองวางแผนจัดพอร์ตด้วยเงินจำลอง **$100,000** 
    โดยใช้โครงสร้าง **80/20 Stock Replacement**
    """)
    
    # 1. Inputs
    st.subheader("1. กำหนดค่าเริ่มต้น (Initialize)")
    capital = st.number_input("เงินทุนตั้งต้น (Capital)", value=100000, step=10000)
    
    st.markdown("---")
    
    # 2. Allocation
    st.subheader("2. จัดสรรเงินทุน (Allocation 80/20)")
    
    leaps_budget = capital * 0.80
    liquidity_budget = capital * 0.20
    
    col1, col2 = st.columns(2)
    col1.metric("LEAPS Budget (80%)", f"${leaps_budget:,.2f}")
    col2.metric("Liquidity Pool (20%)", f"${liquidity_budget:,.2f}")
    
    st.markdown("---")
    
    # 3. LEAPS Selection
    st.subheader("3. เลือก LEAPS (Stock Replacement)")
    strike_price = st.number_input("เลือก Strike Price (Deep ITM)", value=80.0, step=5.0, help="ควรเลือก Delta ~ 0.80-0.90")
    premium_per_share = st.number_input("ราคา LEAPS Premium (ต่อหุ้น)", value=25.0, step=0.5)
    contract_multiplier = 100 # Standard US Options
    cost_per_contract = premium_per_share * contract_multiplier
    
    # Calculate Number of Contracts
    num_contracts = int(leaps_budget // cost_per_contract)
    actual_leaps_cost = num_contracts * cost_per_contract
    remaining_cash_from_leaps = leaps_budget - actual_leaps_cost
    
    total_liquidity = liquidity_budget + remaining_cash_from_leaps
    
    st.info(f"""
    **ผลการคำนวณ LEAPS:**
    - ราคาต่อสัญญา: ${cost_per_contract:,.2f}
    - ซื้อได้สูงสุด: **{num_contracts} สัญญา**
    - ต้นทุนรวม: ${actual_leaps_cost:,.2f}
    - เงินเหลือทบเข้า Liquidity: ${remaining_cash_from_leaps:,.2f}
    """)
    
    st.markdown("---")
    
    # 4. Hedge Protection
    st.subheader("4. สร้างเกราะป้องกัน (Put Hedge x2)")
    st.markdown("ซื้อ Put Options จำนวน 2 เท่าของจำนวนสัญญา LEAPS")
    
    put_premium = st.number_input("ราคา Put Premium (ต่อหุ้น)", value=2.0, step=0.1)
    put_cost_per_contract = put_premium * contract_multiplier
    
    needed_put_contracts = num_contracts * 2
    total_hedge_cost = needed_put_contracts * put_cost_per_contract
    
    st.warning(f"""
    **Hedge Requirement:**
    - จำนวน Put ที่ต้องซื้อ (x2): **{needed_put_contracts} สัญญา**
    - ต้นทุน Hedge รวม: ${total_hedge_cost:,.2f}
    - สถานะเงินสดหลังจ่ายค่า Hedge: ${(total_liquidity - total_hedge_cost):,.2f}
    """)
    
    # Final Summary
    st.markdown("---")
    st.subheader("📊 สรุปพอร์ตโฟลิโอจำลอง (Final Portfolio Status)")
    
    final_data = {
        "Asset": ["LEAPS (Long Call)", "Put Options (Hedge)", "Cash (Liquidity)"],
        "Contracts/Units": [f"{num_contracts} Contracts", f"{needed_put_contracts} Contracts", "-"],
        "Value ($)": [actual_leaps_cost, total_hedge_cost, total_liquidity - total_hedge_cost],
        "Allocation (%)": [
            actual_leaps_cost/capital*100, 
            total_hedge_cost/capital*100, 
            (total_liquidity - total_hedge_cost)/capital*100
        ]
    }
    
    df_final = pd.DataFrame(final_data)
    st.table(df_final)
    
    if (total_liquidity - total_hedge_cost) < 0:
        st.error("⚠️ เงินสดไม่พอจ่ายค่า Hedge! กรุณาลดจำนวนสัญญา LEAPS หรือเพิ่มเงินเริ่มต้น")
    else:
        st.success("✅ พอร์ตพร้อมลุย! (Ready to Deploy)")


# --- Glossary Section ---
def glossary_section():
    st.header("📚 อภิธานศัพท์ (Glossary)")
    
    terms = {
        "Alpha (อัลฟ่า)": "ผลตอบแทนส่วนเกินที่ระบบทำได้เหนือกว่าเส้นมาตรวัด (Benchmark) หรือผลตอบแทนของตลาดโดยรวม",
        "Anti-Fragile (แอนตี้แฟรไจล์)": "คุณสมบัติของระบบที่ยิ่งโดนแรงกระแทกจากวิกฤต (ความผันผวนรุนแรง) ยิ่งแข็งแกร่งและสร้างผลกำไรได้มากขึ้น",
        "Convexity (ความโค้ง)": "ลักษณะกราฟผลตอบแทนที่ 'ขาดทุนช้าลงเมื่อราคาตก (Flatline Risk) และได้กำไรเร็วขึ้นเมื่อราคาขึ้น (Gamma Acceleration)'",
        "Dynamic Hedging": "การปรับเปลี่ยนสัดส่วนของการทำประกัน (Put Options) และขนาดหน้าตัก ให้ยืดหยุ่นตามสภาวะความผันผวน (VIX)",
        "Fix Capital (fix_c)": "มูลค่าหน้าตักเป้าหมายที่ระบบต้องรักษาให้คงที่ผ่านการทำ Rebalance",
        "Gamma": "อัตราเร่งของกำไร (ความชันของ Delta) ช่วยให้พอร์ตกำไรก้าวกระโดดเมื่อถูกทาง",
        "LEAPS": "Long-Term Equity Anticipation Securities (Options อายุยาว 1-3 ปี) ใช้ทำ Stock Replacement",
        "Portfolio Margin": "ระบบคำนวณหลักประกันแบบ 'ความเสี่ยงรวมสุทธิ' ช่วยปลดล็อก Buying Power",
        "Stock Replacement": "การถือ LEAPS 80% + เงินสด 20% แทนการถือหุ้น 100% เพื่อจำกัดความเสี่ยงและเพิ่มสภาพคล่อง",
        "Synthetic Dividend": "กระแสเงินสดจากการขาย Short Call/Put (กิน Theta) เพื่อนำมาจ่ายค่าประกัน",
        "T-Bills (Treasury Bills)": "พันธบัตรรัฐบาลระยะสั้น (Risk-free Asset) ใช้เป็นหลักประกัน (Collateral) ชั้นดีในการเทรด",
        "Theta Decay": "ค่าเสื่อมเวลาของ Option ที่เราได้รับเมื่ออยู่ฝั่ง Short (คนขาย) หรือเสียเมื่ออยู่ฝั่ง Long (คนซื้อ)",
        "Volatility Premium": "กำไรส่วนต่างที่เกิดจากการซื้อถูก-ขายแพง (Rebalance) ในตลาดที่ผันผวน"
    }

    for term, definition in terms.items():
        with st.expander(f"**{term}**"):
            st.write(definition)
