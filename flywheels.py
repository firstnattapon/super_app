
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import json
import os
import re

# ============================================================
# UTILITIES
# ============================================================

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def generate_gbm(S0, mu, sigma, T, dt, n_sims=1):
    """Generate Geometric Brownian Motion price path."""
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S

# ============================================================
# DATA LAYER — trading_data.json
# ============================================================

_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data.json")

def load_trading_data():
    """Load portfolio data from trading_data.json."""
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_trading_data(data):
    """Save portfolio data to trading_data.json."""
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_final(final_str):
    """Parse 'Final' field → (t, c, b).  e.g. '12, 4000, -519.45' → (12.0, 4000.0, -519.45)"""
    if not final_str:
        return None, None, None
    parts = [p.strip() for p in final_str.split(",")]
    try:
        t = float(parts[0]) if len(parts) > 0 else None
        c = float(parts[1]) if len(parts) > 1 else None
        b = float(parts[2]) if len(parts) > 2 else 0.0
        return t, c, b
    except (ValueError, IndexError):
        return None, None, None

def parse_beta_numbers(beta_str):
    """Extract Ev and Lock_P&L from beta_Equation string.
    e.g. ' Ev: -204.00 + Lock_P&L: +0'  → (ev, lock_pnl)
    Ev = Extrinsic Value (มูลค่าทางเวลา / ค่า K จ่ายทิ้ง)
    EV = Premium − Intrinsic Value
    """
    ev, lock_pnl = 0.0, 0.0
    if not beta_str:
        return ev, lock_pnl
    # Extract Ev
    ev_match = re.search(r'Ev:\s*([+-]?[\d,.]+)', beta_str)
    if ev_match:
        try:
            ev = float(ev_match.group(1).replace(",", ""))
        except ValueError:
            pass
    # Extract Lock_P&L (may have multiple values like +1618.48 +498|+231)
    lock_match = re.search(r'Lock_P&L:\s*(.+)', beta_str)
    if lock_match:
        raw = lock_match.group(1).strip()
        # Split by | and + to handle patterns like "+1618.48 +498|+231" or "-519.45-438+599+485"
        # Replace | with +, then split all numbers
        raw = raw.replace("|", "+")
        nums = re.findall(r'[+-]?[\d,.]+', raw)
        for n in nums:
            try:
                lock_pnl += float(n.replace(",", ""))
            except ValueError:
                pass
    return ev, lock_pnl

def parse_beta_net(beta_mem_str):
    """Extract Net value from beta_momory string. e.g. 'Net: -204.00' → -204.0"""
    if not beta_mem_str:
        return 0.0
    m = re.search(r'Net:\s*([+-]?[\d,.]+)', beta_mem_str)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return 0.0
    return 0.0

def parse_surplus_iv(surplus_str):
    """Extract Surplus IV (Put premium income) from Surplus_Iv string.
    e.g. 'Iv_Put: (4.98*100)= 498 | (2.31*100)=231' → 729.0
    """
    if not surplus_str or "No_Expiry" in surplus_str:
        return 0.0
    matches = re.findall(r'=\s*([+-]?\d+(?:\.\d+)?)', surplus_str)
    total = 0.0
    for m in matches:
        try:
            total += float(m)
        except ValueError:
            pass
    return total

def get_rollover_history(ticker_data):
    """Extract all history entries in order, returning list of dicts."""
    history = []
    i = 1
    while True:
        key_desc = f"history_{i}"
        key_calc = f"history_{i}.1"
        if key_desc not in ticker_data and key_calc not in ticker_data:
            break
        entry = {"step": i}
        entry["description"] = ticker_data.get(key_desc, "")
        entry["calculation"] = ticker_data.get(key_calc, "")
        # Parse b value from calculation string
        calc_str = entry["calculation"]
        b_match = re.search(r'b\s*=\s*([+-]?[\d,.]+)', calc_str)
        if b_match:
            try:
                entry["b"] = float(b_match.group(1).replace(",", ""))
            except ValueError:
                entry["b"] = None
        else:
            entry["b"] = None
        # Parse c value
        c_match = re.search(r'\|\s*c\s*=\s*([\d,.]+)', calc_str)
        if c_match:
            try:
                entry["c"] = float(c_match.group(1).replace(",", ""))
            except ValueError:
                entry["c"] = None
        else:
            entry["c"] = None
        # Parse t value
        t_match = re.search(r',\s*t\s*=\s*([\d,.]+)', calc_str)
        if t_match:
            try:
                entry["t"] = float(t_match.group(1).replace(",", ""))
            except ValueError:
                entry["t"] = None
        else:
            entry["t"] = None
        history.append(entry)
        i += 1
    return history

def build_portfolio_df(data):
    """Build a pandas DataFrame summarizing all tickers."""
    rows = []
    for item in data:
        ticker = item.get("ticker", "???")
        t, c, b = parse_final(item.get("Final", ""))
        ev, lock_pnl = parse_beta_numbers(item.get("beta_Equation", ""))
        net = parse_beta_net(item.get("beta_momory", ""))
        surplus_iv = parse_surplus_iv(item.get("Surplus_Iv", ""))
        rows.append({
            "Ticker": ticker,
            "Price (t)": t,
            "Fix_C": c,
            "Baseline (b)": b if b else 0.0,
            "Ev (Extrinsic)": ev,
            "Lock P&L": lock_pnl,
            "Surplus IV": surplus_iv,
            "Net": net,
        })
    return pd.DataFrame(rows)


# ============================================================
# CHAPTERS 0-7 — Placeholders
# ============================================================

def chapter_0_introduction():
    st.header("บทที่ 0: Introduction")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_1_baseline():
    st.header("บทที่ 1: Baseline")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_2_shannon_process():
    st.header("บทที่ 2: Shannon Process")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_3_volatility_harvesting():
    st.header("บทที่ 3: Volatility Harvesting")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_4_black_swan_shield():
    st.header("บทที่ 4: Black Swan Shield")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_5_dynamic_scaling():
    st.header("บทที่ 5: Dynamic Scaling")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_6_synthetic_dividend():
    st.header("บทที่ 6: Synthetic Dividend")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_7_collateral_magic():
    st.header("บทที่ 7: Collateral Magic")
    st.warning("Content currently unavailable. Please restore from backup if needed.")


# ============================================================
# CHAPTER 8: CHAIN SYSTEM (ระบบลูกโซ่) — FINAL PRODUCT
# ============================================================

def chapter_chain_system():
    st.header("บทที่ 8: Chain System (ระบบลูกโซ่)")
    st.markdown("""
    **Concept:** เชื่อมกำไรจากทุก Flywheel เข้าเป็น **ลูกโซ่** (Chain) — 
    กำไรจากขั้นหนึ่งไหลไปเป็น "เชื้อเพลิง" ให้ขั้นถัดไป วนเป็นวงจร **ทั้งขาขึ้น + ขาลง**
    
    > **ขาขึ้น:** กำไร Shannon + Harvest → จ่ายค่า Put Hedge → Surplus → Scale Up fix_c = **Free Risk**
    > 
    > **ขาลง:** Put ระเบิดกำไร → เข้า **Pool CF** → Deploy (เมื่อ Regime กลับ) + Reserve (สำรอง)
    """)

    with st.expander("📐 สมการ Continuous Rollover"):
        st.latex(r"b_{new} = b_{old} + c \cdot \ln(P/t_{old}) - c' \cdot \ln(P/t_{new})")
        st.caption("ปรับ Baseline ให้ต่อเนื่องเมื่อเปลี่ยน fix_c และ re-center ราคา t")
    
    with st.expander("💡 Extrinsic Value (Ev) — ค่า K จ่ายทิ้ง"):
        st.latex(r"\text{Extrinsic Value (Ev)} = \text{Premium} - \text{Intrinsic Value}")
        st.caption("มูลค่าทางเวลาที่จ่ายค่า LEAPS — เป็นต้นทุนที่ต้องชนะให้ได้จากระบบ Chain")

    # --- Load real data ---
    data = load_trading_data()

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Portfolio Dashboard",
        "🔗 Chain Flow (Simulation)",
        "📜 Rollover History",
        "➕ Manage Data"
    ])

    # ==========================================================
    # TAB 1: PORTFOLIO DASHBOARD
    # ==========================================================
    with tab1:
        _render_portfolio_dashboard(data)

    # ==========================================================
    # TAB 2: CHAIN FLOW (PRESERVED SIMULATION)
    # ==========================================================
    with tab2:
        _render_chain_flow()

    # ==========================================================
    # TAB 3: ROLLOVER HISTORY
    # ==========================================================
    with tab3:
        _render_rollover_history(data)

    # ==========================================================
    # TAB 4: MANAGE DATA
    # ==========================================================
    with tab4:
        _render_manage_data(data)


# ----------------------------------------------------------
# TAB 1: Portfolio Dashboard
# ----------------------------------------------------------
def _render_portfolio_dashboard(data):
    if not data:
        st.info("ยังไม่มีข้อมูล — เพิ่มหุ้นที่แท็บ ➕ Manage Data")
        return

    df = build_portfolio_df(data)

    # --- Summary Metrics ---
    total_ev = df["Ev (Extrinsic)"].sum()
    total_lock = df["Lock P&L"].sum()
    total_surplus = df["Surplus IV"].sum()
    total_net = df["Net"].sum()
    total_c = df["Fix_C"].sum()
    n_tickers = len(df)
    n_profit = (df["Net"] > 0).sum()
    n_loss = (df["Net"] < 0).sum()

    st.subheader("Portfolio Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fix_C (Deployed)", f"${total_c:,.0f}", f"{n_tickers} tickers")
    m2.metric("Ev (ค่า K จ่ายทิ้ง)", f"${total_ev:,.2f}",
              delta="Extrinsic Cost", delta_color="inverse")
    m3.metric("Lock P&L + Surplus IV", f"${total_lock + total_surplus:,.2f}",
              delta=f"Lock {total_lock:,.0f} + IV {total_surplus:,.0f}")
    m4.metric("💰 Net P&L (รวมทั้งพอร์ต)", f"${total_net:,.2f}",
              delta=f"🟢{n_profit} 🔴{n_loss}",
              delta_color="normal" if total_net >= 0 else "inverse")

    st.divider()

    # --- Per-Ticker Table ---
    st.subheader("Per-Ticker Breakdown")

    # Style the dataframe
    def color_net(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return "color: #00c853"
            elif val < 0:
                return "color: #ff1744"
        return ""

    styled = df.style.format({
        "Price (t)": "${:,.2f}",
        "Fix_C": "${:,.0f}",
        "Baseline (b)": "${:,.2f}",
        "Ev (Extrinsic)": "${:,.2f}",
        "Lock P&L": "${:,.2f}",
        "Surplus IV": "${:,.2f}",
        "Net": "${:,.2f}",
    })
    # Use map (pandas >= 2.1) with fallback to applymap
    try:
        styled = styled.map(color_net, subset=["Net", "Baseline (b)", "Lock P&L"])
    except AttributeError:
        styled = styled.applymap(color_net, subset=["Net", "Baseline (b)", "Lock P&L"])

    st.dataframe(styled, use_container_width=True, height=400)

    st.divider()

    # --- Bar Chart: Net P&L per Ticker ---
    st.subheader("Net P&L per Ticker")
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in df["Net"]]
    fig_bar = go.Figure(data=[go.Bar(
        x=df["Ticker"], y=df["Net"],
        marker_color=colors,
        text=[f"${v:,.0f}" for v in df["Net"]],
        textposition="outside",
    )])
    fig_bar.update_layout(
        title="Net P&L = Ev + Lock P&L (per ticker)",
        xaxis_title="Ticker", yaxis_title="Net P&L ($)",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Waterfall: Ev vs Lock P&L Breakdown ---
    st.subheader("Waterfall: Ev → Lock P&L → Net")
    fig_wf = go.Figure(go.Waterfall(
        x=["Ev (Cost)", "Lock P&L", "Surplus IV", "Net"],
        y=[total_ev, total_lock, total_surplus, 0],
        measure=["relative", "relative", "relative", "total"],
        text=[f"${total_ev:,.0f}", f"${total_lock:,.0f}",
              f"${total_surplus:,.0f}", f"${total_net:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="gray", width=1)),
        increasing_marker_color="#00c853",
        decreasing_marker_color="#ff1744",
        totals_marker_color="#2196f3",
    ))
    fig_wf.update_layout(title="Portfolio P&L Waterfall", height=380)
    st.plotly_chart(fig_wf, use_container_width=True)


# ----------------------------------------------------------
# TAB 2: Chain Flow (Preserved Simulation Logic)
# ----------------------------------------------------------
def _render_chain_flow():
    """Preserved: Stage 1-4 simulation with Sankey + Payoff charts."""

    # --- Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Shannon Config")
        fix_c = st.number_input("Fixed Capital ($)", 1000, 100000, 10000, 1000, key="chain_c")
        P0 = st.number_input("Initial Price ($)", 10, 500, 100, 10, key="chain_p0")
        sigma = st.slider("Volatility (σ)", 0.1, 2.0, 0.5, 0.1, key="chain_sig")

    with col2:
        st.subheader("2. Hedge Config (Put)")
        hedge_ratio = st.slider("Hedge Ratio (contracts/fix_c unit)", 0.1, 2.0, 1.0, 0.1)
        qty_puts = (fix_c / P0) * hedge_ratio

        st.markdown("---")
        st.subheader("3. Pool CF & Crash Sim")
        deploy_ratio = st.slider("Deploy Ratio (from Pool CF)", 0.0, 1.0, 0.7, 0.1,
                                 help="% of Net Put Profit to Deploy")
        crash_price_pct = st.slider("Simulate Crash Price (%)", 30, 100, 50, 5,
                                    help="% of P0")
        P_crash = P0 * (crash_price_pct / 100.0)
        st.metric("Crash Price Scenario", f"${P_crash:.1f}")

    # --- Calculations ---
    r = 0.04
    T = 1.0
    put_strike_pct = 0.9
    put_strike = P0 * put_strike_pct
    put_premium = black_scholes(P0, put_strike, T, r, sigma, 'put')
    cost_hedge = qty_puts * put_premium

    harvest_profit = fix_c * 0.5 * (sigma ** 2) * T

    st.divider()

    # --- Stage 1-3: Bull/Sideway ---
    st.subheader("Stage 1-3: Bull/Sideway Flow")
    c1, c2, c3 = st.columns(3)
    c1.metric("1. Harvest Profit (Est.)", f"${harvest_profit:.2f}", f"+ Volatility {sigma}")
    c2.metric("2. Hedge Cost", f"${cost_hedge:.2f}", f"- Put Premium")

    surplus = harvest_profit - cost_hedge
    c3.metric("3. Surplus (Fuel)", f"${surplus:.2f}",
              delta="Scale Up Possible" if surplus > 0 else "Deficit",
              delta_color="normal" if surplus > 0 else "inverse")

    # --- Stage 4: Crash Scenario ---
    st.divider()
    st.subheader(f"Stage 4: Downside Scenario (Price Crashes to ${P_crash:.1f})")

    put_payoff_crash = max(0, put_strike - P_crash)
    total_put_payoff = qty_puts * put_payoff_crash

    # Shannon Net: Price Loss + Harvest Profit
    shannon_price_term = fix_c * np.log(P_crash / P0) if P_crash > 0 else 0
    shannon_harvest_term = fix_c * 0.5 * (sigma ** 2) * T
    shannon_net_ref = shannon_price_term + shannon_harvest_term

    # Rolldown Cost
    new_strike_crash = P_crash * put_strike_pct
    rolldown_premium = black_scholes(P_crash, new_strike_crash, T, r, sigma, 'put')
    rolldown_cost = qty_puts * rolldown_premium

    # Pool CF Net
    pool_cf_gross = total_put_payoff
    pool_cf_net = pool_cf_gross - rolldown_cost

    s4a, s4b, s4c = st.columns(3)
    s4a.metric("Put Payoff (Unit)", f"${put_payoff_crash:.2f}", f"Strike {put_strike:.1f}")
    s4b.metric("Total Put Payoff", f"${total_put_payoff:,.2f}", f"{qty_puts:.1f} Puts")
    s4c.metric("Shannon Net (Ref)", f"${shannon_net_ref:,.2f}",
               f"Price {shannon_price_term:,.0f} + Harvest {shannon_harvest_term:,.0f}")

    # --- Pool CF Dashboard ---
    st.markdown("#### 🎱 Pool CF Dashboard")
    with st.container(border=True):
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Pool CF (Gross)", f"${pool_cf_gross:,.2f}")
        pc2.metric("Re-Hedge Cost", f"${rolldown_cost:,.2f}", "- Cost to Armor")
        pc3.metric("Pool CF (Net)", f"${pool_cf_net:,.2f}", "Available for Action")

        deploy_amount = pool_cf_net * deploy_ratio if pool_cf_net > 0 else 0
        reserve_amount = pool_cf_net * (1 - deploy_ratio) if pool_cf_net > 0 else 0

        pc4.caption(f"Action (Ratio {deploy_ratio:.1f})")
        pc4.write(f"**Deploy:** ${deploy_amount:,.2f}")
        pc4.write(f"**Reserve:** ${reserve_amount:,.2f}")

    if pool_cf_net > 0:
        st.success(f"✅ **Survive & Thrive:** กำไรจาก Put (${pool_cf_net:,.2f}) พร้อม Deploy เพื่อ Scale Up fix_c ที่ราคาต่ำ ($ {P_crash:.1f})")
    else:
        st.error("⚠️ **Warning:** Payoff ไม่พอคลุมค่า Re-Hedge")

    # --- Charts ---
    st.divider()

    # 1. Sankey Diagram
    labels = ["Shannon Income", "Harvest (Vol)", "Put Hedge", "Surplus", "Scale Up",
              "Put Payoff", "Pool CF", "Re-Hedge Cost", "Deploy", "Reserve"]

    value_harvest = max(1, harvest_profit)
    value_hedge = max(0.01, cost_hedge)
    value_surplus = max(0.01, surplus) if surplus > 0 else 0.01

    value_put = max(1, total_put_payoff)
    value_rolldown = max(0.01, rolldown_cost)
    value_deploy = max(0.01, deploy_amount)
    value_reserve = max(0.01, reserve_amount)

    sources = [1, 1, 3, 5, 6, 6, 6]
    targets = [2, 3, 4, 6, 7, 8, 9]
    values = [value_hedge, value_surplus, value_surplus,
              value_put, min(value_put, value_rolldown), value_deploy, value_reserve]

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=labels, color=["purple", "green", "red", "blue", "gold",
                                 "red", "orange", "brown", "gold", "gray"]
        ),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig_sankey.update_layout(title="Full Cycle: Upside (Harvest) & Downside (Put → Pool CF)", height=400)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # 2. Payoff Chart
    st.subheader("Payoff Profile Ref")
    prices = np.linspace(P0 * 0.2, P0 * 1.5, 100)
    shannon_val = fix_c * np.log(prices / P0)
    put_val = qty_puts * np.maximum(0, put_strike - prices)
    combined = shannon_val + put_val

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=prices, y=shannon_val, name="Shannon (Unhedged)"))
    fig_payoff.add_trace(go.Scatter(x=prices, y=combined, name="Shannon + Put Shield",
                                    line=dict(width=3, color='green')))
    fig_payoff.add_vline(x=P_crash, line_dash="dash", line_color="red",
                         annotation_text=f"Crash Scenario ({P_crash:.1f})")
    fig_payoff.update_layout(title="Payoff Profile with Crash Scenario",
                             xaxis_title="Price", yaxis_title="Value")
    st.plotly_chart(fig_payoff, use_container_width=True)

    st.info(f"""
    **Chain System — Full Cycle Analysis:**
    
    **ขาขึ้น (Bull/Sideway):** Harvest (${harvest_profit:.2f}) จ่ายค่า Hedge (${cost_hedge:.2f}) เหลือ Surplus Scale Up.
    
    **ขาลง (Bear/Crash):** Put ทำงาน (${total_put_payoff:,.2f}) → เข้า Pool CF → หักลบ Re-Hedge (${rolldown_cost:,.2f})
    → **Valid Net:** ${pool_cf_net:,.2f}
    → **Deploy** ${deploy_amount:,.2f} ({(deploy_ratio*100):.0f}%) + **Reserve** ${reserve_amount:,.2f} ({(100-deploy_ratio*100):.0f}%)
    """)


# ----------------------------------------------------------
# TAB 3: Rollover History
# ----------------------------------------------------------
def _render_rollover_history(data):
    if not data:
        st.info("ยังไม่มีข้อมูล — เพิ่มหุ้นที่แท็บ ➕ Manage Data")
        return

    tickers = [d.get("ticker", "???") for d in data]
    selected = st.selectbox("เลือก Ticker", tickers, key="hist_ticker")
    idx = tickers.index(selected)
    ticker_data = data[idx]

    # --- Current State ---
    t, c, b = parse_final(ticker_data.get("Final", ""))
    ev, lock_pnl = parse_beta_numbers(ticker_data.get("beta_Equation", ""))
    net = parse_beta_net(ticker_data.get("beta_momory", ""))
    surplus_iv = parse_surplus_iv(ticker_data.get("Surplus_Iv", ""))
    comment = ticker_data.get("comment", "")

    st.subheader(f"📌 {selected} — Current State")
    with st.container(border=True):
        cs1, cs2, cs3, cs4 = st.columns(4)
        cs1.metric("Price (t)", f"${t}" if t else "N/A")
        cs2.metric("Fix_C", f"${c:,.0f}" if c else "N/A")
        cs3.metric("Baseline (b)", f"${b:,.2f}" if b is not None else "N/A")
        cs4.metric("Net P&L",  f"${net:,.2f}",
                   delta="Profit" if net > 0 else "Loss",
                   delta_color="normal" if net >= 0 else "inverse")

        cs5, cs6, cs7, cs8 = st.columns(4)
        cs5.metric("Ev (ค่า K)", f"${ev:,.2f}")
        cs6.metric("Lock P&L", f"${lock_pnl:,.2f}")
        cs7.metric("Surplus IV", f"${surplus_iv:,.2f}")
        cs8.metric("Comment", comment if comment else "—")

    # --- Rollover History ---
    history = get_rollover_history(ticker_data)
    if not history:
        st.caption("ยังไม่มีประวัติ Rollover สำหรับ ticker นี้")
        return

    st.subheader("📜 Rollover History Timeline")

    # Table
    rows_for_table = []
    for h in history:
        rows_for_table.append({
            "Step": h["step"],
            "Description": h["description"],
            "Calculation": h["calculation"],
            "b": h["b"],
            "c": h["c"],
            "t": h["t"],
        })
    hist_df = pd.DataFrame(rows_for_table)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    # --- b-Evolution Chart ---
    b_values = [h["b"] for h in history if h["b"] is not None]
    steps = [h["step"] for h in history if h["b"] is not None]

    if b_values:
        st.subheader("📈 Baseline (b) Evolution")
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=steps, y=b_values,
            mode="lines+markers+text",
            text=[f"${v:,.0f}" for v in b_values],
            textposition="top center",
            line=dict(width=3, color="#2196f3"),
            marker=dict(size=10),
            name="Baseline (b)"
        ))
        fig_b.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_b.update_layout(
            title=f"{selected} — Baseline Evolution",
            xaxis_title="Rollover Step", yaxis_title="Baseline (b) Value ($)",
            height=350,
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # --- c-Evolution Chart ---
    c_values = [h["c"] for h in history if h["c"] is not None]
    c_steps = [h["step"] for h in history if h["c"] is not None]
    if c_values:
        st.subheader("📊 Fix_C Evolution")
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            x=c_steps, y=c_values,
            text=[f"${v:,.0f}" for v in c_values],
            textposition="outside",
            marker_color="#ff9800",
            name="Fix_C"
        ))
        fig_c.update_layout(
            title=f"{selected} — Fix_C Changes Over Time",
            xaxis_title="Rollover Step", yaxis_title="Fix_C ($)",
            height=300,
        )
        st.plotly_chart(fig_c, use_container_width=True)


# ----------------------------------------------------------
# TAB 4: Manage Data
# ----------------------------------------------------------
def _render_manage_data(data):
    st.subheader("จัดการข้อมูลพอร์ต")

    action = st.radio("เลือกการทำงาน", [
        "📝 เพิ่ม Rollover Entry (ให้ ticker ที่มีอยู่)",
        "➕ เพิ่ม Ticker ใหม่",
        "🔄 อัพเดทราคาปัจจุบัน (Quick Update)",
    ], key="manage_action")

    if action == "➕ เพิ่ม Ticker ใหม่":
        _form_add_ticker(data)
    elif action == "📝 เพิ่ม Rollover Entry (ให้ ticker ที่มีอยู่)":
        _form_add_rollover(data)
    elif action == "🔄 อัพเดทราคาปัจจุบัน (Quick Update)":
        _form_quick_update(data)


def _form_add_ticker(data):
    with st.form("add_ticker_form", clear_on_submit=True):
        st.markdown("##### ➕ เพิ่ม Ticker ใหม่")
        ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").upper()
        col_a, col_b = st.columns(2)
        with col_a:
            price = st.number_input("ราคาอ้างอิงเริ่มต้น (t)", min_value=0.01, value=10.0, step=0.5)
            fix_c = st.number_input("ทุนคงที่ (c)", min_value=0.01, value=1500.0, step=100.0)
        with col_b:
            ev_val = st.number_input("Ev (Extrinsic Value costs)", value=0.0, step=10.0,
                                     help="EV = Premium − Intrinsic Value (ค่า K จ่ายทิ้ง)")

        submitted = st.form_submit_button("✅ เพิ่ม Ticker", type="primary")
        if submitted and ticker:
            new_entry = {
                "ticker": ticker,
                "Final": f"{price}, {fix_c}, 0",
                "Original": f"ราคาอ้างอิง: {price}, ทุนคงที่: {fix_c}",
                "Equation": "b += c · ln(P / t) - c' · ln(P / t'); แล้วตั้ง P = P', t = t', c = c'",
                "history_1": "",
                "comment": "",
                "beta_Equation": f" Ev: {ev_val:.2f} + Lock_P&L: +0",
                "beta_momory": f"Net: {ev_val:.2f}"
            }
            data.append(new_entry)
            save_trading_data(data)
            st.success(f"✅ เพิ่ม {ticker} สำเร็จ!")
            st.rerun()


def _form_add_rollover(data):
    if not data:
        st.info("ยังไม่มี ticker — เพิ่มที่ '➕ เพิ่ม Ticker ใหม่' ก่อน")
        return

    tickers = [d.get("ticker", "???") for d in data]

    with st.form("add_rollover_form", clear_on_submit=True):
        st.markdown("##### 📝 เพิ่ม Rollover Entry")
        sel_ticker = st.selectbox("Ticker", tickers)

        col_a, col_b = st.columns(2)
        with col_a:
            old_t = st.number_input("t เดิม", min_value=0.01, value=10.0, step=0.5)
            new_t = st.number_input("t ใหม่", min_value=0.01, value=10.0, step=0.5)
            current_p = st.number_input("ราคาปัจจุบัน (P)", min_value=0.01, value=10.0, step=0.5)
        with col_b:
            old_c = st.number_input("c เดิม", min_value=0.01, value=1500.0, step=100.0)
            new_c = st.number_input("c ใหม่", min_value=0.01, value=1500.0, step=100.0)

        submitted = st.form_submit_button("✅ บันทึก Rollover", type="primary")
        if submitted:
            idx = tickers.index(sel_ticker)
            ticker_data = data[idx]

            # Get current b
            _, _, old_b = parse_final(ticker_data.get("Final", ""))
            old_b = old_b if old_b else 0.0

            # Calculate new b
            if current_p > 0 and old_t > 0 and new_t > 0:
                delta_b = old_c * np.log(current_p / old_t) - new_c * np.log(current_p / new_t)
                new_b = old_b + delta_b
            else:
                new_b = old_b

            # Find next history index
            h_idx = 1
            while f"history_{h_idx}" in ticker_data:
                h_idx += 1

            # Write history entry
            desc = f"ราคาอ้างอิง: {old_t} → {new_t} , ทุนคงที่: {old_c} → {new_c} , ราคาปัจจุบัน: {current_p}"
            calc = (f"{old_b:.2f} += ({old_c} × ln({current_p}/{old_t})) − "
                    f"({new_c} × ln({current_p}/{new_t})) | "
                    f"c = {new_c} , t = {new_t} , b = {new_b:.2f}")

            ticker_data[f"history_{h_idx}"] = desc
            ticker_data[f"history_{h_idx}.1"] = calc

            # Update Final
            ticker_data["Final"] = f"{new_t}, {new_c}, {new_b:.2f}"

            data[idx] = ticker_data
            save_trading_data(data)
            st.success(f"✅ Rollover #{h_idx} สำหรับ {sel_ticker} บันทึกแล้ว! b = ${new_b:.2f}")
            st.rerun()


def _form_quick_update(data):
    if not data:
        st.info("ยังไม่มี ticker")
        return

    tickers = [d.get("ticker", "???") for d in data]

    with st.form("quick_update_form", clear_on_submit=True):
        st.markdown("##### 🔄 Quick Update — อัพเดทค่า Ev/Net")
        sel_ticker = st.selectbox("Ticker", tickers, key="qu_ticker")

        col_a, col_b = st.columns(2)
        with col_a:
            new_ev = st.number_input("Ev (Extrinsic Value)", value=0.0, step=10.0,
                                     help="EV = Premium − Intrinsic Value")
        with col_b:
            new_surplus = st.text_input("Surplus IV (เช่น (4.98*100)=498|(2.31*100)=231)",
                                        placeholder="Iv_Put: ...", value="")

        submitted = st.form_submit_button("✅ อัพเดท", type="primary")
        if submitted:
            idx = tickers.index(sel_ticker)
            ticker_data = data[idx]

            # Get current Lock P&L
            _, lock_pnl = parse_beta_numbers(ticker_data.get("beta_Equation", ""))
            _, _, b = parse_final(ticker_data.get("Final", ""))
            b = b if b else 0.0

            # Update Ev
            lock_total = lock_pnl
            if b != 0:
                # Lock P&L includes baseline b
                lock_str = f"+{b:.2f}" if b >= 0 else f"{b:.2f}"
            else:
                lock_str = "+0"

            # Add surplus to lock string if provided
            if new_surplus:
                ticker_data["Surplus_Iv"] = f"Iv_Put: {new_surplus}"
                surplus_val = parse_surplus_iv(f"Iv_Put: {new_surplus}")
                lock_total += surplus_val

            ticker_data["beta_Equation"] = f" Ev: {new_ev:.2f} + Lock_P&L: {lock_str}"
            net = new_ev + lock_pnl + parse_surplus_iv(ticker_data.get("Surplus_Iv", ""))
            ticker_data["beta_momory"] = f"Net: {net:.2f}"

            data[idx] = ticker_data
            save_trading_data(data)
            st.success(f"✅ อัพเดท {sel_ticker} สำเร็จ! Net = ${net:,.2f}")
            st.rerun()


# ============================================================
# MAIN APP NAVIGATION
# ============================================================

def main():
    st.sidebar.title("Flywheel & Shannon's Demon")
    menu = st.sidebar.radio("Menu", [
        "Introduction", "Baseline", "Shannon Process", "Volatility Harvesting",
        "Black Swan Shield", "Dynamic Scaling", "Synthetic Dividend", "Collateral Magic",
        "Chain System (Active)", "Quiz", "Paper Trading", "Glossary"
    ], index=8)  # Default to Chain System

    if menu == "Introduction": chapter_0_introduction()
    elif menu == "Baseline": chapter_1_baseline()
    elif menu == "Shannon Process": chapter_2_shannon_process()
    elif menu == "Volatility Harvesting": chapter_3_volatility_harvesting()
    elif menu == "Black Swan Shield": chapter_4_black_swan_shield()
    elif menu == "Dynamic Scaling": chapter_5_dynamic_scaling()
    elif menu == "Synthetic Dividend": chapter_6_synthetic_dividend()
    elif menu == "Collateral Magic": chapter_7_collateral_magic()
    elif menu == "Chain System (Active)": chapter_chain_system()
    elif menu == "Quiz": master_study_guide_quiz()
    elif menu == "Paper Trading": paper_trading_workshop()
    elif menu == "Glossary": glossary_section()


# Stub functions for missing features
def master_study_guide_quiz(): pass
def paper_trading_workshop(): pass
def glossary_section(): pass

if __name__ == "__main__":
    main()
