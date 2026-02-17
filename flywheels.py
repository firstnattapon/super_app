import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import json
import os
import re  
import copy
from datetime import datetime

# ============================================================ 
# UTILITIES
# ============================================================ 

def sanitize_number_str(s):
    """Normalize number strings: replace Unicode minus, remove commas/spaces."""
    if not s:
        return s
    return s.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-').replace(',', '').strip()

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

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(_DATA_DIR, "trading_data.json")
_BACKUP_FILE = os.path.join(_DATA_DIR, "trading_data.backup.json")

def load_trading_data():
    """Load portfolio data, auto-migrating to V2 if needed."""
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []
    return _migrate_data_if_needed(raw)

def save_trading_data(data):
    """Save portfolio data with auto-backup."""
    if os.path.exists(_DATA_FILE):
        try:
            import shutil
            shutil.copy2(_DATA_FILE, _BACKUP_FILE)
        except Exception:
            pass
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_tickers(data):
    """Get tickers list from V2 data structure."""
    if isinstance(data, dict):
        return data.get("tickers", [])
    return data  # fallback for raw list

def _migrate_data_if_needed(raw_data):
    """Upgrade from V1 (flat list) to V2 (wrapper with tickers/pool_cf).
    Preserves all legacy fields. Adds current_state + empty rounds to each ticker."""
    if isinstance(raw_data, dict) and raw_data.get("version") == 2:
        return raw_data
    tickers = raw_data if isinstance(raw_data, list) else []
    for item in tickers:
        if "current_state" not in item:
            t, c, b = parse_final(item.get("Final", ""))
            ev, _ = parse_beta_numbers(item.get("beta_Equation", ""))
            item["current_state"] = {
                "price": t if t else 0.0,
                "fix_c": c if c else 0.0,
                "baseline": b if b else 0.0,
                "pool_cf_net": 0.0,
                "cumulative_ev": abs(ev) if ev else 0.0,
            }
        if "rounds" not in item:
            item["rounds"] = []
    return {
        "version": 2,
        "tickers": tickers,
        "global_pool_cf": 0.0,
        "settings": {
            "surplus_mode": "auto_compound",
            "default_sigma": 0.5,
            "default_hedge_ratio": 2.0,
        }
    }


# ============================================================
# CHAIN ENGINE — Core Round Logic
# ============================================================

def run_chain_round(ticker_state, p_new, sigma, hedge_ratio, r=0.04, T=1.0):
    """Core engine: compute one chain round.
    Returns a round_data dict (not yet committed)."""
    c = ticker_state["fix_c"]
    t = ticker_state["price"]
    b = ticker_state["baseline"]
    if t <= 0 or p_new <= 0:
        return None

    # 1. Shannon Profit
    shannon = c * np.log(p_new / t)
    # 2. Harvest Profit (Volatility Premium)
    harvest = c * 0.5 * (sigma ** 2) * T
    total_income = shannon + harvest
    # 3. Hedge Cost (Full Upfront Put)
    qty = (c / t) * hedge_ratio
    strike = t * 0.9
    premium = black_scholes(t, strike, T, r, sigma, 'put')
    hedge_cost = qty * premium
    # 4. Surplus
    surplus = total_income - hedge_cost
    scale_up = max(0.0, surplus)
    # 5. New State
    c_new = c + scale_up
    t_new = p_new  # re-center
    # 6. Rollover (b stays continuous)
    # Since t_new = p_new, ln(p_new/t_new) = 0
    rollover_delta = c * np.log(p_new / t) - c_new * np.log(p_new / t_new) if t_new != p_new else c * np.log(p_new / t)
    b_new = b + rollover_delta

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "action": "Chain Round",
        "p_old": round(t, 4),
        "p_new": round(p_new, 4),
        "c_before": round(c, 2),
        "c_after": round(c_new, 2),
        "shannon_profit": round(shannon, 2),
        "harvest_profit": round(harvest, 2),
        "hedge_cost": round(hedge_cost, 2),
        "surplus": round(surplus, 2),
        "scale_up": round(scale_up, 2),
        "b_before": round(b, 2),
        "b_after": round(b_new, 2),
        "hedge_ratio": hedge_ratio,
        "sigma": sigma,
    }


def commit_round(data, ticker_idx, round_data):
    """Save round to ticker's rounds[], update current_state, sync legacy fields."""
    ticker = data["tickers"][ticker_idx]
    if "rounds" not in ticker:
        ticker["rounds"] = []
    round_data["round_id"] = len(ticker["rounds"]) + 1
    ticker["rounds"].append(round_data)

    # Update current_state
    old_ev = ticker.get("current_state", {}).get("cumulative_ev", 0.0)
    ticker["current_state"] = {
        "price": round_data["p_new"],
        "fix_c": round_data["c_after"],
        "baseline": round_data["b_after"],
        "pool_cf_net": ticker.get("current_state", {}).get("pool_cf_net", 0.0),
        "cumulative_ev": old_ev + round_data["hedge_cost"],
    }

    # Sync legacy Final
    ticker["Final"] = f"{round_data['p_new']}, {round_data['c_after']}, {round_data['b_after']}"

    # Write legacy history entry
    h_idx = 1
    while f"history_{h_idx}" in ticker:
        h_idx += 1
    desc = (f"ราคาอ้างอิง: {round_data['p_old']} → {round_data['p_new']} , "
            f"ทุนคงที่: {round_data['c_before']} → {round_data['c_after']} , "
            f"ราคาปัจจุบัน: {round_data['p_new']}")
    calc = (f"{round_data['b_before']:.2f} += ({round_data['c_before']} × ln({round_data['p_new']}/{round_data['p_old']})) − "
            f"({round_data['c_after']} × ln({round_data['p_new']}/{round_data['p_new']})) | "
            f"c = {round_data['c_after']} , t = {round_data['p_new']} , b = {round_data['b_after']}")
    ticker[f"history_{h_idx}"] = desc
    ticker[f"history_{h_idx}.1"] = calc

    data["tickers"][ticker_idx] = ticker
    save_trading_data(data)
    return data


def deploy_pool_cf(data, ticker_idx, amount):
    """Deploy from global Pool CF to a specific ticker's fix_c."""
    if amount <= 0 or amount > data.get("global_pool_cf", 0):
        return data, False
    ticker = data["tickers"][ticker_idx]
    state = ticker.get("current_state", {})
    old_c = state.get("fix_c", 0)
    state["fix_c"] = old_c + amount
    ticker["current_state"] = state
    # Sync Final
    t = state.get("price", 0)
    b = state.get("baseline", 0)
    ticker["Final"] = f"{t}, {state['fix_c']:.2f}, {b:.2f}"
    data["global_pool_cf"] -= amount
    data["tickers"][ticker_idx] = ticker
    save_trading_data(data)
    return data, True


def parse_final(final_str):
    """Parse 'Final' field → (t, c, b).  e.g. '12, 4000, -519.45' → (12.0, 4000.0, -519.45)"""
    if not final_str:
        return None, None, None
    # Split on commas first (delimiter), then sanitize each part for Unicode minus
    parts = [sanitize_number_str(p) for p in final_str.split(",")]
    try:
        t = float(parts[0]) if len(parts) > 0 and parts[0] else None
        c = float(parts[1]) if len(parts) > 1 and parts[1] else None
        b = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
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
    beta_str = sanitize_number_str(beta_str)
    # Extract Ev
    ev_match = re.search(r'Ev:\s*([+-]?[\d.]+)', beta_str)
    if ev_match:
        try:
            ev = float(ev_match.group(1))
        except ValueError:
            pass
    # Extract Lock_P&L (may have multiple values like +1618.48 +498|+231)
    lock_match = re.search(r'Lock_P&L:\s*(.+)', beta_str)
    if lock_match:
        raw = lock_match.group(1).strip()
        raw = raw.replace("|", "+")
        nums = re.findall(r'[+-]?[\d.]+', raw)
        for n in nums:
            try:
                lock_pnl += float(n)
            except ValueError:
                pass
    return ev, lock_pnl

def parse_beta_net(beta_mem_str):
    """Extract Net value from beta_momory string. e.g. 'Net: -204.00' → -204.0"""
    if not beta_mem_str:
        return 0.0
    beta_mem_str = sanitize_number_str(beta_mem_str)
    m = re.search(r'Net:\s*([+-]?[\d.]+)', beta_mem_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0

def parse_surplus_iv(surplus_str):
    """Extract Surplus IV (Put premium income) from Surplus_Iv string.
    e.g. 'Iv_Put: (4.98*100)= 498 | (2.31*100)=231' → 729.0
    """
    if not surplus_str or "No_Expiry" in surplus_str:
        return 0.0
    surplus_str = sanitize_number_str(surplus_str)
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
            "Price (t)": t if t is not None else 0.0,
            "Fix_C": c if c is not None else 0.0,
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
    st.header("⚡ Chain System — Main Engine")
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

    # --- Load V2 data ---
    data = load_trading_data()
    tickers_list = get_tickers(data)

    # --- 6 Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚡ Active Dashboard",
        "🔗 Run Round",
        "🎱 Pool CF",
        "📜 History",
        "🔬 Simulation (Ref)",
        "➕ Manage Data"
    ])

    with tab1:
        _render_active_dashboard(data)
    with tab2:
        _render_run_round_form(data)
    with tab3:
        _render_pool_cf_dashboard(data)
    with tab4:
        _render_rollover_history(tickers_list)
    with tab5:
        _render_chain_flow()
    with tab6:
        _render_manage_data(data)


# ----------------------------------------------------------
# TAB: Active Dashboard (NEW — Home Page)
# ----------------------------------------------------------
def _render_active_dashboard(data):
    """Portfolio overview + Ticker cards + Burn Rate + Net Reality."""
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มีข้อมูล — เพิ่มหุ้นที่แท็บ ➕ Manage Data")
        return

    df = build_portfolio_df(tickers_list)
    total_c = df["Fix_C"].sum()
    total_ev = df["Ev (Extrinsic)"].sum()
    total_lock = df["Lock P&L"].sum()
    total_surplus = df["Surplus IV"].sum()
    total_net = df["Net"].sum()
    pool_cf = data.get("global_pool_cf", 0.0)

    # Cumulative Ev from rounds (Burn Rate)
    total_burn = sum(
        t_data.get("current_state", {}).get("cumulative_ev", 0.0) for t_data in tickers_list
    )

    # --- Summary Metrics ---
    st.subheader("📊 Portfolio Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fix_C (Deployed)", f"${total_c:,.0f}", f"{len(tickers_list)} tickers")
    m2.metric("🎱 Pool CF (War Chest)", f"${pool_cf:,.2f}")
    m3.metric("🔥 Burn Rate (Cum. Ev)", f"${total_burn:,.2f}", delta="Cost of Business", delta_color="inverse")
    m4.metric("💰 Net Reality", f"${total_net:,.2f}",
              delta=f"Lock {total_lock:,.0f} + IV {total_surplus:,.0f} + Ev {total_ev:,.0f}",
              delta_color="normal" if total_net >= 0 else "inverse")

    st.divider()

    # --- Ticker Cards ---
    st.subheader("📋 Ticker Status Cards")
    cols_per_row = 4
    for i in range(0, len(tickers_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(tickers_list):
                break
            t_data = tickers_list[idx]
            ticker = t_data.get("ticker", "???")
            state = t_data.get("current_state", {})
            n_rounds = len(t_data.get("rounds", []))
            net_val = parse_beta_net(t_data.get("beta_momory", ""))

            with col:
                with st.container(border=True):
                    color = "🟢" if net_val >= 0 else "🔴"
                    st.markdown(f"### {color} {ticker}")
                    st.caption(f"Price: ${state.get('price', 0):,.2f} | fix_c: ${state.get('fix_c', 0):,.0f}")
                    st.caption(f"Baseline: ${state.get('baseline', 0):,.2f} | Rounds: {n_rounds}")
                    st.caption(f"Net: ${net_val:,.2f}")

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
        height=400, plot_bgcolor="rgba(0,0,0,0)",
    )
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Waterfall ---
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
# TAB: Run Round (NEW — Core Workflow)
# ----------------------------------------------------------
def _render_run_round_form(data):
    """Select ticker → input P_new + σ → preview calculation → Commit Round."""
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Manage Data ก่อน")
        return

    st.subheader("⚡ Run Chain Round")
    st.markdown("เลือก Ticker → ใส่ราคาปัจจุบัน → ระบบคำนวณอัตโนมัติ → กด Commit")

    ticker_names = [d.get("ticker", "???") for d in tickers_list]
    selected = st.selectbox("เลือก Ticker", ticker_names, key="run_round_ticker")
    idx = ticker_names.index(selected)
    t_data = tickers_list[idx]
    state = t_data.get("current_state", {})

    # Current state display
    with st.container(border=True):
        st.caption(f"🔵 สถานะปัจจุบัน — {selected}")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("fix_c", f"${state.get('fix_c', 0):,.2f}")
        sc2.metric("Price (t)", f"${state.get('price', 0):,.2f}")
        sc3.metric("Baseline (b)", f"${state.get('baseline', 0):,.2f}")
        sc4.metric("Rounds", str(len(t_data.get("rounds", []))))

    # Default settings
    settings = data.get("settings", {})
    default_sigma = settings.get("default_sigma", 0.5)
    default_hr = settings.get("default_hedge_ratio", 2.0)

    with st.form("run_round_form", clear_on_submit=False):
        st.markdown("##### 📊 Input — ราคาและ Config")
        r1, r2, r3 = st.columns(3)
        with r1:
            p_new = st.number_input(
                f"ราคาใหม่ P (ปัจจุบัน t = ${state.get('price', 0):.2f})",
                min_value=0.01, value=round(state.get("price", 10.0) * 1.1, 2), step=1.0,
                key="rr_pnew")
        with r2:
            sigma = st.number_input("Volatility (σ)", min_value=0.05, value=default_sigma, step=0.05, key="rr_sigma")
        with r3:
            hedge_ratio = st.number_input("Hedge Ratio (x Put)", min_value=0.0, value=default_hr, step=0.5, key="rr_hr")

        # Preview button
        preview_btn = st.form_submit_button("🔍 Preview Calculation")

    # Preview logic (outside form so it shows after submit)
    if preview_btn and p_new > 0:
        preview = run_chain_round(state, p_new, sigma, hedge_ratio)
        if preview is None:
            st.error("Invalid price — cannot run round")
            return

        st.session_state["_pending_round"] = preview
        st.session_state["_pending_ticker_idx"] = idx
        st.session_state["_pending_ticker_name"] = selected

    # Show preview if available
    if "_pending_round" in st.session_state and st.session_state.get("_pending_ticker_name") == selected:
        rd = st.session_state["_pending_round"]
        st.markdown("---")
        st.subheader("📋 Preview — ผลลัพธ์ก่อน Commit")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Shannon Profit", f"${rd['shannon_profit']:,.2f}",
                   delta=f"P: {rd['p_old']} → {rd['p_new']}")
        p2.metric("Harvest Profit", f"${rd['harvest_profit']:,.2f}",
                   delta=f"σ={rd['sigma']}")
        p3.metric("Hedge Cost", f"${rd['hedge_cost']:,.2f}",
                   delta=f"-{rd['hedge_ratio']}x Put", delta_color="inverse")
        p4.metric("Surplus (Free Risk)", f"${rd['surplus']:,.2f}",
                   delta="Scale Up!" if rd['surplus'] > 0 else "Deficit",
                   delta_color="normal" if rd['surplus'] > 0 else "inverse")

        p5, p6, p7 = st.columns(3)
        p5.metric("fix_c After", f"${rd['c_after']:,.2f}",
                   delta=f"+${rd['scale_up']:,.2f}" if rd['scale_up'] > 0 else "No change")
        p6.metric("Baseline After", f"${rd['b_after']:,.2f}")
        p7.metric("Price After (re-centered)", f"${rd['p_new']:,.2f}")

        # Commit button
        if st.button("✅ Commit Round — บันทึกถาวร", type="primary", key="commit_round"):
            commit_round(data, st.session_state["_pending_ticker_idx"], rd)
            del st.session_state["_pending_round"]
            del st.session_state["_pending_ticker_idx"]
            del st.session_state["_pending_ticker_name"]
            st.success(f"✅ Round committed for {selected}! fix_c = ${rd['c_after']:,.2f}, b = ${rd['b_after']:,.2f}")
            st.rerun()

    # --- Round History for this ticker ---
    rounds = t_data.get("rounds", [])
    if rounds:
        st.divider()
        st.subheader(f"📜 Chain Rounds — {selected}")
        rows = []
        for rd in rounds:
            rows.append({
                "#": rd.get("round_id", ""),
                "Date": rd.get("date", ""),
                "Price": f"${rd['p_old']} → ${rd['p_new']}",
                "Shannon": f"${rd['shannon_profit']:,.2f}",
                "Harvest": f"${rd['harvest_profit']:,.2f}",
                "Hedge": f"-${rd['hedge_cost']:,.2f}",
                "Surplus": f"${rd['surplus']:,.2f}",
                "fix_c After": f"${rd['c_after']:,.2f}",
                "b After": f"${rd['b_after']:,.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ----------------------------------------------------------
# TAB: Pool CF Dashboard (NEW)
# ----------------------------------------------------------
def _render_pool_cf_dashboard(data):
    """Global Pool CF management — view balance + deploy to tickers."""
    tickers_list = get_tickers(data)
    pool_cf = data.get("global_pool_cf", 0.0)

    st.subheader("🎱 Global Pool CF — War Chest")
    st.markdown("""
    **Pool CF** = กำไรจาก Put ระเบิดตอน Crash → แยกเก็บเป็น **Emergency Fund / War Chest**
    
    ใช้ Deploy เพื่อเพิ่ม fix_c ให้ ticker ที่ราคาต่ำ (Buy the Dip).
    """)

    m1, m2 = st.columns(2)
    m1.metric("💰 Pool CF Balance", f"${pool_cf:,.2f}")
    m2.metric("Tickers", str(len(tickers_list)))

    st.divider()

    # Manual Add to Pool (for put profits etc.)
    with st.form("add_pool_cf_form", clear_on_submit=True):
        st.markdown("##### ➕ Add to Pool CF (e.g., Put payoff profit)")
        amount = st.number_input("Amount to add ($)", min_value=0.0, value=0.0, step=100.0, key="add_pool_amt")
        note = st.text_input("Note (optional)", placeholder="e.g. Put payoff from FLNC crash")
        if st.form_submit_button("💰 Add to Pool", type="primary"):
            if amount > 0:
                data["global_pool_cf"] = data.get("global_pool_cf", 0) + amount
                save_trading_data(data)
                st.success(f"✅ Added ${amount:,.2f} to Pool CF. New balance: ${data['global_pool_cf']:,.2f}")
                st.rerun()

    st.divider()

    # Deploy from Pool to Ticker
    if pool_cf > 0 and tickers_list:
        st.subheader("🚀 Deploy from Pool CF → Ticker")
        ticker_names = [d.get("ticker", "???") for d in tickers_list]

        with st.form("deploy_pool_form", clear_on_submit=True):
            dp1, dp2 = st.columns(2)
            with dp1:
                deploy_ticker = st.selectbox("Deploy to Ticker", ticker_names, key="deploy_ticker")
            with dp2:
                deploy_amount = st.number_input("Amount ($)", min_value=0.0, max_value=float(pool_cf),
                                                 value=0.0, step=100.0, key="deploy_amt")
            if st.form_submit_button("🚀 Deploy", type="primary"):
                if deploy_amount > 0:
                    d_idx = ticker_names.index(deploy_ticker)
                    data, success = deploy_pool_cf(data, d_idx, deploy_amount)
                    if success:
                        st.success(f"✅ Deployed ${deploy_amount:,.2f} to {deploy_ticker}")
                        st.rerun()
                    else:
                        st.error("❌ Insufficient Pool CF balance")
    elif pool_cf <= 0:
        st.info("Pool CF ว่าง — ยังไม่มีเงินสำหรับ Deploy")


# ----------------------------------------------------------
# TAB: Portfolio Dashboard (Legacy — preserved)
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

    # 2. Payoff Chart — 4 เส้นเปรียบเทียบ (เวอร์ชันเต็ม)
    st.subheader("Payoff Profile Ref")
    prices = np.linspace(P0 * 0.2, P0 * 1.5, 200)

    # เส้น 1: Stock Only (Linear) — ถือหุ้นจริง 100%
    stock_only = fix_c * (prices / P0 - 1)

    # เส้น 2: Base 80/20 (Unhedged) — Shannon log baseline
    base_80_20 = fix_c * np.log(prices / P0)

    # เส้น 3: Dynamic Shield (+Vol Premium) — Shannon + Volatility Harvest
    vol_premium = fix_c * 0.5 * (sigma ** 2) * T
    dynamic_shield = base_80_20 + vol_premium

    # เส้น 4: Shielded 80/20 (+2.0x Puts) — Shannon + Put Hedge (anti-fragile)
    put_val = qty_puts * np.maximum(0, put_strike - prices)
    shielded_80_20 = dynamic_shield + put_val - cost_hedge  # หักค่า Hedge แล้ว

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=stock_only, name="Stock Only (Linear)",
        line=dict(width=1, color='gray', dash='dot')))
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=base_80_20, name="เส้น Base 80/20 (Unhedged)",
        line=dict(width=2, color='#ff9800')))
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=dynamic_shield, name="Dynamic Shield (+Vol Premium)",
        line=dict(width=2, color='#2196f3', dash='dash')))
    fig_payoff.add_trace(go.Scatter(
        x=prices, y=shielded_80_20, name="Shielded 80/20 (+2.0x Puts)",
        line=dict(width=3, color='#00c853')))

    # Crash marker
    fig_payoff.add_vline(x=P_crash, line_dash="dash", line_color="red",
                         annotation_text=f"Crash Scenario ({P_crash:.1f})")
    # Break-even line
    fig_payoff.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig_payoff.update_layout(
        title="Payoff Profile — 4 เส้นเปรียบเทียบ (Stock vs 80/20 vs Shield vs Anti-Fragile)",
        xaxis_title="Price ($)", yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500,
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

    # =================================================================
    # 3. ระบบลูกโซ่ (Chain Simulation) — จำลองรอบต่อรอบ
    # =================================================================
    st.divider()
    st.subheader("🔗 ระบบลูกโซ่ — Chain Simulation (Round-by-Round)")
    st.markdown("""
    **หลักการ:** ราคาขึ้น → กำไร Shannon + Harvest → จ่ายค่า Put Hedge → 
    **Surplus = Free Risk** → Scale Up fix_c ด้วย Rollover Equation
    """)

    # Initialize session state for chain rounds
    # Auto-reset chain if config changed
    config_key = f"{fix_c}_{P0}_{sigma}"
    if "chain_rounds" not in st.session_state or st.session_state.get("_chain_config") != config_key:
        st.session_state.chain_rounds = []
        st.session_state.chain_current_c = fix_c
        st.session_state.chain_current_t = float(P0)
        st.session_state.chain_current_b = 0.0
        st.session_state._chain_config = config_key

    # Reset button
    if st.button("🔄 Reset Chain", key="reset_chain"):
        st.session_state.chain_rounds = []
        st.session_state.chain_current_c = fix_c
        st.session_state.chain_current_t = float(P0)
        st.session_state.chain_current_b = 0.0
        st.session_state._chain_config = config_key
        st.rerun()

    # Current state display
    cur_c = st.session_state.chain_current_c
    cur_t = st.session_state.chain_current_t
    cur_b = st.session_state.chain_current_b

    with st.container(border=True):
        st.caption(f"🔵 สถานะปัจจุบัน — Round #{len(st.session_state.chain_rounds)}")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("fix_c (ทุนคงที่)", f"${cur_c:,.2f}")
        sc2.metric("t (ราคาอ้างอิง)", f"${cur_t:,.2f}")
        sc3.metric("b (Baseline)", f"${cur_b:,.2f}")

    # --- New Round Input ---
    with st.form("chain_round_form", clear_on_submit=True):
        st.markdown("##### ➕ เพิ่ม Round ใหม่ — ราคาเปลี่ยนจาก t → P")
        cr1, cr2 = st.columns(2)
        with cr1:
            new_price = st.number_input(
                f"ราคาใหม่ P (ปัจจุบัน t = ${cur_t:.2f})",
                min_value=0.01, value=round(cur_t * 1.2, 2), step=1.0,
                key="chain_new_p")
        with cr2:
            chain_hedge_ratio = st.number_input(
                "Hedge Ratio (x Put)", min_value=0.0, value=2.0, step=0.5,
                key="chain_hr", help="2.0 = Over-hedge 2 เท่า (Anti-Fragile)")

        submitted = st.form_submit_button("⚡ Run Chain Round", type="primary")
        if submitted and new_price > 0:
            P_new = new_price

            # === STEP 1: Shannon Profit (Simple Reference) ===
            shannon_profit = cur_c * np.log(P_new / cur_t) if P_new > 0 and cur_t > 0 else 0.0

            # === STEP 2: Harvest Profit (Volatility Premium) ===
            harvest = cur_c * 0.5 * (sigma ** 2) * T

            total_income = shannon_profit + harvest

            # === STEP 3: Fund Put Hedge ===
            qty = (cur_c / cur_t) * chain_hedge_ratio
            strike = cur_t * put_strike_pct
            premium = black_scholes(cur_t, strike, T, r, sigma, 'put')
            hedge_cost = qty * premium

            # === STEP 4: Surplus → Scale Up (FREE RISK!) ===
            surplus_val = total_income - hedge_cost
            scale_up = max(0, surplus_val)  # Only scale up if positive

            new_c = cur_c + scale_up
            new_t = P_new  # Re-center price

            # === STEP 5: Rollover Equation (keep baseline continuous) ===
            # b_new = b_old + c_old * ln(P/t_old) - c_new * ln(P/t_new)
            # Since t_new = P (re-center), ln(P/t_new) = ln(1) = 0
            rollover_delta = cur_c * np.log(P_new / cur_t) - new_c * np.log(P_new / new_t)
            new_b = cur_b + rollover_delta

            # Save round
            round_data = {
                "round": len(st.session_state.chain_rounds) + 1,
                "P_from": cur_t,
                "P_to": P_new,
                "c_before": cur_c,
                "shannon": shannon_profit,
                "harvest": harvest,
                "total_income": total_income,
                "hedge_cost": hedge_cost,
                "surplus": surplus_val,
                "scale_up": scale_up,
                "c_after": new_c,
                "t_after": new_t,
                "b_after": new_b,
                "hedge_ratio": chain_hedge_ratio,
            }
            st.session_state.chain_rounds.append(round_data)
            st.session_state.chain_current_c = new_c
            st.session_state.chain_current_t = new_t
            st.session_state.chain_current_b = new_b
            st.rerun()

    # --- Chain History Table ---
    if st.session_state.chain_rounds:
        st.subheader("📋 Chain History — ลูกโซ่ทุก Round")

        rows = []
        for rd in st.session_state.chain_rounds:
            rows.append({
                "Round": rd["round"],
                "Price": f"${rd['P_from']:.2f} → ${rd['P_to']:.2f}",
                "Shannon": f"${rd['shannon']:,.2f}",
                "Harvest": f"${rd['harvest']:,.2f}",
                "Total": f"${rd['total_income']:,.2f}",
                "Hedge (x{:.1f})".format(rd["hedge_ratio"]): f"-${rd['hedge_cost']:,.2f}",
                "Surplus": f"${rd['surplus']:,.2f}",
                "Scale Up": f"+${rd['scale_up']:,.2f}" if rd['scale_up'] > 0 else "—",
                "fix_c After": f"${rd['c_after']:,.2f}",
                "b After": f"${rd['b_after']:,.2f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Chain Evolution Chart ---
        rounds_x = [0] + [rd["round"] for rd in st.session_state.chain_rounds]
        c_vals = [fix_c] + [rd["c_after"] for rd in st.session_state.chain_rounds]
        b_vals = [0] + [rd["b_after"] for rd in st.session_state.chain_rounds]

        fig_chain = make_subplots(rows=1, cols=2,
                                  subplot_titles=["fix_c Growth (Free Risk)", "Baseline (b) Evolution"])

        fig_chain.add_trace(go.Bar(
            x=rounds_x, y=c_vals,
            text=[f"${v:,.0f}" for v in c_vals],
            textposition="outside",
            marker_color=["#ff9800"] + ["#00c853" if rd["surplus"] > 0 else "#ff1744"
                                        for rd in st.session_state.chain_rounds],
            name="fix_c",
        ), row=1, col=1)

        fig_chain.add_trace(go.Scatter(
            x=rounds_x, y=b_vals,
            mode="lines+markers+text",
            text=[f"${v:,.0f}" for v in b_vals],
            textposition="top center",
            line=dict(width=3, color="#2196f3"),
            marker=dict(size=10),
            name="Baseline (b)",
        ), row=1, col=2)

        fig_chain.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=2)
        fig_chain.update_layout(height=350, showlegend=False)
        fig_chain.update_xaxes(title_text="Round", row=1, col=1)
        fig_chain.update_xaxes(title_text="Round", row=1, col=2)
        fig_chain.update_yaxes(title_text="fix_c ($)", row=1, col=1)
        fig_chain.update_yaxes(title_text="b ($)", row=1, col=2)
        st.plotly_chart(fig_chain, use_container_width=True)

        # Summary
        total_scaled = st.session_state.chain_current_c - fix_c
        st.success(f"""
        **🔗 Chain Summary:**
        เริ่มต้น fix_c = **${fix_c:,.2f}** → ปัจจุบัน fix_c = **${st.session_state.chain_current_c:,.2f}**
        
        ↑ Scale Up รวม **${total_scaled:,.2f}** (Free Risk — มาจากกำไรล้วนๆ ไม่ใช่เงินต้น!)
        
        Baseline (b) = **${st.session_state.chain_current_b:,.2f}** (Rollover Equation ต่อเนื่อง)
        """)

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
    tickers_list = get_tickers(data)

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
                "beta_momory": f"Net: {ev_val:.2f}",
                "current_state": {
                    "price": price,
                    "fix_c": fix_c,
                    "baseline": 0.0,
                    "pool_cf_net": 0.0,
                    "cumulative_ev": abs(ev_val),
                },
                "rounds": [],
            }
            data["tickers"].append(new_entry)
            save_trading_data(data)
            st.success(f"✅ เพิ่ม {ticker} สำเร็จ!")
            st.rerun()


def _form_add_rollover(data):
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่ '➕ เพิ่ม Ticker ใหม่' ก่อน")
        return

    tickers = [d.get("ticker", "???") for d in tickers_list]

    with st.form("add_rollover_form", clear_on_submit=True):
        st.markdown("##### 📝 เพิ่ม Rollover Entry (Manual)")
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
            ticker_data = tickers_list[idx]

            _, _, old_b = parse_final(ticker_data.get("Final", ""))
            old_b = old_b if old_b else 0.0

            if current_p > 0 and old_t > 0 and new_t > 0:
                delta_b = old_c * np.log(current_p / old_t) - new_c * np.log(current_p / new_t)
                new_b = old_b + delta_b
            else:
                new_b = old_b

            h_idx = 1
            while f"history_{h_idx}" in ticker_data:
                h_idx += 1

            desc = f"ราคาอ้างอิง: {old_t} → {new_t} , ทุนคงที่: {old_c} → {new_c} , ราคาปัจจุบัน: {current_p}"
            calc = (f"{old_b:.2f} += ({old_c} × ln({current_p}/{old_t})) − "
                    f"({new_c} × ln({current_p}/{new_t})) | "
                    f"c = {new_c} , t = {new_t} , b = {new_b:.2f}")

            ticker_data[f"history_{h_idx}"] = desc
            ticker_data[f"history_{h_idx}.1"] = calc
            ticker_data["Final"] = f"{new_t}, {new_c}, {new_b:.2f}"
            # Update current_state
            ticker_data["current_state"] = {
                "price": new_t,
                "fix_c": new_c,
                "baseline": new_b,
                "pool_cf_net": ticker_data.get("current_state", {}).get("pool_cf_net", 0.0),
                "cumulative_ev": ticker_data.get("current_state", {}).get("cumulative_ev", 0.0),
            }

            data["tickers"][idx] = ticker_data
            save_trading_data(data)
            st.success(f"✅ Rollover #{h_idx} สำหรับ {sel_ticker} บันทึกแล้ว! b = ${new_b:.2f}")
            st.rerun()


def _form_quick_update(data):
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มี ticker")
        return

    tickers = [d.get("ticker", "???") for d in tickers_list]

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
            ticker_data = tickers_list[idx]

            _, lock_pnl = parse_beta_numbers(ticker_data.get("beta_Equation", ""))
            _, _, b = parse_final(ticker_data.get("Final", ""))
            b = b if b else 0.0

            existing_beta = ticker_data.get("beta_Equation", "")
            lock_match = re.search(r'Lock_P&L:\s*(.+)', sanitize_number_str(existing_beta))
            lock_str = lock_match.group(1).strip() if lock_match else "+0"

            if new_surplus:
                ticker_data["Surplus_Iv"] = f"Iv_Put: {new_surplus}"

            ticker_data["beta_Equation"] = f" Ev: {new_ev:.2f} + Lock_P&L: {lock_str}"
            net = new_ev + lock_pnl + parse_surplus_iv(ticker_data.get("Surplus_Iv", ""))
            ticker_data["beta_momory"] = f"Net: {net:.2f}"

            data["tickers"][idx] = ticker_data
            save_trading_data(data)
            st.success(f"✅ อัพเดท {sel_ticker} สำเร็จ! Net = ${net:,.2f}")
            st.rerun()


# ============================================================
# FUNCTION ALIASES — for Manual page compatibility
# ============================================================
chapter_2_volatility_harvest = chapter_2_shannon_process
chapter_3_convexity_engine = chapter_3_volatility_harvesting

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
