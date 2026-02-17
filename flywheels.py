
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import json
import os
from datetime import datetime

# ============================================================
# 1. CORE MATH & UTILITIES
# ============================================================

def sanitize_number_str(s):
    """Normalize number strings."""
    if not s: return s
    return s.replace('\u2212', '-').replace('\u2013', '-').replace(',', '').strip()

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0: return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# ============================================================
# 2. DATA LAYER (V2 Schema)
# ============================================================

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(_DATA_DIR, "trading_data_v2.json")
_BACKUP_FILE = os.path.join(_DATA_DIR, "trading_data_v2.backup.json")
_LEGACY_FILE = os.path.join(_DATA_DIR, "trading_data.json")

def load_data():
    """Load data with migration support."""
    if os.path.exists(_DATA_FILE):
        try:
            with open(_DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading V2: {e}")
            return {"tickers": [], "global_pool": 0.0}
    
    # Migration
    if os.path.exists(_LEGACY_FILE):
        st.toast("Migrating Legacy Data to Chain V2...", icon="📦")
        return _migrate_v1_to_v2()

    return {"tickers": [], "global_pool": 0.0}

def save_data(data):
    """Save with backup."""
    if os.path.exists(_DATA_FILE):
        try:
            import shutil
            shutil.copy2(_DATA_FILE, _BACKUP_FILE)
        except: pass
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _migrate_v1_to_v2():
    """Convert V1 schema to V2."""
    try:
        with open(_LEGACY_FILE, "r", encoding="utf-8") as f:
            v1 = json.load(f)
    except:
        return {"tickers": [], "global_pool": 0.0}

    v2_tickers = []
    for item in v1:
        # Parse old "Final" string: t, c, b
        final_str = item.get("Final", "")
        parts = [sanitize_number_str(p) for p in final_str.split(",")]
        try:
            t = float(parts[0]) if len(parts) > 0 else 0.0
            c = float(parts[1]) if len(parts) > 1 else 0.0
            b = float(parts[2]) if len(parts) > 2 else 0.0
        except:
            t, c, b = 10.0, 1000.0, 0.0
        
        v2_tickers.append({
            "symbol": item.get("ticker", "UNKNOWN"),
            "state": {
                "t": t, "c": c, "b": b, 
                "sigma": 0.5, "hedge_ratio": 2.0
            },
            "rounds": [],
            "metrics": {"total_ev_paid": 0.0, "total_surplus": 0.0}
        })
    return {"tickers": v2_tickers, "global_pool": 0.0}

# ============================================================
# 3. CHAIN ENGINE (Business Logic)
# ============================================================

class ChainEngine:
    @staticmethod
    def run_round(ticker, current_price, sigma, hedge_ratio, days_passed=30):
        """
        Executes Chain Logic:
        1. Shannon Profit (Log Path)
        2. Harvest (Vol Premium)
        3. Hedge Cost (Upfront for Surplus)
        4. Put Payoff (Market Crash) -> Pool CF
        5. Rollover -> New State
        """
        state = ticker["state"]
        t_old = state["t"]
        c_old = state["c"] # Start capital
        b_old = state["b"]
        
        dt = days_passed / 365.0
        r = 0.04
        
        # 1. Shannon Profit (Price Action)
        # Shannon = c * ln(P/t)
        shannon_profit = c_old * np.log(current_price / t_old)
        
        # 2. Harvest (Vol Premium)
        harvest_profit = c_old * 0.5 * (sigma ** 2) * dt
        
        # 3. Hedge Cost (Upfront for Surplus calculation)
        strike_price_old = t_old * 0.9 # Protective Put (10% OTM) from START
        put_premium = black_scholes(t_old, strike_price_old, 1.0, r, sigma, 'put') 
        qty_puts = (c_old / t_old) * hedge_ratio
        
        # Cost Estimate: Pro-rated premium (Budget) + Buffer
        # Treating it as "Cost of Carry" for this period
        hedge_cost = (qty_puts * put_premium) * dt * 1.5 
        
        # 4. Net Surplus (Internal Fuel)
        # Surplus = Shannon + Harvest - Hedge Cost
        surplus = shannon_profit + harvest_profit - hedge_cost
        
        # 5. Put Payoff (External Safety Net)
        # Did price crash below Strike?
        put_payoff = 0.0
        if current_price < strike_price_old:
            put_payoff = qty_puts * (strike_price_old - current_price)
            
        # 6. Rollover Logic
        # If Surplus > 0 -> Scale Up (Reinvest)
        # If Surplus < 0 -> Absorb Cost (No Scale Up)
        scale_up = max(0, surplus)
        
        c_new = c_old + scale_up # Price appreciation/loss doesn't change c automatically, only Surplus does.
        t_new = current_price # Re-center
        
        # Baseline Update (Log Path Continuity)
        # b_new = b_old + c_old*ln(P/t_old) - c_new*ln(P/t_new)
        # Since t_new = P, ln(P/t_new) = 0
        term1 = c_old * np.log(current_price / t_old)
        b_new = b_old + term1
        
        return {
            "round_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "p_old": t_old,
            "p_new": current_price,
            "shannon": shannon_profit,
            "harvest": harvest_profit,
            "hedge_cost": hedge_cost,
            "put_payoff": put_payoff,
            "surplus": surplus,
            "scale_up": scale_up,
            "c_old": c_old,
            "c_new": c_new,
            "b_old": b_old,
            "b_new": b_new,
            "note": f"Round: {days_passed} days"
        }

# ============================================================
# 4. UI COMPONENTS
# ============================================================

def setup_page():
    st.set_page_config(page_title="Chain System Core V2", layout="wide")
    st.markdown("""
    <style>
        .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;}
        .surplus-pos {color: #00c853; font-weight: bold;}
        .surplus-neg {color: #ff1744; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

def render_dashboard(data):
    st.title("🔗 Chain System: Operation Core (V2)")
    
    # Global Metrics
    total_c = sum(t["state"]["c"] for t in data["tickers"])
    pool_cf = data.get("global_pool", 0.0)
    total_ev = sum(t["metrics"].get("total_ev_paid", 0) for t in data["tickers"])
    
    # Top Bar
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🛡️ Total Fix_C (Deployed)", f"${total_c:,.0f}")
        c2.metric("🎱 Global Pool CF", f"${pool_cf:,.2f}", "Internal War Chest")
        c3.metric("🔥 Total Ev Burned", f"${total_ev:,.2f}", "Realized Hazard Cost")
        c4.button("Refresh Data", on_click=lambda: st.rerun())

    st.divider()

    # Ticker Grid
    if not data["tickers"]:
        st.info("No tickers active. Add a new Chain below.")
    else:
        st.subheader("Active Chains")
        for i, ticker in enumerate(data["tickers"]):
            with st.container(border=True):
                # Layout: Info | Metrics | Surplus | Actions
                c1, c2, c3, c4 = st.columns([1.5, 2.5, 1.5, 2.5])
                
                state = ticker["state"]
                rnd = ticker.get("rounds", [{}])[-1]
                
                # C1: Symbol
                c1.markdown(f"### {ticker['symbol']}")
                c1.caption(f"Update: {rnd.get('round_date', 'New')}")
                
                # C2: State
                c2.metric("Price (t)", f"${state['t']:,.2f}", f"Base: ${state['b']:,.2f}")
                c2.caption(f"Capital: ${state['c']:,.0f} | Hedge: {state['hedge_ratio']}x")
                
                # C3: Last Surplus
                last_surplus = rnd.get("surplus", 0)
                c3.metric("Last Surplus", f"${last_surplus:,.2f}", 
                          delta_color="normal" if last_surplus >= 0 else "inverse")
                
                # C4: Actions
                if c4.button("⚡ RUN ROUND", key=f"run_{i}", type="primary"):
                    st.session_state["modal_idx"] = i
                    st.session_state["modal_type"] = "run"
                    st.rerun()
                
                if c4.button("💰 Deploy Pool", key=f"deploy_{i}"):
                    st.session_state["modal_idx"] = i
                    st.session_state["modal_type"] = "deploy"
                    st.rerun()

def render_run_round_modal(data, idx):
    ticker = data["tickers"][idx]
    state = ticker["state"]
    
    with st.expander(f"⚡ Run Round: {ticker['symbol']}", expanded=True):
        st.info("Operating Round: Calculate Surplus & Pool CF")
        with st.form("round_form"):
            c1, c2 = st.columns(2)
            current_sim_price = state['t']
            
            p_new = c1.number_input("Current Market Price ($)", value=float(current_sim_price), step=0.5)
            days = c2.number_input("Days since last round", value=30, step=1)
            
            # Preview Logic
            res = ChainEngine.run_round(ticker, p_new, state['sigma'], state['hedge_ratio'], days)
            
            st.markdown("---")
            st.markdown("#### Preview Result")
            
            # Row 1: Components
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Shannon (Price)", f"${res['shannon']:,.2f}",
                       delta="Negative OK if Crash" if res['shannon'] < 0 else None, delta_color="inverse")
            rc2.metric("Harvest (Vol)", f"${res['harvest']:,.2f}")
            rc3.metric("Hedge Cost (Ev)", f"-${res['hedge_cost']:,.2f}")
            
            # Row 2: Final Split (Surplus vs Pool CF)
            st.markdown("#### Final Split")
            sc1, sc2 = st.columns(2)
            
            surplus = res['surplus']
            sc1.metric("💰 Surplus (Internal)", f"${surplus:,.2f}",
                       delta="Scale Up Fuel" if surplus > 0 else "Drag",
                       delta_color="normal" if surplus > 0 else "off")
            
            payoff = res['put_payoff']
            sc2.metric("🎱 Pool CF (External)", f"${payoff:,.2f}",
                       delta="Crash Protection Payout" if payoff > 0 else "Expired",
                       delta_color="normal")
            
            st.caption(f"New fix_c: ${res['c_new']:,.2f} | New Baseline: ${res['b_new']:,.2f}")
            
            if st.form_submit_button("✅ COMMIT ROUND"):
                # Update State
                ticker["state"].update({"t": res["p_new"], "c": res["c_new"], "b": res["b_new"]})
                
                # Update Metrics
                m = ticker["metrics"]
                m["total_ev_paid"] = m.get("total_ev_paid", 0) + res["hedge_cost"]
                if surplus > 0: m["total_surplus"] = m.get("total_surplus", 0) + surplus
                
                # Update Global Pool
                if payoff > 0:
                    data["global_pool"] = data.get("global_pool", 0.0) + payoff
                
                # Save Round
                ticker["rounds"].append(res)
                data["tickers"][idx] = ticker
                save_data(data)
                
                st.session_state["modal_type"] = None
                st.success(f"Round Saved! Pool CF +${payoff:,.2f}")
                st.rerun()

def render_deploy_modal(data, idx):
    ticker = data["tickers"][idx]
    pool_avail = data.get("global_pool", 0.0)
    
    with st.expander(f"💰 Deploy Capital: {ticker['symbol']}", expanded=True):
        st.info(f"Inject capital from Global Pool (${pool_avail:,.2f}) into this Ticker.")
        
        with st.form("deploy_form"):
            amount = st.number_input("Amount to Deploy ($)", min_value=0.0, max_value=pool_avail, step=100.0)
            
            if st.form_submit_button("✅ CONFIRM DEPLOY"):
                if amount > 0:
                    # Logic
                    data["global_pool"] -= amount
                    ticker["state"]["c"] += amount
                    
                    # Log Event as a specialized round
                    ticker["rounds"].append({
                        "round_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "note": f"⚡ DEPLOYED ${amount:,.2f} from Pool",
                        "scale_up": amount,
                        "surplus": 0, "put_payoff": 0, "hedge_cost": 0, "shannon": 0, "harvest": 0,
                        "c_old": ticker["state"]["c"] - amount, "c_new": ticker["state"]["c"],
                        "p_old": ticker["state"]["t"], "p_new": ticker["state"]["t"],
                        "b_new": ticker["state"]["b"] # Baseline unchanged if injected at same price
                    })
                    
                    data["tickers"][idx] = ticker
                    save_data(data)
                    st.session_state["modal_type"] = None
                    st.success("Capital Deployed!")
                    st.rerun()
                else:
                    st.error("Amount must be > 0")

def render_add_ticker(data):
    with st.expander("➕ Add New Chain Ticker"):
        with st.form("new_ticker"):
            sym = st.text_input("Symbol (e.g. BTC)")
            c1, c2 = st.columns(2)
            c = c1.number_input("Initial Capital (fix_c)", value=10000.0)
            t = c2.number_input("Initial Price (t)", value=100.0)
            
            if st.form_submit_button("Create Chain"):
                if sym:
                    data["tickers"].append({
                        "symbol": sym.upper(),
                        "state": {"t": t, "c": c, "b": 0.0, "sigma": 0.5, "hedge_ratio": 2.0},
                        "rounds": [],
                        "metrics": {"total_ev_paid": 0.0, "total_surplus": 0.0}
                    })
                    save_data(data)
                    st.rerun()

# ============================================================
# MAIN
# ============================================================

def main():
    setup_page()
    data = load_data()
    
    # Sidebar
    st.sidebar.title("Flywheel Core V2")
    menu = st.sidebar.radio("Mode", ["Active Dashboard", "Analytics", "Manual Page"])
    
    if menu == "Active Dashboard":
        render_dashboard(data)
        render_add_ticker(data)
        
        # Modal System
        m_type = st.session_state.get("modal_type")
        m_idx = st.session_state.get("modal_idx")
        
        if m_type == "run" and m_idx is not None:
            render_run_round_modal(data, m_idx)
            if st.button("Close Modal"):
                st.session_state["modal_type"] = None
                st.rerun()
        
        elif m_type == "deploy" and m_idx is not None:
            render_deploy_modal(data, m_idx)
            if st.button("Close Modal"):
                st.session_state["modal_type"] = None
                st.rerun()

    elif menu == "Analytics":
        st.header("Analytics (Coming Soon)")
        st.info("Visualizations for Equity Curve & Ev Burn.")

    elif menu == "Manual Page":
        st.info("Legacy Manual Content.")
        # Re-import original manual logic here if needed, or keep separate.

if __name__ == "__main__":
    main()
