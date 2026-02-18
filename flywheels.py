
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


def allocate_pool_funds(data, ticker_idx, amount, action_type="Scale Up", note=""):
    """
    Allocate funds from global Pool CF to a specific ticker with various objectives.
    action_type:
      - "Scale Up": Increase fix_c (Traditional Deploy)
      - "Buy Puts": Expense for hedging (Reduces Pool, doesn't increase fix_c)
      - "Buy Calls": Expense for speculation (Reduces Pool)
      - "Pay Ev": Pay off cumulative Ev debt (Reduces Pool, Reduces cumulative_ev)
    """
    if amount <= 0 or amount > data.get("global_pool_cf", 0):
        return data, False
    
    ticker = data["tickers"][ticker_idx]
    state = ticker.get("current_state", {})
    
    # 1. Deduct from Global Pool
    data["global_pool_cf"] -= amount
    
    # 2. Apply Logic based on Action
    if action_type == "Scale Up":
        old_c = state.get("fix_c", 0)
        state["fix_c"] = old_c + amount
        # Sync Final
        t = state.get("price", 0)
        b = state.get("baseline", 0)
        ticker["Final"] = f"{t}, {state['fix_c']:.2f}, {b:.2f}"
    
    elif action_type == "Pay Ev":
        # Reducing the visible "Burn Rate" debt
        old_ev_debt = state.get("cumulative_ev", 0.0)
        state["cumulative_ev"] = max(0.0, old_ev_debt - amount)
        
    elif action_type in ["Buy Puts", "Buy Calls"]:
        # Just an expenditure, maybe track in a log if we had one, 
        # but for now it just consumes Pool CF to support the ticker positions.
        pass
        
    ticker["current_state"] = state
    data["tickers"][ticker_idx] = ticker
    
    # 3. Log History (Optional but good for tracking)
    # We'll use the legacy history log for now to show the event
    h_idx = 1
    while f"history_{h_idx}" in ticker:
        h_idx += 1
    
    if action_type == "Scale Up":
        desc = f"🎱 Pool Allocation: {action_type} +${amount:,.2f}"
        calc = f"fix_c updated to {state['fix_c']:.2f} | Note: {note}"
    elif action_type == "Pay Ev":
        desc = f"🎱 Pool Allocation: {action_type} -${amount:,.2f}"
        calc = f"Burn Rate (Ev Debt) reduced by {amount:,.2f} | Note: {note}"
    else:
        desc = f"🎱 Pool Allocation: {action_type} -${amount:,.2f}"
        calc = f"Funded via Pool CF | Note: {note}"

    ticker[f"history_{h_idx}"] = desc
    ticker[f"history_{h_idx}.1"] = calc
    
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
# CHAPTER 8: CHAIN SYSTEM — imported from chain_system.py
# ============================================================
def chapter_chain_system():
    from chain_system import chapter_chain_system as _run_chapter
    _run_chapter()



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
