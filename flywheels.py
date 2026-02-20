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
from typing import Dict, List, Tuple, Optional, Any, Union

# ============================================================
# UTILITIES
# ============================================================

def sanitize_number_str(s: Optional[str]) -> str:
    """Normalize number strings: replace Unicode minus, remove commas/spaces."""
    if not s:
        return ""
    return str(s).replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-').replace(',', '').strip()

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes option pricing."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    except Exception:
        return 0.0

def generate_gbm(S0: float, mu: float, sigma: float, T: float, dt: float, n_sims: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Geometric Brownian Motion price path."""
    N = int(T / dt)
    if N <= 0:
        return np.array([0.0]), np.array([S0])
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return t, S

# ============================================================
# DATA LAYER — trading_data.json
# ============================================================

_DATA_DIR: str = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE: str = os.path.join(_DATA_DIR, "trading_data.json")
_BACKUP_FILE: str = os.path.join(_DATA_DIR, "trading_data.backup.json")

def load_trading_data() -> Dict[str, Any]:
    """Load portfolio data, auto-migrating to V2 if needed."""
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []
    
    return _migrate_data_if_needed(raw)

def save_trading_data(data: Dict[str, Any]) -> None:
    """Save portfolio data with auto-backup and atomic write pattern safely."""
    if os.path.exists(_DATA_FILE):
        try:
            import shutil
            shutil.copy2(_DATA_FILE, _BACKUP_FILE)
        except Exception:
            pass
            
    tmp_file = f"{_DATA_FILE}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_file, _DATA_FILE)
    except Exception as e:
        st.error(f"Error saving data: {e}")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

def get_tickers(data: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    """Get tickers list from V2 data structure."""
    if isinstance(data, dict):
        return data.get("tickers", [])
    return data if isinstance(data, list) else []

def _migrate_data_if_needed(raw_data: Union[Dict, List]) -> Dict[str, Any]:
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
                "price": t if t is not None else 0.0,
                "fix_c": c if c is not None else 0.0,
                "baseline": b if b is not None else 0.0,
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
        },
        "treasury_history": []
    }


# ============================================================
# CHAIN ENGINE — Core Round Logic
# ============================================================

def run_chain_round(ticker_state: Dict[str, float], p_new: float, sigma: float, hedge_ratio: float, r: float = 0.04, T: float = 1.0) -> Optional[Dict[str, Any]]:
    """Core engine: compute one chain round.
    Returns a round_data dict (not yet committed)."""
    try:
        c = float(ticker_state.get("fix_c", 0.0))
        t = float(ticker_state.get("price", 0.0))
        b = float(ticker_state.get("baseline", 0.0))
    except (ValueError, TypeError):
        return None

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
    if t_new != p_new:
        rollover_delta = c * np.log(p_new / t) - c_new * np.log(p_new / t_new)
    else:
        rollover_delta = c * np.log(p_new / t)
        
    b_new = b + rollover_delta

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
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


def commit_round(data: Dict[str, Any], ticker_idx: int, round_data: Dict[str, Any]) -> Dict[str, Any]:
    """Save round to ticker's rounds[], update current_state, sync legacy fields."""
    try:
        ticker = data["tickers"][ticker_idx]
    except (IndexError, KeyError):
        return data
        
    if "rounds" not in ticker:
        ticker["rounds"] = []
        
    round_data["round_id"] = len(ticker["rounds"]) + 1
    ticker["rounds"].append(round_data)

    # Update current_state
    old_ev = float(ticker.get("current_state", {}).get("cumulative_ev", 0.0))
    hedge_cost = float(round_data.get("hedge_cost", 0.0))
    ev_change = float(round_data.get("ev_change", hedge_cost))
    
    ticker["current_state"] = {
        "price": float(round_data.get("p_new", 0.0)),
        "fix_c": float(round_data.get("c_after", 0.0)),
        "baseline": float(round_data.get("b_after", 0.0)),
        "pool_cf_net": float(ticker.get("current_state", {}).get("pool_cf_net", 0.0)),
        "cumulative_ev": max(0.0, old_ev + ev_change),
    }

    # Sync legacy Final
    ticker["Final"] = f"{round_data.get('p_new', 0)}, {round_data.get('c_after', 0)}, {round_data.get('b_after', 0)}"

    # Write legacy history entry
    h_idx = 1
    while f"history_{h_idx}" in ticker:
        h_idx += 1
        
    desc = (f"ราคาอ้างอิง: {round_data.get('p_old', 0)} → {round_data.get('p_new', 0)} , "
            f"ทุนคงที่: {round_data.get('c_before', 0)} → {round_data.get('c_after', 0)} , "
            f"ราคาปัจจุบัน: {round_data.get('p_new', 0)}")
            
    calc = (f"{round_data.get('b_before', 0):.2f} += ({round_data.get('c_before', 0)} × ln({round_data.get('p_new', 1)}/{round_data.get('p_old', 1) if round_data.get('p_old') else 1})) − "
            f"({round_data.get('c_after', 0)} × ln({round_data.get('p_new', 1)}/{round_data.get('p_new', 1)})) | "
            f"c = {round_data.get('c_after', 0)} , t = {round_data.get('p_new', 0)} , b = {round_data.get('b_after', 0)}")
            
    ticker[f"history_{h_idx}"] = desc
    ticker[f"history_{h_idx}.1"] = calc

    data["tickers"][ticker_idx] = ticker
    save_trading_data(data)
    return data


def allocate_pool_funds(data: Dict[str, Any], ticker_idx: int, amount: float, action_type: str = "Scale Up", note: str = "") -> Tuple[Dict[str, Any], bool]:
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
    
    try:
        ticker = data["tickers"][ticker_idx]
    except (IndexError, KeyError):
        return data, False
        
    state = ticker.get("current_state", {})
    
    # 1. Deduct from Global Pool
    data["global_pool_cf"] -= amount
    
    # 2. Apply Logic based on Action
    if action_type == "Scale Up":
        old_c = float(state.get("fix_c", 0.0))
        state["fix_c"] = old_c + amount
        # Sync Final
        t = float(state.get("price", 0.0))
        b = float(state.get("baseline", 0.0))
        ticker["Final"] = f"{t}, {state['fix_c']:.2f}, {b:.2f}"
    
    elif action_type == "Pay Ev":
        # Reducing the visible "Burn Rate" debt
        old_ev_debt = float(state.get("cumulative_ev", 0.0))
        state["cumulative_ev"] = max(0.0, old_ev_debt - amount)
        
    elif action_type in ["Buy Puts", "Buy Calls"]:
        # Just an expenditure, maybe track in a log if we had one.
        pass
        
    ticker["current_state"] = state
    data["tickers"][ticker_idx] = ticker
    
    # 3. Log History (Optional but good for tracking)
    h_idx = 1
    while f"history_{h_idx}" in ticker:
        h_idx += 1
    
    if action_type == "Scale Up":
        desc = f"🎱 Pool Allocation: {action_type} +${amount:,.2f}"
        calc = f"fix_c updated to {state.get('fix_c', 0):.2f} | Note: {note}"
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


def parse_final(final_str: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse 'Final' field → (t, c, b).  e.g. '12, 4000, -519.45' → (12.0, 4000.0, -519.45)"""
    if not final_str:
        return None, None, None
    parts = [sanitize_number_str(p) for p in final_str.split(",")]
    try:
        t = float(parts[0]) if len(parts) > 0 and parts[0] else None
        c = float(parts[1]) if len(parts) > 1 and parts[1] else None
        b = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        return t, c, b
    except (ValueError, IndexError):
        return None, None, None

def parse_beta_numbers(beta_str: str) -> Tuple[float, float]:
    """Extract Ev and Lock_P&L from beta_Equation string."""
    ev, lock_pnl = 0.0, 0.0
    if not beta_str:
        return ev, lock_pnl
    try:
        beta_str = sanitize_number_str(beta_str)
        # Extract Ev
        ev_match = re.search(r'Ev:\s*([+-]?[\d.]+)', beta_str)
        if ev_match:
            ev = float(ev_match.group(1))
        # Extract Lock_P&L
        lock_match = re.search(r'Lock_P&L:\s*(.+)', beta_str)
        if lock_match:
            raw = lock_match.group(1).strip()
            raw = raw.replace("|", "+")
            nums = re.findall(r'[+-]?[\d.]+', raw)
            for n in nums:
                lock_pnl += float(n)
        return ev, lock_pnl
    except Exception:
        return 0.0, 0.0

def parse_beta_net(beta_mem_str: str) -> float:
    """Extract Net value from beta_momory string."""
    if not beta_mem_str:
        return 0.0
    try:
        beta_mem_str = sanitize_number_str(beta_mem_str)
        m = re.search(r'Net:\s*([+-]?[\d.]+)', beta_mem_str)
        if m:
            return float(m.group(1))
        return 0.0
    except Exception:
        return 0.0

def parse_surplus_iv(surplus_str: str) -> float:
    """Extract Surplus IV (Put premium income) from Surplus_Iv string."""
    if not surplus_str or "No_Expiry" in surplus_str:
        return 0.0
    try:
        surplus_str = sanitize_number_str(surplus_str)
        matches = re.findall(r'=\s*([+-]?\d+(?:\.\d+)?)', surplus_str)
        total = sum(float(m) for m in matches)
        return total
    except Exception:
        return 0.0

def get_rollover_history(ticker_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract all history entries in order, returning list of dicts."""
    history = []
    i = 1
    while True:
        key_desc = f"history_{i}"
        key_calc = f"history_{i}.1"
        if key_desc not in ticker_data and key_calc not in ticker_data:
            break
        entry: Dict[str, Any] = {"step": i}
        entry["description"] = ticker_data.get(key_desc, "")
        entry["calculation"] = ticker_data.get(key_calc, "")
        
        calc_str = entry["calculation"]
        try:
            b_match = re.search(r'b\s*=\s*([+-]?[\d,.]+)', calc_str)
            entry["b"] = float(b_match.group(1).replace(",", "")) if b_match else None
            
            c_match = re.search(r'\|\s*c\s*=\s*([\d,.]+)', calc_str)
            entry["c"] = float(c_match.group(1).replace(",", "")) if c_match else None
            
            t_match = re.search(r',\s*t\s*=\s*([\d,.]+)', calc_str)
            entry["t"] = float(t_match.group(1).replace(",", "")) if t_match else None
        except Exception:
            entry["b"], entry["c"], entry["t"] = None, None, None
            
        history.append(entry)
        i += 1
    return history

def build_portfolio_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
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
            "Baseline (b)": b if b is not None else 0.0,
            "Ev (Extrinsic)": ev,
            "Lock P&L": lock_pnl,
            "Surplus IV": surplus_iv,
            "Net": net,
        })
    return pd.DataFrame(rows)


# ============================================================
# CHAPTERS 0-7 — Placeholders
# ============================================================

def chapter_0_introduction() -> None:
    st.header("บทที่ 0: Introduction")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_1_baseline() -> None:
    st.header("บทที่ 1: Baseline")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_2_shannon_process() -> None:
    st.header("บทที่ 2: Shannon Process")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_3_volatility_harvesting() -> None:
    st.header("บทที่ 3: Volatility Harvesting")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_4_black_swan_shield() -> None:
    st.header("บทที่ 4: Black Swan Shield")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_5_dynamic_scaling() -> None:
    st.header("บทที่ 5: Dynamic Scaling")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_6_synthetic_dividend() -> None:
    st.header("บทที่ 6: Synthetic Dividend")
    st.warning("Content currently unavailable. Please restore from backup if needed.")

def chapter_7_collateral_magic() -> None:
    st.header("บทที่ 7: Collateral Magic")
    st.warning("Content currently unavailable. Please restore from backup if needed.")


# ============================================================
# CHAPTER 8: CHAIN SYSTEM — imported from chain_system.py
# ============================================================
def chapter_chain_system() -> None:
    try:
        from chain_system import chapter_chain_system as _run_chapter
        _run_chapter()
    except ImportError:
        st.warning("File `chain_system.py` not found or contains errors. Falling back to active Engine.")
        st.info("💡 Plase run `app.py` for the main Engine view.")


# ============================================================
# FUNCTION ALIASES — for Manual page compatibility
# ============================================================
chapter_2_volatility_harvest = chapter_2_shannon_process
chapter_3_convexity_engine = chapter_3_volatility_harvesting

# ============================================================
# MAIN APP NAVIGATION
# ============================================================

def main() -> None:
    st.sidebar.title("Flywheel & Shannon's Demon")
    menu = st.sidebar.radio("Menu", [
        "Introduction", "Baseline", "Shannon Process", "Volatility Harvesting",
        "Black Swan Shield", "Dynamic Scaling", "Synthetic Dividend", "Collateral Magic",
        "Chain System (Active)", "Quiz", "Paper Trading", "Glossary"
    ], index=8)

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

def master_study_guide_quiz() -> None: pass
def paper_trading_workshop() -> None: pass
def glossary_section() -> None: pass

if __name__ == "__main__":
    main()
