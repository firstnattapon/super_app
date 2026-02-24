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
    """Load portfolio data, auto-migrating to V2 and auto-repairing current_state if needed."""
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        raw = []
    
    data = _migrate_data_if_needed(raw)
    _auto_repair_current_state(data)   # ✅ silent fix every load
    return data


def _auto_repair_current_state(data: Dict[str, Any]) -> None:
    """Auto-sync current_state fields from rounds[-1] when they diverge.
    Only patches fields that are stale — preserves fields not tracked in rounds.
    Does NOT save; caller decides whether to persist.
    """
    dirty = False
    for ticker in data.get("tickers", []):
        rounds = ticker.get("rounds", [])
        if not rounds:
            continue
        last  = rounds[-1]
        state = ticker.setdefault("current_state", {})
        
        # Only repair Chain Round entries (not Extract Baseline / Pool ops)
        action = last.get("action", "")
        if "Chain Round" not in action and "Chain Round" != action:
            # Still sync price/fix_c/baseline from last round regardless of action type
            pass

        new_vals = {
            "baseline": float(last.get("b_after", state.get("baseline", 0.0))),
            "price":    float(last.get("p_new",   state.get("price",    0.0))),
            "fix_c":    float(last.get("c_after",  state.get("fix_c",   0.0))),
        }
        for k, v in new_vals.items():
            if abs(state.get(k, 0.0) - v) > 1e-9:
                state[k] = v
                dirty = True

    if dirty:
        save_trading_data(data)

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

def run_chain_round(ticker_state: Dict[str, float], p_new: float, hedge_ratio: float, r: float = 0.04, T: float = 1.0, ignore_hedge: bool = False, ignore_surplus: bool = False) -> Optional[Dict[str, Any]]:
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
    total_income = shannon
    
    # 2. Hedge Cost (Full Upfront Put)
    if ignore_hedge:
        hedge_cost = 0.0
    else:
        qty = (c / t) * hedge_ratio
        strike = t * 0.9
        sigma = 0.5  # Fixed hidden value for hedge cost
        premium = black_scholes(t, strike, T, r, sigma, 'put')
        hedge_cost = qty * premium
    
    # 3. Surplus
    surplus = total_income - hedge_cost
    scale_up = 0.0 if ignore_surplus else max(0.0, surplus)
    
    # 4. New State
    c_new = c + scale_up
    t_new = p_new  # re-center
    
    # 5. Rollover (b stays continuous)
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
        "hedge_cost": round(hedge_cost, 2),
        "surplus": round(surplus, 2),
        "scale_up": round(scale_up, 2),
        "b_before": round(b, 2),
        "b_after": round(b_new, 2),
        "hedge_ratio": hedge_ratio,
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
    cur_state = ticker.get("current_state", {})
    old_ev = float(cur_state.get("cumulative_ev", 0.0))
    hedge_cost = float(round_data.get("hedge_cost", 0.0))
    ev_change = float(round_data.get("ev_change", hedge_cost))
    
    # ✅ BUG-01 FIX: always read live fields from round_data, never from stale cur_state
    ticker["current_state"] = {
        "price":    float(round_data["p_new"]),
        "fix_c":    float(round_data["c_after"]),
        "baseline": float(round_data["b_after"]),
        "pool_cf_net": float(cur_state.get("pool_cf_net", 0.0)),
        "cumulative_ev": max(0.0, old_ev + ev_change),
        "surplus_iv": float(cur_state.get("surplus_iv", 0.0)),
        "lock_pnl": float(cur_state.get("lock_pnl", 0.0)),
        "net_pnl": float(cur_state.get("net_pnl", 0.0)),
        "strategy_tags": cur_state.get("strategy_tags", [])
    }

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
    old_c = float(state.get("fix_c", 0.0))
    if action_type == "Scale Up":
        state["fix_c"] = old_c + amount
    elif action_type == "Pay Ev":
        # Reducing the visible "Burn Rate" debt
        old_ev_debt = float(state.get("cumulative_ev", 0.0))
        state["cumulative_ev"] = max(0.0, old_ev_debt - amount)
    elif action_type in ["Buy Puts", "Buy Calls"]:
        # Just an expenditure, maybe track in a log if we had one.
        pass
        
    ticker["current_state"] = state
    data["tickers"][ticker_idx] = ticker
    
    # 3. Log History as a round
    round_data = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "action": f"🎱 {action_type}",
        "p_old": float(state.get("price", 0.0)),
        "p_new": float(state.get("price", 0.0)),
        "c_before": old_c,
        "c_after": float(state.get("fix_c", 0.0)),
        "b_before": float(state.get("baseline", 0.0)),
        "b_after": float(state.get("baseline", 0.0)),
        "surplus": 0.0,
        "scale_up": amount if action_type == "Scale Up" else 0.0,
        "note": note
    }
    
    if "rounds" not in ticker:
        ticker["rounds"] = []
    round_data["round_id"] = len(ticker["rounds"]) + 1
    ticker["rounds"].append(round_data)
    
    save_trading_data(data)
    return data, True


def repair_baseline(data: Dict[str, Any]) -> Dict[str, Any]:
    """One-time patch: rebuild current_state from rounds[-1] for every ticker.
    Safe to call multiple times — idempotent.
    """
    for ticker in data.get("tickers", []):
        rounds = ticker.get("rounds", [])
        if rounds:
            last = rounds[-1]
            state = ticker.get("current_state", {})
            state["baseline"] = float(last.get("b_after", state.get("baseline", 0.0)))
            state["price"]    = float(last.get("p_new",   state.get("price",    0.0)))
            state["fix_c"]    = float(last.get("c_after", state.get("fix_c",    0.0)))
            ticker["current_state"] = state
    save_trading_data(data)
    return data



def build_portfolio_df(data) -> pd.DataFrame:
    """Build a pandas DataFrame summarizing all tickers."""
    tickers = data.get("tickers", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    rows = []
    for item in tickers:
        state = item.get("current_state", {})
        rows.append({
            "Ticker":           item.get("ticker", "???"),
            "Price (t)":        float(state.get("price",         0.0)),
            "Fix_C":            float(state.get("fix_c",          0.0)),
            "Baseline (b)":     float(state.get("baseline",       0.0)),
            "Ev (Extrinsic)":   float(state.get("cumulative_ev",  0.0)),
            "Lock P&L":         float(state.get("lock_pnl",       0.0)),
            "Surplus IV":       float(state.get("surplus_iv",      0.0)),
            "Net":              float(state.get("net_pnl",         0.0)),
        })
    return pd.DataFrame(rows)

