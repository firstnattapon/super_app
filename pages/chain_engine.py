"""
chain_engine.py — Chain Engine v3.0-minimal
Commercial Grade · Single File · Self-Contained
State: (price, fix_c, baseline, cumulative_ev)
Engine: surplus = fix_c × ln(p_new/price) × (1 - hedge_ratio)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import math
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Chain Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_FILE = Path("chain_data.json")

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c14;
    color: #c9d1d9;
}

/* Top metric bar */
.metric-block {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 4px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    font-weight: 600;
    line-height: 1.1;
}
.metric-value.pos { color: #3fb950; }
.metric-value.neg { color: #f85149; }
.metric-value.neu { color: #e6edf3; }
.metric-value.cost { color: #d29922; }

/* Watchlist */
.watch-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    border-radius: 4px;
    cursor: pointer;
    border-left: 2px solid transparent;
    margin-bottom: 2px;
    background: #0d1117;
    transition: all 0.15s;
}
.watch-row:hover { background: #161b22; border-left-color: #388bfd; }
.watch-row.active { background: #132338; border-left-color: #1f6feb; }
.watch-ticker { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600; color: #e6edf3; }
.watch-net.pos { color: #3fb950; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }
.watch-net.neg { color: #f85149; font-size: 12px; font-family: 'IBM Plex Mono', monospace; }

/* Preview card */
.preview-card {
    background: #0d1117;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 14px 18px;
}
.preview-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #1c2a3a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
}
.preview-row:last-child { border-bottom: none; }
.preview-key { color: #4a6080; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.preview-val { font-weight: 600; }
.preview-val.pos { color: #3fb950; }
.preview-val.neg { color: #f85149; }
.preview-val.cost { color: #d29922; }
.preview-val.neu  { color: #e6edf3; }

/* History table */
.history-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 6px 0 4px;
    border-bottom: 1px solid #1c2a3a;
}

/* Input styling */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
    border-radius: 4px !important;
}
div[data-testid="stSlider"] { margin-top: 4px; }

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    transition: all 0.15s !important;
}
.stButton > button[kind="primary"] {
    background: #1f6feb !important;
    border: none !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    background: #388bfd !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
}

/* Divider */
hr { border-color: #1c2a3a !important; margin: 10px 0 !important; }

/* Radio buttons in watchlist */
div[data-testid="stRadio"] label { cursor: pointer; }
div[data-testid="stRadio"] { gap: 2px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #080c14 !important; }

/* Top bar */
.topbar {
    background: #0d1117;
    border-bottom: 1px solid #1c2a3a;
    padding: 10px 0 12px;
    margin-bottom: 14px;
}
.engine-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.01em;
}
.engine-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3d5570;
    margin-top: 1px;
}
.badge {
    display: inline-block;
    background: #132338;
    border: 1px solid #1f3a5f;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #79c0ff;
    margin-left: 8px;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #1c2a3a !important;
    border-radius: 6px !important;
}
.stDataFrame thead tr th {
    background: #0d1117 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #4a6080 !important;
}
.stDataFrame tbody tr td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    background: #080c14 !important;
}

/* Form */
div[data-testid="stForm"] { border: none !important; padding: 0 !important; background: transparent !important; }

/* Remove default streamlit padding */
.block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; }

/* Chain equation strip */
.eq-strip {
    background: #0d1117;
    border: 1px solid #1c2a3a;
    border-radius: 6px;
    padding: 8px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #3d5570;
    margin-bottom: 12px;
}
.eq-strip span.eq-hl { color: #79c0ff; }
.eq-strip span.eq-op { color: #d29922; }
.eq-strip span.eq-result { color: #3fb950; font-weight: 600; }

/* No-ticker state */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    font-family: 'IBM Plex Mono', monospace;
    color: #3d5570;
    font-size: 13px;
}

/* Section header */
.sec-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3d5570;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid #1c2a3a;
}

/* Commit banner */
.commit-ok {
    background: #0f2a1a;
    border: 1px solid #238636;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #3fb950;
    margin-top: 8px;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #1c2a3a; border-radius: 4px; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PERSISTENCE  (JSON flat file)
# ─────────────────────────────────────────────

def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tickers": [], "pool_cf": 0.0}


def save_data(d: dict) -> None:
    DATA_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


# ─────────────────────────────────────────────
# PURE MATH ENGINE  — f(state, input) → result
# ─────────────────────────────────────────────

def compute_round(state: dict, p_new: float, hedge_ratio: float, ev_cost: float) -> dict:
    """
    Pure function. No side effects.
    Returns full derived + proposed next state.
    Equation: surplus = fix_c × ln(p_new / price) × (1 − hedge_ratio)
    """
    price   = float(state["price"])
    fix_c   = float(state["fix_c"])
    base    = float(state["baseline"])
    cum_ev  = float(state["cumulative_ev"])

    if p_new <= 0 or price <= 0:
        raise ValueError("Price must be > 0")

    shannon  = fix_c * math.log(p_new / price)
    hedge    = shannon * hedge_ratio
    surplus  = shannon - hedge
    delta_fc = max(0.0, surplus)

    fix_c2   = fix_c + delta_fc
    base2    = base + surplus
    cum_ev2  = cum_ev + ev_cost
    net_pnl2 = base2 - cum_ev2
    net_pnl0 = base - cum_ev

    ev_eff   = (base2 / cum_ev2) if cum_ev2 > 0 else float("inf")

    return {
        # inputs echo
        "p_old":        price,
        "p_new":        p_new,
        "hedge_ratio":  hedge_ratio,
        "ev_cost":      ev_cost,
        # derived
        "shannon":      shannon,
        "hedge":        hedge,
        "surplus":      surplus,
        "delta_fix_c":  delta_fc,
        # next state
        "fix_c_after":  fix_c2,
        "base_after":   base2,
        "cum_ev_after": cum_ev2,
        "net_pnl_before": net_pnl0,
        "net_pnl_after":  net_pnl2,
        "ev_efficiency":  ev_eff,
    }


def commit_round(data: dict, t_idx: int, result: dict) -> None:
    """Apply computed result to state and append history row."""
    t = data["tickers"][t_idx]
    rounds = t.get("rounds", [])

    t["current_state"]["price"]         = result["p_new"]
    t["current_state"]["fix_c"]         = result["fix_c_after"]
    t["current_state"]["baseline"]      = result["base_after"]
    t["current_state"]["cumulative_ev"] = result["cum_ev_after"]

    rounds.append({
        "round":       len(rounds) + 1,
        "date":        datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price":       result["p_old"],
        "p_new":       result["p_new"],
        "shannon":     round(result["shannon"], 4),
        "hedge":       round(result["hedge"], 4),
        "surplus":     round(result["surplus"], 4),
        "delta_fix_c": round(result["delta_fix_c"], 4),
        "fix_c":       round(result["fix_c_after"], 4),
        "baseline":    round(result["base_after"], 4),
        "net_pnl":     round(result["net_pnl_after"], 4),
    })
    t["rounds"] = rounds
    save_data(data)


# ─────────────────────────────────────────────
# HELPER RENDERERS
# ─────────────────────────────────────────────

def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:,.{decimals}f}"


def _color_class(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")


def _metric(label: str, value: str, cls: str = "neu") -> str:
    return f"""
    <div class="metric-block">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
    </div>"""


def _preview_row(key: str, val: str, cls: str = "neu") -> str:
    return f"""
    <div class="preview-row">
        <span class="preview-key">{key}</span>
        <span class="preview-val {cls}">{val}</span>
    </div>"""


def _net_pnl(t: dict) -> float:
    s = t.get("current_state", {})
    return float(s.get("baseline", 0)) - float(s.get("cumulative_ev", 0))


# ─────────────────────────────────────────────
# SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────

def boot():
    if "data" not in st.session_state:
        st.session_state["data"] = load_data()
    if "active_idx" not in st.session_state:
        st.session_state["active_idx"] = 0
    if "pending" not in st.session_state:
        st.session_state["pending"] = None
    if "last_commit" not in st.session_state:
        st.session_state["last_commit"] = None


# ─────────────────────────────────────────────
# PANELS
# ─────────────────────────────────────────────

def render_topbar(data: dict):
    tickers = data.get("tickers", [])
    total_net  = sum(_net_pnl(t) for t in tickers)
    total_fc   = sum(float(t.get("current_state", {}).get("fix_c", 0)) for t in tickers)
    total_ev   = sum(float(t.get("current_state", {}).get("cumulative_ev", 0)) for t in tickers)
    total_base = sum(float(t.get("current_state", {}).get("baseline", 0)) for t in tickers)
    eff = (total_base / total_ev) if total_ev > 0 else float("inf")
    n_rounds = sum(len(t.get("rounds", [])) for t in tickers)

    c0, c1, c2, c3, c4, c5, c6 = st.columns([3, 2, 2, 2, 2, 2, 2])

    with c0:
        st.markdown("""
        <div style="padding-top:4px">
            <div class="engine-title">⚡ Chain Engine</div>
            <div class="engine-sub">v3.0-minimal · surplus = C·ln(p'/p)·(1−h)</div>
        </div>""", unsafe_allow_html=True)

    nc = _color_class(total_net)
    ec = "pos" if eff >= 1 else "neg"
    pool_cf = data.get("pool_cf", 0.0)

    c1.markdown(_metric("Net P&L", f"${_fmt(total_net)}", nc), unsafe_allow_html=True)
    c2.markdown(_metric("Total fix_c", f"${_fmt(total_fc)}"), unsafe_allow_html=True)
    c3.markdown(_metric("Ev Efficiency", "∞" if eff == float("inf") else f"{eff:.2f}×", ec), unsafe_allow_html=True)
    c4.markdown(_metric("Ev Burn", f"${_fmt(total_ev)}", "cost"), unsafe_allow_html=True)
    c5.markdown(_metric("Pool CF", f"${_fmt(pool_cf)}"), unsafe_allow_html=True)
    c6.markdown(_metric("Rounds", str(n_rounds), "neu"), unsafe_allow_html=True)


def render_watchlist(data: dict) -> int:
    tickers = data.get("tickers", [])
    st.markdown('<div class="sec-header">Watchlist</div>', unsafe_allow_html=True)

    if not tickers:
        st.markdown('<div class="empty-state">No tickers<br>Add below ↓</div>', unsafe_allow_html=True)
        return 0

    labels = []
    for t in tickers:
        ticker = t.get("ticker", "???")
        net    = _net_pnl(t)
        sign   = "+" if net >= 0 else ""
        cls    = "pos" if net >= 0 else "neg"
        labels.append(f"{ticker}  ${sign}{net:,.0f}")

    active_idx = st.session_state.get("active_idx", 0)
    active_idx = min(active_idx, len(tickers) - 1)

    choice = st.radio(
        "ticker_list", labels,
        index=active_idx,
        label_visibility="collapsed",
        key="watchlist_radio"
    )
    new_idx = labels.index(choice)
    if new_idx != st.session_state["active_idx"]:
        st.session_state["active_idx"] = new_idx
        st.session_state["pending"] = None
        st.rerun()

    st.markdown("---")

    # Pool CF management
    st.markdown('<div class="sec-header">Pool CF</div>', unsafe_allow_html=True)
    pool_cf = data.get("pool_cf", 0.0)
    st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:13px;color:#79c0ff;margin-bottom:6px'>${pool_cf:,.2f}</div>", unsafe_allow_html=True)

    with st.expander("± Adjust", expanded=False):
        adj = st.number_input("Amount", value=0.0, step=100.0, key="pool_adj", label_visibility="collapsed")
        col_a, col_b = st.columns(2)
        if col_a.button("＋ Add", use_container_width=True):
            if adj > 0:
                data["pool_cf"] = pool_cf + adj
                save_data(data)
                st.rerun()
        if col_b.button("－ Use", use_container_width=True):
            if adj > 0:
                data["pool_cf"] = max(0, pool_cf - adj)
                save_data(data)
                st.rerun()

    return new_idx


def render_input_panel(data: dict, t_idx: int):
    """CENTER: 1-row input form — the entire engine UI."""
    tickers = data.get("tickers", [])
    if not tickers:
        st.markdown('<div class="empty-state">Add a ticker to begin.</div>', unsafe_allow_html=True)
        return

    t      = tickers[t_idx]
    state  = t.get("current_state", {})
    ticker = t.get("ticker", "???")
    price  = float(state.get("price", 100.0))
    fix_c  = float(state.get("fix_c", 1.0))

    # Equation strip
    st.markdown(f"""
    <div class="eq-strip">
        <span class="eq-hl">{ticker}</span>
        &nbsp;·&nbsp;
        surplus <span class="eq-op">=</span>
        <span class="eq-hl">fix_c</span>
        <span class="eq-op">×</span> ln(p_new / price)
        <span class="eq-op">×</span> (1 − hedge_ratio)
        &nbsp;→&nbsp;
        <span class="eq-result">chain ↻</span>
    </div>""", unsafe_allow_html=True)

    # ── 1-ROW FORM ─────────────────────────────────────────────────────
    with st.container(border=True):
        fa, fb, fc, fd, fe = st.columns([2.5, 2.5, 2, 1.8, 1.4])

        with fa:
            st.caption(f"p_new  (now: ${price:,.2f})")
            p_new = st.number_input(
                "p_new", min_value=0.01,
                value=float(round(price * 1.05, 2)),
                step=1.0, format="%.2f",
                label_visibility="collapsed",
                key=f"inp_pnew_{t_idx}"
            )

        with fb:
            st.caption(f"fix_c  (now: {fix_c:,.2f})")
            fix_c_override = st.number_input(
                "fix_c override", min_value=0.0,
                value=float(fix_c),
                step=0.1, format="%.4f",
                label_visibility="collapsed",
                key=f"inp_fixc_{t_idx}",
                help="Override fix_c for this preview only — doesn't commit until you press Commit"
            )

        with fc:
            st.caption("hedge_ratio  [0–1]")
            hedge_ratio = st.slider(
                "hedge", min_value=0.0, max_value=1.0,
                value=0.0, step=0.05,
                label_visibility="collapsed",
                key=f"inp_hedge_{t_idx}"
            )

        with fd:
            st.caption("ev_cost  (θ decay)")
            ev_cost = st.number_input(
                "ev_cost", min_value=0.0,
                value=0.0, step=10.0, format="%.2f",
                label_visibility="collapsed",
                key=f"inp_ev_{t_idx}"
            )

        with fe:
            st.write("")
            st.write("")
            preview_clicked = st.button(
                "▶ Preview", type="primary",
                use_container_width=True,
                key=f"btn_preview_{t_idx}"
            )

    if preview_clicked:
        state_for_calc = dict(state)
        state_for_calc["fix_c"] = fix_c_override  # allow override
        result = compute_round(state_for_calc, p_new, hedge_ratio, ev_cost)
        st.session_state["pending"] = result
        st.session_state["pending_ticker"] = t_idx
        st.session_state["last_commit"] = None

    # ── TICKER STATE QUICK METRICS ─────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:10px">Current State</div>', unsafe_allow_html=True)
    baseline = float(state.get("baseline", 0))
    cum_ev   = float(state.get("cumulative_ev", 0))
    net_pnl  = baseline - cum_ev
    rounds_n = len(t.get("rounds", []))
    eff      = (baseline / cum_ev) if cum_ev > 0 else float("inf")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(_metric("Price",    f"${_fmt(price)}"),                          unsafe_allow_html=True)
    m2.markdown(_metric("fix_c",    f"{_fmt(fix_c, 4)}"),                        unsafe_allow_html=True)
    m3.markdown(_metric("Baseline", f"${_fmt(baseline)}", _color_class(baseline)), unsafe_allow_html=True)
    m4.markdown(_metric("Net P&L",  f"${_fmt(net_pnl)}",  _color_class(net_pnl)), unsafe_allow_html=True)
    m5.markdown(_metric("Rounds",   str(rounds_n)),                              unsafe_allow_html=True)

    # Ev row
    e1, e2, _ = st.columns([2, 2, 6])
    e1.markdown(_metric("Ev Burn",    f"${_fmt(cum_ev)}", "cost"),                                          unsafe_allow_html=True)
    e2.markdown(_metric("Efficiency", "∞" if eff == float("inf") else f"{eff:.2f}×", "pos" if eff >= 1 else "neg"), unsafe_allow_html=True)

    # Add-ticker / manage section
    st.markdown("---")
    with st.expander("⊕ Add Ticker  /  ✕ Remove  /  ⚙ Settings", expanded=False):
        tab_a, tab_b, tab_c = st.tabs(["Add", "Remove", "Import/Export"])

        with tab_a:
            with st.form("add_ticker_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                new_ticker = c1.text_input("Symbol").upper()
                init_price = c2.number_input("Entry Price", min_value=0.01, value=100.0, step=1.0)
                init_fixc  = c3.number_input("Initial fix_c", min_value=0.001, value=1.0, step=0.1)
                if st.form_submit_button("Add", type="primary"):
                    if new_ticker:
                        existing = [t["ticker"] for t in data.get("tickers", [])]
                        if new_ticker not in existing:
                            data.setdefault("tickers", []).append({
                                "ticker": new_ticker,
                                "current_state": {
                                    "price":         init_price,
                                    "fix_c":         init_fixc,
                                    "baseline":      0.0,
                                    "cumulative_ev": 0.0,
                                },
                                "rounds": []
                            })
                            save_data(data)
                            st.session_state["active_idx"] = len(data["tickers"]) - 1
                            st.rerun()
                        else:
                            st.error("Already exists")

        with tab_b:
            if tickers:
                del_ticker = st.selectbox("Select to delete", [t["ticker"] for t in tickers])
                if st.button("🗑 Delete", type="primary"):
                    data["tickers"] = [t for t in tickers if t["ticker"] != del_ticker]
                    st.session_state["active_idx"] = 0
                    st.session_state["pending"] = None
                    save_data(data)
                    st.rerun()

        with tab_c:
            export_str = json.dumps(data, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇ Export JSON", export_str,
                file_name=f"chain_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            uploaded = st.file_uploader("⬆ Import JSON", type=["json"])
            if uploaded:
                try:
                    imported = json.load(uploaded)
                    if "tickers" in imported:
                        save_data(imported)
                        st.session_state["data"] = imported
                        st.rerun()
                except Exception as e:
                    st.error(str(e))


def render_preview_and_history(data: dict, t_idx: int):
    """RIGHT: Preview card + commit + history table."""
    tickers = data.get("tickers", [])
    if not tickers:
        return

    t       = tickers[t_idx]
    pending = st.session_state.get("pending")
    is_mine = (st.session_state.get("pending_ticker") == t_idx)

    # ── PREVIEW CARD ───────────────────────────────────────────────────
    if pending and is_mine:
        st.markdown('<div class="sec-header">Preview — not committed</div>', unsafe_allow_html=True)

        r = pending
        sc = _color_class(r["surplus"])
        nc = _color_class(r["net_pnl_after"])

        # Arrow: price movement
        arrow = "↑" if r["p_new"] > r["p_old"] else ("↓" if r["p_new"] < r["p_old"] else "→")
        price_color = "#3fb950" if r["p_new"] > r["p_old"] else ("#f85149" if r["p_new"] < r["p_old"] else "#8b949e")

        preview_html = f"""
        <div class="preview-card">
            {_preview_row("Price",
                f"<span style='color:{price_color}'>{arrow} ${_fmt(r['p_old'])} → ${_fmt(r['p_new'])}</span>", "neu")}
            {_preview_row("Shannon",   f"${_fmt(r['shannon'])}",   _color_class(r['shannon']))}
            {_preview_row("Hedge",     f"−${_fmt(r['hedge'])}",    "cost")}
            {_preview_row("Surplus",   f"${_fmt(r['surplus'])}",   sc)}
            {_preview_row("Δ fix_c",   f"+${_fmt(r['delta_fix_c'])}", "pos" if r['delta_fix_c'] > 0 else "neu")}
            {_preview_row("fix_c →",   f"{_fmt(r['fix_c_after'], 4)}", "neu")}
            {_preview_row("Baseline →",f"${_fmt(r['base_after'])}", "neu")}
            {_preview_row("Net P&L",   f"${_fmt(r['net_pnl_after'])}", nc)}
            {_preview_row("Ev Efficiency",
                "∞" if r["ev_efficiency"] == float("inf") else f"{r['ev_efficiency']:.3f}×",
                "pos" if r["ev_efficiency"] >= 1 else "neg")}
        </div>"""
        st.markdown(preview_html, unsafe_allow_html=True)

        # Commit / Cancel
        bc, cc = st.columns([3, 1])
        with bc:
            if st.button("✅  COMMIT — Save Round", type="primary", use_container_width=True):
                commit_round(data, t_idx, pending)
                st.session_state["pending"]       = None
                st.session_state["last_commit"]   = pending
                st.session_state["pending_ticker"] = None
                st.rerun()
        with cc:
            if st.button("✕ Cancel", use_container_width=True):
                st.session_state["pending"] = None
                st.rerun()

    elif st.session_state.get("last_commit"):
        r = st.session_state["last_commit"]
        st.markdown(f"""
        <div class="commit-ok">
            ✅ Committed — Surplus ${_fmt(r['surplus'])}
            &nbsp;·&nbsp; Δ fix_c +${_fmt(r['delta_fix_c'])}
            &nbsp;·&nbsp; Net ${_fmt(r['net_pnl_after'])}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#0d1117;border:1px dashed #1c2a3a;border-radius:8px;
                    padding:20px;text-align:center;font-family:'IBM Plex Mono',monospace;
                    font-size:12px;color:#3d5570;">
            ▶ Press Preview to calculate
        </div>""", unsafe_allow_html=True)

    # ── PAYOFF MINI-CHART ──────────────────────────────────────────────
    rounds = t.get("rounds", [])
    if len(rounds) >= 2:
        st.markdown('<div class="sec-header" style="margin-top:14px">Equity Curve</div>', unsafe_allow_html=True)
        xs  = [r["round"] for r in rounds]
        ys  = [r["net_pnl"] for r in rounds]
        fcs = [r["fix_c"] for r in rounds]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name="Net P&L",
            line=dict(color="#3fb950", width=2),
            fill="tozeroy", fillcolor="rgba(63,185,80,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=fcs, name="fix_c",
            line=dict(color="#1f6feb", width=1.5, dash="dot"),
            yaxis="y2"
        ))
        fig.add_hline(y=0, line_color="#1c2a3a", line_dash="dot")
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            showlegend=False,
            xaxis=dict(showgrid=False, color="#3d5570", tickfont=dict(family="IBM Plex Mono", size=9)),
            yaxis=dict(showgrid=False, color="#3d5570", tickfont=dict(family="IBM Plex Mono", size=9)),
            yaxis2=dict(overlaying="y", side="right", showgrid=False,
                        color="#1f6feb", tickfont=dict(family="IBM Plex Mono", size=9)),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── HISTORY TABLE ──────────────────────────────────────────────────
    st.markdown('<div class="sec-header" style="margin-top:6px">Round History</div>', unsafe_allow_html=True)

    if rounds:
        rows = []
        for r in reversed(rounds):
            rows.append({
                "#":         r["round"],
                "Date":      r.get("date", "")[:10],
                "Price →":   f"${r['price']:,.2f} → ${r['p_new']:,.2f}",
                "Shannon":   f"${r['shannon']:,.4f}",
                "Hedge":     f"${r['hedge']:,.4f}",
                "Surplus":   f"${r['surplus']:,.4f}",
                "Δ fix_c":   f"+${r['delta_fix_c']:,.4f}",
                "fix_c":     f"{r['fix_c']:,.4f}",
                "Net P&L":   f"${r['net_pnl']:,.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=220)
    else:
        st.markdown("""
        <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                    color:#3d5570;padding:16px 0;">
            No rounds yet — Preview then Commit.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    boot()
    data = st.session_state["data"]

    # ── TOP BAR ───────────────────────────────────────────────────────
    render_topbar(data)
    st.markdown("---")

    # ── 3-COLUMN LAYOUT ──────────────────────────────────────────────
    col_l, col_c, col_r = st.columns([12, 44, 44], gap="medium")

    with col_l:
        t_idx = render_watchlist(data)

    with col_c:
        render_input_panel(data, t_idx)

    with col_r:
        render_preview_and_history(data, t_idx)


if __name__ == "__main__":
    main()
