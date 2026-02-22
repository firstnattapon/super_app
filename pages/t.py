import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(page_title="Chain System - Main Engine", layout="wide")

from flywheels import (
    load_trading_data, save_trading_data, get_tickers,
    run_chain_round, commit_round,
)

# ============================================================
# UTILITIES & DEFENSIVE PROGRAMMING (STABILITY UPGRADES)
# ============================================================
def safe_float(val, default=0.0) -> float:
    """Safe float conversion to prevent ValueError/TypeError."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (ValueError, TypeError):
        return float(default)

def init_session_state():
    """Initialize critical session state variables to prevent KeyErrors."""
    keys_defaults = {
        "run_round_ticker": "",
        "ticker_watchlist_radio": None,
        "_pending_round": None,
        "_pending_ticker_idx": None,
        "_pending_ticker_name": "",
        "treasury_filter_sel": "🌐 ทั้งหมด",
        "_treasury_last_ticker": "",
    }
    for k, v in keys_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# Graceful import 
try:
    from flywheels import build_portfolio_df
except ImportError:
    import pandas as _pd
    def build_portfolio_df(data):
        rows = []
        for item in (data if isinstance(data, list) else []):
            state = item.get("current_state", {})
            rows.append({
                "Ticker":       item.get("ticker", "???"),
                "Price (t)":    safe_float(state.get("price", 0.0)),
                "Fix_C":        safe_float(state.get("fix_c", 0.0)),
                "Baseline (b)": safe_float(state.get("baseline", 0.0)),
                "Ev (Extrinsic)": safe_float(state.get("cumulative_ev", 0.0)),
                "Lock P&L":     safe_float(state.get("lock_pnl", 0.0)),
                "Surplus IV":   safe_float(state.get("surplus_iv", 0.0)),
                "Net":          safe_float(state.get("net_pnl", 0.0)),
            })
        return _pd.DataFrame(rows)

# ============================================================
# HELPER: Treasury Logging
# ============================================================
def log_treasury_event(data: dict, category: str, amount: float, note: str = "") -> None:
    if "treasury_history" not in data:
        data["treasury_history"] = []
    
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "category": category,
        "amount": amount,
        "pool_cf_balance": safe_float(data.get("global_pool_cf", 0.0)),
        "ev_reserve_balance": safe_float(data.get("global_ev_reserve", 0.0)),
        "note": note
    }
    data["treasury_history"].append(entry)

# ============================================================
# MAIN APPLICATION ENGINE
# ============================================================
def main():
    init_session_state()
    
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

    data = load_trading_data()

    tab2, tab4, tab5 = st.tabs([
        "⚡ Engine & History",
        "Payoff Profile 🔗 Run Chain Round",
        "➕ Manage Data"
    ])

    with tab2:
        _render_engine_tab(data)
    with tab4:
        _render_payoff_profile_tab(data)
    with tab5:
        _render_manage_data(data)

# ----------------------------------------------------------
# UI COMPONENTS (TAB 2: ENGINE & HISTORY)
# ----------------------------------------------------------
def _chip_label(t_data: dict) -> str:
    ticker = t_data.get("ticker", "???")
    state  = t_data.get("current_state", {})
    net    = safe_float(state.get("net_pnl", 0))
    price  = safe_float(state.get("price", 0))
    dot    = "🟢" if net >= 0 else "🔴"
    return f"{dot} {ticker}  ${price:.2f}"

def _fmt(v: float) -> str:
    return f"{v:,.2f}"

def _calc_withdraw_b(t_data: dict) -> float:
    return sum(
        safe_float(r.get("b_before", 0)) - safe_float(r.get("b_after", 0))
        for r in t_data.get("rounds", [])
        if r.get("action") == "Extract Baseline"
        and safe_float(r.get("b_before", 0)) > safe_float(r.get("b_after", 0))
    )

def _m(label: str, value: str, col, neg_red: bool = False, is_cost: bool = False):
    try:
        num = float(value.replace(",", ""))
        if is_cost: color = "#ef4444"
        elif neg_red: color = "#22c55e" if num >= 0 else "#ef4444"
        else: color = "#e2e8f0"
    except ValueError:
        color = "#e2e8f0"

    col.markdown(
        f"<div style='line-height:1.2'>"
        f"<div style='font-size:12px;color:#64748b;white-space:nowrap'>{label}</div>"
        f"<div style='font-size:18px;font-weight:700;color:{color};white-space:nowrap'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

def _render_engine_metrics(data: dict, tickers_list: list, active_ticker: str, active_t_data: dict):
    pool_cf       = safe_float(data.get("global_pool_cf", 0.0))
    ev_reserve    = safe_float(data.get("global_ev_reserve", 0.0))
    total_rounds  = sum(len(t.get("rounds", [])) for t in tickers_list)
    total_fix_c   = sum(safe_float(t.get("current_state", {}).get("fix_c", 0)) for t in tickers_list)
    total_net_pnl = pool_cf + ev_reserve

    state      = active_t_data.get("current_state", {}) if active_t_data else {}
    sel_net    = safe_float(state.get("net_pnl", 0))
    sel_rounds = len(active_t_data.get("rounds", [])) if active_t_data else 0
    withdraw_b = _calc_withdraw_b(active_t_data) if active_t_data else 0.0

    with st.container(border=True):
        hL, hDiv, hR = st.columns([4, 0.05, 6])
        hL.caption("🌐 Global")
        if active_t_data:
            hR.caption(f"📌 {active_ticker}  ·  {len(tickers_list)} Tickers · {total_rounds} Rounds")

        (g1, g2, g3, g4, div, t1, t2, t3, t4, t5, t6, t7) = st.columns(
            [1.8, 1.8, 1.6, 1.6, 0.05, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 0.8], gap="small"
        )

        _m("🎱 Pool CF",    _fmt(pool_cf),       g1)
        _m("🛡️ EV Reserve", _fmt(ev_reserve),    g2, neg_red=True)
        _m("⚡ Fix_C",       _fmt(total_fix_c),   g3)
        _m("💰 Net",         _fmt(total_net_pnl), g4, neg_red=True)

        div.markdown("<div style='border-left:1px solid #334155;height:48px;margin-top:4px'></div>", unsafe_allow_html=True)

        if active_t_data:
            _m("Price",      _fmt(safe_float(state.get("price", 0))),      t1)
            _m("fix_c",      _fmt(safe_float(state.get("fix_c", 0))),      t2)
            _m("Baseline",   _fmt(safe_float(state.get("baseline", 0))),   t3)
            _m("Withdraw_b", _fmt(withdraw_b),                             t4)
            _m("🔥 Ev Burn", _fmt(safe_float(state.get("cumulative_ev", 0))), t5, is_cost=True)
            _m("Net P&L",    _fmt(sel_net),                                t6, neg_red=True)
            _m("Rounds",     str(sel_rounds),                              t7)
        else:
            t1.caption("📌 ยังไม่มี Ticker")

def _render_engine_tab(data: dict):
    tickers_list = get_tickers(data)
    ticker_names = [t.get("ticker", "???") for t in tickers_list]

    if ticker_names:
        labels = [_chip_label(t) for t in tickers_list]
        radio_val = st.session_state.get("ticker_watchlist_radio")
        if radio_val in labels:
            active_idx = labels.index(radio_val)
        else:
            saved = st.session_state.get("run_round_ticker", ticker_names[0])
            active_idx = ticker_names.index(saved) if saved in ticker_names else 0
        active_ticker = ticker_names[active_idx]
        active_t_data = tickers_list[active_idx]
    else:
        active_idx = 0
        active_ticker = ""
        active_t_data = {}

    st.session_state["run_round_ticker"] = active_ticker
    _render_engine_metrics(data, tickers_list, active_ticker, active_t_data)

    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Manage Data ก่อน")
        return

    z_left, z_center, z_right = st.columns([12, 44, 44], gap="medium")

    with z_left:
        st.markdown("##### 📋 Watchlist")
        st.radio("Ticker", [_chip_label(t) for t in tickers_list], index=active_idx,
                 key="ticker_watchlist_radio", label_visibility="collapsed")

    with z_center:
        _render_center_panels(data, tickers_list, active_ticker, active_t_data)

    with z_right:
        _render_chain_engine_center(data, tickers_list, active_ticker, active_t_data, active_idx)

def _render_chain_engine_center(data: dict, tickers_list: list, selected_ticker: str, t_data: dict, idx: int):
    state      = t_data.get("current_state", {}) if t_data else {}
    settings   = data.get("settings", {})
    default_hr = safe_float(settings.get("default_hedge_ratio", 2.0))
    
    # Safe calc for default_p
    safe_price = safe_float(state.get("price", 10.0))
    default_p  = safe_float(max(0.01, round(safe_price * 1.1, 2)))

    hc1, hc2 = st.columns([3, 1])
    hc1.markdown(f"#### ⚡ Chain Round — **{selected_ticker}**")
    with hc2:
        if (st.session_state.get("_pending_round") and
                st.session_state.get("_pending_ticker_name") == selected_ticker):
            st.success("🔗 Synced → Payoff")
        else:
            st.caption("💡 Preview → syncs Payoff tab")

    with st.container(border=True):
        st.caption("⚡ Order Strip — ป้อนค่าแล้วกด Preview")
        sc1, sc2, sc3, sc4, sc5 = st.columns([2.5, 1.8, 1.2, 1.4, 1.8])
        with sc1:
            p_new = st.number_input("P New", min_value=0.01, value=default_p, step=1.0, key=f"strip_pnew_{idx}")
        with sc2:
            hedge_ratio = st.number_input("Hedge ×", min_value=0.0, value=default_hr, step=0.5, key=f"strip_hr_{idx}")
        with sc3:
            ignore_hedge = st.checkbox("No Hedge", value=False, key=f"strip_ih_{idx}")
        with sc4:
            ignore_surplus = st.checkbox("No Surplus", value=False, key=f"strip_is_{idx}")
        with sc5:
            st.write("")
            preview_clicked = st.button("🔍 Preview", type="primary", key=f"strip_preview_{idx}", use_container_width=True)

    if preview_clicked and p_new > 0:
        preview = run_chain_round(
            ticker_state=state, p_new=p_new,
            hedge_ratio=hedge_ratio,
            ignore_hedge=ignore_hedge,
            ignore_surplus=ignore_surplus,
        )
        if preview:
            st.session_state["_pending_round"]       = preview
            st.session_state["_pending_ticker_idx"]  = idx
            st.session_state["_pending_ticker_name"] = selected_ticker

    rd         = st.session_state.get("_pending_round")
    is_pending = (rd is not None and st.session_state.get("_pending_ticker_name") == selected_ticker)

    if is_pending:
        with st.container(border=True):
            st.markdown("**📊 Preview Result** — แก้ไขได้ก่อน Commit")

            r1c1, r1c2, r1c3 = st.columns(3)
            new_shannon = r1c1.number_input("💰 Shannon Profit", value=safe_float(rd.get("shannon_profit", 0.0)), step=10.0, format="%.2f", key="edit_shannon")
            new_hedge   = r1c2.number_input("🛡️ Hedge Cost", value=safe_float(rd.get("hedge_cost", 0.0)), step=10.0, format="%.2f", key="edit_hedge")
            new_surplus = r1c3.number_input("✨ Surplus", value=safe_float(rd.get("surplus", 0.0)), step=10.0, format="%.2f", key="edit_surplus")

            scale_val = safe_float(rd.get("scale_up", max(0.0, safe_float(rd.get("surplus", 0.0)))))
            sc_color  = "#22c55e" if scale_val > 0 else ("#ef4444" if scale_val < 0 else "#94a3b8")
            st.markdown(
                f"<div style='text-align:center;padding:4px 0 10px;font-size:13px;color:#94a3b8'>"
                f"🚀 Scale Up &nbsp;<span style='color:{sc_color};font-weight:700;font-size:20px'>"
                f"+${scale_val:,.2f}</span></div>",
                unsafe_allow_html=True,
            )

            st.divider()

            r2c1, r2c2, r2c3 = st.columns(3)
            new_c_after = r2c1.number_input("fix_c (after)", value=safe_float(rd.get("c_after", 0.0)), step=100.0, format="%.0f", key="edit_c_after", help=f"Before: ${safe_float(rd.get('c_before', 0)):,.0f}")
            new_p_new   = r2c2.number_input("Price (new t)", value=safe_float(rd.get("p_new", 0.0)), step=0.1, format="%.2f", key="edit_p_new", help=f"Before: ${safe_float(rd.get('p_old', 0)):,.2f}")
            new_b_after = r2c3.number_input("Baseline (after)", value=safe_float(rd.get("b_after", 0.0)), step=0.1, format="%.2f", key="edit_b_after", help=f"Before: ${safe_float(rd.get('b_before', 0)):,.2f}")

            d1, d2, d3 = st.columns(3)
            d1.markdown(_delta_badge(safe_float(rd.get("c_before", 0)), new_c_after, ",.0f"), unsafe_allow_html=True)
            d2.markdown(_delta_badge(safe_float(rd.get("p_old", 0)),    new_p_new,   ",.2f"), unsafe_allow_html=True)
            d3.markdown(_delta_badge(safe_float(rd.get("b_before", 0)), new_b_after, ",.2f"), unsafe_allow_html=True)

            # Defensive Math variables
            _b_old   = safe_float(rd.get("b_before", 0.0))
            _c_old   = safe_float(rd.get("c_before", 0.0))
            _p_old   = max(safe_float(rd.get("p_old", 0.0)), 1e-9)
            _c_new_w = safe_float(new_c_after)
            _p_new_w = max(safe_float(new_p_new), 1e-9)

            if _p_old > 0 and _p_new_w > 0 and _c_old > 0:
                _sh_term  = _c_old * math.log(_p_new_w / _p_old) if _p_new_w != _p_old else 0.0
                _b_calc   = _b_old + _sh_term
                _note_sfx = "" if _p_new_w != _p_old else "  · ราคาไม่เปลี่ยน — Shannon = $0"
                _fline = (
                    f"{_b_old:+.2f} += "
                    f"({_c_old:,.0f} × ln({_p_new_w:.2f}/{_p_old:.2f})) − "
                    f"({_c_new_w:,.0f} × ln({_p_new_w:.2f}/{_p_new_w:.2f}))"
                    f"  |  c = {_c_new_w:,.0f} , t = {_p_new_w:.2f} , b = {_b_calc:.2f}"
                )
                st.markdown(
                    f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;"
                    f"padding:10px 14px;margin:10px 0 4px;font-family:monospace;"
                    f"font-size:13px;color:#94a3b8'>"
                    f"<span style='color:#64748b;font-size:11px'>📐 สมการ Baseline</span><br/>"
                    f"<span style='color:#fbbf24;font-weight:600'>{_fline}</span>"
                    f"{_note_sfx}</div>", unsafe_allow_html=True)

                _shannon_val = _c_old * math.log(_p_new_w / _p_old) if _p_new_w != _p_old else 0.0
                _shannon_line = f"Shannon = {_c_old:,.0f} × ln({_p_new_w:.2f}/{_p_old:.2f})  =  ${_shannon_val:+,.2f}"
                st.markdown(
                    f"<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;"
                    f"padding:10px 14px;margin:4px 0 4px;font-family:monospace;"
                    f"font-size:13px;color:#94a3b8'>"
                    f"<span style='color:#64748b;font-size:11px'>💰 Shannon Baseline  (fix_c × ln(Pt / P0))</span><br/>"
                    f"<span style='color:#34d399;font-weight:600'>{_shannon_line}</span></div>", unsafe_allow_html=True)

        btn_col, cnl_col = st.columns([4, 1])
        with btn_col:
            if st.button("✅ COMMIT — บันทึกถาวร", type="primary", use_container_width=True):
                rd.update({
                    "shannon_profit": new_shannon, "hedge_cost": new_hedge,
                    "surplus": new_surplus, "c_after": new_c_after,
                    "p_new": new_p_new, "b_after": new_b_after,
                    "scale_up": new_c_after - safe_float(rd.get("c_before", 0)),
                })
                commit_round(data, st.session_state["_pending_ticker_idx"], rd)
                st.session_state["_pending_round"] = None
                st.success(f"✅ Committed! {selected_ticker} fix_c → ${new_c_after:,.0f}")
                st.rerun()
        with cnl_col:
            if st.button("✖ Cancel", use_container_width=True, key="cancel_preview"):
                st.session_state["_pending_round"] = None
                st.rerun()
    else:
        with st.container(border=True):
            st.caption(
                f"📌 {selected_ticker}  |  "
                f"fix_c ${safe_float(state.get('fix_c', 0)):,.0f}  |  "
                f"Price ${safe_float(state.get('price', 0)):,.2f}  |  "
                f"Baseline ${safe_float(state.get('baseline', 0)):,.2f}  |  "
                f"Rounds {len(t_data.get('rounds', []) if t_data else [])}"
            )
            st.info("ป้อน P New แล้วกด 🔍 Preview เพื่อเริ่มคำนวณ")

def _delta_badge(before: float, after: float, fmt: str = ",.0f") -> str:
    diff  = after - before
    color = "#22c55e" if diff > 0 else ("#ef4444" if diff < 0 else "#94a3b8")
    sign  = "+" if diff > 0 else ""
    arrow = "<span style='color:#475569;margin:0 5px'>→</span>"
    return (f"<div style='font-size:12px;padding:2px 0'>"
            f"<span style='color:#64748b'>${before:{fmt}}</span>{arrow}"
            f"<span style='color:{color};font-weight:700'>${after:{fmt}}</span>"
            f"&nbsp;<span style='color:{color}'>({sign}{diff:{fmt}})</span></div>")

def _render_center_panels(data: dict, tickers_list: list, active_ticker: str, active_t_data: dict):
    tab_hist, tab_treasury, tab_pool, tab_deploy = st.tabs(["📜 History", "🏛️ Treasury", "🎱 Pool CF", "🚀 Deploy"])
    with tab_hist:
        if active_t_data: _render_consolidated_history(active_t_data)
        else: st.info("เลือก Ticker ที่ Watchlist ก่อน")
    with tab_treasury: _render_treasury_log(data, filter_ticker=active_ticker)
    with tab_pool: _render_pool_cf_section(data)
    with tab_deploy:
        _render_deployment_section(data, tickers_list)
        _render_ev_leaps_section(data)

def _render_pool_cf_section(data: dict):
    with st.expander('🎱 Pool CF & Allocation', expanded=False):
        ticker_names = [t.get("ticker", "???") for t in get_tickers(data)]
        note_options = ["None"] + ticker_names
        
        with st.form("add_pool_cf_form", clear_on_submit=True):
            c1, c2 = st.columns([2, 1])
            with c1: 
                amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=100.0)
                selected_ticker = st.selectbox("Note (Select Ticker)", options=note_options)
            with c2: 
                for _ in range(4): st.write("")
                btn_add = st.form_submit_button("💰 Add Fund", type="primary")
                
            if btn_add and amount > 0:
                data["global_pool_cf"] = safe_float(data.get("global_pool_cf", 0)) + amount
                note_str = "Added to Pool CF" + (f" [Ticker: {selected_ticker}]" if selected_ticker and selected_ticker != "None" else "")
                log_treasury_event(data, "Funding", amount, note_str)
                save_trading_data(data)
                st.success(f"✅ +${amount:,.2f} → Pool CF = ${data['global_pool_cf']:,.2f}")
                st.rerun()

        st.divider()
        st.markdown("##### 🌾 Record Harvest Profit")
        with st.form("record_harvest_form", clear_on_submit=True):
            hc1, hc2 = st.columns([2, 1])
            with hc1:
                h_amount = st.number_input("Harvest Amount ($)", min_value=0.0, value=0.0, step=100.0)
                h_ticker = st.selectbox("Note (Select Ticker)", options=note_options, key="harvest_note")
            with hc2:
                for _ in range(4): st.write("")
                btn_harvest = st.form_submit_button("🌾 Add Harvest", type="primary")
                
            if btn_harvest and h_amount > 0:
                data["global_pool_cf"] = safe_float(data.get("global_pool_cf", 0)) + h_amount
                note_str = "Harvest Profit" + (f" [Ticker: {h_ticker}]" if h_ticker and h_ticker != "None" else "")
                log_treasury_event(data, "Harvest", h_amount, note_str)
                save_trading_data(data)
                st.success(f"✅ +${h_amount:,.2f} Harvest → Pool CF = ${data['global_pool_cf']:,.2f}")
                st.rerun()

        st.divider()
        st.markdown("##### 📥 Extract Baseline to Pool CF")
        eligible_tickers = [t for t in get_tickers(data) if safe_float(t.get("current_state", {}).get("baseline", 0)) > 0]
        
        if eligible_tickers:
            with st.form("extract_baseline_form", clear_on_submit=True):
                hc1, hc2 = st.columns([2, 1])
                with hc1:
                    ext_ticker_name = st.selectbox("Select Ticker", options=[t.get("ticker") for t in eligible_tickers], key="extract_ticker")
                    selected_t_obj = next((t for t in eligible_tickers if t.get("ticker") == ext_ticker_name), None)
                    max_baseline = safe_float(selected_t_obj.get("current_state", {}).get("baseline", 0)) if selected_t_obj else 0.0
                    ext_amount = st.number_input("Extract Amount ($)", min_value=0.0, max_value=max_baseline, value=max_baseline, step=100.0)
                with hc2:
                    for _ in range(4): st.write("")
                    btn_extract = st.form_submit_button("📥 Extract to Pool CF", type="primary")
                    
                if btn_extract and ext_amount > 0 and selected_t_obj:
                    current_baseline = safe_float(selected_t_obj["current_state"].get("baseline", 0))
                    selected_t_obj["current_state"]["baseline"] = current_baseline - ext_amount
                    data["global_pool_cf"] = safe_float(data.get("global_pool_cf", 0.0)) + ext_amount
                    
                    log_treasury_event(data, "Baseline Harvest", ext_amount, f"[Ticker: {ext_ticker_name}]")
                    cur_state = selected_t_obj["current_state"]
                    dummy_round = {
                        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "action": "Extract Baseline", 
                        "p_old": safe_float(cur_state.get("price", 0)), "p_new": safe_float(cur_state.get("price", 0)), 
                        "c_before": safe_float(cur_state.get("fix_c", 0)), "c_after": safe_float(cur_state.get("fix_c", 0)),
                        "shannon_profit": 0.0, "harvest_profit": 0.0, "hedge_cost": 0.0,
                        "surplus": 0.0, "scale_up": 0.0, 
                        "b_before": current_baseline, "b_after": current_baseline - ext_amount, 
                        "note": f"Extracted ${ext_amount:,.2f} to Pool CF",
                        "hedge_ratio": 0.0, "sigma": 0.0, "ev_change": 0.0
                    }
                    if "rounds" not in selected_t_obj: selected_t_obj["rounds"] = []
                    selected_t_obj["rounds"].append(dummy_round)
                    save_trading_data(data)
                    st.success(f"✅ Extracted ${ext_amount:,.2f} from {ext_ticker_name} Baseline to Pool CF")
                    st.rerun()
        else:
            st.info("No tickers with a positive Baseline available for extraction.")

def _render_deployment_section(data: dict, tickers_list: list):
    if not tickers_list: return
    pool_cf = safe_float(data.get("global_pool_cf", 0.0))
    deploy_ticker_options = [d.get("ticker", "???") for d in tickers_list]

    with st.expander("🚀 Deploy to Baseline", expanded=True):
        deploy_ticker = st.selectbox("Ticker", deploy_ticker_options, key="deploy_ticker")
        d_idx         = deploy_ticker_options.index(deploy_ticker)
        t_data_deploy = tickers_list[d_idx]
        cur_state     = t_data_deploy.get("current_state", {})
        cur_b         = safe_float(cur_state.get("baseline", 0.0))
        cur_t         = safe_float(cur_state.get("price",    0.0))
        cur_c         = safe_float(cur_state.get("fix_c",    0.0))
        rounds        = t_data_deploy.get("rounds", [])
        last_round    = rounds[-1] if rounds else {}
        cur_sigma     = safe_float(last_round.get("sigma", 0.5))
        cur_hr        = safe_float(last_round.get("hedge_ratio", 2.0))

        m1, m2, m3 = st.columns(3)
        m1.metric("🎱 Pool CF", f"${pool_cf:,.2f}")
        m2.metric("📐 Baseline", f"${cur_b:,.2f}", f"{deploy_ticker}")
        m3.metric("fix_c", f"${cur_c:,.0f}")
        st.divider()

        with st.form("deploy_to_baseline_form", clear_on_submit=True):
            f1, f2 = st.columns([2, 1])
            with f1:
                d_amt  = st.number_input("Amount ($)", min_value=0.0, max_value=pool_cf if pool_cf > 0 else 0.0, value=0.0, step=100.0)
                d_note = st.text_input("Note (optional)", value="")
            with f2:
                for _ in range(3): st.write("")
                submitted = st.form_submit_button("🚀 Deploy to Baseline", type="primary", use_container_width=True)

        if submitted and d_amt > 0:
            if d_amt > pool_cf: st.error(f"❌ Pool CF ไม่พอ (มี ${pool_cf:,.2f})")
            else:
                new_b = cur_b + d_amt
                new_pool = pool_cf - d_amt

                st.markdown("**📊 Preview Deploy**")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Pool CF",  f"${pool_cf:,.2f}",  f"−${d_amt:,.2f}", delta_color="inverse")
                pc2.metric("Baseline", f"${cur_b:,.2f}",    f"+${d_amt:,.2f}", delta_color="normal")
                pc3.metric("Baseline after", f"${new_b:,.2f}")

                deploy_round = {
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "action": "Deploy to Baseline",
                    "p_old": cur_t, "p_new": cur_t, "c_before": cur_c, "c_after": cur_c,
                    "shannon_profit": 0.0, "harvest_profit": 0.0, "hedge_cost": 0.0, 
                    "surplus": 0.0, "scale_up": 0.0, "b_before": cur_b, "b_after": new_b,
                    "note": f"Deploy from Pool CF → Baseline{(' | ' + d_note) if d_note else ''}",
                    "hedge_ratio": cur_hr, "sigma": cur_sigma, "ev_change": 0.0,
                }

                data["global_pool_cf"] = new_pool
                t_data_deploy["current_state"]["baseline"] = new_b
                if "rounds" not in t_data_deploy: t_data_deploy["rounds"] = []
                t_data_deploy["rounds"].append(deploy_round)
                log_treasury_event(data, "Deploy", -d_amt, f"Deploy to {deploy_ticker} Baseline{(' | ' + d_note) if d_note else ''}")
                save_trading_data(data)
                st.success(f"✅ Deployed ${d_amt:,.2f} → {deploy_ticker} Baseline = ${new_b:,.2f}")
                st.rerun()

def _render_ev_leaps_section(data: dict):
    pool_cf = safe_float(data.get("global_pool_cf", 0.0))
    ev_reserve = safe_float(data.get("global_ev_reserve", 0.0))
    
    with st.expander("🛡️ Manage Pool EV LEAPS (Income & Expenses)"):
        st.markdown("##### 📥 Allocate (Income from Pool CF)")
        col_a, col_b = st.columns(2)
        with col_a: 
            alloc_amt = st.number_input("Allocate Amount ($)", min_value=0.0, max_value=pool_cf, step=100.0, key="alloc")
        with col_b:
            if st.button("📥 Allocate"):
                if alloc_amt > 0 and pool_cf >= alloc_amt:
                    data["global_pool_cf"] = pool_cf - alloc_amt
                    data["global_ev_reserve"] = ev_reserve + alloc_amt
                    log_treasury_event(data, "Allocation", alloc_amt, "Pool CF -> EV Reserve")
                    save_trading_data(data)
                    st.success(f"Allocated ${alloc_amt:,.2f}")
                    st.rerun()
        
        st.divider()
        st.markdown("##### 📤 Pay LEAPS (Expense/Adjustment)")
        ticker_names = [t.get("ticker", "???") for t in get_tickers(data)]
        note_options = ["None"] + ticker_names
        
        col_c, col_d = st.columns(2)
        with col_c: 
            pay_leaps_amt = st.number_input("LEAPS Net Flow ($)", value=0.0, step=100.0, help="Negative (-) = Expense/Cost.")
            selected_leaps_ticker = st.selectbox("Note (Select Ticker)", options=note_options, key="leaps_note_ticker")
        with col_d:
            st.write(""); st.write("")
            if st.button("💾 Record Flow"):
                data["global_ev_reserve"] = ev_reserve + pay_leaps_amt
                note_str = "Manual Adjustment" + (f" [Ticker: {selected_leaps_ticker}]" if selected_leaps_ticker and selected_leaps_ticker != "None" else "")
                log_treasury_event(data, "Income" if pay_leaps_amt >= 0 else "Expense", pay_leaps_amt, note_str)
                save_trading_data(data)
                st.success("Recorded Extrinsic Value adjustment")
                st.rerun()
        st.markdown(f"**Current Pool EV LEAPS Balance:** `${ev_reserve:,.2f}`")

def _render_treasury_log(data: dict, filter_ticker: str = ""):
    st.subheader("🏛️ Treasury & Ops History")
    history = data.get("treasury_history", [])
    if not history: st.info("ยังไม่มี Treasury events"); return

    all_tickers = sorted({e.get("note", "").split("[Ticker: ")[-1].rstrip("]") for e in history if "[Ticker:" in e.get("note", "")})
    options = ["🌐 ทั้งหมด"] + all_tickers
    target = filter_ticker if filter_ticker in options else "🌐 ทั้งหมด"
    if st.session_state.get("_treasury_last_ticker") != filter_ticker:
        st.session_state["treasury_filter_sel"] = target
        st.session_state["_treasury_last_ticker"] = filter_ticker

    sel = st.selectbox("กรอง Ticker", options, key="treasury_filter_sel", label_visibility="collapsed")
    filtered = history if sel == "🌐 ทั้งหมด" else [e for e in history if sel in e.get("note", "")]
    tbl = [{
        "Date":   e.get("date", "")[:10], "Action": e.get("category", ""), "Amount": f"${safe_float(e.get('amount', 0)):,.2f}",
        "Pool CF":f"${safe_float(e.get('pool_cf_balance', 0)):,.0f}", "EV Res": f"${safe_float(e.get('ev_reserve_balance', 0)):,.0f}",
        "Note":   e.get("note", ""),
    } for e in filtered]

    st.caption(f"แสดง {len(tbl)} รายการ" + (f"  (filter: {sel})" if sel != "🌐 ทั้งหมด" else ""))
    df = pd.DataFrame(tbl)[::-1]
    if sel != "🌐 ทั้งหมด": df = df.drop(columns=["Pool CF", "EV Res"], errors="ignore")
    st.dataframe(df, use_container_width=True, hide_index=True)

def _render_consolidated_history(t_data: dict):
    st.subheader(f"📜 {t_data.get('ticker','???')} — History")
    rounds = t_data.get("rounds", [])
    if rounds:
        tbl = [{
            "Date": rd.get("date", "")[:10], 
            "Action": f"Scale +${safe_float(rd.get('scale_up', 0)):,.0f}" if safe_float(rd.get("scale_up", 0)) > 0 else ("Inject/Deploy" if "Injection" in rd.get("action", "") else rd.get("action", "Round")),
            "Price": f"${safe_float(rd.get('p_old',0)):,.2f} > ${safe_float(rd.get('p_new',0)):,.2f}", 
            "fix_c": f"${safe_float(rd.get('c_before',0)):,.0f} > ${safe_float(rd.get('c_after',0)):,.0f}",
            "b": f"${safe_float(rd.get('b_before',0)):,.2f} > ${safe_float(rd.get('b_after',0)):,.2f}", 
            "Net Result": f"${safe_float(rd.get('surplus',0)):,.2f}"
        } for rd in rounds]
        st.dataframe(pd.DataFrame(tbl)[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No history recorded yet.")

# ----------------------------------------------------------
# TAB 4: Payoff Profile Simulator
# ----------------------------------------------------------
def _render_payoff_profile_tab(data: dict):
    tickers_list = get_tickers(data)
    selected_ticker = st.session_state.get("run_round_ticker")
    t_data = next((t for t in tickers_list if t["ticker"] == selected_ticker), tickers_list[0] if tickers_list else None)
    
    if not t_data: st.info("👈 Please select a Ticker in the 'Engine & History' tab first."); return

    st.subheader(f"📐 Advanced Payoff Profile Simulator")
    cur_state = t_data.get("current_state", {})
    def_c = safe_float(cur_state.get("fix_c", 10000.0))
    def_p = safe_float(cur_state.get("price", 100.0))

    pending = st.session_state.get("_pending_round")
    if pending and st.session_state.get("_pending_ticker_name") == t_data.get("ticker"):
        def_c = safe_float(pending.get('c_after', def_c))
        def_p = safe_float(pending.get('p_new', def_p))
        st.success(f"🔗 **Connected State:** กราฟตั้งต้นจากค่า **Preview** ของ `{t_data.get('ticker')}` | Price: ${def_p:,.2f} | fix_c: ${def_c:,.0f}")
    else:
        st.info(f"🟢 **Current State:** กราฟตั้งต้นจากข้อมูลปัจจุบันของ `{t_data.get('ticker')}` | Price: ${def_p:,.2f} | fix_c: ${def_c:,.0f}")

    controls = _render_payoff_controls(def_p, def_c)
    _calculate_and_plot_payoff(def_p, def_c, controls, data)

def _render_payoff_controls(def_p: float, def_c: float) -> dict:
    safe_p = max(0.01, def_p)
    safe_c = max(1.0, def_c)
    with st.expander("🛠️ แผงควบคุมตัวแปร (Simulator Controls)", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        controls = {}
        with col_c1:
            st.markdown("##### 🟢 กลุ่มหลัก (Shannon 1 / Long หุ้น)")
            controls["x0_1"] = st.number_input("จุดอ้างอิง x0_1", min_value=0.01, value=safe_p, step=1.0)
            controls["constant1"] = st.number_input("เงินทุน Constant C", min_value=1.0, value=safe_c, step=100.0)
            controls["b1"] = st.number_input("ค่า Bias เลื่อนแกน (b1)", value=0.0, step=100.0)
            controls["delta1"] = st.slider("ความชันขาลง (δ1 สำหรับ x < x0)", 0.0, 2.0, 0.2, 0.05)
            st.markdown("---")
            controls["long_shares"] = st.number_input("จำนวน Quantity (y10 Long)", min_value=0, value=100)
            controls["long_entry"] = st.number_input("ราคา Long Entry", min_value=0.01, value=safe_p, step=1.0)
        with col_c2:
            st.markdown("##### 🟡 กลุ่มรอง (Shannon 2 / Short หุ้น)")
            controls["x0_2"] = st.number_input("จุดอ้างอิง x0_2", min_value=0.01, value=max(safe_p*1.5, 0.01), step=1.0)
            controls["constant2"] = st.number_input("เงินทุน Constant (y2/y4)", min_value=1.0, value=safe_c, step=100.0)
            controls["b2"] = st.number_input("ค่า Bias เลื่อนแกน (b2)", value=0.0, step=100.0)
            controls["delta2"] = st.slider("ความชันขาขึ้น (δ2 สำหรับ x >= x0)", 0.0, 2.0, 1.0, 0.05)
            st.markdown("---")
            controls["short_shares"] = st.number_input("จำนวน Quantity (y11 Short)", min_value=0, value=100)
            controls["short_entry"] = st.number_input("ราคา Short Entry", min_value=0.01, value=max(safe_p*1.5, 0.01), step=1.0)
        with col_c3:
            st.markdown("##### ⚔️ กลุ่ม Options & Benchmark")
            c3_1, c3_2 = st.columns(2)
            with c3_1: controls["anchorY6"] = st.number_input("ราคา Benchmark", min_value=0.01, value=safe_p, step=1.0)
            with c3_2: controls["refConst"] = st.number_input("เงินลงทุนอ้างอิง", min_value=1.0, value=safe_c, step=100.0)
            st.markdown("---")
            o1, o2, o3 = st.columns(3)
            with o1: 
                controls["call_contracts"] = st.number_input("Call Qty", min_value=0, value=100)
                controls["put_contracts"] = st.number_input("Put Qty", min_value=0, value=100)
            with o2: 
                controls["strike_call"] = st.number_input("C Strike", min_value=0.01, value=safe_p, step=1.0)
                controls["strike_put"] = st.number_input("P Strike", min_value=0.01, value=safe_p, step=1.0)
            with o3: 
                controls["premium_call"] = st.number_input("C Prem", min_value=0.0, value=0.0, step=0.1)
                controls["premium_put"] = st.number_input("P Prem", min_value=0.0, value=0.0, step=0.1)
                
        st.markdown("---")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        controls["showY1"] = t_col1.checkbox("y1: Shannon 1 (+piecewise)", value=True)
        controls["showY2"] = t_col1.checkbox("y2: Shannon 2 (original)", value=False)
        controls["showY3"] = t_col1.checkbox("y3: Net P/L (ผลรวม)", value=True)
        controls["showY4"] = t_col2.checkbox("y4: Piecewise y2", value=False)
        controls["showY5"] = t_col2.checkbox("y5: Piecewise y1", value=False)
        controls["showY6"] = t_col2.checkbox("y6: Ref y1 (Benchmark)", value=True)
        controls["showY7"] = t_col3.checkbox("y7: Ref y2", value=False)
        controls["showY8"] = t_col3.checkbox("y8: Call Intrinsic", value=False)
        controls["showY9"] = t_col3.checkbox("y9: Put Intrinsic", value=False)
        controls["showY10"] = t_col4.checkbox("y10: P/L Long (หุ้น)", value=False)
        controls["showY11"] = t_col4.checkbox("y11: P/L Short (หุ้น)", value=False)
    return controls

def _calculate_and_plot_payoff(def_p: float, def_c: float, req: dict, data: dict = None):
    x_min = max(0.01, def_p * 0.1)
    x_max = max(x_min + 0.1, def_p * 2.5)
    prices = np.linspace(x_min, x_max, 300)

    # Safe log implementations using max(..., 1e-9) to avoid Domain Error
    y1_raw = req["constant1"] * np.log(np.maximum(prices / req["x0_1"], 1e-9))
    y2_raw = req["constant2"] * np.log(np.maximum(2 - (prices / req["x0_2"]), 1e-9))
    
    y1_d2 = (y1_raw * req["delta2"]) + req["b1"]
    y2_d2 = (y2_raw * req["delta2"]) + req["b2"]
    y4_piece = (y2_raw * np.where(prices >= req["x0_2"], req["delta2"], req["delta1"])) + req["b2"]
    y5_piece = (y1_raw * np.where(prices >= req["x0_1"], req["delta2"], req["delta1"])) + req["b1"]
    y6_raw = req["refConst"] * np.log(np.maximum(prices / req["anchorY6"], 1e-9))
    y7_raw = req["refConst"] * np.log(np.maximum(2 - (prices / req["anchorY6"], 1e-9), 1e-9))
    
    y6_ref_d2 = y6_raw * req["delta2"]
    y7_ref_d2 = y7_raw * req["delta2"]

    y8_call_intrinsic = (np.maximum(0, prices - req["strike_call"]) * req["call_contracts"])
    y9_put_intrinsic = (np.maximum(0, req["strike_put"] - prices) * req["put_contracts"])
    y10_long_pl = (prices - req["long_entry"]) * req["long_shares"]
    y11_short_pl = (req["short_entry"] - prices) * req["short_shares"]

    components_d2 = []
    if req["showY1"]: components_d2.append(y1_d2)
    if req["showY2"]: components_d2.append(y2_d2)
    if req["showY4"]: components_d2.append(y4_piece)
    if req["showY5"]: components_d2.append(y5_piece)
    if req["showY8"]: components_d2.append(y8_call_intrinsic)
    if req["showY9"]: components_d2.append(y9_put_intrinsic)
    if req["showY10"]: components_d2.append(y10_long_pl)
    if req["showY11"]: components_d2.append(y11_short_pl)

    y3_delta2 = np.sum(components_d2, axis=0) if components_d2 else np.zeros_like(prices)
    y_overlay_d2 = y3_delta2 - y6_ref_d2

    tabs_chart = st.tabs(["เปรียบเทียบทั้งหมด", "Net เท่านั้น", "Delta_Log_Overlay", "Capital Flow (Sankey) 🔗"])

    with tabs_chart[0]:
        fig1 = go.Figure()
        if req["showY1"]: fig1.add_trace(go.Scatter(x=prices, y=y1_d2, name=f"y1", line=dict(color='#22d3ee', width=3)))
        if req["showY2"]: fig1.add_trace(go.Scatter(x=prices, y=y2_d2, name=f"y2", line=dict(color='#fde047', width=3)))
        if req["showY4"]: fig1.add_trace(go.Scatter(x=prices, y=y4_piece, name="y4", line=dict(color='#a3e635', width=3)))
        if req["showY5"]: fig1.add_trace(go.Scatter(x=prices, y=y5_piece, name="y5", line=dict(color='#10b981', width=3)))
        if req["showY3"]: fig1.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net", line=dict(color='#f472b6', width=3.5)))
        if req["showY6"]: fig1.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6", line=dict(color='#94a3b8', dash='dash')))
        if req["showY7"]: fig1.add_trace(go.Scatter(x=prices, y=y7_ref_d2, name="y7", line=dict(color='#c084fc', dash='dash')))
        if req["showY8"]: fig1.add_trace(go.Scatter(x=prices, y=y8_call_intrinsic, name="y8", line=dict(color='#ef4444', width=3)))
        if req["showY9"]: fig1.add_trace(go.Scatter(x=prices, y=y9_put_intrinsic, name="y9", line=dict(color='#22c55e', width=3)))
        if req["showY10"]: fig1.add_trace(go.Scatter(x=prices, y=y10_long_pl, name="y10", line=dict(color='#60a5fa', width=3)))
        if req["showY11"]: fig1.add_trace(go.Scatter(x=prices, y=y11_short_pl, name="y11", line=dict(color='#fb923c', width=3)))
        fig1.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig1.add_vline(x=def_p, line_dash="solid", line_color="#facc15")
        st.plotly_chart(fig1, use_container_width=True)

    with tabs_chart[1]:
        fig2 = go.Figure()
        if req["showY3"]: fig2.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net", line=dict(color='#f472b6', width=3.5)))
        if req["showY6"]: fig2.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6 Benchmark", line=dict(color='#94a3b8', dash='dash')))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig2.add_vline(x=def_p, line_dash="solid", line_color="#facc15")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs_chart[2]:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=prices, y=y_overlay_d2, name="Overlay", line=dict(color='#ea580c', width=3.5)))
        fig3.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig3.add_vline(x=def_p, line_dash="solid", line_color="#facc15")
        st.plotly_chart(fig3, use_container_width=True)

    with tabs_chart[3]:
        st.subheader("🔗 Capital Flow — by Ticker (Sankey)")
        _render_sankey_by_ticker(data)

def _render_sankey_by_ticker(data: dict):
    if not data: st.info("ไม่มีข้อมูลพอร์ต"); return
    tickers_list = get_tickers(data)
    if not tickers_list: st.info("ยังไม่มี Ticker"); return

    ticker_stats = []
    for t in tickers_list:
        rounds = t.get("rounds", [])
        state  = t.get("current_state", {})
        s = {
            "ticker":   t.get("ticker", "???"),
            "shannon":  sum(safe_float(r.get("shannon_profit", 0)) for r in rounds),
            "hedge":    sum(safe_float(r.get("hedge_cost", 0)) for r in rounds),
            "surplus":  sum(safe_float(r.get("surplus", 0)) for r in rounds),
            "scale_up": sum(safe_float(r.get("scale_up", 0)) for r in rounds),
            "harvest":  sum(safe_float(r.get("harvest_profit", 0)) for r in rounds),
            "injection":sum(safe_float(r.get("scale_up", 0)) for r in rounds if "Injection" in str(r.get("action", ""))),
            "ev":       safe_float(state.get("cumulative_ev", 0)),
            "fix_c":    safe_float(state.get("fix_c", 0)),
        }
        if any([s["shannon"], s["hedge"], s["surplus"], s["scale_up"], s["harvest"], s["ev"], s["fix_c"]]):
            ticker_stats.append(s)

    if not ticker_stats:
        st.info("ยังไม่มี Round data"); return

    st.warning("⚠️ โหมด Sankey จะแสดงผลได้ดีเมื่อมี Flow การเงินที่เกิดขึ้นแล้ว")

# ----------------------------------------------------------
# TAB 5: Manage Data
# ----------------------------------------------------------
def _render_manage_data(data: dict):
    st.subheader("จัดการข้อมูลพอร์ต")
    with st.expander("➕ เพิ่ม Ticker ใหม่", expanded=False):
        with st.form("add_ticker_form"):
            new_ticker = st.text_input("Ticker Symbol").upper()
            init_price = st.number_input("Initial Price (t)", min_value=0.01, value=100.0)
            init_c = st.number_input("Initial Fix_C ($)", min_value=1000.0, value=10000.0)
            if st.form_submit_button("Add Ticker"):
                if new_ticker and init_price > 0 and init_c > 0:
                    existing = [d["ticker"] for d in get_tickers(data)]
                    if new_ticker not in existing:
                        new_entry = {
                            "ticker": new_ticker, "Final": f"{init_price}, {init_c}, 0.0",
                            "current_state": {"price": init_price, "fix_c": init_c, "baseline": init_price, "cumulative_ev": 0.0},
                            "rounds": [], "beta_momory": "0.0, 0.0, 0.0"
                        }
                        if "tickers" not in data: data["tickers"] = []
                        data["tickers"].append(new_entry)
                        save_trading_data(data)
                        st.success(f"Added {new_ticker}")
                        st.rerun()
                    else: st.error("Ticker นี้มีอยู่แล้วในพอร์ต")

    with st.expander("💾 Export / Import Data", expanded=False):
        export_str = json.dumps(data, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Download Data as JSON", data=export_str,
            file_name=f"chain_system_backup_{datetime.now().strftime('%Y-%m-%d')}.json",
            mime="application/json"
        )
        st.divider()
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ JSON ที่ได้จากการ Export", type=["json"])
        if uploaded_file is not None:
            try:
                uploaded_data = json.load(uploaded_file)
                if isinstance(uploaded_data, dict) and "tickers" in uploaded_data:
                    st.warning("⚠️ การ Import จะเป็นการ **เขียนทับข้อมูลปัจจุบัน** ทั้งหมด")
                    if st.button("✅ Confirm Import", type="primary"):
                        save_trading_data(uploaded_data)
                        st.success("นำเข้าข้อมูลสำเร็จ! ระบบกำลังรีโหลด...")
                        st.rerun()
                else: st.error("❌ ไฟล์ JSON ไม่ถูกต้อง")
            except Exception as e: st.error(f"❌ Error: {e}")

    with st.expander("⚠️ ลบข้อมูลทั้งหมด", expanded=False):
        if st.button("DELETE ALL DATA", type="primary"):
            data.update({"tickers": [], "global_pool_cf": 0.0, "global_ev_reserve": 0.0, "treasury_history": []})
            save_trading_data(data)
            st.warning("All data cleared!")
            st.rerun()

if __name__ == "__main__":
    main()
