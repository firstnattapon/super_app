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
    build_portfolio_df
)

# ============================================================
# HELPER: Treasury Logging
# ============================================================
def log_treasury_event(data: dict, category: str, amount: float, note: str = "") -> None:
    """Log global treasury events (Funding, Allocation, Expense, Deploy)."""
    if "treasury_history" not in data:
        data["treasury_history"] = []
    
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "category": category,
        "amount": amount,
        "pool_cf_balance": data.get("global_pool_cf", 0.0),
        "ev_reserve_balance": data.get("global_ev_reserve", 0.0),
        "note": note
    }
    data["treasury_history"].append(entry)

# ============================================================
# MAIN APPLICATION ENGINE
# ============================================================
def main():
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
# TAB 1: Active Dashboard
# ----------------------------------------------------------
def _render_active_dashboard(data: dict):
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มีข้อมูล — เพิ่มหุ้นที่แท็บ ➕ Manage Data")
        return

    df = build_portfolio_df(tickers_list)
    _render_portfolio_summary(data, tickers_list, df)
    
    st.divider()
    _render_ticker_cards(tickers_list)
    
    st.divider()
    _render_pnl_waterfall(df)

def _render_portfolio_summary(data: dict, tickers_list: list, df: pd.DataFrame):
    total_c = df["Fix_C"].sum()
    total_ev = df["Ev (Extrinsic)"].sum()
    total_lock = df["Lock P&L"].sum()
    total_surplus = df["Surplus IV"].sum()
    total_net = df["Net"].sum()
    pool_cf = data.get("global_pool_cf", 0.0)
    total_burn = sum(t.get("current_state", {}).get("cumulative_ev", 0.0) for t in tickers_list)

    st.subheader("📊 Portfolio Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fix_C (Deployed)", f"${total_c:,.0f}", f"{len(tickers_list)} tickers")
    m2.metric("🎱 Pool CF (War Chest)", f"${pool_cf:,.2f}")
    m3.metric("🔥 Burn Rate (Cum. Ev)", f"${total_burn:,.2f}", delta="Cost of Business", delta_color="inverse")
    m4.metric("💰 Net Reality", f"${total_net:,.2f}",
              delta=f"Lock {total_lock:,.0f} + IV {total_surplus:,.0f} + Ev {total_ev:,.0f}",
              delta_color="normal" if total_net >= 0 else "inverse")

    gross_profit = total_lock + total_surplus
    efficiency = (gross_profit / total_burn) if total_burn > 0 else 0.0
    
    st.markdown("#### ⏱️ Ev Efficiency (Winning against Time)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Profit (No Ev)", f"${gross_profit:,.2f}", "Harvest + Shannon")
    c2.metric("Total Burn (Cost)", f"${total_burn:,.2f}", "Cumulative Theta Decay")
    c3.metric("Ev Efficiency Ratio", f"{efficiency:.2f}x", 
              delta="Sustainable" if efficiency >= 1.0 else "Bleeding", delta_color="normal" if efficiency >= 1.0 else "inverse")
    c4.caption("**Ratio > 1.0** = กำไรจากระบบชนะค่าเช่า (Time Decay)\n\n**Ratio < 1.0** = ยังต้องการ Direction ช่วย")

def _render_ticker_cards(tickers_list: list):
    st.subheader("📋 Ticker Status Cards")
    cols_per_row = 4
    for i in range(0, len(tickers_list), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(tickers_list): break
            t_data = tickers_list[idx]
            ticker = t_data.get("ticker", "???")
            state = t_data.get("current_state", {})
            n_rounds = len(t_data.get("rounds", []))
            net_val = state.get("net_pnl", 0.0)
            
            with col:
                with st.container(border=True):
                    color = "🟢" if net_val >= 0 else "🔴"
                    st.markdown(f"### {color} {ticker}")
                    st.caption(f"Price: ${state.get('price', 0):,.2f} | fix_c: ${state.get('fix_c', 0):,.0f}")
                    st.caption(f"Baseline: ${state.get('baseline', 0):,.2f} | Rounds: {n_rounds}")
                    st.caption(f"Net: ${net_val:,.2f}")

def _render_pnl_waterfall(df: pd.DataFrame):
    st.subheader("Waterfall: Ev → Lock P&L → Net")
    total_ev = df["Ev (Extrinsic)"].sum()
    total_lock = df["Lock P&L"].sum()
    total_surplus = df["Surplus IV"].sum()
    total_net = df["Net"].sum()
    
    fig_wf = go.Figure(go.Waterfall(
        x=["Ev (Cost)", "Lock P&L", "Surplus IV", "Net"],
        y=[total_ev, total_lock, total_surplus, 0],
        measure=["relative", "relative", "relative", "total"],
        text=[f"${total_ev:,.0f}", f"${total_lock:,.0f}", f"${total_surplus:,.0f}", f"${total_net:,.0f}"],
        textposition="outside",
        connector=dict(line=dict(color="gray", width=1)),
        increasing_marker_color="#00c853", decreasing_marker_color="#ff1744", totals_marker_color="#2196f3",
    ))
    fig_wf.update_layout(title="Portfolio P&L Waterfall", height=380)
    st.plotly_chart(fig_wf, use_container_width=True)


# ----------------------------------------------------------
# TAB 2: Engine & History  —  IB Workspace Layout (3-Zone)
# ----------------------------------------------------------

def _render_engine_tab(data: dict):
    tickers_list = get_tickers(data)
    ticker_names = [t.get("ticker", "???") for t in tickers_list]

    # ── Resolve active ticker ONCE at top — single source of truth ──────
    # Read from watchlist radio (index-based) if available, else fallback
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
        active_ticker = ""
        active_t_data = {}

    # Sync session_state so Payoff tab also reads correct ticker
    st.session_state["run_round_ticker"] = active_ticker

    # ── TOP BAR: Global (left) + Quick Stats (right) ────────────────────
    _render_engine_metrics(data, tickers_list, active_ticker, active_t_data)

    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Manage Data ก่อน")
        return

    # ── 3-Zone Horizontal Split ─────────────────────────────────────────
    z_left, z_center, z_right = st.columns([12, 44, 44], gap="medium")

    with z_left:
        _render_ticker_watchlist(tickers_list, active_idx)

    with z_center:
        _render_center_panels(data, tickers_list, active_ticker, active_t_data)

    with z_right:
        _render_chain_engine_center(data, tickers_list, active_ticker, active_t_data, active_idx)


def _chip_label(t_data: dict) -> str:
    """Short chip label — ticker + price only, readable in narrow column."""
    ticker = t_data.get("ticker", "???")
    state  = t_data.get("current_state", {})
    net    = float(state.get("net_pnl", 0))
    price  = float(state.get("price", 0))
    dot    = "🟢" if net >= 0 else "🔴"
    return f"{dot} {ticker}  ${price:.2f}"


# ── TOP METRICS BAR: Left = Global, Right = Per-Ticker ────────────────
def _fmt(v: float) -> str:
    """Integer format without symbols or commas.
    Shows full value: -13500 instead of -$13,500.00
    """
    return f"{int(round(v))}"


def _render_engine_metrics(data: dict, tickers_list: list,
                            active_ticker: str, active_t_data: dict):
    pool_cf      = data.get("global_pool_cf", 0.0)
    ev_reserve   = data.get("global_ev_reserve", 0.0)
    total_rounds = sum(len(t.get("rounds", [])) for t in tickers_list)

    state      = active_t_data.get("current_state", {}) if active_t_data else {}
    sel_net    = float(state.get("net_pnl", 0))
    sel_rounds = len(active_t_data.get("rounds", [])) if active_t_data else 0

    with st.container(border=True):
        col_global, col_div, col_ticker = st.columns([4, 0.1, 5])

        with col_global:
            st.caption("🌐 Global")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("🎱 Pool CF",   _fmt(pool_cf))
            g2.metric("🛡️ EV LEAPS", _fmt(ev_reserve))
            g3.metric("Tickers",      str(len(tickers_list)))
            g4.metric("Total Rounds", str(total_rounds))

        with col_div:
            st.markdown(
                "<div style='border-left:1px solid #334155;height:80px;margin-top:8px'></div>",
                unsafe_allow_html=True
            )

        with col_ticker:
            if active_t_data:
                st.caption(f"📌 {active_ticker}")
                t1, t2, t3, t4, t5, t6 = st.columns(6)
                t1.metric("Price",      _fmt(float(state.get("price", 0))))
                t2.metric("fix_c",      _fmt(float(state.get("fix_c", 0))))
                t3.metric("Baseline",   _fmt(float(state.get("baseline", 0))))
                t4.metric("Ev Burn 🔥", _fmt(float(state.get("cumulative_ev", 0))),
                          delta_color="inverse")
                t5.metric("Net P&L",    _fmt(sel_net),
                          delta_color="normal" if sel_net >= 0 else "inverse")
                t6.metric("Rounds",     str(sel_rounds))
            else:
                st.caption("📌 ยังไม่มี Ticker")


# ── ZONE LEFT: Ticker Watchlist ────────────────────────────────────────
def _render_ticker_watchlist(tickers_list: list, active_idx: int):
    labels = [_chip_label(t) for t in tickers_list]
    st.markdown("##### 📋 Watchlist")
    st.radio(
        "Ticker", labels, index=active_idx,
        key="ticker_watchlist_radio",
        label_visibility="collapsed"
    )


# ── ZONE CENTER: Chain Round Engine ───────────────────────────────────
def _render_chain_engine_center(data: dict, tickers_list: list,
                                 selected_ticker: str, t_data: dict, idx: int):
    state      = t_data.get("current_state", {}) if t_data else {}
    settings   = data.get("settings", {})
    default_hr = float(settings.get("default_hedge_ratio", 2.0))
    default_p  = float(max(0.01, round(float(state.get("price", 10.0)) * 1.1, 2)))

    # Header
    hc1, hc2 = st.columns([3, 1])
    hc1.markdown(f"#### ⚡ Chain Round — **{selected_ticker}**")
    with hc2:
        if (st.session_state.get("_pending_round") and
                st.session_state.get("_pending_ticker_name") == selected_ticker):
            st.success("🔗 Synced → Payoff")
        else:
            st.caption("💡 Preview → syncs Payoff tab")

    # ── 1-ROW COMMAND STRIP ────────────────────────────────────────────
    with st.container(border=True):
        st.caption("⚡ Order Strip — ป้อนค่าแล้วกด Preview")
        sc1, sc2, sc3, sc4, sc5 = st.columns([2.5, 1.8, 1.2, 1.4, 1.8])
        with sc1:
            p_new = st.number_input("P New", min_value=0.01, value=default_p,
                                    step=1.0, key=f"strip_pnew_{idx}")
        with sc2:
            hedge_ratio = st.number_input("Hedge ×", min_value=0.0,
                                          value=default_hr, step=0.5,
                                          key=f"strip_hr_{idx}")
        with sc3:
            ignore_hedge   = st.checkbox("No Hedge",   value=False, key=f"strip_ih_{idx}")
        with sc4:
            ignore_surplus = st.checkbox("No Surplus", value=False, key=f"strip_is_{idx}")
        with sc5:
            st.write("")
            preview_clicked = st.button("🔍 Preview", type="primary",
                                        key=f"strip_preview_{idx}",
                                        use_container_width=True)

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

    # ── PREVIEW RESULT PANEL ──────────────────────────────────────────
    rd         = st.session_state.get("_pending_round")
    is_pending = (rd is not None and
                  st.session_state.get("_pending_ticker_name") == selected_ticker)

    if is_pending:
        with st.container(border=True):
            st.markdown("**📊 Preview Result** — แก้ไขได้ก่อน Commit")

            # ── ROW 1: P&L ─────────────────────────────────────────────
            r1c1, r1c2, r1c3 = st.columns(3)
            new_shannon = r1c1.number_input("💰 Shannon Profit",
                value=float(rd["shannon_profit"]), step=10.0, format="%.2f", key="edit_shannon")
            new_hedge   = r1c2.number_input("🛡️ Hedge Cost",
                value=float(rd["hedge_cost"]),     step=10.0, format="%.2f", key="edit_hedge")
            new_surplus = r1c3.number_input("✨ Surplus",
                value=float(rd["surplus"]),        step=10.0, format="%.2f", key="edit_surplus")

            scale_val = float(rd.get("scale_up", max(0.0, float(rd.get("surplus", 0.0)))))
            sc_color  = "#22c55e" if scale_val > 0 else ("#ef4444" if scale_val < 0 else "#94a3b8")
            st.markdown(
                f"<div style='text-align:center;padding:4px 0 10px;font-size:13px;color:#94a3b8'>"
                f"🚀 Scale Up &nbsp;<span style='color:{sc_color};font-weight:700;font-size:20px'>"
                f"+${scale_val:,.2f}</span></div>",
                unsafe_allow_html=True,
            )

            st.divider()

            r2c1, r2c2, r2c3 = st.columns(3)
            new_c_after = r2c1.number_input("fix_c (after)",
                value=float(rd["c_after"]),  step=100.0, format="%.0f", key="edit_c_after",
                help=f"Before: ${rd['c_before']:,.0f}")
            new_p_new   = r2c2.number_input("Price (new t)",
                value=float(rd["p_new"]),    step=0.1,   format="%.2f", key="edit_p_new",
                help=f"Before: ${rd['p_old']:,.2f}")
            new_b_after = r2c3.number_input("Baseline (after)",
                value=float(rd["b_after"]),  step=0.1,   format="%.2f", key="edit_b_after",
                help=f"Before: ${rd['b_before']:,.2f}")

            d1, d2, d3 = st.columns(3)
            d1.markdown(_delta_badge(rd["c_before"], new_c_after, ",.0f"), unsafe_allow_html=True)
            d2.markdown(_delta_badge(rd["p_old"],    new_p_new,   ",.2f"), unsafe_allow_html=True)
            d3.markdown(_delta_badge(rd["b_before"], new_b_after, ",.2f"), unsafe_allow_html=True)

        btn_col, cnl_col = st.columns([4, 1])
        with btn_col:
            if st.button("✅ COMMIT — บันทึกถาวร", type="primary", use_container_width=True):
                rd.update({
                    "shannon_profit": new_shannon,
                    "hedge_cost":     new_hedge,
                    "surplus":        new_surplus,
                    "c_after":        new_c_after,
                    "p_new":          new_p_new,
                    "b_after":        new_b_after,
                    "scale_up":       new_c_after - rd["c_before"],
                })
                commit_round(data, st.session_state["_pending_ticker_idx"], rd)
                del st.session_state["_pending_round"]
                st.success(f"✅ Committed! {selected_ticker} fix_c → ${new_c_after:,.0f}")
                st.rerun()
        with cnl_col:
            if st.button("✖ Cancel", use_container_width=True, key="cancel_preview"):
                del st.session_state["_pending_round"]
                st.rerun()

    else:
        with st.container(border=True):
            st.caption(
                f"📌 {selected_ticker}  |  "
                f"fix_c ${state.get('fix_c', 0):,.0f}  |  "
                f"Price ${state.get('price', 0):,.2f}  |  "
                f"Baseline ${state.get('baseline', 0):,.2f}  |  "
                f"Rounds {len(t_data.get('rounds', []) if t_data else [])}"
            )
            st.info("ป้อน P New แล้วกด 🔍 Preview เพื่อเริ่มคำนวณ")


def _delta_badge(before: float, after: float, fmt: str = ",.0f") -> str:
    """Inline-style delta badge — safe against Streamlit HTML sanitizer."""
    diff  = after - before
    color = "#22c55e" if diff > 0 else ("#ef4444" if diff < 0 else "#94a3b8")
    sign  = "+" if diff > 0 else ""
    arrow = "<span style='color:#475569;margin:0 5px'>→</span>"
    return (
        f"<div style='font-size:12px;padding:2px 0'>"
        f"<span style='color:#64748b'>${before:{fmt}}</span>"
        f"{arrow}"
        f"<span style='color:{color};font-weight:700'>${after:{fmt}}</span>"
        f"&nbsp;<span style='color:{color}'>({sign}{diff:{fmt}})</span>"
        f"</div>"
    )


# ── ZONE CENTER: Tabbed Panels ────────────────────────────────────────
def _render_center_panels(data: dict, tickers_list: list,
                           active_ticker: str, active_t_data: dict):
    tab_hist, tab_treasury, tab_pool, tab_deploy = st.tabs([
        "📜 History", "🏛️ Treasury", "🎱 Pool CF", "🚀 Deploy"
    ])

    with tab_hist:
        if active_t_data:
            _render_consolidated_history(active_t_data)
        else:
            st.info("เลือก Ticker ที่ Watchlist ก่อน")

    with tab_treasury:
        _render_treasury_log(data, filter_ticker=active_ticker)

    with tab_pool:
        _render_pool_cf_section(data)

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
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                btn_add = st.form_submit_button("💰 Add Fund", type="primary")
                
            if btn_add and amount > 0:
                data["global_pool_cf"] = data.get("global_pool_cf", 0) + amount
                note_str = "Added to Pool CF"
                if selected_ticker and selected_ticker != "None":
                    note_str += f" [Ticker: {selected_ticker}]"
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
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                btn_harvest = st.form_submit_button("🌾 Add Harvest", type="primary")
                
            if btn_harvest and h_amount > 0:
                data["global_pool_cf"] = data.get("global_pool_cf", 0) + h_amount
                note_str = "Harvest Profit"
                if h_ticker and h_ticker != "None":
                    note_str += f" [Ticker: {h_ticker}]"
                log_treasury_event(data, "Harvest", h_amount, note_str)
                save_trading_data(data)
                st.success(f"✅ +${h_amount:,.2f} Harvest → Pool CF = ${data['global_pool_cf']:,.2f}")
                st.rerun()

        st.divider()
        st.markdown("##### 📥 Extract Baseline to Pool CF")
        # Filter tickers that actually have a positive baseline
        eligible_tickers = [t for t in get_tickers(data) if t.get("current_state", {}).get("baseline", 0) > 0]
        
        if eligible_tickers:
            with st.form("extract_baseline_form", clear_on_submit=True):
                hc1, hc2 = st.columns([2, 1])
                with hc1:
                    ext_ticker_name = st.selectbox("Select Ticker", options=[t.get("ticker") for t in eligible_tickers], key="extract_ticker")
                    # Find the selected ticker object to get its max baseline
                    selected_t_obj = next((t for t in eligible_tickers if t.get("ticker") == ext_ticker_name), None)
                    max_baseline = float(selected_t_obj.get("current_state", {}).get("baseline", 0)) if selected_t_obj else 0.0
                    
                    ext_amount = st.number_input("Extract Amount ($)", min_value=0.0, max_value=max_baseline, value=max_baseline, step=100.0)
                with hc2:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.write("")
                    btn_extract = st.form_submit_button("📥 Extract to Pool CF", type="primary")
                    
                if btn_extract and ext_amount > 0 and selected_t_obj:
                    # 1. Update Ticker state
                    current_baseline = selected_t_obj["current_state"]["baseline"]
                    selected_t_obj["current_state"]["baseline"] -= ext_amount
                    
                    # 2. Add to Pool CF
                    data["global_pool_cf"] = data.get("global_pool_cf", 0.0) + ext_amount
                    
                    # 3. Log event
                    log_treasury_event(data, "Baseline Harvest", ext_amount, f"[Ticker: {ext_ticker_name}]")
                    
                    # 4. Record dummy round to preserve history of baseline change
                    cur_state = selected_t_obj["current_state"]
                    dummy_round = {
                        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), 
                        "action": "Extract Baseline", 
                        "p_old": cur_state.get("price", 0), 
                        "p_new": cur_state.get("price", 0), 
                        "c_before": cur_state.get("fix_c", 0), 
                        "c_after": cur_state.get("fix_c", 0),
                        "shannon_profit": 0.0, 
                        "harvest_profit": 0.0, 
                        "hedge_cost": 0.0,
                        "surplus": 0.0, 
                        "scale_up": 0.0, 
                        "b_before": current_baseline, 
                        "b_after": current_baseline - ext_amount, 
                        "note": f"Extracted ${ext_amount:,.2f} to Pool CF",
                        "hedge_ratio": 0.0, 
                        "sigma": 0.0, 
                        "ev_change": 0.0
                    }
                    if "rounds" not in selected_t_obj:
                        selected_t_obj["rounds"] = []
                    selected_t_obj["rounds"].append(dummy_round)

                    # 5. Save & Rerun
                    save_trading_data(data)
                    st.success(f"✅ Extracted ${ext_amount:,.2f} from {ext_ticker_name} Baseline to Pool CF")
                    st.rerun()
        else:
            st.info("No tickers with a positive Baseline available for extraction.")


def _render_deployment_section(data: dict, tickers_list: list):
    if not tickers_list:
        return
        
    with st.expander("🔍 Preview Deployment", expanded=False):
        deploy_ticker_options = [d.get("ticker", "???") for d in tickers_list]
        deploy_ticker = st.selectbox("Select Ticker", deploy_ticker_options, key="deploy_ticker")
        d_idx = deploy_ticker_options.index(deploy_ticker)
        t_data_deploy = tickers_list[d_idx]
        
        cur_state = t_data_deploy.get("current_state", {})
        cur_c = cur_state.get("fix_c", 0)
        cur_t = cur_state.get("price", 0)
        cur_b = cur_state.get("baseline", 0)
        cur_ev_debt = cur_state.get("cumulative_ev", 0.0)
        pool_cf = data.get("global_pool_cf", 0.0)
        
        rounds = t_data_deploy.get("rounds", [{}])
        last_round = rounds[-1] if rounds else {}
        cur_sigma = last_round.get("sigma", 0.5)
        cur_hr = last_round.get("hedge_ratio", 2.0)
    
        with st.form("deploy_round_form", clear_on_submit=False):
            d1, d2 = st.columns(2)
            with d1: 
                action_type = st.selectbox("Objective", ["📈 Scale Up", "🛡️ Buy Puts", "🎯 Buy Calls", "⏳ Pay Ev"])
            with d2: 
                d_amt = st.number_input("Amount ($) [Pool Funding]", min_value=0.0, max_value=float(pool_cf) if pool_cf > 0 else 0.0, value=0.0, step=100.0)
                
            manual_new_c = st.number_input("Target fix_c (Optional Override)", min_value=0.0, value=0.0, step=100.0)
            d_note = st.text_input("Note", value="")
            submitted_deploy = st.form_submit_button("🔍 Preview Deployment")
    
        if submitted_deploy and d_amt > 0:
            mock_scale_up, mock_new_c = (manual_new_c - cur_c, manual_new_c) if manual_new_c > cur_c else (d_amt, cur_c + d_amt)
            mock_ev_change = -d_amt if "Pay Ev" in action_type else 0.0
            
            injection_round = {
                "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), 
                "action": "Injection", "p_old": cur_t, "p_new": cur_t, 
                "c_before": cur_c, "c_after": mock_new_c,
                "shannon_profit": 0.0, "harvest_profit": 0.0, "hedge_cost": 0.0,
                "surplus": mock_scale_up if "Scale Up" in action_type else -d_amt, 
                "scale_up": mock_scale_up if "Scale Up" in action_type else 0.0, 
                "b_before": cur_b, "b_after": cur_b, 
                "note": f"[{action_type.split()[1]}] {d_note}",
                "hedge_ratio": cur_hr, "sigma": cur_sigma, "ev_change": mock_ev_change
            }
            st.session_state["_pending_injection"] = injection_round
            st.session_state["_pending_injection_idx"] = d_idx
            st.session_state["_pending_injection_amt"] = d_amt
            st.session_state["_pending_injection_type"] = action_type
    
        if "_pending_injection" in st.session_state and st.session_state.get("_pending_injection_idx") == d_idx:
            p_inj = st.session_state["_pending_injection"]
            p_type = st.session_state.get("_pending_injection_type", "")
            
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("fix_c Change", f"${p_inj['c_before']:,.0f} → ${p_inj['c_after']:,.0f}")
            dc2.metric("Pool Deduction", f"-${st.session_state['_pending_injection_amt']:,.2f}")
            
            if "Pay Ev" in p_type:
                dc3.metric("Burn Rate (Ev)", f"${cur_ev_debt:,.2f} → ${max(0, cur_ev_debt - st.session_state['_pending_injection_amt']):,.2f}")
            
            if st.button("🚀 Confirm Deployment", type="primary"):
                amt = st.session_state["_pending_injection_amt"]
                if data["global_pool_cf"] >= amt:
                    data["global_pool_cf"] -= amt
                    if "Pay Ev" in p_type:
                        t_data_deploy["current_state"]["cumulative_ev"] = max(0.0, cur_ev_debt - amt)
                    
                    log_treasury_event(data, "Deploy", -amt, f"Deployed to {deploy_ticker} ({p_type})")
                    commit_round(data, d_idx, p_inj)
                    
                    del st.session_state["_pending_injection"]
                    st.success(f"✅ Complete: {st.session_state['_pending_injection_type']} ${amt:,.2f}!")
                    st.rerun()

def _render_ev_leaps_section(data: dict):
    pool_cf = data.get("global_pool_cf", 0.0)
    ev_reserve = data.get("global_ev_reserve", 0.0)
    
    with st.expander("🛡️ Manage Pool EV LEAPS (Income & Expenses)"):
        st.markdown("##### 📥 Allocate (Income from Pool CF)")
        col_a, col_b = st.columns(2)
        with col_a: 
            alloc_amt = st.number_input("Allocate Amount ($)", min_value=0.0, max_value=float(pool_cf), step=100.0, key="alloc")
        with col_b:
            if st.button("📥 Allocate"):
                if alloc_amt > 0 and pool_cf >= alloc_amt:
                    data["global_pool_cf"] -= alloc_amt
                    data["global_ev_reserve"] = data.get("global_ev_reserve", 0.0) + alloc_amt
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
            st.write("")
            st.write("")
            if st.button("💾 Record Flow"):
                data["global_ev_reserve"] = data.get("global_ev_reserve", 0.0) + pay_leaps_amt
                note_str = "Manual Adjustment"
                if selected_leaps_ticker and selected_leaps_ticker != "None":
                    note_str += f" [Ticker: {selected_leaps_ticker}]"
                log_treasury_event(data, "Income" if pay_leaps_amt >= 0 else "Expense", pay_leaps_amt, note_str)
                save_trading_data(data)
                st.success("Recorded Extrinsic Value adjustment")
                st.rerun()
        
        st.markdown(f"**Current Pool EV LEAPS Balance:** `${ev_reserve:,.2f}`")

def _render_treasury_log(data: dict, filter_ticker: str = ""):
    st.subheader("🏛️ Treasury & Ops History")
    history = data.get("treasury_history", [])
    if not history:
        st.info("ยังไม่มี Treasury events")
        return

    all_tickers = sorted({
        e.get("note", "").split("[Ticker: ")[-1].rstrip("]")
        for e in history if "[Ticker:" in e.get("note", "")
    })
    options = ["🌐 ทั้งหมด"] + all_tickers

    # ── Force sync with active Watchlist ticker ──────────────────────────
    # st.selectbox ignores `index` after first render when key is set.
    # Solution: detect ticker change → overwrite session_state key directly.
    target = filter_ticker if filter_ticker in options else "🌐 ทั้งหมด"
    if st.session_state.get("_treasury_last_ticker") != filter_ticker:
        st.session_state["treasury_filter_sel"] = target
        st.session_state["_treasury_last_ticker"] = filter_ticker

    sel = st.selectbox(
        "กรอง Ticker", options,
        key="treasury_filter_sel",
        label_visibility="collapsed"
    )

    filtered = history if sel == "🌐 ทั้งหมด" else [
        e for e in history if sel in e.get("note", "")
    ]
    tbl = [{
        "Date":   e.get("date", "")[:10],
        "Action": e.get("category", ""),
        "Amount": f"${e.get('amount', 0):,.2f}",
        "Pool CF":f"${e.get('pool_cf_balance', 0):,.0f}",
        "EV Res": f"${e.get('ev_reserve_balance', 0):,.0f}",
        "Note":   e.get("note", ""),
    } for e in filtered]

    st.caption(f"แสดง {len(tbl)} รายการ" +
               (f"  (filter: {sel})" if sel != "🌐 ทั้งหมด" else ""))
               
    # --- Modification applied here ---
    df = pd.DataFrame(tbl)[::-1]
    
    if sel != "🌐 ทั้งหมด":
        df = df.drop(columns=["Pool CF", "EV Res"], errors="ignore")
        
    st.dataframe(df, use_container_width=True, hide_index=True)
    # ---------------------------------

def _render_consolidated_history(t_data: dict):
    st.subheader(f"📜 {t_data.get('ticker','???')} — History")
    rounds = t_data.get("rounds", [])
    if rounds:
        tbl = [{
            "Date": rd.get("date", "")[:10], 
            "Action": f"Scale +${rd.get('scale_up', 0):,.0f}" if rd.get("scale_up", 0) > 0 else ("Inject/Deploy" if "Injection" in rd.get("action", "") else rd.get("action", "Round")),
            "Price": f"${rd.get('p_old',0):,.2f} > ${rd.get('p_new',0):,.2f}", 
            "fix_c": f"${rd.get('c_before',0):,.0f} > ${rd.get('c_after',0):,.0f}",
            "b": f"${rd.get('b_before',0):,.2f} > ${rd.get('b_after',0):,.2f}", 
            "Net Result": f"${rd.get('surplus',0):,.2f}"
        } for rd in rounds]
        df = pd.DataFrame(tbl)[::-1]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No history recorded yet.")


# ----------------------------------------------------------
# TAB 4: Payoff Profile Simulator
# ----------------------------------------------------------
def _render_payoff_profile_tab(data: dict):
    tickers_list = get_tickers(data)
    selected_ticker = st.session_state.get("run_round_ticker")
    t_data = next((t for t in tickers_list if t["ticker"] == selected_ticker), tickers_list[0] if tickers_list else None)
    
    if not t_data:
        st.info("👈 Please select a Ticker in the 'Engine & History' tab first.")
        return

    st.subheader(f"📐 Advanced Payoff Profile Simulator")
    st.caption("จำลองโครงสร้างผลตอบแทน (Logarithmic 11-Line Model - React Port)")

    cur_state = t_data.get("current_state", {})
    def_c = float(cur_state.get("fix_c", 10000.0))
    def_p = float(cur_state.get("price", 100.0))

    pending = st.session_state.get("_pending_round")
    if pending and st.session_state.get("_pending_ticker_name") == t_data.get("ticker"):
        def_c = float(pending.get('c_after', def_c))
        def_p = float(pending.get('p_new', def_p))
        st.success(f"🔗 **Connected State:** กราฟตั้งต้นจากค่า **Preview Calculation** ของ `{t_data.get('ticker')}` | New Price: ${def_p:,.2f} | New fix_c: ${def_c:,.0f}")
    else:
        st.info(f"🟢 **Current State:** กราฟตั้งต้นจากข้อมูลปัจจุบันของ `{t_data.get('ticker')}` | Price: ${def_p:,.2f} | fix_c: ${def_c:,.0f}")

    controls = _render_payoff_controls(def_p, def_c)
    _calculate_and_plot_payoff(def_p, def_c, controls, data)

def _render_payoff_controls(def_p: float, def_c: float) -> dict:
    # Ensure def_p and def_c are within safe bounds for Streamlit UI
    safe_p = float(max(0.01, def_p))
    safe_c = float(max(1.0, def_c))

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
            controls["x0_2"] = st.number_input("จุดอ้างอิง x0_2", min_value=0.01, value=float(max(safe_p*1.5, 0.01)), step=1.0)
            controls["constant2"] = st.number_input("เงินทุน Constant (y2/y4)", min_value=1.0, value=safe_c, step=100.0)
            controls["b2"] = st.number_input("ค่า Bias เลื่อนแกน (b2)", value=0.0, step=100.0)
            controls["delta2"] = st.slider("ความชันขาขึ้น (δ2 สำหรับ x >= x0)", 0.0, 2.0, 1.0, 0.05)
            st.markdown("---")
            controls["short_shares"] = st.number_input("จำนวน Quantity (y11 Short)", min_value=0, value=100)
            controls["short_entry"] = st.number_input("ราคา Short Entry", min_value=0.01, value=float(max(safe_p*1.5, 0.01)), step=1.0)

        with col_c3:
            st.markdown("##### ⚔️ กลุ่ม Options & Benchmark")
            c3_1, c3_2 = st.columns(2)
            with c3_1: 
                controls["anchorY6"] = st.number_input("ราคา Benchmark", min_value=0.01, value=safe_p, step=1.0)
            with c3_2: 
                controls["refConst"] = st.number_input("เงินลงทุนอ้างอิง", min_value=1.0, value=safe_c, step=100.0)
            
            st.markdown("---")
            st.caption("Call Options (y8) | Put Options (y9)")
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
    # Retrieve control values
    x0_1, constant1, b1, delta1 = req["x0_1"], req["constant1"], req["b1"], req["delta1"]
    x0_2, constant2, b2, delta2 = req["x0_2"], req["constant2"], req["b2"], req["delta2"]
    anchorY6, refConst = req["anchorY6"], req["refConst"]
    strike_call, strike_put = req["strike_call"], req["strike_put"]
    call_contracts, put_contracts = req["call_contracts"], req["put_contracts"]
    premium_call, premium_put = req["premium_call"], req["premium_put"]
    long_shares, long_entry = req["long_shares"], req["long_entry"]
    short_shares, short_entry = req["short_shares"], req["short_entry"]

    # Mathematics Engine (numpy optimized)
    x_min = float(max(0.01, def_p * 0.1))
    x_max = float(max(x_min + 0.1, def_p * 2.5))
    prices = np.linspace(x_min, x_max, 300)

    y1_raw = constant1 * np.where(prices > 0, np.log(prices / x0_1), 0)
    y2_raw = constant2 * np.log(np.where(prices / x0_2 < 2, 2 - (prices / x0_2), 1e-9))

    y1_d2 = (y1_raw * delta2) + b1
    y2_d2 = (y2_raw * delta2) + b2

    y4_piece = (y2_raw * np.where(prices >= x0_2, delta2, delta1)) + b2
    y5_piece = (y1_raw * np.where(prices >= x0_1, delta2, delta1)) + b1

    y6_raw = refConst * np.where(prices > 0, np.log(prices / anchorY6), 0)
    y7_raw = refConst * np.log(np.where(prices / anchorY6 < 2, 2 - (prices / anchorY6), 1e-9))

    y6_ref_d2 = y6_raw * delta2
    y7_ref_d2 = y7_raw * delta2

    y8_call_intrinsic = (np.maximum(0, prices - strike_call) * call_contracts)
    y9_put_intrinsic = (np.maximum(0, strike_put - prices) * put_contracts)

    y10_long_pl = (prices - long_entry) * long_shares
    y11_short_pl = (short_entry - prices) * short_shares

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

    # Plotly Visualization
    tabs_chart = st.tabs(["เปรียบเทียบทั้งหมด", "Net เท่านั้น", "Delta_Log_Overlay", "Capital Flow (Sankey) 🔗"])

    with tabs_chart[0]:
        fig1 = go.Figure()
        if req["showY1"]: fig1.add_trace(go.Scatter(x=prices, y=y1_d2, name=f"y1 (δ={delta2:.2f})", line=dict(color='#22d3ee', width=3)))
        if req["showY2"]: fig1.add_trace(go.Scatter(x=prices, y=y2_d2, name=f"y2 (δ={delta2:.2f})", line=dict(color='#fde047', width=3)))
        if req["showY4"]: fig1.add_trace(go.Scatter(x=prices, y=y4_piece, name="y4 (piecewise δ y2)", line=dict(color='#a3e635', width=3)))
        if req["showY5"]: fig1.add_trace(go.Scatter(x=prices, y=y5_piece, name="y5 (piecewise δ y1)", line=dict(color='#10b981', width=3)))
        if req["showY3"]: fig1.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net (δ2 base)", line=dict(color='#f472b6', width=3.5)))
        if req["showY6"]: fig1.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6 (Benchmark, δ2)", line=dict(color='#94a3b8', width=2.5, dash='dash')))
        if req["showY7"]: fig1.add_trace(go.Scatter(x=prices, y=y7_ref_d2, name="y7 (Ref y2, δ2)", line=dict(color='#c084fc', width=2.5, dash='dash')))
        if req["showY8"]: fig1.add_trace(go.Scatter(x=prices, y=y8_call_intrinsic, name="y8 (Call)", line=dict(color='#ef4444', width=3)))
        if req["showY9"]: fig1.add_trace(go.Scatter(x=prices, y=y9_put_intrinsic, name="y9 (Put)", line=dict(color='#22c55e', width=3)))
        if req["showY10"]: fig1.add_trace(go.Scatter(x=prices, y=y10_long_pl, name="y10 (Long)", line=dict(color='#60a5fa', width=3)))
        if req["showY11"]: fig1.add_trace(go.Scatter(x=prices, y=y11_short_pl, name="y11 (Short)", line=dict(color='#fb923c', width=3)))
        
        fig1.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig1.add_vline(x=def_p, line_dash="solid", line_color="#facc15", opacity=0.8, annotation_text="t (current)")
        fig1.update_layout(title="Full Custom Comparison Model", xaxis_title="Price (x)", yaxis_title="P/L (y)", height=600)
        st.plotly_chart(fig1, use_container_width=True)

    with tabs_chart[1]:
        fig2 = go.Figure()
        if req["showY3"]: fig2.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net (δ2 base)", line=dict(color='#f472b6', width=3.5)))
        if req["showY6"]: fig2.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6 (Benchmark, δ2)", line=dict(color='#94a3b8', width=3, dash='dash')))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig2.add_vline(x=def_p, line_dash="solid", line_color="#facc15", opacity=0.8)
        fig2.update_layout(title="Net (y3) vs Benchmark", xaxis_title="Price (x)", yaxis_title="P/L (y)", height=500)
        st.plotly_chart(fig2, use_container_width=True)

    with tabs_chart[2]:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=prices, y=y_overlay_d2, name="Delta Log Overlay (Net - Benchmark)", line=dict(color='#ea580c', width=3.5)))
        fig3.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig3.add_vline(x=def_p, line_dash="solid", line_color="#facc15", opacity=0.8)
        fig3.update_layout(title="Delta Log Overlay", xaxis_title="Price (x)", yaxis_title="P/L (y)", height=500)
        st.plotly_chart(fig3, use_container_width=True)

    with tabs_chart[3]:
        st.subheader("🔗 Capital Flow — by Ticker (Sankey)")
        _render_sankey_by_ticker(data)


def _render_sankey_by_ticker(data: dict):
    if not data:
        st.info("ไม่มีข้อมูลพอร์ต"); return
    tickers_list = get_tickers(data)
    if not tickers_list:
        st.info("ยังไม่มี Ticker"); return

    ticker_stats = []
    for t in tickers_list:
        rounds = t.get("rounds", [])
        state  = t.get("current_state", {})
        s = {
            "ticker":   t.get("ticker", "???"),
            "shannon":  sum(float(r.get("shannon_profit", 0)) for r in rounds),
            "hedge":    sum(float(r.get("hedge_cost",    0)) for r in rounds),
            "surplus":  sum(float(r.get("surplus",       0)) for r in rounds),
            "scale_up": sum(float(r.get("scale_up",      0)) for r in rounds),
            "harvest":  sum(float(r.get("harvest_profit",0)) for r in rounds),
            "injection":sum(float(r.get("scale_up", 0))     for r in rounds
                            if "Injection" in str(r.get("action", ""))),
            "ev":       float(state.get("cumulative_ev", 0)),
            "fix_c":    float(state.get("fix_c", 0)),
        }
        if any([s["shannon"], s["hedge"], s["surplus"], s["scale_up"],
                s["harvest"], s["ev"], s["fix_c"]]):
            ticker_stats.append(s)

    if not ticker_stats:
        st.info("ยังไม่มี Round data — กรุณา Run Chain Round ก่อน")
        _render_ticker_state_overview(tickers_list, data); return

    all_names = [s["ticker"] for s in ticker_stats]
    selected  = st.multiselect("🔍 เลือก Ticker (ว่าง = ทั้งหมด)", all_names,
                                default=[], key="sankey_ticker_filter")
    active    = [s for s in ticker_stats if s["ticker"] in selected] if selected else ticker_stats

    mode = st.radio("โหมด", ["📊 ภาพรวมพอร์ต", "🎯 รายละเอียด by Ticker"],
                    horizontal=True, key="sankey_mode")

    if "ภาพรวม" in mode:
        _render_sankey_aggregated(active, data)
    else:
        _render_sankey_per_ticker(active)

    st.divider()
    show_t = (tickers_list if not selected
              else [t for t in tickers_list if t.get("ticker") in [s["ticker"] for s in active]])
    _render_ticker_state_overview(show_t, data)


def _render_sankey_aggregated(ticker_stats: list, data: dict):
    pool_cf = data.get("global_pool_cf", 0.0)
    n       = len(ticker_stats)
    SH, HA, HG, SU, SC, PL, EV = n, n+1, n+2, n+3, n+4, n+5, n+6
    TCOLORS = ["#22d3ee","#fbbf24","#34d399","#f472b6","#60a5fa",
               "#a78bfa","#fb923c","#4ade80","#e879f9","#f87171",
               "#38bdf8","#facc15","#a3e635","#c084fc","#fb7185","#67e8f9"]
    node_labels = [s["ticker"] for s in ticker_stats] + [
        "Shannon Engine","Harvest Income","Hedge Costs 🛡️","Net Surplus",
        "Scale Up 🚀", f"Pool CF 🎱 ${pool_cf:,.0f}", "Ev Burn 🔥"
    ]
    node_colors = [TCOLORS[i % len(TCOLORS)] for i in range(n)] + [
        "#94a3b8","#10b981","#ef4444","#f472b6","#22c55e","#3b82f6","#dc2626"
    ]
    src, tgt, val, lnk = [], [], [], []

    def lk(s, t, v, label=""):
        if v > 0.001: src.append(s); tgt.append(t); val.append(round(v,2)); lnk.append(label)

    tot_sh = tot_ha = tot_hg = tot_sc = tot_ev = 0
    for i, s in enumerate(ticker_stats):
        if s["shannon"] > 0: lk(i, SH, s["shannon"], f"{s['ticker']} Shannon"); tot_sh += s["shannon"]
        if s["harvest"] > 0: lk(i, HA, s["harvest"], f"{s['ticker']} Harvest"); tot_ha += s["harvest"]
        tot_hg += s["hedge"]; tot_sc += s["scale_up"]; tot_ev += s["ev"]

    net_s = tot_sh - tot_hg
    if tot_hg > 0 and tot_sh > 0: lk(SH, HG, min(tot_hg, tot_sh), "Hedge Premium")
    if net_s > 0:                  lk(SH, SU, net_s, "Surplus")
    if tot_ha > 0:                 lk(HA, PL, tot_ha, "Harvest → Pool CF")
    if tot_sc > 0:                 lk(SU, SC, min(tot_sc, max(0, net_s)), "Auto Scale Up")
    if net_s - tot_sc > 0:        lk(SU, PL, net_s - tot_sc, "Overflow → Pool CF")
    if tot_ev > 0 and tot_sc > 0: lk(SC, EV, min(tot_ev, tot_sc), "Ev Theta Decay")

    if not val:
        st.info("ยังไม่มี Flow — กรุณา Run Chain Round ก่อน"); return

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=18, thickness=22, line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                  label=node_labels, color=node_colors),
        link=dict(source=src, target=tgt, value=val, label=lnk,
                  color=["rgba(148,163,184,0.35)"] * len(val))
    )])
    fig.update_layout(title_text="📊 Capital Flow — Portfolio Aggregated",
                      font=dict(size=12, color="#e2e8f0"),
                      paper_bgcolor="#0f172a", height=600)
    st.plotly_chart(fig, use_container_width=True)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Shannon",     f"${tot_sh:,.0f}")
    m2.metric("Harvest",     f"${tot_ha:,.0f}")
    m3.metric("Hedge 🛡️",   f"${tot_hg:,.0f}", delta_color="inverse")
    m4.metric("Net Surplus", f"${net_s:,.0f}",
              delta_color="normal" if net_s >= 0 else "inverse")
    m5.metric("Scale Up 🚀", f"${tot_sc:,.0f}")
    m6.metric("Ev Burn 🔥",  f"${tot_ev:,.0f}", delta_color="inverse")


def _render_sankey_per_ticker(ticker_stats: list):
    if len(ticker_stats) > 8:
        st.warning("⚠️ เลือก ≤ 8 Ticker"); ticker_stats = ticker_stats[:8]
    TCOLORS = ["#22d3ee","#fbbf24","#34d399","#f472b6","#60a5fa","#a78bfa","#fb923c","#4ade80"]
    cols    = st.columns(min(2, len(ticker_stats)))

    for i, s in enumerate(ticker_stats):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"#### {s['ticker']}")
                NL = [s["ticker"],"Shannon","Hedge 🛡️","Surplus","Scale Up 🚀","Ev Burn 🔥","Pool CF 🎱"]
                NC = [TCOLORS[i%len(TCOLORS)],"#94a3b8","#ef4444","#f472b6","#22c55e","#dc2626","#3b82f6"]
                src2,tgt2,val2,clr2 = [],[],[],[]

                def tl(s_,t_,v_,c_="rgba(148,163,184,0.5)"):
                    if v_>0.001: src2.append(s_);tgt2.append(t_);val2.append(round(v_,2));clr2.append(c_)

                if s["shannon"] > 0:
                    tl(0,1,s["shannon"])
                    if s["hedge"] > 0: tl(1,2,min(s["hedge"],s["shannon"]),"rgba(239,68,68,0.5)")
                    ns = s["shannon"]-s["hedge"]
                    if ns > 0:
                        tl(1,3,ns,"rgba(244,114,182,0.5)")
                        if s["scale_up"] > 0: tl(3,4,min(s["scale_up"],ns),"rgba(34,197,94,0.5)")
                        if ns-s["scale_up"]>0: tl(3,6,ns-s["scale_up"],"rgba(59,130,246,0.4)")
                if s["injection"]>0: tl(6,4,s["injection"],"rgba(34,197,94,0.4)")
                if s["ev"]>0 and s["fix_c"]>0: tl(4,5,min(s["ev"],s["fix_c"]),"rgba(220,38,38,0.4)")

                if not val2:
                    st.caption("ยังไม่มี Round data"); st.metric("fix_c",f"${s['fix_c']:,.0f}"); continue

                fig_t = go.Figure(data=[go.Sankey(
                    node=dict(pad=12,thickness=18,line=dict(color="rgba(255,255,255,0.1)",width=0.5),
                              label=NL,color=NC),
                    link=dict(source=src2,target=tgt2,value=val2,color=clr2)
                )])
                fig_t.update_layout(font=dict(size=10,color="#e2e8f0"),
                                    paper_bgcolor="#1e293b",
                                    margin=dict(l=10,r=10,t=30,b=10),height=350)
                st.plotly_chart(fig_t, use_container_width=True)
                mc1,mc2,mc3 = st.columns(3)
                mc1.metric("Shannon",f"${s['shannon']:,.0f}")
                mc2.metric("fix_c",  f"${s['fix_c']:,.0f}")
                mc3.metric("Ev Burn",f"${s['ev']:,.0f}",delta_color="inverse")


def _render_ticker_state_overview(tickers_list: list, data: dict):
    st.markdown("##### 📋 Current State Overview")
    rows = []
    for t in tickers_list:
        state  = t.get("current_state", {})
        rounds = t.get("rounds", [])
        rows.append({
            "Ticker":   t.get("ticker","???"),
            "Price":    f"${state.get('price',0):,.2f}",
            "fix_c":    f"${state.get('fix_c',0):,.0f}",
            "Baseline": f"${state.get('baseline',0):,.2f}",
            "Ev Burn":  f"${state.get('cumulative_ev',0):,.2f}",
            "Net P&L":  f"${state.get('net_pnl',0):,.2f}",
            "Σ Shannon":f"${sum(float(r.get('shannon_profit',0)) for r in rounds):,.2f}",
            "Σ Hedge":  f"${sum(float(r.get('hedge_cost',0)) for r in rounds):,.2f}",
            "Rounds":   len(rounds),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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
                    else:
                        st.error("Ticker นี้มีอยู่แล้วในพอร์ต")

    with st.expander("💾 Export / Import Data", expanded=False):
        st.markdown("##### 📤 Export Data (ดาวน์โหลดข้อมูล)")
        export_str = json.dumps(data, indent=2, ensure_ascii=False)
        st.download_button(
            label="💾 Download Data as JSON",
            data=export_str,
            file_name=f"chain_system_backup_{datetime.now().strftime('%Y-%m-%d')}.json",
            mime="application/json"
        )
        
        st.divider()
        st.markdown("##### 📥 Import Data (นำเข้าข้อมูล)")
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ JSON ที่ได้จากการ Export", type=["json"])
        if uploaded_file is not None:
            try:
                uploaded_data = json.load(uploaded_file)
                if isinstance(uploaded_data, dict) and "tickers" in uploaded_data:
                    st.warning("⚠️ การ Import จะเป็นการ **เขียนทับข้อมูลปัจจุบัน** ทั้งหมด คุณแน่ใจหรือไม่?")
                    if st.button("✅ Confirm Import", type="primary"):
                        save_trading_data(uploaded_data)
                        st.success("นำเข้าข้อมูลสำเร็จ! ระบบกำลังรีโหลด...")
                        st.rerun()
                else:
                    st.error("❌ ไฟล์ JSON ไม่ถูกต้อง หรือไม่มีโครงสร้างข้อมูลของระบบ Chain (ขาดคีย์ 'tickers')")
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

    with st.expander("⚠️ ลบข้อมูลทั้งหมด", expanded=False):
        if st.button("DELETE ALL DATA", type="primary"):
            data.update({"tickers": [], "global_pool_cf": 0.0, "global_ev_reserve": 0.0, "treasury_history": []})
            save_trading_data(data)
            st.warning("All data cleared!")
            st.rerun()

if __name__ == "__main__":
    main()
