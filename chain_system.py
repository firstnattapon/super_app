
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime

from flywheels import (
    load_trading_data, save_trading_data, get_tickers,
    run_chain_round, commit_round, allocate_pool_funds,
    parse_final, parse_beta_numbers, parse_beta_net,
    parse_surplus_iv, get_rollover_history, build_portfolio_df,
    black_scholes, sanitize_number_str,
)
 

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

    data = load_trading_data()
    tickers_list = get_tickers(data)

    # REFACTORED: Combined Engine & History into one tab
    tab1, tab2, tab4, tab5 = st.tabs([
        "⚡ Active Dashboard",
        "⚡ Engine & History",
        "Payoff Profile 🔗 Run Chain Round",
        "➕ Manage Data"
    ])

    with tab1:
        _render_active_dashboard(data)
    with tab2:
        _render_engine_tab(data)
    with tab4:
        _render_payoff_profile_tab(data)
    with tab5:
        _render_manage_data(data)


# ----------------------------------------------------------
# TAB: Active Dashboard
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
    total_burn = sum(
        t_data.get("current_state", {}).get("cumulative_ev", 0.0) for t_data in tickers_list
    )

    st.subheader("📊 Portfolio Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fix_C (Deployed)", f"${total_c:,.0f}", f"{len(tickers_list)} tickers")
    m2.metric("🎱 Pool CF (War Chest)", f"${pool_cf:,.2f}")
    m3.metric("🔥 Burn Rate (Cum. Ev)", f"${total_burn:,.2f}", delta="Cost of Business", delta_color="inverse")
    m4.metric("💰 Net Reality", f"${total_net:,.2f}",
              delta=f"Lock {total_lock:,.0f} + IV {total_surplus:,.0f} + Ev {total_ev:,.0f}",
              delta_color="normal" if total_net >= 0 else "inverse")

    # ── Ev Efficiency Analysis ──
    gross_profit = total_lock + total_surplus
    efficiency = (gross_profit / total_burn) if total_burn > 0 else 0.0
    
    st.markdown("#### ⏱️ Ev Efficiency (Winning against Time)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Profit (No Ev)", f"${gross_profit:,.2f}", "Harvest + Shannon")
    c2.metric("Total Burn (Cost)", f"${total_burn:,.2f}", "Cumulative Theta Decay")
    c3.metric("Ev Efficiency Ratio", f"{efficiency:.2f}x", 
              delta="Sustainable" if efficiency >= 1.0 else "Bleeding",
              delta_color="normal" if efficiency >= 1.0 else "inverse")
    c4.caption(f"""
    **Ratio > 1.0** = กำไรจากระบบชนะค่าเช่า (Time Decay)
    **Ratio < 1.0** = ยังต้องการ Direction ช่วย
    """)

    st.divider()

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

    st.subheader("Net P&L per Ticker")
    colors = ["#00c853" if v >= 0 else "#ff1744" for v in df["Net"]]
    fig_bar = go.Figure(data=[go.Bar(
        x=df["Ticker"], y=df["Net"], marker_color=colors,
        text=[f"${v:,.0f}" for v in df["Net"]], textposition="outside",
    )])
    fig_bar.update_layout(title="Net P&L = Ev + Lock P&L (per ticker)",
        xaxis_title="Ticker", yaxis_title="Net P&L ($)", height=400, plot_bgcolor="rgba(0,0,0,0)")
    fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Waterfall: Ev → Lock P&L → Net")
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
# TAB: ⚡ Engine (Run Round + Pool CF) & History Consolidated
# ----------------------------------------------------------
def _render_engine_tab(data):
    """Unified Engine: Run Chain Round (left) + Pool CF (right) + History (Bottom)."""
    tickers_list = get_tickers(data)
    pool_cf = data.get("global_pool_cf", 0.0)
    ev_reserve = data.get("global_ev_reserve", 0.0)

    # ── Top bar: Global Stats ──
    with st.container(border=True):
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("🎱 Pool CF (War Chest)", f"${pool_cf:,.2f}")
        top2.metric("🛡️ Pool LEAPS", f"${ev_reserve:,.2f}")
        top3.metric("Tickers", str(len(tickers_list)))
        total_rounds = sum(len(t.get("rounds", [])) for t in tickers_list)
        top4.metric("Total Rounds", str(total_rounds))

    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Manage Data ก่อน")
        return

    # ── Two-column layout: Run Round | Pool CF ──
    col_left, col_right = st.columns([3, 2], gap="large")

    # ==============================
    # LEFT COLUMN — Run Chain Round
    # ==============================
    with col_left:
        st.subheader("🔗 Run Chain Round")
        st.caption("เลือก Ticker → ใส่ราคาปัจจุบัน → ระบบคำนวณอัตโนมัติ")

        ticker_names = [d.get("ticker", "???") for d in tickers_list]
        selected = st.selectbox("เลือก Ticker", ticker_names, key="run_round_ticker")
        idx = ticker_names.index(selected)
        t_data = tickers_list[idx]
        state = t_data.get("current_state", {})

        with st.container(border=True):
            st.caption(f"🔵 สถานะปัจจุบัน — {selected}")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("fix_c", f"${state.get('fix_c', 0):,.2f}")
            sc2.metric("Price (t)", f"${state.get('price', 0):,.2f}")
            sc3.metric("Baseline (b)", f"${state.get('baseline', 0):,.2f}")
            sc4.metric("Rounds", str(len(t_data.get("rounds", []))))

        settings = data.get("settings", {})
        default_sigma = settings.get("default_sigma", 0.5)
        default_hr = settings.get("default_hedge_ratio", 2.0)

        with st.form("run_round_form", clear_on_submit=False):
            st.markdown("##### 📊 Input — ราคาและ Config")
            r1, r2, r3 = st.columns(3)
            with r1:
                p_new = st.number_input(
                    f"ราคาใหม่ P (ปัจจุบัน t = ${state.get('price', 0):.2f})",
                    min_value=0.01, value=round(state.get("price", 10.0) * 1.1, 2), step=1.0, key="rr_pnew")
            with r2:
                sigma = st.number_input("Volatility (σ)", min_value=0.05, value=default_sigma, step=0.05, key="rr_sigma")
            with r3:
                hedge_ratio = st.number_input("Hedge Ratio (x Put)", min_value=0.0, value=default_hr, step=0.5, key="rr_hr")
            preview_btn = st.form_submit_button("🔍 Preview Calculation")

        if preview_btn and p_new > 0:
            preview = run_chain_round(state, p_new, sigma, hedge_ratio)
            if preview is None:
                st.error("Invalid price — cannot run round")
                return
            st.session_state["_pending_round"] = preview
            st.session_state["_pending_ticker_idx"] = idx
            st.session_state["_pending_ticker_name"] = selected

        if "_pending_round" in st.session_state and st.session_state.get("_pending_ticker_name") == selected:
            rd = st.session_state["_pending_round"]
            st.markdown("---")
            st.subheader("📋 Preview — ผลลัพธ์ก่อน Commit")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Shannon Profit", f"${rd['shannon_profit']:,.2f}", delta=f"P: {rd['p_old']} → {rd['p_new']}")
            p2.metric("Harvest Profit", f"${rd['harvest_profit']:,.2f}", delta=f"σ={rd['sigma']}")
            p3.metric("Hedge Cost", f"${rd['hedge_cost']:,.2f}", delta=f"-{rd['hedge_ratio']}x Put", delta_color="inverse")
            p4.metric("Surplus (Free Risk)", f"${rd['surplus']:,.2f}",
                       delta="Scale Up!" if rd['surplus'] > 0 else "Deficit",
                       delta_color="normal" if rd['surplus'] > 0 else "inverse")

            st.caption("ผลลัพธ์ของ Round นี้:")
            p5, p6, p7, p8 = st.columns(4)
            p5.metric("c Before", f"${rd['c_before']:,.0f}")
            p6.metric("c After", f"${rd['c_after']:,.2f}",
                       delta=f"+${rd['scale_up']:,.2f}" if rd['scale_up'] > 0 else "No change")
            p7.metric("Price (New)", f"${rd['p_new']:,.2f}")
            p8.metric("Baseline (New)", f"${rd['b_after']:,.2f}")

            if st.button("✅ Commit Round — บันทึกถาวร", type="primary", key="commit_round"):
                commit_round(data, st.session_state["_pending_ticker_idx"], rd)
                del st.session_state["_pending_round"]
                del st.session_state["_pending_ticker_idx"]
                del st.session_state["_pending_ticker_name"]
                st.success(f"✅ Round committed for {selected}! fix_c = ${rd['c_after']:,.2f}, b = ${rd['b_after']:,.2f}")
                st.rerun()

    # ==============================
    # RIGHT COLUMN — Pool CF & Allocation
    # ==============================
    with col_right:
        st.subheader("🎱 Pool CF & Allocation")
        st.caption("เติมเงิน (Funding) หรือ เบิกจ่าย (Deploy)")

        # ── 1. Add to Pool (Funding) ──
        with st.form("add_pool_cf_form", clear_on_submit=True):
            st.markdown("##### 📥 Add Funding to Pool")
            c1, c2 = st.columns([2, 1])
            with c1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=100.0, key="add_pool_amt")
            with c2:
                st.write("")
                st.write("")
                btn_add = st.form_submit_button("💰 Add Fund", type="primary")
            
            if btn_add and amount > 0:
                data["global_pool_cf"] = data.get("global_pool_cf", 0) + amount
                note = "Funding Injection"
                save_trading_data(data)
                st.success(f"✅ +${amount:,.2f} → Pool CF = ${data['global_pool_cf']:,.2f}")
                st.rerun()

        st.divider()

        # ── 2. Manual Round Injection (Deployment) ──
        if tickers_list:
            st.markdown("##### 🚀 Manual Round Injection (Deploy)")
            st.caption("ยิง `fix_c` หรือจ่าย Ev เข้า Ticker โดยตรง")

            ticker_names_deploy = [d.get("ticker", "???") for d in tickers_list]
            deploy_ticker = st.selectbox("Select Ticker", ticker_names_deploy, key="deploy_ticker")
            
            d_idx = ticker_names_deploy.index(deploy_ticker)
            t_data_deploy = tickers_list[d_idx]
            state_deploy = t_data_deploy.get("current_state", {})
            cur_c = state_deploy.get("fix_c", 0)
            cur_t = state_deploy.get("price", 0)
            cur_b = state_deploy.get("baseline", 0)
            cur_ev_debt = state_deploy.get("cumulative_ev", 0.0)

            with st.form("deploy_round_form", clear_on_submit=False):
                action_type = st.selectbox("Action", [
                    "📈 Scale Up (เพิ่มทุน fix_c)",
                    "🛡️ Buy Puts (ซื้อประกัน)",
                    "🎯 Buy Calls (เพิ่มของ/Speculate)",
                    "⏳ Pay Ev (จ่ายค่าเช่า/ลด Burn Rate)"
                ], key="deploy_action_round")

                d_amt = st.number_input(
                    f"Deploy Amount ($)", 
                    min_value=0.0, max_value=float(pool_cf) if pool_cf > 0 else 0.0,
                    value=0.0, step=100.0, key="deploy_amt_round"
                )
                d_note = st.text_input("Note", value="", placeholder="Reason...")
                
                submitted_deploy = st.form_submit_button("🔍 Preview Injection")
            
            if submitted_deploy and d_amt > 0:
                mock_scale_up = 0.0
                mock_new_c = cur_c
                mock_ev_change = 0.0
                note_prefix = ""

                if "Scale Up" in action_type:
                    mock_scale_up = d_amt
                    mock_new_c = cur_c + d_amt
                    note_prefix = "[Scale Up]"
                elif "Buy Puts" in action_type:
                    note_prefix = "[Buy Puts]"
                elif "Buy Calls" in action_type:
                    note_prefix = "[Buy Calls]"
                elif "Pay Ev" in action_type:
                    mock_ev_change = -d_amt
                    note_prefix = "[Pay Ev]"

                final_note = f"{note_prefix} {d_note}".strip()
                
                st.info(f"💡 **Preview:** {action_type} ${d_amt:,.2f} to {deploy_ticker}")
                
                injection_round = {
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "action": "Injection",
                    "p_old": cur_t,
                    "p_new": cur_t,
                    "c_before": cur_c,
                    "c_after": mock_new_c,
                    "shannon_profit": 0.0,
                    "harvest_profit": 0.0,
                    "hedge_cost": 0.0,
                    "surplus": d_amt if mock_scale_up > 0 else -d_amt,
                    "scale_up": mock_scale_up,
                    "b_before": cur_b,
                    "b_after": cur_b,
                    "hedge_ratio": 0.0,
                    "sigma": 0.0,
                    "note": final_note,
                    "ev_change": mock_ev_change
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
                    dc3.metric("Burn Rate (Ev)", f"${cur_ev_debt:,.2f} → ${max(0, cur_ev_debt - st.session_state['_pending_injection_amt']):,.2f}", delta="Reduced Debt")
                
                if st.button("🚀 Confirm Transaction", type="primary", key="confirm_deploy"):
                    amt = st.session_state["_pending_injection_amt"]
                    if data["global_pool_cf"] >= amt:
                        data["global_pool_cf"] -= amt
                        if "Pay Ev" in p_type:
                             t_data_deploy["current_state"]["cumulative_ev"] = max(0.0, cur_ev_debt - amt)
                        commit_round(data, d_idx, p_inj)
                        del st.session_state["_pending_injection"]
                        del st.session_state["_pending_injection_idx"]
                        del st.session_state["_pending_injection_amt"]
                        del st.session_state["_pending_injection_type"]
                        st.success(f"✅ Transaction Complete: {p_type} ${amt:,.2f}!")
                        st.rerun()
                    else:
                        st.error("❌ Pool fund insufficient")
        
        st.divider()

        # ── 3. Pool LEAPS (Balance & Expenses) ──
        with st.expander("🛡️ Manage Pool LEAPS (Income & Expenses)"):
            st.caption("จัดการกองทุน LEAPS: รับเงินจาก Pool CF หรือ จ่ายค่า LEAPS")
            
            # Allocation (Income)
            st.markdown("##### 📥 Allocate (Income from Pool CF)")
            col_a, col_b = st.columns(2)
            with col_a:
                alloc_amt = st.number_input("Allocate Amount ($)", min_value=0.0, max_value=float(pool_cf), step=100.0, key="alloc_leaps")
            with col_b:
                if st.button("📥 Allocate to Pool LEAPS"):
                    if alloc_amt > 0 and pool_cf >= alloc_amt:
                        data["global_pool_cf"] -= alloc_amt
                        data["global_ev_reserve"] = data.get("global_ev_reserve", 0.0) + alloc_amt
                        save_trading_data(data)
                        st.success(f"Allocated ${alloc_amt:,.2f} to Pool LEAPS")
                        st.rerun()
            
            st.divider()
            
            # Expenses (Pay LEAPS)
            st.markdown("##### 📤 Pay LEAPS (Expense)")
            col_c, col_d = st.columns(2)
            with col_c:
                pay_leaps_amt = st.number_input("LEAPS Cost ($)", min_value=0.0, max_value=float(ev_reserve), step=100.0, key="pay_leaps_cost")
            with col_d:
                if st.button("📤 Pay LEAPS (Deduct)"):
                    if pay_leaps_amt > 0 and ev_reserve >= pay_leaps_amt:
                        data["global_ev_reserve"] -= pay_leaps_amt
                        save_trading_data(data)
                        st.success(f"Paid LEAPS Cost ${pay_leaps_amt:,.2f} from Pool")
                        st.rerun()
            
            st.markdown(f"**Current Pool LEAPS Balance:** `${ev_reserve:,.2f}`")

    # ==============================
    # BOTTOM SECTION — Consolidated History
    # ==============================
    st.divider()
    _render_consolidated_history(t_data)


def _render_consolidated_history(t_data):
    """
    Combined History View:
    Table only (Graphs moved to Payoff Profile).
    """
    ticker = t_data.get("ticker", "???")
    rounds = t_data.get("rounds", [])

    st.subheader(f"📜 {ticker} — Operations History")

    if not rounds:
        # Fallback to Legacy if no structured rounds
        legacy_hist = get_rollover_history(t_data)
        if legacy_hist:
            st.warning("⚠️ Showing Legacy History (No structured rounds yet). Run a round to upgrade.")
            st.dataframe(pd.DataFrame(legacy_hist))
        else:
            st.info("No history available yet.")
        return
    
    table_rows = []

    for i, rd in enumerate(rounds):
        hedge = rd.get("hedge_cost", 0.0)
        
        # ── Calculate Real Ev Burn (Time Value Only) ──
        # Intrinsic = Max(0, Strike - Price) * Qty
        p_old = rd.get("p_old", 0)
        p_new = rd.get("p_new", 0)
        c_before = rd.get("c_before", 0)
        h_ratio = rd.get("hedge_ratio", 2.0)
        
        # Estimate Strike if not saved (Standard 0.9x)
        strike = rd.get("strike_price", p_old * 0.9 if p_old > 0 else 0)
        qty = (c_before / p_old * h_ratio) if p_old > 0 else 0
        
        intrinsic_val = max(0, strike - p_new) * qty
        ev_burn = max(0, hedge - intrinsic_val) # Pure Time Value Burn

        # Check action type
        ev_impact = 0.0
        ev_label = "—"
        
        if "Pay Ev" in rd.get("note", "") or rd.get("ev_change", 0) < 0:
             payment = abs(rd.get("ev_change", 0))
             ev_impact = -payment
             ev_label = f"🟢 Paid -${payment:,.0f}"
        elif hedge > 0:
             ev_impact = ev_burn
             ev_label = f"🔴 Burn -${ev_burn:,.0f}"
             if intrinsic_val > 0:
                 ev_label += f" (In: ${intrinsic_val:,.0f})"

        action_short = rd.get("action", "Round")
        if rd.get("scale_up", 0) > 0:
            action_short = f"Scale +${rd['scale_up']:,.0f}"
        if "Injection" in rd.get("action", ""):
            action_short = "Inject/Deploy"
            
        table_rows.append({
            "Date": rd.get("date", "")[:10],  # Short date
            "Action": action_short,
            "Price": f"${rd.get('p_new', 0):,.2f}",
            "fix_c": f"${rd.get('c_after', 0):,.0f}",
            "b": f"${rd.get('b_after', 0):,.2f}",
            "Ev Impact": ev_label,
            "Net Result": f"${rd.get('surplus', 0):,.2f}",
            "Note": rd.get("note", "")
        })

    # Render Table
    st.markdown("##### 🧾 Detailed Log")
    df_table = pd.DataFrame(table_rows[::-1]) # Reverse order (Newest first)
    
    st.dataframe(
        df_table, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Ev Impact": st.column_config.TextColumn("Ev Impact", width="medium", help="Red = Burn (Time Value), Green = Pay/Recover"),
            "Net Result": st.column_config.TextColumn("Surplus", help="Positive = Scale Up, Negative = Cost")
        }
    )


# ----------------------------------------------------------
# TAB: Payoff Profile 🔗 Run Chain Round
# ----------------------------------------------------------
def _render_payoff_profile_tab(data):
    """
    Visual Analytics Hub:
    1. History Graphs (Ev Battle, Growth) - Linked to Selected Ticker
    2. Payoff Profile & Sankey (Simulation/Projections)
    """
    tickers_list = get_tickers(data)
    
    # Try to get selected ticker from Session State
    selected_ticker = st.session_state.get("run_round_ticker")
    t_data = None
    
    if selected_ticker:
        for t in tickers_list:
            if t["ticker"] == selected_ticker:
                t_data = t
                break
    
    if not t_data:
        st.info("👈 Please select a Ticker in the 'Engine & History' tab first.")
        if tickers_list:
            t_data = tickers_list[0]
            st.caption(f"Showing default: {t_data['ticker']}")
        else:
            return

    # 📊 SECTION 1: HISTORY GRAPHS (Moved from Engine)
    st.subheader(f"📈 History Analysis: {t_data['ticker']}")
    
    rounds = t_data.get("rounds", [])
    if rounds:
        cum_ev_burn = 0.0
        cum_profit = 0.0
        chart_data = []

        for i, rd in enumerate(rounds):
            hedge = rd.get("hedge_cost", 0.0)
            shannon = rd.get("shannon_profit", 0)
            harvest = rd.get("harvest_profit", 0)
            total_income = shannon + harvest
            
            # ── Calculate Ev Burn (Logic Match) ──
            p_old = rd.get("p_old", 0)
            p_new = rd.get("p_new", 0)
            c_before = rd.get("c_before", 0)
            h_ratio = rd.get("hedge_ratio", 2.0)
            strike = rd.get("strike_price", p_old * 0.9 if p_old > 0 else 0)
            qty = (c_before / p_old * h_ratio) if p_old > 0 else 0
            
            intrinsic_val = max(0, strike - p_new) * qty
            ev_burn = max(0, hedge - intrinsic_val)

            if hedge > 0:
                 cum_ev_burn += ev_burn  # Track ONLY extrinsic burn
            
            if total_income > 0:
                cum_profit += total_income
                
            chart_data.append({
                "Round": i+1,
                "Burn": cum_ev_burn,
                "Profit": cum_profit,
                "c": rd.get("c_after", 0),
                "b": rd.get("b_after", 0)
            })
            
        df_chart = pd.DataFrame(chart_data)

        c1, c2 = st.columns(2)
        with c1:
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(x=df_chart["Round"], y=df_chart["Burn"], 
                                        mode='lines', name='Cum. Ev Burn', line=dict(color='#ff1744', width=2)))
            fig_ev.add_trace(go.Scatter(x=df_chart["Round"], y=df_chart["Profit"], 
                                        mode='lines', name='Cum. Harvest', line=dict(color='#00c853', width=2)))
            fig_ev.update_layout(title="⚔️ Ev Battle: Cost vs. Harvest", 
                                 xaxis_title="Round", yaxis_title="Cumulative ($)", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_ev, use_container_width=True)

        with c2:
            fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
            fig_evo.add_trace(go.Scatter(x=df_chart["Round"], y=df_chart["c"], 
                                         name="fix_c", line=dict(color='#ff9800')), secondary_y=False)
            fig_evo.add_trace(go.Scatter(x=df_chart["Round"], y=df_chart["b"], 
                                         name="Baseline", line=dict(color='#2196f3', dash='dot')), secondary_y=True)
            fig_evo.update_layout(title="📈 Growth Evolution (c & b)", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.warning("No history data to plot.")

    st.divider()

    # 📊 SECTION 2: PAYOFF PROFILE
    st.subheader(f"📐 Payoff Profile Simulator")
    st.caption("จำลองโครงสร้างผลตอบแทน (อ้างอิงสถานะปัจจุบันได้)")

    cur_state = t_data.get("current_state", {})
    def_c = cur_state.get("fix_c", 10000.0)
    def_p = cur_state.get("price", 100.0)
    
    col1, col2 = st.columns(2)
    with col1:
        fix_c = st.number_input("Fixed Capital ($)", 1000.0, 1000000.0, float(def_c), 1000.0, key="chain_c_payoff")
        P0 = st.number_input("Current Price ($)", 0.1, 10000.0, float(def_p), 1.0, key="chain_p0_payoff")
        sigma = st.slider("Volatility (σ)", 0.1, 2.0, 0.5, 0.1, key="chain_sig_payoff")

    with col2:
        hedge_ratio = st.slider("Hedge Ratio (contracts/fix_c unit)", 0.1, 3.0, 2.0, 0.1, key="chain_hr_payoff")
        qty_puts = (fix_c / P0) * hedge_ratio
        
        crash_price_pct = st.slider("Simulate Crash Price (%)", 10, 100, 50, 5, key="chain_crash_pct")
        P_crash = P0 * (crash_price_pct / 100.0)
        st.metric("Crash Price Scenario", f"${P_crash:.1f}")

    r = 0.04
    T = 1.0
    put_strike_pct = 0.9
    put_strike = P0 * put_strike_pct
    put_premium = black_scholes(P0, put_strike, T, r, sigma, 'put')
    cost_hedge = qty_puts * put_premium
    harvest_profit = fix_c * 0.5 * (sigma ** 2) * T
    surplus = harvest_profit - cost_hedge

    prices = np.linspace(P0 * 0.2, P0 * 1.5, 200)
    base_80_20 = fix_c * np.log(prices / P0)
    vol_premium = fix_c * 0.5 * (sigma ** 2) * T
    dynamic_shield = base_80_20 + vol_premium
    put_val = qty_puts * np.maximum(0, put_strike - prices)
    shielded_80_20 = dynamic_shield + put_val - cost_hedge

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=prices, y=base_80_20, name="Base 80/20 (Unhedged)",
        line=dict(width=2, color='#ff9800')))
    fig_payoff.add_trace(go.Scatter(x=prices, y=dynamic_shield, name="Dynamic (+Vol)",
        line=dict(width=2, color='#2196f3', dash='dash')))
    fig_payoff.add_trace(go.Scatter(x=prices, y=shielded_80_20, name="Shielded (+Puts)",
        line=dict(width=3, color='#00c853')))
    fig_payoff.add_vline(x=P_crash, line_dash="dash", line_color="red",
                         annotation_text=f"Crash {P_crash:.1f}")
    fig_payoff.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig_payoff.update_layout(
        title="Payoff Profile (4-Line Model)",
        xaxis_title="Price ($)", yaxis_title="P&L ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=500,
    )
    st.plotly_chart(fig_payoff, use_container_width=True)

    st.divider()

    st.subheader("💧 Capital Flow (Sankey)")
    
    put_payoff_crash = max(0, put_strike - P_crash)
    total_put_payoff = qty_puts * put_payoff_crash
    
    new_strike_crash = P_crash * put_strike_pct
    rolldown_premium = black_scholes(P_crash, new_strike_crash, T, r, sigma, 'put')
    rolldown_cost = qty_puts * rolldown_premium
    
    deploy_ratio = 0.7
    pool_cf_net = total_put_payoff - rolldown_cost
    deploy_amount = pool_cf_net * deploy_ratio if pool_cf_net > 0 else 0
    reserve_amount = pool_cf_net * (1 - deploy_ratio) if pool_cf_net > 0 else 0

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
    fig_sankey.update_layout(title="Full Cycle Flow", height=400)
    st.plotly_chart(fig_sankey, use_container_width=True)


# ----------------------------------------------------------
# TAB: Manage Data
# ----------------------------------------------------------
def _render_manage_data(data):
    st.subheader("จัดการข้อมูลพอร์ต")
    
    # Add new Ticker
    with st.expander("➕ เพิ่ม Ticker ใหม่", expanded=False):
        with st.form("add_ticker_form"):
            new_ticker = st.text_input("Ticker Symbol").upper()
            init_price = st.number_input("Initial Price (t)", min_value=0.01, value=100.0)
            init_c = st.number_input("Initial Fix_C ($)", min_value=1000.0, value=10000.0)
            
            if st.form_submit_button("Add Ticker"):
                if new_ticker and init_price > 0 and init_c > 0:
                    # Check if exists
                    existing = [d["ticker"] for d in get_tickers(data)]
                    if new_ticker in existing:
                        st.error("Ticker มีอยู่แล้ว")
                    else:
                        new_entry = {
                            "ticker": new_ticker,
                            "Final": f"{init_price}, {init_c}, 0.0",
                            "current_state": {
                                "price": init_price,
                                "fix_c": init_c,
                                "baseline": 0.0,
                                "cumulative_ev": 0.0
                            },
                            "rounds": []
                        }
                        data["tickers"].append(new_entry)
                        save_trading_data(data)
                        st.success(f"Added {new_ticker}!")
                        st.rerun()

    st.divider()
    
    # Manage Existing
    tickers = get_tickers(data)
    if tickers:
        st.write("##### รายชื่อ Tickers")
        for i, t in enumerate(tickers):
            with st.expander(f"{t['ticker']}"):
                if st.button(f"🗑️ ลบ {t['ticker']}", key=f"del_{i}"):
                    data["tickers"].pop(i)
                    save_trading_data(data)
                    st.success(f"Deleted {t['ticker']}")
                    st.rerun()
