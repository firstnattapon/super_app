import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Chain System - Flywheel Shannon's Demon", layout="wide")

from flywheels import (
    load_trading_data, save_trading_data, get_tickers,
    run_chain_round, commit_round,
    parse_beta_net, build_portfolio_df, get_rollover_history,
    chapter_0_introduction, chapter_1_baseline, chapter_2_volatility_harvest,
    chapter_3_convexity_engine, chapter_4_black_swan_shield,
    chapter_5_dynamic_scaling, chapter_6_synthetic_dividend,
    chapter_7_collateral_magic, master_study_guide_quiz,
    paper_trading_workshop, glossary_section
)

def log_treasury_event(data, category, amount, note=""):
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

    tab1, tab2, tab4, tab5 = st.tabs([
        "⚡ Active Dashboard",
        "⚡ Engine & History",
        "Payoff Profile 🔗 Run Chain Round",
        "➕ Manage Data"
    ])

    with tab1: _render_active_dashboard(data)
    with tab2: _render_engine_tab(data)
    with tab4: _render_payoff_profile_tab(data)
    with tab5: _render_manage_data(data)

def _render_active_dashboard(data):
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
    total_burn = sum(t_data.get("current_state", {}).get("cumulative_ev", 0.0) for t_data in tickers_list)

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
              delta="Sustainable" if efficiency >= 1.0 else "Bleeding",
              delta_color="normal" if efficiency >= 1.0 else "inverse")
    c4.caption("**Ratio > 1.0** = กำไรชนะ Time Decay\n\n**Ratio < 1.0** = ต้องการ Direction ช่วย")

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

def _render_engine_tab(data):
    tickers_list = get_tickers(data)
    pool_cf = data.get("global_pool_cf", 0.0)
    ev_reserve = data.get("global_ev_reserve", 0.0)

    with st.container(border=True):
        top1, top2, top3, top4 = st.columns(4)
        top1.metric("🎱 Pool CF (War Chest)", f"${pool_cf:,.2f}")
        top2.metric("🛡️ Pool EV LEAPS", f"${ev_reserve:,.2f}")
        top3.metric("Tickers", str(len(tickers_list)))
        total_rounds = sum(len(t.get("rounds", [])) for t in tickers_list)
        top4.metric("Total Rounds", str(total_rounds))

    if not tickers_list:
        st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Manage Data ก่อน")
        return

    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.subheader("🔗 Run Chain Round")
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
            r1, r2, r3 = st.columns(3)
            with r1: p_new = st.number_input(f"ราคาใหม่ P", min_value=0.01, value=round(state.get("price", 10.0) * 1.1, 2), step=1.0)
            with r2: sigma = st.number_input("Volatility (σ)", min_value=0.05, value=default_sigma, step=0.05)
            with r3: hedge_ratio = st.number_input("Hedge Ratio (x Put)", min_value=0.0, value=default_hr, step=0.5)
            preview_btn = st.form_submit_button("🔍 Preview Calculation")

        if preview_btn and p_new > 0:
            preview = run_chain_round(state, p_new, sigma, hedge_ratio)
            if preview:
                st.session_state["_pending_round"] = preview
                st.session_state["_pending_ticker_idx"] = idx
                st.session_state["_pending_ticker_name"] = selected

        if "_pending_round" in st.session_state and st.session_state.get("_pending_ticker_name") == selected:
            rd = st.session_state["_pending_round"]
            st.markdown("---")
            st.info("💡 **Connected Simulator:** คุณสามารถเปิดแท็บ **Payoff Profile 🔗 Run Chain Round** ด้านบน เพื่อดูกราฟล่วงหน้าอิงจากค่า Preview นี้ได้ทันที")
            
            p1, p2, p3, p4 = st.columns(4)
            new_shannon = p1.number_input("Shannon Profit", value=float(rd['shannon_profit']), step=10.0, format="%.2f")
            new_harvest = p2.number_input("Harvest Profit", value=float(rd['harvest_profit']), step=10.0, format="%.2f")
            new_hedge = p3.number_input("Hedge Cost", value=float(rd['hedge_cost']), step=10.0, format="%.2f")
            new_surplus = p4.number_input("Surplus (Free Risk)", value=float(rd['surplus']), step=10.0, format="%.2f")

            p5, p6, p7, p8 = st.columns(4)
            new_c_after = p5.number_input("New fix_c", value=float(rd['c_after']), step=100.0, format="%.0f")
            new_p_new = p6.number_input("New Price", value=float(rd['p_new']), step=0.1, format="%.2f")
            new_b_after = p7.number_input("New Baseline", value=float(rd['b_after']), step=0.1, format="%.2f")
            new_sigma = p8.number_input("Volatility (σ)", value=float(rd['sigma']), step=0.01, format="%.2f")

            if st.button("✅ Commit Round — บันทึกถาวร", type="primary"):
                rd.update({'shannon_profit': new_shannon, 'harvest_profit': new_harvest, 'hedge_cost': new_hedge, 'surplus': new_surplus, 'c_after': new_c_after, 'p_new': new_p_new, 'b_after': new_b_after, 'sigma': new_sigma, 'scale_up': new_c_after - rd['c_before']})
                commit_round(data, st.session_state["_pending_ticker_idx"], rd)
                del st.session_state["_pending_round"]
                st.success(f"✅ Round committed for {selected}! fix_c = ${rd['c_after']:,.2f}")
                st.rerun()

    with col_right:
        st.subheader("🎱 Pool CF & Allocation")
        with st.form("add_pool_cf_form", clear_on_submit=True):
            c1, c2 = st.columns([2, 1])
            with c1: amount = st.number_input("Amount ($)", min_value=0.0, value=0.0, step=100.0)
            with c2: 
                st.write("")
                st.write("")
                btn_add = st.form_submit_button("💰 Add Fund", type="primary")
            if btn_add and amount > 0:
                data["global_pool_cf"] = data.get("global_pool_cf", 0) + amount
                log_treasury_event(data, "Funding", amount, "Added to Pool CF")
                save_trading_data(data)
                st.success(f"✅ +${amount:,.2f} → Pool CF = ${data['global_pool_cf']:,.2f}")
                st.rerun()

        st.divider()
        if tickers_list:
            deploy_ticker = st.selectbox("Select Ticker", ticker_names_deploy := [d.get("ticker", "???") for d in tickers_list], key="deploy_ticker")
            d_idx = ticker_names_deploy.index(deploy_ticker)
            t_data_deploy = tickers_list[d_idx]
            cur_c = t_data_deploy.get("current_state", {}).get("fix_c", 0)
            cur_t = t_data_deploy.get("current_state", {}).get("price", 0)
            cur_b = t_data_deploy.get("current_state", {}).get("baseline", 0)

            with st.form("deploy_round_form", clear_on_submit=False):
                d1, d2 = st.columns(2)
                with d1: action_type = st.selectbox("Objective", ["📈 Scale Up", "🛡️ Buy Puts", "🎯 Buy Calls", "⏳ Pay Ev"])
                with d2: d_amt = st.number_input("Amount ($) [Pool Funding]", min_value=0.0, max_value=float(pool_cf) if pool_cf > 0 else 0.0, value=0.0, step=100.0)
                manual_new_c = st.number_input("Target fix_c (Optional Override)", min_value=0.0, value=0.0, step=100.0)
                d_note = st.text_input("Note", value="")
                submitted_deploy = st.form_submit_button("🔍 Preview Deployment")

            if submitted_deploy and d_amt > 0:
                mock_scale_up, mock_new_c = (manual_new_c - cur_c, manual_new_c) if manual_new_c > cur_c else (d_amt, cur_c + d_amt)
                injection_round = {
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "action": "Injection", "p_old": cur_t, "p_new": cur_t, "c_before": cur_c, "c_after": mock_new_c,
                    "surplus": mock_scale_up if "Scale Up" in action_type else -d_amt, "scale_up": mock_scale_up if "Scale Up" in action_type else 0, "b_before": cur_b, "b_after": cur_b, "note": f"[{action_type.split()[1]}] {d_note}"
                }
                st.session_state["_pending_injection"] = injection_round
                st.session_state["_pending_injection_idx"] = d_idx
                st.session_state["_pending_injection_amt"] = d_amt
                st.session_state["_pending_injection_type"] = action_type

            if "_pending_injection" in st.session_state and st.session_state.get("_pending_injection_idx") == d_idx:
                p_inj = st.session_state["_pending_injection"]
                if st.button("🚀 Confirm Deployment", type="primary"):
                    amt = st.session_state["_pending_injection_amt"]
                    if data["global_pool_cf"] >= amt:
                        data["global_pool_cf"] -= amt
                        log_treasury_event(data, "Deploy", -amt, f"Deployed to {deploy_ticker}")
                        commit_round(data, d_idx, p_inj)
                        del st.session_state["_pending_injection"]
                        st.success(f"✅ Complete: {st.session_state['_pending_injection_type']} ${amt:,.2f}!")
                        st.rerun()

    st.divider()
    h1, h2 = st.columns([3, 2], gap="large")
    with h1: _render_consolidated_history(t_data)
    with h2: _render_treasury_log(data)

def _render_treasury_log(data):
    st.subheader("🏛️ Treasury & Ops History")
    history = data.get("treasury_history", [])
    if history:
        st.dataframe(pd.DataFrame([{"Date": e.get("date","")[5:], "Action": e.get("category",""), "Amount": e.get("amount",0), "Note": e.get("note","")} for e in history])[::-1], use_container_width=True, hide_index=True)

def _render_consolidated_history(t_data):
    st.subheader(f"📜 {t_data.get('ticker','???')} — History")
    rounds = t_data.get("rounds", [])
    if not rounds:
        legacy = get_rollover_history(t_data)
        if legacy: st.dataframe(pd.DataFrame(legacy))
        return
    st.dataframe(pd.DataFrame([{
        "Date": rd.get("date", "")[:10], "Action": f"Scale +${rd['scale_up']:,.0f}" if rd.get("scale_up", 0) > 0 else rd.get("action", "Round"),
        "Price": f"${rd.get('p_old',0):,.2f} > ${rd.get('p_new',0):,.2f}", "fix_c": f"${rd.get('c_before',0):,.0f} > ${rd.get('c_after',0):,.0f}",
        "b": f"${rd.get('b_before',0):,.2f} > ${rd.get('b_after',0):,.2f}", "Net Result": f"${rd.get('surplus',0):,.2f}"
    } for rd in rounds])[::-1], use_container_width=True, hide_index=True)

# ----------------------------------------------------------
# TAB: Payoff Profile Simulator
# ----------------------------------------------------------
def _render_payoff_profile_tab(data):
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

    # ---- 🔗 เชื่อมต่อ Data Integration (Goal 1) 🔗 ----
    pending = st.session_state.get("_pending_round")
    if pending and st.session_state.get("_pending_ticker_name") == t_data.get("ticker"):
        def_c = float(pending.get('c_after', def_c))
        def_p = float(pending.get('p_new', def_p))
        st.success(f"🔗 **Connected State:** กราฟตั้งต้นจากค่า **Preview Calculation** ของ `{t_data.get('ticker')}` | New Price: ${def_p:,.2f} | New fix_c: ${def_c:,.0f}")
    else:
        st.info(f"🟢 **Current State:** กราฟตั้งต้นจากข้อมูลปัจจุบันของ `{t_data.get('ticker')}` | Price: ${def_p:,.2f} | fix_c: ${def_c:,.0f}")

    with st.expander("📚 อธิบายหลักการและสมการคณิตศาสตร์ (Principles & Formulas) - คลิกเพื่ออ่าน", expanded=False):
        st.markdown("""
        **1. ฟังก์ชัน Logarithmic หลัก (y1, y2)**
        สมการหลักของการสร้าง Cashflow (Rebalance) แบบสมการเชิงลอการิทึม 
        - $y = C \\cdot \\ln(P / x_0)$   *(C = เงินทุน fix_c, $x_0$ = ราคาจุดอ้างอิงเริ่มต้น)*
        
        **2. การประยุกต์ Delta ทิศทางเดียว (δ1, δ2)**
        การปรับความชัน (Slope) ของกราฟ ให้ลาดชันหรือหนืดลงตามเป้าหมาย (เช่น ใช้ดึง Cashflow หรือป้องกันความเสี่ยง)
        - $y_{delta} = (C \\cdot \\ln(P / x_0) \\times \\delta) + bias$
        
        **3. Piecewise Delta แบบ 2 ทิศทาง (y4, y5)**
        การใช้ความชัน 2 ระดับในเส้นเดียวกัน: แบ่งเป็น $\\delta_1$ อัตราเติบโตช่วงราคาขาลง และ $\\delta_2$ อัตราเติบโตช่วงราคาขาขึ้น:
        - ถ้าราคา $P < x_0 \\rightarrow \\text{ใช้ } \\delta_1$ (เช่น $\\delta=0.2$ ทน Drawdown ให้พอร์ตลดทอนลงช้าๆ)
        - ถ้าราคา $P \\ge x_0 \\rightarrow \\text{ใช้ } \\delta_2$ (เช่น $\\delta=1.0$ ขาขึ้นเก็บกำไรได้เต็มเม็ดเต็มหน่วย)
        
        **4. Benchmark เส้นอ้างอิง (y6, y7)**
        กราฟอ้างอิงเปรียบเทียบจากตัวแปร $P$ พื้นฐาน เพื่อเทียบกับพอร์ตปัจจุบันของเรา
        
        **5. Options Intrinsic Value (y8, y9)**
        มูลค่าการใช้สิทธิที่แท้จริง (Intrinsic Value) ของ Call และ Put Options (คำนวณหักลบด้วย Premium Cost)
        - Call (ได้กำไรขาขึ้น): $(\\max(0, P - \\text{Strike}) \\times Qty) - Premium$
        - Put (ได้กำไรขาลง): $(\\max(0, \\text{Strike} - P) \\times Qty) - Premium$
        
        **6. Stock / Synthetic ขาตรง (y10, y11)**
        จำลองกำไรขาดทุนจากการถือหุ้นตรงๆ (Linear) แบบ Long หรือ Short โดยกำไรขาดทุนเป็นเส้นตรง
        - P/L Long: $(P - Entry) \\times Qty$
        """)

    with st.expander("🛠️ แผงควบคุมตัวแปร (Simulator Controls)", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown("##### 🟢 กลุ่มหลัก (Shannon 1 / Long หุ้น)")
            x0_1 = st.number_input("จุดอ้างอิง x0_1 (แกนศูนย์ y1/y5)", min_value=0.1, max_value=1000.0, value=def_p, step=1.0, help="ราคา Threshold ตัดขาดทุนกำไรของระบบรอบแกน 1")
            constant1 = st.number_input("เงินทุน Constant C (y1/y5)", min_value=100.0, value=def_c, step=100.0, help="ขนาดเงินทุนคงที่ C ของสมการตัวที่ 1")
            b1 = st.number_input("ค่า Bias เลื่อนแกน (b1)", min_value=-10000.0, max_value=10000.0, value=0.0, step=100.0)
            delta1 = st.slider("ความชันขาลง (δ1 สำหรับ x < x0)", 0.0, 2.0, 0.2, 0.05, help="อัตราเร่งตอนที่ราคาสินทรัพยร่วงลง")
            st.markdown("---")
            long_shares = st.number_input("จำนวน Quantity (y10 Long)", min_value=0, value=100, help="ปริมาณสัญญาหรือหุ้น Long สด")
            long_entry = st.number_input("ราคา Long Entry", min_value=0.1, value=def_p, step=1.0)
            
        with col_c2:
            st.markdown("##### 🟡 กลุ่มรอง (Shannon 2 / Short หุ้น)")
            x0_2 = st.number_input("จุดอ้างอิง x0_2 (แกนศูนย์ y2/y4)", min_value=0.1, max_value=1000.0, value=max(def_p*1.5, 0.1), step=1.0, help="ราคา Threshold โซนที่ 2 (เช่น แนวต้านใหญ่)")
            constant2 = st.number_input("เงินทุน Constant (y2/y4)", min_value=100.0, value=def_c, step=100.0)
            b2 = st.number_input("ค่า Bias เลื่อนแกน (b2)", min_value=-10000.0, max_value=10000.0, value=0.0, step=100.0)
            delta2 = st.slider("ความชันขาขึ้น (δ2 สำหรับ x >= x0)", 0.0, 2.0, 1.0, 0.05, help="อัตราเร่งตอนที่ราคาสินทรัพย์พุ่งทะยานผ่านจุด x0")
            st.markdown("---")
            short_shares = st.number_input("จำนวน Quantity (y11 Short)", min_value=0, value=100)
            short_entry = st.number_input("ราคา Short Entry", min_value=0.1, value=max(def_p*1.5, 0.1), step=1.0)

        with col_c3:
            st.markdown("##### ⚔️ กลุ่ม Options & Benchmark")
            anchorY6 = st.number_input("ราคาอ้างอิง Benchmark", min_value=0.1, value=def_p, step=1.0, help="ราคาเริ่มต้นของ Benchmark เส้นทึบดั้งเดิม")
            refConst = st.number_input("เงินทุนอ้างอิง (Ref Const)", min_value=100.0, value=def_c, step=100.0)
            st.markdown("---")
            call_contracts = st.number_input("จำนวน Call Option (y8)", min_value=0, value=100)
            premium_call = st.number_input("Premium ที่จ่าย (Call)", min_value=0.0, value=0.0, step=0.1)
            put_contracts = st.number_input("จำนวน Put Option (y9)", min_value=0, value=100)
            premium_put = st.number_input("Premium ที่จ่าย (Put)", min_value=0.0, value=0.0, step=0.1)
            st.markdown("---")
            st.markdown("##### 🌪️ Volatility Harvest")
            sigma = st.slider("Volatility (σ)", 0.0, 2.0, 0.5, 0.05, help="ความผันผวนสำหรับคำนวณ Harvest Profit")

        st.markdown("---")
        st.caption("👁️ ควบคุมการแสดงผลของเส้นกราฟที่คำนวณแล้ว (Toggle Active Lines)")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        showY1 = t_col1.checkbox("y1: Shannon 1 (+piecewise)", value=True, help="เส้น Base ลอการิทึม 1")
        showY2 = t_col1.checkbox("y2: Shannon 2 (original)", value=False, help="เส้น Base ลอการิทึม 2")
        showY3 = t_col1.checkbox("y3: Net P/L (ผลรวม)", value=True, help="พอร์ตโฟลิโอ Net P/L (ผลรวมจากเส้นสีอื่นทั้งหมดที่เปิดติ๊กถูกไว้)")
        
        showY4 = t_col2.checkbox("y4: Piecewise y2", value=False, help="แยกความชันบนล่างของ y2")
        showY5 = t_col2.checkbox("y5: Piecewise y1", value=False, help="แยกความชันบนล่างของ y1")
        showY6 = t_col2.checkbox("y6: Ref y1 (Benchmark)", value=True, help="เส้นอ้างอิงเทียบ y1")
        
        showY7 = t_col3.checkbox("y7: Ref y2", value=False, help="เส้นอ้างอิงเทียบ y2")
        showY8 = t_col3.checkbox("y8: Call Intrinsic", value=False, help="สมมติฐานผลกำไรจากฝั่ง Long Call")
        showY9 = t_col3.checkbox("y9: Put Intrinsic", value=False, help="สมมติฐานผลกำไรจากฝั่ง Long Put")
        
        showY10 = t_col4.checkbox("y10: P/L Long (หุ้น)", value=False, help="เส้นกำไรสมมติจากการถือหุ้นเต็มเม็ด")
        showY11 = t_col4.checkbox("y11: P/L Short (หุ้น)", value=False, help="เส้นกำไรสมมติจากการ Short สินทรัพย์")
        showY12 = t_col4.checkbox("y12: Harvest Profit", value=True, help="Harvest Profit (Vol Premium)")
        includePremium = t_col4.checkbox("คำนวณหักต้นทุน Premium ในตระกูล Option", value=True)

    # ---------------- Mathematics Engine ----------------
    x_min, x_max = max(0.1, def_p * 0.1), def_p * 2.5
    prices = np.linspace(x_min, x_max, 300)

    ln1 = np.where(prices > 0, np.log(prices / x0_1), 0)
    ln2_inner = np.where(prices / x0_2 < 2, 2 - (prices / x0_2), 1e-9)
    ln2 = np.log(ln2_inner)

    y1_raw = constant1 * ln1
    y2_raw = constant2 * ln2

    y1_d2 = (y1_raw * delta2) + b1
    y2_d2 = (y2_raw * delta2) + b2

    d_y4 = np.where(prices >= x0_2, delta2, delta1)
    y4_piece = (y2_raw * d_y4) + b2

    d_y5 = np.where(prices >= x0_1, delta2, delta1)
    y5_piece = (y1_raw * d_y5) + b1

    ln6 = np.where(prices > 0, np.log(prices / anchorY6), 0)
    y6_raw = refConst * ln6
    ln7_inner = np.where(prices / anchorY6 < 2, 2 - (prices / anchorY6), 1e-9)
    y7_raw = refConst * np.log(ln7_inner)

    y6_ref_d2 = y6_raw * delta2
    y7_ref_d2 = y7_raw * delta2

    premCallCost = call_contracts * premium_call if includePremium else 0
    premPutCost = put_contracts * premium_put if includePremium else 0
    y8_call_intrinsic = (np.maximum(0, prices - x0_1) * call_contracts) - premCallCost
    y9_put_intrinsic = (np.maximum(0, x0_2 - prices) * put_contracts) - premPutCost

    y10_long_pl = (prices - long_entry) * long_shares
    y11_short_pl = (short_entry - prices) * short_shares

    harvest_profit = constant1 * 0.5 * (sigma ** 2) * 1.0
    y12_dynamic = np.full_like(prices, harvest_profit)

    components_d2 = []
    if showY1: components_d2.append(y1_d2)
    if showY2: components_d2.append(y2_d2)
    if showY4: components_d2.append(y4_piece)
    if showY5: components_d2.append(y5_piece)
    if showY8: components_d2.append(y8_call_intrinsic)
    if showY9: components_d2.append(y9_put_intrinsic)
    if showY10: components_d2.append(y10_long_pl)
    if showY11: components_d2.append(y11_short_pl)
    if showY12: components_d2.append(y12_dynamic)

    y3_delta2 = np.sum(components_d2, axis=0) if components_d2 else np.zeros_like(prices)
    y_overlay_d2 = y3_delta2 - y6_ref_d2

    # ---------------- Plotly Visualization ----------------
    tabs_chart = st.tabs(["เปรียบเทียบทั้งหมด", "Net เท่านั้น", "Delta_Log_Overlay"])

    with tabs_chart[0]:
        fig1 = go.Figure()
        if showY1: fig1.add_trace(go.Scatter(x=prices, y=y1_d2, name=f"y1 (δ={delta2:.2f})", line=dict(color='#22d3ee', width=3)))
        if showY2: fig1.add_trace(go.Scatter(x=prices, y=y2_d2, name=f"y2 (δ={delta2:.2f})", line=dict(color='#fde047', width=3)))
        if showY4: fig1.add_trace(go.Scatter(x=prices, y=y4_piece, name="y4 (piecewise δ y2)", line=dict(color='#a3e635', width=3)))
        if showY5: fig1.add_trace(go.Scatter(x=prices, y=y5_piece, name="y5 (piecewise δ y1)", line=dict(color='#10b981', width=3)))
        if showY12: fig1.add_trace(go.Scatter(x=prices, y=y12_dynamic, name="y12 (Harvest Profit)", line=dict(color='#2196f3', width=3, dash='dash')))
        if showY3: fig1.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net (δ2 base)", line=dict(color='#f472b6', width=3.5)))
        if showY6: fig1.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6 (Benchmark, δ2)", line=dict(color='#94a3b8', width=2.5, dash='dash')))
        if showY7: fig1.add_trace(go.Scatter(x=prices, y=y7_ref_d2, name="y7 (Ref y2, δ2)", line=dict(color='#c084fc', width=2.5, dash='dash')))
        if showY8: fig1.add_trace(go.Scatter(x=prices, y=y8_call_intrinsic, name="y8 (Call)", line=dict(color='#ef4444', width=3)))
        if showY9: fig1.add_trace(go.Scatter(x=prices, y=y9_put_intrinsic, name="y9 (Put)", line=dict(color='#22c55e', width=3)))
        if showY10: fig1.add_trace(go.Scatter(x=prices, y=y10_long_pl, name="y10 (Long)", line=dict(color='#60a5fa', width=3)))
        if showY11: fig1.add_trace(go.Scatter(x=prices, y=y11_short_pl, name="y11 (Short)", line=dict(color='#fb923c', width=3)))
        
        fig1.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig1.add_vline(x=def_p, line_dash="solid", line_color="#facc15", opacity=0.8, annotation_text="t (current)")
        fig1.update_layout(title="Full Custom Comparison Model", xaxis_title="Price (x)", yaxis_title="P/L (y)", height=600)
        st.plotly_chart(fig1, use_container_width=True)

    with tabs_chart[1]:
        fig2 = go.Figure()
        if showY3: fig2.add_trace(go.Scatter(x=prices, y=y3_delta2, name="Net (δ2 base)", line=dict(color='#f472b6', width=3.5)))
        if showY6: fig2.add_trace(go.Scatter(x=prices, y=y6_ref_d2, name="y6 (Benchmark, δ2)", line=dict(color='#94a3b8', width=3, dash='dash')))
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

def _render_manage_data(data):
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

    with st.expander("⚠️ ลบข้อมูลทั้งหมด", expanded=False):
        if st.button("DELETE ALL DATA", type="primary"):
            data.update({"tickers": [], "global_pool_cf": 0.0, "global_ev_reserve": 0.0, "treasury_history": []})
            save_trading_data(data)
            st.warning("All data cleared!")
            st.rerun()

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("เลือกบทเรียน:", [
        "บทนำ: Flywheel 0 (Dragon Portfolio)", "บทที่ 1: The Baseline", "บทที่ 2: Volatility Harvest",
        "บทที่ 3: Convexity Engine", "บทที่ 4: The Black Swan Shield", "บทที่ 5: Dynamic Scaling",
        "บทที่ 6: Synthetic Dividend", "บทที่ 7: Collateral Magic", "บทที่ 8: Chain System (ลูกโซ่)",
        "📝 แบบทดสอบ (Quiz)", "🛠️ Workshop: จัดพอร์ตจริง", "📚 อภิธานศัพท์ (Glossary)"
    ])
    st.sidebar.markdown("---")
    st.sidebar.info("Application to demonstrate the concepts of Shannon's Demon strategy.")

    if page == "บทนำ: Flywheel 0 (Dragon Portfolio)": chapter_0_introduction()
    elif page == "บทที่ 1: The Baseline": chapter_1_baseline()
    elif page == "บทที่ 2: Volatility Harvest": chapter_2_volatility_harvest()
    elif page == "บทที่ 3: Convexity Engine": chapter_3_convexity_engine()
    elif page == "บทที่ 4: The Black Swan Shield": chapter_4_black_swan_shield()
    elif page == "บทที่ 5: Dynamic Scaling": chapter_5_dynamic_scaling()
    elif page == "บทที่ 6: Synthetic Dividend": chapter_6_synthetic_dividend()
    elif page == "บทที่ 7: Collateral Magic": chapter_7_collateral_magic()
    elif page == "บทที่ 8: Chain System (ลูกโซ่)": chapter_chain_system()
    elif page == "📝 แบบทดสอบ (Quiz)": master_study_guide_quiz()
    elif page == "🛠️ Workshop: จัดพอร์ตจริง": paper_trading_workshop()
    elif page == "📚 อภิธานศัพท์ (Glossary)": glossary_section()

if __name__ == "__main__":
    main()
