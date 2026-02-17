
import streamlit as st
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from scipy.stats import norm

# --- Utilities ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def generate_gbm(S0, mu, sigma, T, dt, n_sims=1):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) 
    W = np.cumsum(W) * np.sqrt(dt) 
    X = (mu - 0.5 * sigma**2) * t + sigma * W 
    S = S0 * np.exp(X) 
    return t, S

# --- Placeholders for Chapters 0-7 (Code temporarily unavailable) ---
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

# --- Chapter 8: Chain System (Restored & Updated) ---
def chapter_chain_system():
    st.header("บทที่ 8: Chain System (ระบบลูกโซ่)")
    st.markdown("""
    **Concept:** เชื่อมกำไรจากทุก Flywheel เข้าเป็น **ลูกโซ่** (Chain) — 
    กำไรจากขั้นหนึ่งไหลไปเป็น "เชื้อเพลิง" ให้ขั้นถัดไป วนเป็นวงจร **ทั้งขาขึ้น + ขาลง**
    
    > **ขาขึ้น:** กำไร Shannon + Harvest → จ่ายค่า Put Hedge → Surplus → Scale Up fix_c = **Free Risk**
    > 
    > **ขาลง:** Put ระเบิดกำไร → เข้า **Pool CF** → Deploy (เมื่อ Regime กลับ) + Reserve (สำรอง)
    """)
    
    with st.expander("สมการ Continuous Rollover"):
        st.latex(r"b_{new} = b_{old} + c \cdot \ln(P/t_{old}) - c' \cdot \ln(P/t_{new})")
        st.caption("ปรับ Baseline ให้ต่อเนื่องเมื่อเปลี่ยน fix_c และ re-center ราคา t")

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
        
        # Pool CF Config
        st.markdown("---")
        st.subheader("3. Pool CF & Crash Sim")
        deploy_ratio = st.slider("Deploy Ratio (from Pool CF)", 0.0, 1.0, 0.7, 0.1, help="% of Net Put Profit to Deploy")
        crash_price_pct = st.slider("Simulate Crash Price (%)", 30, 100, 50, 5, help="% of P0")
        P_crash = P0 * (crash_price_pct / 100.0)
        st.metric("Crash Price Scenario", f"${P_crash:.1f}")

    # Calculations
    r = 0.04
    T = 1.0
    put_strike_pct = 0.9
    put_strike = P0 * put_strike_pct
    put_premium = black_scholes(P0, put_strike, T, r, sigma, 'put')
    cost_hedge = qty_puts * put_premium
    
    harvest_profit = fix_c * 0.5 * (sigma ** 2) * T
    
    st.divider()
    
    # --- Stage 1 & 2 ---
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
    
    # Put Payoff Calculation
    put_payoff_crash = max(0, put_strike - P_crash)
    total_put_payoff = qty_puts * put_payoff_crash
    
    # Shannon Net: Price Loss + Harvest Profit (Ref)
    shannon_price_term = fix_c * np.log(P_crash / P0) if P_crash > 0 else 0
    shannon_harvest_term = fix_c * 0.5 * (sigma ** 2) * T  # Harvest accumulated over T
    shannon_net_ref = shannon_price_term + shannon_harvest_term
    
    # Rolldown Cost (Simulated cost to re-hedge at crash price)
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
    
    # Simple Logic for Sankey Flows
    # Source -> Target, Value
    # 0->1 (Shannon->Harvest) ??? No, Harvest IS Source.
    # Let's use: Harvest -> Put Hedge, Harvest -> Surplus
    # Put Payoff -> Pool CF
    # Pool CF -> Re-hedge, Pool CF -> Net -> Deploy, Reserve
    
    # Simplified for Demo
    value_harvest = max(1, harvest_profit)
    value_hedge = cost_hedge
    value_surplus = max(0, surplus)
    
    value_put = max(1, total_put_payoff)
    value_rolldown = rolldown_cost
    value_deploy = deploy_amount
    value_reserve = reserve_amount

    # Flows (Indices based on labels list)
    # Harvest(1) -> Put Hedge(2)
    # Harvest(1) -> Surplus(3)
    # Surplus(3) -> Scale Up(4)
    # Put Payoff(5) -> Pool CF(6)
    # Pool CF(6) -> Re-Hedge(7)
    # Pool CF(6) -> Deploy(8)
    # Pool CF(6) -> Reserve(9)

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
    
    # 2. Payoff Chart
    st.subheader("Payoff Profile Ref")
    prices = np.linspace(P0*0.2, P0*1.5, 100)
    shannon_val = fix_c * np.log(prices / P0)
    put_val = qty_puts * np.maximum(0, put_strike - prices)
    combined = shannon_val + put_val
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=prices, y=shannon_val, name="Shannon (Unhedged)"))
    fig_payoff.add_trace(go.Scatter(x=prices, y=combined, name="Shannon + Put Shield", line=dict(width=3, color='green')))
    
    # Crash Marker
    fig_payoff.add_vline(x=P_crash, line_dash="dash", line_color="red", 
                         annotation_text=f"Crash Scenario ({P_crash:.1f})")
    
    fig_payoff.update_layout(title="Payoff Profile with Crash Scenario", xaxis_title="Price", yaxis_title="Value")
    st.plotly_chart(fig_payoff, use_container_width=True)

    st.info(f"""
    **Chain System — Full Cycle Analysis:**
    
    **ขาขึ้น (Bull/Sideway):** Harvest (${harvest_profit:.2f}) จ่ายค่า Hedge (${cost_hedge:.2f}) เหลือ Surplus Scale Up.
    
    **ขาลง (Bear/Crash):** Put ทำงาน (${total_put_payoff:,.2f}) → เข้า Pool CF → หักลบ Re-Hedge (${rolldown_cost:,.2f})
    → **Valid Net:** ${pool_cf_net:,.2f}
    → **Deploy** ${deploy_amount:,.2f} ({(deploy_ratio*100):.0f}%) + **Reserve** ${reserve_amount:,.2f} ({(100-deploy_ratio*100):.0f}%)
    """)

# --- Main App Navigation ---
def main():
    st.sidebar.title("Flywheel & Shannon's Demon")
    menu = st.sidebar.radio("Menu", [
        "Introduction", "Baseline", "Shannon Process", "Volatility Harvesting",
        "Black Swan Shield", "Dynamic Scaling", "Synthetic Dividend", "Collateral Magic",
        "Chain System (Active)", "Quiz", "Paper Trading", "Glossary"
    ], index=8) # Default to Chain System

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

# Stub functions for missing chapters if any called
def master_study_guide_quiz(): pass
def paper_trading_workshop(): pass
def glossary_section(): pass

if __name__ == "__main__":
    main()
