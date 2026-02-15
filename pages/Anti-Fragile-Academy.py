import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="Anti-Fragile Strategy Canvas",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Styling (CSS for Canvas Look) ---
st.markdown("""
<style>
    .canvas-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #e0e0e0;
        transition: transform 0.2s;
        height: 100%;
    }
    .canvas-box:hover {
        transform: scale(1.02);
        border-color: #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .box-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .box-desc {
        font-size: 0.9rem;
        color: #666;
    }
    .header-zone { background-color: #e3f2fd; border-color: #2196f3; } /* Macro */
    .engine-zone { background-color: #e8f5e9; border-color: #4caf50; } /* Core */
    .shield-zone { background-color: #fce4ec; border-color: #e91e63; } /* Defense */
    .fuel-zone { background-color: #fff3e0; border-color: #ff9800; } /* Optimization */
    
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Math Helper Functions ---
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    # Approximation of N(x) using math.erf
    cdf_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    cdf_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    return S * cdf_d1 - K * np.exp(-r * T) * cdf_d2

# --- Session State Management ---
if 'active_flywheel' not in st.session_state:
    st.session_state.active_flywheel = 0

def set_flywheel(index):
    st.session_state.active_flywheel = index

# --- Main Canvas Layout ---
def render_canvas():
    st.title("🗺️ Anti-Fragile Strategy Canvas")
    st.markdown("แผนผังกลยุทธ์แบบองค์รวม: คลิกที่กล่องเพื่อเจาะลึกรายละเอียดและใช้งาน Simulator")
    
    # --- ROW 1: Foundation (Flywheel 0 & 7) ---
    col1, col2 = st.columns(2)
    
    with col1: # FW 0
        st.markdown(f"""
        <div class="canvas-box header-zone">
            <div class="box-title">🐲 FW 0: Dragon Portfolio</div>
            <div class="box-desc">รากฐานการจัดพอร์ตแบบ All-Weather (Equity, Gold, Volatility, etc.)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Dragon", key="btn_fw0"): set_flywheel(0)

    with col2: # FW 7
        st.markdown(f"""
        <div class="canvas-box header-zone">
            <div class="box-title">🏦 FW 7: Collateral Magic</div>
            <div class="box-desc">เงินต้น 100% อยู่ใน T-Bills + Portfolio Margin</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Collateral", key="btn_fw7"): set_flywheel(7)

    st.markdown("---")

    # --- ROW 2: The Engine & The Shield (FW 3, 2, 4) ---
    col3, col4, col5 = st.columns(3)
    
    with col3: # FW 3
        st.markdown(f"""
        <div class="canvas-box engine-zone">
            <div class="box-title">🏎️ FW 3: Convexity Engine</div>
            <div class="box-desc">Stock Replacement: 80% LEAPS + 20% Liquidity</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก 80/20", key="btn_fw3"): set_flywheel(3)

    with col4: # FW 2 (Center Piece)
        st.markdown(f"""
        <div class="canvas-box engine-zone">
            <div class="box-title">🌊 FW 2: Vol Rebalance</div>
            <div class="box-desc">สกัด Cashflow จากการแกว่งตัว (Continuous Rebalancing)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Rebalance", key="btn_fw2"): set_flywheel(2)

    with col5: # FW 4
        st.markdown(f"""
        <div class="canvas-box shield-zone">
            <div class="box-title">🛡️ FW 4: Black Swan Shield</div>
            <div class="box-desc">ซื้อ Put x2 (Over-hedge) ด้วยกำไรส่วนเกิน</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Shield", key="btn_fw4"): set_flywheel(4)

    st.markdown("---")
    
    # --- ROW 3: Optimization & Metric (FW 6, 5, 1) ---
    col6, col7, col8 = st.columns(3)
    
    with col6: # FW 6
        st.markdown(f"""
        <div class="canvas-box fuel-zone">
            <div class="box-title">💵 FW 6: Synthetic Div</div>
            <div class="box-desc">ขาย Short Call/Put เพื่อสร้างรายได้ (Theta Harvesting)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Income", key="btn_fw6"): set_flywheel(6)

    with col7: # FW 5
        st.markdown(f"""
        <div class="canvas-box fuel-zone">
            <div class="box-title">⚙️ FW 5: Dynamic Scaling</div>
            <div class="box-desc">ปรับเกียร์ตาม VIX (Scale Up/Down)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Dynamic", key="btn_fw5"): set_flywheel(5)
        
    with col8: # FW 1
        st.markdown(f"""
        <div class="canvas-box" style="border-color: #9e9e9e;">
            <div class="box-title">📏 FW 1: The Baseline</div>
            <div class="box-desc">ไม้บรรทัดวัดผล: fix_c * ln(Pt/P0)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔍 เจาะลึก Baseline", key="btn_fw1"): set_flywheel(1)

# --- Detail Views (Interactivity) ---

def render_details():
    fw = st.session_state.active_flywheel
    st.markdown("---")
    
    if fw == 0:
        st.header("🐲 Flywheel 0: Dragon Portfolio Simulator")
        st.info("โครงสร้างพอร์ตที่ออกแบบมาเพื่ออยู่รอดในทุกสภาวะเศรษฐกิจ")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Asset Allocation:**")
            alloc = {
                "Equity (LEAPS)": 20,
                "Fixed Income": 20,
                "Gold": 20,
                "Long Volatility": 20,
                "Commodity Trend": 20
            }
            df_alloc = pd.DataFrame(alloc.items(), columns=["Asset", "Weight (%)"])
            st.dataframe(df_alloc, hide_index=True)
        with col2:
            fig = px.pie(df_alloc, values='Weight (%)', names='Asset', title='The Dragon Allocation', hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

    elif fw == 1:
        st.header("📏 Flywheel 1: The Baseline Calculator")
        st.markdown("คำนวณผลตอบแทนทางทฤษฎีของ Shannon's Demon")
        
        c1, c2 = st.columns(2)
        fix_c = c1.number_input("Fix Capital ($)", value=100000)
        p0 = c2.number_input("Start Price ($)", value=100)
        pt = c2.slider("Ending Price ($)", 50, 200, 120)
        
        benchmark = fix_c * np.log(pt / p0)
        buy_hold = fix_c * (pt / p0 - 1) # Assuming started with cash equivalent to fix_c invested
        
        st.metric("Shannon Benchmark (Log)", f"${benchmark:,.2f}", 
                  delta=f"Diff vs Buy&Hold: ${benchmark - buy_hold:,.2f}")
        
        st.caption("ถ้าเส้นกำไรจริงของคุณ (Realized PnL) สูงกว่าค่านี้ แสดงว่าคุณชนะตลาดด้วย Volatility Premium")

    elif fw == 2:
        st.header("🌊 Flywheel 2: Volatility Harvest Simulator")
        st.markdown("จำลองการ Rebalance เพื่อดู Cashflow สะสม")
        
        vol = st.slider("Volatility (%)", 10, 100, 40) / 100
        days = 252
        
        # Simple GBM Sim
        prices = [100]
        np.random.seed(42)
        for _ in range(days):
            change = np.random.normal(0, vol/np.sqrt(252))
            prices.append(prices[-1] * (1 + change))
            
        # Rebalance Logic
        cashflow = []
        fix_c = 100000
        shares = fix_c / 100
        cum_cf = 0
        
        for p in prices:
            val = shares * p
            diff = val - fix_c
            cum_cf += diff # Sell excess (positive diff) or Buy deficit (negative diff costs money? No, diff is cash generated/used)
            # Actually, "Cashflow" in Shannon's Demon is withdrawing excess. 
            # If val < fix_c, we inject cash. So net cashflow tracks extraction.
            shares = fix_c / p
            cashflow.append(cum_cf)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cashflow, name="Cumulative Cashflow", fill='tozeroy'))
        fig.add_trace(go.Scatter(y=prices, name="Price", yaxis="y2"))
        fig.update_layout(title="Volatility Premium Accumulation", yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)

    elif fw == 3:
        st.header("🏎️ Flywheel 3: 80/20 Stock Replacement")
        st.markdown("คำนวณจำนวนสัญญา LEAPS และเงินสดคงเหลือ")
        
        capital = st.number_input("Total Capital ($)", 100000)
        price = st.number_input("Stock Price", 100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 80% Engine")
            budget_leaps = capital * 0.8
            leaps_premium = st.number_input("LEAPS Premium ($/share)", 20.0)
            contracts = int(budget_leaps / (leaps_premium * 100))
            cost = contracts * leaps_premium * 100
            st.metric("Contracts", f"{contracts}", f"Cost: ${cost:,.0f}")
            
        with col2:
            st.markdown("### 20% Liquidity")
            remaining = capital - cost
            st.metric("Liquidity Pool", f"${remaining:,.2f}", f"Pct: {remaining/capital*100:.1f}%")
            st.success("เงินสดส่วนนี้ใช้สำหรับ FW 2 (Rebalance) และ FW 4 (Put Hedge)")

    elif fw == 4:
        st.header("🛡️ Flywheel 4: Black Swan Shield (x2)")
        st.markdown("ตรวจสอบประสิทธิภาพการป้องกันพอร์ต")
        
        contracts = st.number_input("จำนวนสัญญา LEAPS ที่มี", 10)
        hedge_ratio = st.slider("Hedge Ratio (Put:LEAPS)", 1.0, 3.0, 2.0, step=0.1)
        
        put_contracts = int(contracts * hedge_ratio)
        st.metric("จำนวน Put ที่ต้องซื้อ", f"{put_contracts} สัญญา", "Over-hedging")
        
        st.markdown("### 💥 Crash Simulator")
        crash_pct = st.slider("Market Crash (%)", 10, 80, 40)
        
        # Simple Payoff Logic
        start_price = 100
        end_price = start_price * (1 - crash_pct/100)
        
        leaps_loss = -20 * 100 * contracts # Max loss assume premium lost approx
        put_gain = (90 - end_price) * 100 * put_contracts # Strike 90
        
        net_pnl = put_gain + leaps_loss
        
        col1, col2 = st.columns(2)
        col1.error(f"LEAPS Loss (Est): ${leaps_loss:,.0f}")
        col2.success(f"Put Gain (Est): ${put_gain:,.0f}")
        st.metric("Net Result", f"${net_pnl:,.0f}", delta_color="normal")

    elif fw == 5:
        st.header("⚙️ Flywheel 5: Dynamic Scaling Matrix")
        
        vix = st.slider("Current VIX Index", 10, 60, 25)
        
        col1, col2 = st.columns(2)
        
        if vix < 20:
            regime = "Low Volatility (Green)"
            action = "Scale Down"
            desc = "ลดการซื้อ Put, เน้นเก็บ Cashflow, ถือ 80/20 ปกติ"
            color = "green"
        elif vix < 35:
            regime = "Normal/Choppy (Yellow)"
            action = "Maintain"
            desc = "ถือ Put สัดส่วน 1:1 หรือ 1.5:1, เตรียม Rebalance"
            color = "orange"
        else:
            regime = "High Volatility (Red)"
            action = "Scale Up (Attack)"
            desc = "อัด Put x2, ขยาย fix_c, พร้อมสวนตลาด"
            color = "red"
            
        with col1:
            st.markdown(f"### Regime: :{color}[{regime}]")
        with col2:
            st.info(f"**Action:** {action}\n\n{desc}")

    elif fw == 6:
        st.header("💵 Flywheel 6: Synthetic Dividend Planner")
        
        st.markdown("คำนวณรายรับจากการขาย Short Volatility")
        
        c1, c2, c3 = st.columns(3)
        call_prem = c1.number_input("Short Call Premium", 1.5)
        put_prem = c2.number_input("Short Put Premium", 1.0)
        contracts = c3.number_input("Number of Sets", 10)
        
        weekly_income = (call_prem + put_prem) * 100 * contracts
        monthly_income = weekly_income * 4
        
        st.metric("Weekly Income", f"${weekly_income:,.2f}")
        st.metric("Monthly Income", f"${monthly_income:,.2f}")
        
        st.warning(f"รายได้ต่อเดือน ${monthly_income:,.2f} นี้ จะถูกนำไปจ่ายค่า Put Hedge ใน FW 4")

    elif fw == 7:
        st.header("🏦 Flywheel 7: Collateral & Margin")
        
        capital = st.number_input("Total Capital ($)", 100000)
        tbill_rate = 0.05
        
        passive_income = capital * tbill_rate
        
        st.markdown(f"""
        ### The Setup:
        1. นำเงิน **${capital:,.0f}** ซื้อ T-Bills (Yield 5%)
        2. ได้ดอกเบี้ยปีละ **${passive_income:,.0f}** (โดยไม่เสี่ยง)
        3. ใช้มูลค่า T-Bills ค้ำประกัน (Margin 90-95%) เพื่อเปิด LEAPS 80/20
        """)
        
        st.success("ผลลัพธ์: เงินต้นปลอดภัย 100% + พอร์ต Options ทำกำไร On Top")

# --- Run App ---
render_canvas()
render_details()
