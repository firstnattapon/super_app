import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math

# --- Page Config ---
st.set_page_config(
    page_title="Anti-Fragile Wealth Academy",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions (Math) ---
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, r, sigma):
    """Black-Scholes Call Option Price"""
    if T <= 0: return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_put(S, K, T, r, sigma):
    """Black-Scholes Put Option Price"""
    if T <= 0: return max(0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

# --- Content Data ---
glossary = {
    "Anti-Fragile": "ระบบที่ยิ่งเจอความผันผวนหรือวิกฤต ยิ่งเติบโตและแข็งแกร่งขึ้น",
    "Baseline": "เส้นอ้างอิงผลตอบแทนทางทฤษฎี คำนวณจาก fix_c * ln(Pt/P0)",
    "Convexity (Gamma)": "คุณสมบัติที่ทำให้กำไรเร่งตัวขึ้นเร็วในขาขึ้น และขาดทุนช้าลงในขาลง",
    "Delta": "อัตราการเปลี่ยนแปลงของราคา Option เทียบกับราคาสินทรัพย์อ้างอิง",
    "Theta": "ค่าเสื่อมเวลาของ Option (ยิ่งใกล้หมดอายุ มูลค่ายิ่งลดลง)",
    "LEAPS": "Option อายุยาว (1-2 ปี) ใช้แทนการถือหุ้นเพื่อลดความเสี่ยงขาลง",
    "Portfolio Margin": "ระบบคำนวณหลักประกันตามความเสี่ยงจริงของพอร์ต ทำให้ใช้เงินทุนได้มีประสิทธิภาพสูง",
    "Synthetic Dividend": "การสร้างกระแสเงินสด (ปันผลเทียม) จากการขาย Short Call/Put"
}

# --- Section: Home ---
def render_home():
    st.title("🛡️ Anti-Fragile Wealth Machine Academy")
    st.markdown("""
    ยินดีต้อนรับสู่หลักสูตรวิศวกรรมการเงินขั้นสูง ที่จะเปลี่ยนคุณจาก **"นักเก็งกำไร"** ให้เป็น **"ผู้บริหารกองทุนระดับโลก"**
    
    แอปพลิเคชันนี้ออกแบบมาเพื่อสอนโครงสร้าง **Flywheel 0-7** ซึ่งเป็นพิมพ์เขียวของการสร้างพอร์ตโฟลิโอที่:
    * ✅ **อยู่รอด** ในทุกสภาวะเศรษฐกิจ (เงินเฟ้อ/เงินฝืด)
    * ✅ **เติบโต** จากความผันผวน (Volatility Harvesting)
    * ✅ **กำไรมหาศาล** เมื่อเกิดวิกฤต (Black Swan Event)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📚 **บทเรียน (Modules)**\n\nเรียนรู้ทีละขั้นตอนจากรากฐานสู่ขั้นสูง")
    with col2:
        st.warning("🧪 **ห้องทดลอง (The Lab)**\n\nจำลองพอร์ตจริง ปรับค่าความเสี่ยง และทดสอบ Crash")
    with col3:
        st.success("📝 **แบบทดสอบ (Quiz)**\n\nวัดระดับความเข้าใจของคุณ")

    st.divider()
    st.image("https://images.unsplash.com/photo-1611974765270-ca1258634369?q=80&w=2064&auto=format&fit=crop", caption="The Infinite Game of Finance", use_container_width=True)

# --- Section: Lessons ---
def render_lessons():
    st.header("📚 บทเรียน: 7 เฟืองจักรแห่งความมั่งคั่ง")
    
    tabs = st.tabs(["0. Dragon", "1. Baseline", "2. Rebalance", "3. LEAPS 80/20", "4. Put Shield", "5. Dynamic", "6. Yield", "7. Collateral"])
    
    with tabs[0]:
        st.subheader("🐲 Flywheel 0: The Dragon Portfolio")
        st.markdown("รากฐานที่ออกแบบมาให้อยู่รอดในรอบ 100 ปี ไม่ว่าเศรษฐกิจจะเป็นอย่างไร")
        
        # Dragon Portfolio Chart
        labels = ['Equity (Growth)', 'Fixed Income (Deflation)', 'Gold (Devaluation)', 'Long Volatility (Crisis)', 'Commodity Trend (Inflation)']
        values = [20, 20, 20, 20, 20]
        fig = px.pie(values=values, names=labels, title='โครงสร้าง Dragon Portfolio (All-Weather)', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("เราจะนำระบบ Options เข้าไปสวมในส่วนของ Equity และ Long Volatility")

    with tabs[1]:
        st.subheader("📏 Flywheel 1: The Baseline")
        st.markdown("ไม้บรรทัดวัดผล: ถ้าเราไม่ทำอะไรเลย เราควรได้เท่าไหร่?")
        st.latex(r"Benchmark = fix\_c \cdot \ln \left( \frac{P_t}{P_0} \right)")
        st.info("เราใช้ **Natural Logarithm (ln)** เพราะสะท้อนผลตอบแทนแบบทบต้นต่อเนื่อง (Continuous Compounding) ได้ดีที่สุด")

    with tabs[2]:
        st.subheader("🌊 Flywheel 2: Volatility Rebalance")
        st.markdown("**การสกัดเงินจากคลื่น:** ซื้อถูก-ขายแพง อัตโนมัติ")
        st.markdown("""
        * **หลักการ:** รักษามูลค่าพอร์ตให้คงที่ ($fix\_c$)
        * **หุ้นขึ้น:** ขายส่วนเกิน -> เก็บเงินสด
        * **หุ้นลง:** เอาเงินสด -> ช้อนซื้อ
        * **ผลลัพธ์:** เกิด **Volatility Premium** (ส่วนต่างกำไรที่ไม่เกี่ยวกับทิศทาง)
        """)

    with tabs[3]:
        st.subheader("🏎️ Flywheel 3: Convexity Engine (80/20)")
        st.markdown("**เปลี่ยนเครื่องยนต์:** เลิกถือหุ้น 100% มาใช้โครงสร้าง 80/20")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 80% LEAPS")
            st.write("- ใช้ Option อายุยาว (1-2 ปี) แทนหุ้น")
            st.write("- **ข้อดี:** ล็อกขาดทุนสูงสุด (Flatline) แต่กำไรไม่จำกัด")
        with col2:
            st.markdown("### 20% Liquidity")
            st.write("- เงินสดสำหรับทำ Rebalance")
            st.write("- เป็นเชื้อเพลิงในการช้อนซื้อตอนตลาดย่อ")

    with tabs[4]:
        st.subheader("🛡️ Flywheel 4: The Black Swan Shield")
        st.markdown("**เปลี่ยนวิกฤตเป็นโอกาส:** ซื้อ Put Option สัดส่วน x2 (Over-hedging)")
        st.success("💰 จ่ายค่าประกันด้วย 'กำไร' จาก Volatility Premium (Flywheel 2) ไม่ใช่เงินต้น!")
        st.markdown("เมื่อตลาดพังพินาศ (Crash):")
        st.write("1. LEAPS ขาดทุนชนพื้น (จำกัด)")
        st.write("2. Put x2 ระเบิดกำไรมหาศาล (ไม่จำกัด)")
        st.write("3. **ผลลัพธ์:** พอร์ตโตสวนกระแสตลาด (Anti-Fragile)")

    with tabs[5]:
        st.subheader("⚙️ Flywheel 5: Dynamic Hedging")
        st.markdown("**ระบบเกียร์อัจฉริยะ:** ปรับตัวตาม VIX Index")
        st.table(pd.DataFrame({
            "สภาวะตลาด": ["ตลาดนิ่ง/ซึม (Low Vol)", "ตลาดผันผวน/ใกล้วิกฤต (High Vol)"],
            "Action": ["Scale Down (ถอด Put)", "Scale Up (ซื้อ Put x2)"],
            "เป้าหมาย": ["ประหยัดต้นทุน", "ดักจับ Black Swan"]
        }).set_index("สภาวะตลาด"))

    with tabs[6]:
        st.subheader("💵 Flywheel 6: Synthetic Dividend")
        st.markdown("**เสือนอนกิน:** สร้างปันผลเทียมมาจ่ายค่าประกัน")
        st.markdown("""
        * **Short Call:** ปล่อยเช่า LEAPS (เก็บค่าพรีเมียมรายสัปดาห์)
        * **Short Put:** รับจ้างรอซื้อหุ้น (เก็บค่าพรีเมียมรายสัปดาห์)
        * **ผลลัพธ์:** นำเงินฟรีนี้ไปจ่ายค่า Long Put ใน Flywheel 4 = **Zero-Cost Hedge**
        """)

    with tabs[7]:
        st.subheader("🏦 Flywheel 7: Collateral Magic")
        st.markdown("**ความลับระดับสถาบัน:** เงินต้นปลอดภัย 100%")
        st.markdown("""
        1.  นำเงิน $fix\_c$ ไปซื้อ **T-Bills (พันธบัตร)** -> กินดอกเบี้ย 5%
        2.  ใช้ T-Bills เป็น **หลักประกัน (Collateral)** ในระบบ Portfolio Margin
        3.  เทรด Options ทั้งหมดบน "เงา" ของเงิน โดยเงินจริงไม่เสี่ยง
        """)

# --- Section: Simulator ---
def render_simulator():
    st.header("🧪 ห้องทดลอง: Anti-Fragile Crash Simulator")
    st.markdown("จำลองเหตุการณ์ **Market Crash** เพื่อดูว่าพอร์ต 80/20 + Put Hedge ทำงานอย่างไร")

    # --- Sidebar Inputs for Sim ---
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        fix_c = st.number_input("เงินทุนตั้งต้น (fix_c)", value=1000000, step=100000)
    with col_input2:
        crash_magnitude = st.slider("ความรุนแรงของวิกฤต (% หุ้นร่วง)", 10, 80, 40)
    with col_input3:
        put_budget_pct = st.slider("งบซื้อ Put Hedge (% ต่อเดือน)", 0.1, 2.0, 0.5) / 100

    if st.button("🚀 รันการจำลอง (Run Simulation)"):
        # Logic from previous "Anti-Fragile" script
        days = 365
        P0 = 100
        sigma = 0.4
        r = 0.03
        dt = 1/252
        crash_day = 180
        
        # Generate Price Path
        np.random.seed(42)
        Z = np.random.normal(0, 1, days)
        P_t = np.zeros(days)
        P_t[0] = P0
        
        for t in range(1, days):
            if t == crash_day:
                P_t[t] = P_t[t-1] * (1 - (crash_magnitude / 100))
            else:
                P_t[t] = P_t[t-1] * np.exp((0.05 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t])
        
        # 1. Stock Rebalance
        ret_stock = np.diff(P_t) / P_t[:-1]
        cum_pnl_stock = np.insert(np.cumsum(fix_c * ret_stock), 0, 0)
        
        # 2. LEAPS 80/20
        leaps_weight = 0.8
        T_array = np.linspace(1.0, 0.001, days)
        C_t = np.array([bs_call(p, P0*0.8, t_exp, r, sigma) for p, t_exp in zip(P_t, T_array)])
        ret_opt = np.diff(C_t) / np.maximum(C_t[:-1], 1e-8)
        daily_pnl_leaps = (leaps_weight * fix_c) * ret_opt
        cum_pnl_leaps = np.insert(np.cumsum(daily_pnl_leaps), 0, 0)
        
        # 3. Put Hedge (Simplified Monthly Roll)
        put_pnl_cum = np.zeros(days)
        monthly_budget = fix_c * put_budget_pct
        current_put_pnl = 0
        
        # Simplified Hedge Logic: Assume we buy 10% OTM puts every 21 days
        # If crash happens, Put value explodes
        for t in range(days):
            if t == crash_day:
                # Put Payoff on crash day: Massive gain
                # Approximation: Delta becomes -1 approx, Gamma explodes
                crash_gain = (P_t[t-1] - P_t[t]) * (fix_c / P0) * 2 # x2 Leverage roughly
                current_put_pnl += crash_gain
            elif t % 21 == 0:
                current_put_pnl -= monthly_budget # Pay premium
            
            put_pnl_cum[t] = current_put_pnl

        # Total Anti-Fragile PnL
        cum_pnl_anti = cum_pnl_leaps + put_pnl_cum

        # Create DataFrame
        df = pd.DataFrame({
            "Day": range(days),
            "Price": P_t,
            "Stock Rebalance": cum_pnl_stock,
            "LEAPS Only": cum_pnl_leaps,
            "Anti-Fragile": cum_pnl_anti
        })

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Day"], y=df["Stock Rebalance"], name="1. Stock Rebalance (รับมีด)", line=dict(color='blue', dash='dot')))
        fig.add_trace(go.Scatter(x=df["Day"], y=df["LEAPS Only"], name="2. LEAPS 80/20 (มีพื้น)", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=df["Day"], y=df["Anti-Fragile"], name="3. Anti-Fragile (กำไรขาลง)", line=dict(color='green', width=3)))
        
        fig.add_vline(x=crash_day, line_dash="dash", line_color="red", annotation_text="MARKET CRASH")
        fig.update_layout(title="เปรียบเทียบผลตอบแทนเมื่อเกิดวิกฤต", height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("ผลลัพธ์สุดท้าย (Anti-Fragile)", f"{cum_pnl_anti[-1]:,.2f} บาท", 
                  delta=f"ชนะตลาด: {cum_pnl_anti[-1] - cum_pnl_stock[-1]:,.2f} บาท")

# --- Section: Quiz ---
def render_quiz():
    st.header("📝 แบบทดสอบความเข้าใจ")
    
    questions = {
        "q1": {
            "question": "1. สมการ Baseline (fix_c * ln(Pt/P0)) ใช้เพื่ออะไร?",
            "options": ["คำนวณกำไรจริง", "เป็นไม้บรรทัดวัดผลตอบแทนทางทฤษฎี", "คำนวณภาษี", "หาจุด Stop Loss"],
            "answer": "เป็นไม้บรรทัดวัดผลตอบแทนทางทฤษฎี",
            "explanation": "ถูกต้อง! มันคือเส้นอ้างอิงเพื่อดูว่าระบบ Rebalance ของเราทำได้ดีกว่าการถือเฉยๆ หรือไม่"
        },
        "q2": {
            "question": "2. ในระบบ 80/20 ส่วน 20% (Liquidity) มีหน้าที่หลักคืออะไร?",
            "options": ["เอาไปซื้อหวย", "เก็บไว้เฉยๆ ห้ามใช้", "เป็นเชื้อเพลิงทำ Volatility Rebalance", "เอาไปซื้อหุ้น 100%"],
            "answer": "เป็นเชื้อเพลิงทำ Volatility Rebalance",
            "explanation": "ถูกต้อง! เงินส่วนนี้จะถูกดึงไปช้อนซื้อ LEAPS ตอนราคาลง และรับเงินตอนขาย LEAPS เมื่อราคาขึ้น"
        },
        "q3": {
            "question": "3. เหตุใดเราจึงควรซื้อ Put Option ในสัดส่วน x2 (Over-hedging)?",
            "options": ["เพื่อให้พอร์ตเป็น Anti-Fragile (กำไรเมื่อเกิดวิกฤต)", "เพราะโบรกเกอร์บังคับ", "เพื่อให้ขาดทุนน้อยลงนิดหน่อย", "เพื่อเก็งกำไรระยะสั้น"],
            "answer": "เพื่อให้พอร์ตเป็น Anti-Fragile (กำไรเมื่อเกิดวิกฤต)",
            "explanation": "ถูกต้อง! สัดส่วน x2 จะทำให้กำไรขาลงมากกว่าผลขาดทุนของพอร์ต ทำให้พอร์ตโตสวนกระแสตลาดได้"
        },
         "q4": {
            "question": "4. Flywheel 7 (Collateral Magic) ใช้สินทรัพย์ใดเป็นฐานเพื่อความปลอดภัยสูงสุด?",
            "options": ["Bitcoin", "หุ้นกู้เอกชน", "พันธบัตรรัฐบาล (T-Bills)", "ที่ดิน"],
            "answer": "พันธบัตรรัฐบาล (T-Bills)",
            "explanation": "ถูกต้อง! T-Bills ถือเป็น Risk-free asset ที่รับดอกเบี้ยแน่นอนและใช้ค้ำประกันมาร์จิ้นได้เกือบ 100%"
        }
    }
    
    score = 0
    for q_key, q_val in questions.items():
        st.subheader(q_val["question"])
        choice = st.radio("คำตอบ:", q_val["options"], key=q_key)
        
        if st.button(f"ตรวจคำตอบข้อ {q_key[-1]}", key=f"btn_{q_key}"):
            if choice == q_val["answer"]:
                st.success(q_val["explanation"])
                score += 1
            else:
                st.error(f"ผิดครับ! คำตอบที่ถูกคือ: {q_val['answer']}")
    
    # Note: State management for score requires session_state, simplified here for display.

# --- Section: Glossary ---
def render_glossary():
    st.header("📖 พจนานุกรมการเงิน (Glossary)")
    search = st.text_input("ค้นหาคำศัพท์...", "")
    
    for term, definition in glossary.items():
        if search.lower() in term.lower() or search.lower() in definition.lower():
            with st.expander(f"**{term}**"):
                st.write(definition)

# --- Main App ---
def main():
    with st.sidebar:
        st.title("🎓 เมนูการเรียนรู้")
        page = st.radio("เลือกหัวข้อ:", ["🏠 หน้าหลัก", "📚 บทเรียน (Flywheels)", "🧪 ห้องทดลอง (Lab)", "📝 แบบทดสอบ", "📖 พจนานุกรม"])
        
        st.divider()
        st.caption("Developed based on Anti-Fragile Wealth Machine concept.")

    if page == "🏠 หน้าหลัก":
        render_home()
    elif page == "📚 บทเรียน (Flywheels)":
        render_lessons()
    elif page == "🧪 ห้องทดลอง (Lab)":
        render_simulator()
    elif page == "📝 แบบทดสอบ":
        render_quiz()
    elif page == "📖 พจนานุกรม":
        render_glossary()

if __name__ == "__main__":
    main()
