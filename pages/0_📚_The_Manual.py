import sys
import os

# Add root directory to sys.path to allow importing modules from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import flywheels

st.set_page_config(
    page_title="7 Flywheel Shannon's Demon",
    page_icon="💸",
    layout="wide",
)

st.title("💸 7 Flywheel Shannon's Demon")
st.markdown("### The Anti-Fragile Wealth Machine")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("เลือกบทเรียน:", 
    [
        "บทนำ: Flywheel 0 (Dragon Portfolio)",
        "บทที่ 1: The Baseline",
        "บทที่ 2: Volatility Harvest",
        "บทที่ 3: Convexity Engine",
        "บทที่ 4: The Black Swan Shield",
        "บทที่ 5: Dynamic Scaling",
        "บทที่ 6: Synthetic Dividend",
        "บทที่ 7: Collateral Magic",
        "📝 แบบทดสอบ (Quiz)",
        "🛠️ Workshop: จัดพอร์ตจริง",
        "📚 อภิธานศัพท์ (Glossary)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Application to demonstrate the concepts of Shannon's Demon strategy.")

# Page Routing
if page == "บทนำ: Flywheel 0 (Dragon Portfolio)":
    flywheels.chapter_0_introduction()

elif page == "บทที่ 1: The Baseline":
    flywheels.chapter_1_baseline()

elif page == "บทที่ 2: Volatility Harvest":
    flywheels.chapter_2_volatility_harvest()

elif page == "บทที่ 3: Convexity Engine":
    flywheels.chapter_3_convexity_engine()

elif page == "บทที่ 4: The Black Swan Shield":
    flywheels.chapter_4_black_swan_shield()

elif page == "บทที่ 5: Dynamic Scaling":
    flywheels.chapter_5_dynamic_scaling()

elif page == "บทที่ 6: Synthetic Dividend":
    flywheels.chapter_6_synthetic_dividend()

elif page == "บทที่ 7: Collateral Magic":
    flywheels.chapter_7_collateral_magic()

elif page == "📝 แบบทดสอบ (Quiz)":
    flywheels.master_study_guide_quiz()

elif page == "🛠️ Workshop: จัดพอร์ตจริง":
    flywheels.paper_trading_workshop()

elif page == "📚 อภิธานศัพท์ (Glossary)":
    flywheels.glossary_section()
