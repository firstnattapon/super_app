import streamlit as st
import os
import sys

# Add parent directory to path if needed, though Streamlit usually handles this
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from professional_system.data_manager import DataManager
from professional_system.models import SystemConfig
from professional_system.flywheel_views import (
    render_dashboard_summary,
    render_dragon_portfolio,
    render_shannon_journal,
    render_options_inventory,
    render_collateral_yield
)

# Initialize Logic
# pages/professional_system.py -> pages -> root
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trading_data.json")

@st.cache_resource
def get_manager():
    dm = DataManager(DATA_PATH)
    dm.load_data()
    return dm

def main():
    st.title("Professional 7-Flywheel System")
    
    # Load Data
    try:
        dm = get_manager()
        tickers = dm.get_all_tickers()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar Navigation (Local to this page)
    st.sidebar.title("🚀 Flywheel System")
    st.sidebar.markdown("---")
    
    menu = st.sidebar.radio("Navigation", [
        "Dashboard (Pool CF)",
        "FW0: Dragon Portfolio",
        "FW1-2: Shannon's Journal",
        "FW3-4-6: Options Inventory",
        "FW5: Dynamic Scaling",
        "FW7: Collateral Magic",
        "Configuration"
    ])
    
    # Global Config (Mock)
    config = SystemConfig()
    
    # Routing
    if menu == "Dashboard (Pool CF)":
        render_dashboard_summary(tickers, config)
        st.markdown("### 📌 Active Flywheels Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Active Tickers: {len(tickers)}")
        with col2:
            st.success(f"Total Cash Flow: ${dm.get_pool_cf():,.2f}")

    elif menu == "FW0: Dragon Portfolio":
        render_dragon_portfolio()

    elif menu == "FW1-2: Shannon's Journal":
        render_shannon_journal(tickers)

    elif menu == "FW3-4-6: Options Inventory":
        render_options_inventory(tickers)

    elif menu == "FW5: Dynamic Scaling":
        st.header("🌊 Flywheel 5: Dynamic Scaling")
        vix = st.slider("Current VIX", 10, 60, 20)
        if vix < 20:
            st.success("Market Status: Calm (Green Zone). Focus on Income.")
        elif vix < 40:
            st.warning("Market Status: Choppy (c). Prepare Hedges.")
        else:
            st.error("Market Status: Panic (Red Zone). Monetize Hedges!")

    elif menu == "FW7: Collateral Magic":
        render_collateral_yield(config)
        
    elif menu == "Configuration":
        st.header("⚙️ System Configuration")
        st.text_input("Data File Path", value=DATA_PATH, disabled=True)
        if st.button("Reload Data"):
            st.cache_resource.clear()
            st.rerun()

main()
