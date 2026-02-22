import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime

# ==========================================
# ⚙️ CONFIG & PERSISTENCE
# ==========================================
st.set_page_config(page_title="Chain Engine 3.0", layout="wide")
DATA_FILE = "chain_data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tickers": [], "pool_cf": 0.0}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ==========================================
# 🧮 CORE ENGINE (PURE FUNCTION)
# ==========================================
def compute_round(state: dict, p_new: float, hedge_ratio: float, ev_cost: float, fix_c_override: float = None):
    """
    Pure transition function: (state, input) -> (next_state, row_data)
    Equation: surplus = fix_c * ln(p_new / price) * (1 - hedge_ratio)
    """
    price = state["price"]
    fix_c = fix_c_override if fix_c_override is not None else state["fix_c"]
    baseline = state["baseline"]
    cum_ev = state["cumulative_ev"]

    # 1. Compute
    shannon = fix_c * np.log(p_new / price) if price > 0 else 0.0
    hedge = shannon * hedge_ratio
    surplus = shannon - hedge
    delta_fc = max(0.0, surplus) # fix_c only scales up

    # 2. Next State
    next_state = {
        "price": p_new,
        "fix_c": fix_c + delta_fc,
        "baseline": baseline + surplus,
        "cumulative_ev": cum_ev + ev_cost
    }

    # 3. Row Data (for history)
    net_pnl = next_state["baseline"] - next_state["cumulative_ev"]
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "price_transition": f"{price:.2f} → {p_new:.2f}",
        "shannon": round(shannon, 2),
        "hedge": round(hedge, 2),
        "surplus": round(surplus, 2),
        "delta_fc": round(delta_fc, 2),
        "fix_c": round(next_state["fix_c"], 2),
        "baseline": round(next_state["baseline"], 2),
        "net_pnl": round(net_pnl, 2)
    }
    
    return {"next_state": next_state, "row": row}

# ==========================================
# 🖥️ UI LAYOUT & STATE MANAGEMENT
# ==========================================
def main():
    # Initialize Session State
    if "data" not in st.session_state:
        st.session_state.data = load_data()
    if "active_idx" not in st.session_state:
        st.session_state.active_idx = 0
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "pending_ticker" not in st.session_state:
        st.session_state.pending_ticker = None

    data = st.session_state.data
    tickers = data.get("tickers", [])

    st.title("⚡ Chain Engine — 1-Row State Machine")
    st.markdown(r"**Axiom:** 1 equation $\cdot$ 4 state vars $\cdot$ 2 inputs $\cdot$ 1 row per round. _Pure determinism._")

    # Layout: 12% | 44% | 44%
    c_left, c_center, c_right = st.columns([1.2, 4.4, 4.4], gap="medium")

    # ---------------------------------------------------------
    # ZONE LEFT: Watchlist (12%)
    # ---------------------------------------------------------
    with c_left:
        st.markdown("##### 📋 Watchlist")
        if tickers:
            labels = []
            for i, t in enumerate(tickers):
                st_data = t["state"]
                net_pnl = st_data["baseline"] - st_data["cumulative_ev"]
                dot = "🟢" if net_pnl >= 0 else "🔴"
                labels.append(f"{dot} {t['ticker']}")
            
            # Change active ticker
            selected_label = st.radio("Tickers", labels, index=st.session_state.active_idx, label_visibility="collapsed")
            new_idx = labels.index(selected_label)
            
            if new_idx != st.session_state.active_idx:
                st.session_state.active_idx = new_idx
                st.session_state.pending = None  # Clear stale preview
                st.rerun()
                
            active_t = tickers[st.session_state.active_idx]
            active_state = active_t["state"]
            net_pnl = active_state["baseline"] - active_state["cumulative_ev"]
        else:
            st.info("No tickers yet.")
            active_t = None

    # ---------------------------------------------------------
    # ZONE CENTER: Input & Status (44%)
    # ---------------------------------------------------------
    with c_center:
        if active_t:
            st.markdown(f"#### ⚡ {active_t['ticker']} — Order Strip")
            
            # Equation Strip
            with st.form("preview_form", border=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                p_new = col1.number_input("P New", min_value=0.01, value=float(active_state["price"]), step=1.0)
                fc_over = col2.number_input("Fix C (Override)", min_value=1.0, value=float(active_state["fix_c"]), step=100.0)
                hr = col3.number_input("Hedge Ratio", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
                ev = col4.number_input("EV Cost", min_value=0.0, value=0.0, step=10.0)
                
                col5.write("")
                col5.write("")
                preview_btn = col5.form_submit_button("▶ Preview", use_container_width=True)
                
                if preview_btn:
                    # Resolve override
                    fc_actual = fc_over if fc_over != active_state["fix_c"] else None
                    st.session_state.pending = compute_round(active_state, p_new, hr, ev, fc_actual)
                    st.session_state.pending_ticker = st.session_state.active_idx
                    st.rerun()

            # Current State Metrics
            st.markdown("##### 🟢 Current State")
            with st.container(border=True):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Price", f"${active_state['price']:,.2f}")
                m2.metric("Fix C", f"${active_state['fix_c']:,.0f}")
                m3.metric("Baseline", f"${active_state['baseline']:,.2f}")
                m4.metric("Net PnL", f"${net_pnl:,.2f}", delta_color="normal" if net_pnl >= 0 else "inverse")
                m5.metric("Rounds", len(active_t.get("history", [])))

        st.divider()
        # Manage Data Expander
        with st.expander("⚙️ Manage Data (Add Ticker)"):
            with st.form("add_ticker_form", clear_on_submit=True):
                new_t = st.text_input("Ticker Symbol").upper()
                init_p = st.number_input("Initial Price", min_value=0.01, value=100.0)
                init_c = st.number_input("Initial Fix C", min_value=1.0, value=10000.0)
                if st.form_submit_button("➕ Add Ticker"):
                    if new_t and new_t not in [t["ticker"] for t in tickers]:
                        tickers.append({
                            "ticker": new_t,
                            "state": {"price": init_p, "fix_c": init_c, "baseline": 0.0, "cumulative_ev": 0.0},
                            "history": []
                        })
                        save_data(st.session_state.data)
                        st.success(f"Added {new_t}")
                        st.rerun()

    # ---------------------------------------------------------
    # ZONE RIGHT: Preview, Charts & History (44%)
    # ---------------------------------------------------------
    with c_right:
        if active_t:
            # 1. Preview Card
            is_pending = st.session_state.pending and st.session_state.pending_ticker == st.session_state.active_idx
            
            if is_pending:
                with st.container(border=True):
                    st.markdown("### ⚠️ Pending Preview")
                    row = st.session_state.pending["row"]
                    
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Shannon", f"${row['shannon']:,.2f}")
                    p2.metric("Hedge", f"${row['hedge']:,.2f}")
                    p3.metric("Surplus", f"${row['surplus']:,.2f}")
                    p4.metric("Δ Fix C", f"+${row['delta_fc']:,.2f}")
                    
                    bc1, bc2 = st.columns(2)
                    if bc1.button("✅ COMMIT ROUND", type="primary", use_container_width=True):
                        # Apply transition
                        active_t["state"] = st.session_state.pending["next_state"]
                        # Prepend history (newest first)
                        row["#round"] = len(active_t["history"]) + 1
                        active_t["history"].insert(0, row)
                        
                        save_data(st.session_state.data)
                        st.session_state.pending = None
                        st.rerun()
                        
                    if bc2.button("✕ Cancel", use_container_width=True):
                        st.session_state.pending = None
                        st.rerun()

            # 2. Equity Curve Chart
            history = active_t.get("history", [])
            if history:
                df = pd.DataFrame(history)[::-1] # Reverse to chronological for charting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["#round"], y=df["net_pnl"], name="Net PnL", line=dict(color="#22c55e", width=3)))
                fig.add_trace(go.Scatter(x=df["#round"], y=df["fix_c"], name="Fix C (Capacity)", yaxis="y2", line=dict(color="#3b82f6", dash="dot")))
                
                fig.update_layout(
                    title="Equity Curve & Capital Scale",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(title="Net PnL ($)"),
                    yaxis2=dict(title="Fix C ($)", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # 3. History Dataframe
                st.markdown("##### 📜 Immutable History")
                df_display = pd.DataFrame(history)
                st.dataframe(df_display, use_container_width=True, hide_index=True, height=250)
            else:
                st.info("No history yet. Run a round to see charts and data.")

if __name__ == "__main__":
    main()
