"""⚡ Chain System — Main Engine · v2  (simple · stable · fast)

Same portfolio file as the original Main Engine (trading_data.json), so there is nothing to migrate.
  simple — one page, six views, native Streamlit widgets only (no HTML / CSS injection)
  stable — every write is atomic and all-or-nothing, previews can't go stale, bad data can't crash a view
  fast   — only the selected view runs (no hidden tabs building charts), the data file is parsed once per change
Logic lives in ``engine_v2.py`` (pure Python, unit-tested: ``python -m unittest discover -s tests``).
"""
import json
import os
import sys
import zlib
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import engine_v2 as eng  # noqa: E402

st.set_page_config(page_title="Chain System v2", page_icon="⚡", layout="wide")

DATA_FILE = os.environ.get("CHAIN_V2_DATA_FILE") or str(ROOT / "trading_data.json")
DEFAULT_IMPORT_URL = ("https://raw.githubusercontent.com/firstnattapon/super_app/"
                      "refs/heads/main/data/chain_system_backup.json")
VIEWS = ["⚡ Engine", "📜 History", "📊 Portfolio", "🏛️ Treasury", "📐 Payoff", "➕ Data"]

money = eng.money
CC = st.column_config


@st.cache_resource(show_spinner=False)
def get_store(path: str) -> eng.Store:
    return eng.Store(path)


# ── small UI helpers ─────────────────────────────────────────────────────────────
def usd(label: str):
    return CC.NumberColumn(label, format="dollar")


def esc(text: str) -> str:
    """Streamlit markdown treats ``$…$`` as LaTeX — escape dollars in captions / alerts / toasts."""
    return text.replace("$", "\\$")


def px(x: float) -> str:
    return f"${x:,.2f}" if x >= 1 or x == 0 else f"${x:,.4f}"


def table(rows: list, cols: dict = None, **kw) -> None:
    st.dataframe(rows, hide_index=True, width="stretch", column_config=cols, **kw)


def flash(msg: str) -> None:
    st.session_state["_v2_flash"] = esc(msg)


def attempt(write, ok: str) -> None:
    """Run a store write: show the error, or flash a toast and rerun on success."""
    try:
        write()
    except (ValueError, eng.DataError) as e:   # ChainError is a ValueError
        st.error(f"❌ {e}")
        return
    except OSError as e:
        st.error(f"💾 บันทึกไฟล์ไม่สำเร็จ: {e}")
        return
    flash(ok)
    st.rerun()


def run(fn, ok: str) -> None:
    attempt(lambda: get_store(DATA_FILE).update(fn), ok)


def chip(t: dict) -> str:
    s = t["current_state"]
    return f"{'🟢' if s['net_pnl'] >= 0 else '🔴'} {t['ticker']} · {px(s['price'])}"


ROUND_COLS = {"#": CC.NumberColumn("#", format="%d", width="small"),
              "Date": CC.TextColumn("Date", width="medium"), "Note": CC.TextColumn("Note", width="large"),
              "P old": CC.NumberColumn("P old", format="%.6g"), "P new": CC.NumberColumn("P new", format="%.6g"),
              "fix_c old": usd("fix_c old"), "fix_c new": usd("fix_c new"),
              "b old": usd("b old"), "b new": usd("b new"),
              "Shannon": usd("Shannon"), "Hedge": usd("Hedge"), "Surplus": usd("Surplus")}
MONEY_COLS = {c: usd(c) for c in ("Amount", "Pool CF", "EV Reserve")}


# ── ⚡ Engine ─────────────────────────────────────────────────────────────────────
def view_engine(store, d, t):
    if not t:
        return st.info("ยังไม่มี ticker — เพิ่มที่แท็บ ➕ Data หรือ Import ข้อมูล")
    s = t["current_state"]
    c = st.columns(5)
    c[0].metric("Price", px(s["price"]))
    c[1].metric("fix_c", money(s["fix_c"]))
    c[2].metric("Baseline", money(s["baseline"]))
    c[3].metric("🔥 Ev burn", money(s["cumulative_ev"]))
    c[4].metric("Net P&L", money(s["net_pnl"]))
    if (out := eng.withdrawn_baseline(t)) > 0:
        st.caption(esc(f"📥 Withdraw_b — Baseline ที่ดึงออกไปแล้ว: {money(out)}"))
    left, right = st.columns([3, 2], gap="large")
    with left:
        order_panel(d, t)
    with right:
        st.subheader(f"📜 Recent rounds ({len(t['rounds'])})")
        rows = eng.round_rows(t, limit=8)
        if rows:
            table([{"Date": r["Date"][:10], **{k: r[k] for k in ("Action", "P new", "fix_c new")}} for r in rows],
                  {**ROUND_COLS, "Date": CC.TextColumn("Date", width="small"),
                   "Action": CC.TextColumn("Action", width="medium")})
        else:
            st.caption("ยังไม่มี round")


def order_panel(d: dict, t: dict) -> None:
    name, s = t["ticker"], t["current_state"]
    price = s["price"]
    st.subheader(f"🔗 Chain Round — {name}")
    sig = f"{name}|{price}|{s['fix_c']}|{s['baseline']}"       # a new state ⇒ fresh widgets with fresh defaults
    k = lambda x: f"v2|{x}|{sig}"                              # noqa: E731
    dec = 2 if price >= 1 or price == 0 else 4

    st.session_state.setdefault(k("pnew"), max(float(price), 1e-4))   # default lives in state (Yahoo writes it too)
    c_px, c_btn, c_hr = st.columns([3, 1.4, 2], vertical_alignment="bottom")
    with c_btn:   # rendered before P New so a fetched price can be written into its widget state
        if st.button("🔄 Yahoo", key=k("yf"), width="stretch", help="ดึงราคาล่าสุดจาก Yahoo Finance"):
            with st.spinner(f"ดึงราคา {name}…"):
                got, src = eng.fetch_price(name)
            if got > 0:
                st.session_state[k("pnew")] = round(got, 4)
                st.session_state[k("yf_info")] = esc(f"📡 {name}: ${got:,.2f} ({src})")
            else:
                st.session_state[k("yf_info")] = f"❌ Yahoo: {src}"
    with c_px:
        p_new = st.number_input("P New", min_value=1e-4, step=10.0 ** -dec,
                                format=f"%.{dec}f", key=k("pnew"),
                                help="ราคาใหม่ (Target Price) ของรอบนี้ — เทียบกับราคาเดิม t เพื่อคำนวณ Shannon = fix_c × ln(P/t)")
    with c_hr:
        ratio = st.number_input("Hedge ×", min_value=0.0, value=float(d["settings"]["default_hedge_ratio"]),
                                step=0.5, key=k("hr"),
                                help="สัดส่วน Put Hedge (1.0 = พอดีตัว, 2.0 = Over-hedge เพื่อเร่ง Convexity)")
    if st.session_state.get(k("yf_info")):
        st.caption(st.session_state[k("yf_info")])
    f1, f2 = st.columns(2)
    no_hedge = f1.checkbox("No Hedge", key=k("nh"))
    no_surplus = f2.checkbox("No Surplus", key=k("ns"), help="ไม่ Scale Up fix_c จาก Surplus")

    flags = dict(no_hedge=no_hedge, no_surplus=no_surplus)
    try:
        model = eng.plan_round(s, p_new, ratio, **flags)
    except eng.ChainError as e:
        return st.error(f"❌ {e}")

    # Override: type the real numbers; untouched fields keep following the model.
    msig = f"{sig}|{p_new}|{ratio}|{no_hedge}|{no_surplus}"
    changed = lambda v, ref: None if abs(v - ref) < 0.005 else v    # noqa: E731
    with st.expander("✏️ Override — แก้ให้ตรงกับตัวเลขจริงก่อน Commit"):
        o1, o2, o3 = st.columns(3)
        h_in = o1.number_input("Hedge cost ($)", min_value=0.0, value=model["hedge_cost"], step=10.0,
                               format="%.2f", key=f"v2|oh|{msig}", help="ค่า Put ที่จ่ายจริง (เปลี่ยนแล้ว Surplus/Scale Up คำนวณใหม่)")
        hedge = changed(h_in, model["hedge_cost"])
        c_def = eng.plan_round(s, p_new, ratio, hedge=hedge, **flags)["c_after"]
        c_in = o2.number_input("fix_c after", min_value=0.0, value=c_def, step=100.0, format="%.2f",
                               key=f"v2|oc|{msig}|{h_in}", help="ใส่ 0 เพื่อปิด position หรือกำหนด fix_c ใหม่เอง")
        b_in = o3.number_input("Baseline after", value=model["b_after"], step=10.0, format="%.2f",
                               key=f"v2|ob|{msig}")
    plan = eng.plan_round(s, p_new, ratio, hedge=hedge, c_after=changed(c_in, c_def),
                          b_after=changed(b_in, model["b_after"]), **flags)
    overridden = any(plan[x] != model[x] for x in ("hedge_cost", "c_after", "b_after"))
    st.session_state["v2_last_plan"] = {"ticker": name, "p_new": plan["p_new"], "c_after": plan["c_after"]}

    if price <= 0:
        st.info("ticker นี้ยังไม่มีราคา (t = 0) — Commit จะตั้งราคาเริ่มต้น (Shannon = 0) · ใส่ fix_c ที่ Override")
    elif s["fix_c"] <= 0:
        st.info("fix_c = 0 — รอบนี้ไม่มี Shannon/Hedge · ใช้ Override → fix_c after เพื่อเปิด position ใหม่")

    c0, c1, b0, b1, p0, p1 = (plan[x] for x in ("c_before", "c_after", "b_before", "b_after", "p_old", "p_new"))
    sg = lambda x: ("+" if x > 0 else "") + money(x)                                    # noqa: E731
    dl = lambda x, f=money: "–" if abs(x) < 0.005 else (                                 # noqa: E731
        f":{'green' if x > 0 else 'red'}[{'+' if x > 0 else ''}{f(x)}]")
    st.markdown(esc(
        f"**💰 Shannon** {sg(plan['shannon_profit'])} − **🛡️ Hedge** {money(plan['hedge_cost'])} "
        f"= **✨ Surplus** {sg(plan['surplus'])} → **🚀 Scale Up** {sg(plan['scale_up'])}\n\n"
        "| | ก่อน | หลัง | Δ |\n|:--|--:|--:|--:|\n"
        f"| **fix_c** | {money(c0)} | **{money(c1)}** | {dl(c1 - c0)} |\n"
        f"| **Baseline** | {money(b0)} | **{money(b1)}** | {dl(b1 - b0)} |\n"
        f"| **Price t** | {px(p0)} | **{px(p1)}** | {dl(p1 - p0, lambda x: f'{x:,.4g}')} |"))
    st.code("\n".join([
        f"Shannon = {c0:,.2f} × ln({p1:g} / {p0:g}) = {money(plan['shannon_profit'])}" if p0 > 0
        else "Shannon = 0   (ยังไม่มีราคาเดิม)",
        f"Surplus = Shannon − Hedge = {money(plan['shannon_profit'])} − {money(plan['hedge_cost'])}"
        f" = {money(plan['surplus'])}",
        f"fix_c'  = {money(c0)} + {money(plan['scale_up'])} = {money(c1)}",
        f"b'      = {b0:,.2f} + {plan['shannon_profit']:,.2f} = {b1:,.2f}   (t' = P ⇒ c'·ln(P/t') = 0)",
    ]), language=None, wrap_lines=True)
    if overridden:
        st.caption("✏️ ใช้ตัวเลข Override บางส่วน")

    # The key carries a hash of the plan, so a click on an outdated button (double click, other tab) is ignored.
    plan_id = zlib.crc32(repr(sorted(plan.items())).encode())
    if st.button("✅ COMMIT — บันทึกถาวร", type="primary", width="stretch", key=f"v2|commit|{name}|{plan_id:08x}"):
        run(lambda d_: eng.apply_round(d_, name, plan), f"✅ Committed {name}: fix_c → {money(plan['c_after'])}")


# ── 📜 History ────────────────────────────────────────────────────────────────────
def view_history(store, d, t):
    if not t:
        return st.info("ยังไม่มี ticker")
    rows = eng.round_rows(t)
    st.subheader(f"📜 {t['ticker']} — History ({len(rows)} rounds)")
    if rows:
        table(rows, ROUND_COLS, height=min(640, 38 + 35 * len(rows)))
    else:
        st.info("ยังไม่มี round")


# ── 📊 Portfolio ──────────────────────────────────────────────────────────────────
def view_portfolio(store, d, t):
    if not d["tickers"]:
        return st.info("ยังไม่มี ticker")
    tot = eng.totals(d)
    c = st.columns(5)
    c[0].metric("Total Fix_C", money(tot["fix_c"]))
    c[1].metric("🔥 Burn (Cum. Ev)", money(tot["burn"]))
    c[2].metric("Gross (Lock + IV)", money(tot["gross"]))
    eff = tot["efficiency"]
    verdict = "" if eff is None else (" · Sustainable ✅" if eff >= 1 else " · Bleeding 🩸")
    c[3].metric(f"Ev Efficiency{verdict}", "—" if eff is None else f"{eff:.2f}x",
                help="(Lock P&L + Surplus IV) ÷ Burn — ≥ 1.0 = กำไรชนะค่าเช่าเวลา (Time Decay)")
    c[4].metric("💰 Net P&L", money(tot["net"]))
    table(eng.portfolio_rows(d), {"Price": CC.NumberColumn("Price", format="%.6g"),
                                  "Rounds": CC.NumberColumn("Rounds", format="%d"),
                                  **{x: usd(x) for x in ("fix_c", "Baseline", "Ev burn", "Lock P&L",
                                                         "Surplus IV", "Net P&L")}})
    import plotly.graph_objects as go   # charts are only built in the views that show them

    adj = tot["net"] - (tot["gross"] - tot["burn"])
    steps = [("Lock P&L", tot["lock"]), ("Surplus IV", tot["iv"]), ("Ev burn", -tot["burn"]),
             ("Adjustments", adj), ("Net P&L", 0.0)]
    fig = go.Figure(go.Waterfall(
        x=[s for s, _ in steps], y=[v for _, v in steps], measure=["relative"] * 4 + ["total"],
        text=[money(v) for _, v in steps[:4]] + [money(tot["net"])], textposition="outside",
        increasing_marker_color="#22c55e", decreasing_marker_color="#ef4444", totals_marker_color="#3b82f6"))
    fig.update_layout(title="P&L Waterfall (Ev เป็นต้นทุน)", height=360, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Adjustments = ส่วนต่างระหว่าง Net P&L ที่บันทึกไว้กับ Lock + IV − Ev (ข้อมูล legacy ที่ปรับมือ)")

    if st.toggle("🌊 Capital Flow (Sankey)", key="v2_sankey"):
        names = st.multiselect("เลือก Ticker (ว่าง = ทั้งหมด)", [x["ticker"] for x in d["tickers"]], key="v2_sankey_t")
        nodes, links = eng.capital_flow(d, names)
        if not links:
            return st.info("ยังไม่มี flow — Commit Chain Round ก่อน")
        fig = go.Figure(go.Sankey(
            arrangement="snap", node=dict(label=nodes, pad=18, thickness=20),
            link=dict(source=[x[0] for x in links], target=[x[1] for x in links],
                      value=[x[2] for x in links], label=[x[3] for x in links])))
        fig.update_layout(title="Capital Flow", height=520, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")


# ── 🏛️ Treasury ───────────────────────────────────────────────────────────────────
TREASURY = {   # label → (id, ticker: None | "opt" | "req", hint)
    "💰 Funding → Pool CF": ("funding", "opt", "เติมเงิน / กำไรส่วนเกิน (Surplus) เข้า Pool CF"),
    "🌾 Harvest → Pool CF": ("harvest", "opt", "บันทึกกำไรจริงที่ Harvest จาก Rebalancing"),
    "📥 Extract Baseline → Pool CF": ("extract", "req", "ดึง Baseline ของ ticker ออกมาเป็น Pool CF"),
    "🚀 Deploy Pool CF → Baseline": ("deploy", "req", "ย้าย Pool CF เข้า Baseline ของ ticker"),
    "🛡️ Allocate Pool CF → EV Reserve": ("allocate", None, "สำรองเงินจาก Pool CF ไว้จ่าย LEAPS"),
    "💸 EV Reserve flow (±)": ("ev_flow", "opt", "บวก = Income · ลบ = Expense (เช่น จ่าย LEAPS)"),
}


def view_treasury(store, d, t):
    tot = eng.totals(d)
    c = st.columns(3)
    c[0].metric("🎱 Pool CF", money(tot["pool"]))
    c[1].metric("🛡️ EV Reserve", money(tot["reserve"]))
    c[2].metric("💵 Cash (Pool + Reserve)", money(tot["cash"]))
    names = [x["ticker"] for x in d["tickers"]]

    left, right = st.columns(2, gap="large")
    with left:   # choices live outside the form so the hints below update immediately
        label = st.selectbox("การดำเนินการ", list(TREASURY), key="v2_tr_act")
        act, need, hint = TREASURY[label]
        st.caption(hint)
        pick, cap = None, None
        if need:
            opts = names if need == "req" else ["—"] + names
            if st.session_state.get("v2_tr_tk") not in opts:
                st.session_state["v2_tr_tk"] = t["ticker"] if (t and need == "req") else (opts[0] if opts else None)
            if opts:
                pick = st.selectbox("Ticker", opts, key="v2_tr_tk")
                pick = None if pick == "—" else pick
        if act == "extract" and pick:
            cap = next(x["current_state"]["baseline"] for x in d["tickers"] if x["ticker"] == pick)
            st.caption(esc(f"Baseline ที่ดึงได้: {money(cap)}"))
        elif act in ("deploy", "allocate"):
            cap = tot["pool"]
            st.caption(esc(f"Pool CF ที่ใช้ได้: {money(cap)}"))
    with right:
        if need == "req" and not names:
            st.info("ยังไม่มี ticker")
        else:
            with st.form("v2_tr_form", clear_on_submit=True):
                amount = st.number_input("Amount ($)", min_value=None if act == "ev_flow" else 0.0,
                                         max_value=None if cap is None else max(cap, 0.0), value=0.0,
                                         step=100.0, format="%.2f")
                note = st.text_input("Note (optional)") if act == "deploy" else ""
                go = st.form_submit_button(label, type="primary")
            if go:
                ops = {"funding": lambda d_: eng.add_to_pool(d_, amount, "Funding", pick),
                       "harvest": lambda d_: eng.add_to_pool(d_, amount, "Harvest", pick),
                       "extract": lambda d_: eng.extract_baseline(d_, pick, amount),
                       "deploy": lambda d_: eng.deploy(d_, pick, amount, note),
                       "allocate": lambda d_: eng.allocate_ev(d_, amount),
                       "ev_flow": lambda d_: eng.ev_flow(d_, amount, pick)}
                run(ops[act], f"✅ {label}: {money(amount)}")

    st.subheader("🏛️ Treasury log")
    opts = ["🌐 ทั้งหมด"] + names
    if st.session_state.get("v2_tr_flt") not in opts:
        st.session_state["v2_tr_flt"] = opts[0]
    sel = st.selectbox("กรอง Ticker", opts, key="v2_tr_flt", label_visibility="collapsed")
    rows = eng.treasury_rows(d, None if sel == opts[0] else sel)
    if rows:
        table(rows, MONEY_COLS, height=min(560, 38 + 35 * len(rows)))
    else:
        st.info("ยังไม่มี Treasury events")


# ── 📐 Payoff ─────────────────────────────────────────────────────────────────────
PAYOFF_COLORS = {"y1": "#22d3ee", "y2": "#fde047", "y3": "#f472b6", "y4": "#a3e635", "y5": "#10b981",
                 "y6": "#94a3b8", "y7": "#c084fc", "y8": "#ef4444", "y9": "#22c55e", "y10": "#60a5fa",
                 "y11": "#fb923c"}


def view_payoff(store, d, t):
    if not t:
        return st.info("ยังไม่มี ticker")
    name, s = t["ticker"], t["current_state"]
    p0 = s["price"] if s["price"] > 0 else 100.0
    c0 = s["fix_c"] if s["fix_c"] >= 1 else 10000.0          # closed / dust position → illustrative capital
    last = st.session_state.get("v2_last_plan")
    if last and last["ticker"] == name and st.toggle("🔗 ใช้ผลจาก Order Strip (P new / fix_c after)",
                                                     value=True, key="v2_po_sync"):
        p0, c0 = last["p_new"], last["c_after"] if last["c_after"] >= 1 else c0
        st.success(esc(f"Connected — {name} · P {px(p0)} · fix_c {money(c0)}"))
    else:
        st.info(esc(f"Current state — {name} · P {px(p0)} · fix_c {money(c0)}"))
    p0, c0 = max(0.01, p0), max(1.0, c0)

    k = lambda x: f"v2|po|{name}|{p0:.4f}|{c0:.2f}|{x}"            # noqa: E731
    with st.expander("🛠️ Simulator controls"):
        col1, col2, col3 = st.columns(3)
        ni = lambda col, label, key, value, **kw: col.number_input(label, value=value, key=k(key), **kw)   # noqa: E731
        c = {}
        col1.markdown("**🟢 Shannon 1 / Long**")
        c["x0_1"] = ni(col1, "x0₁ จุดอ้างอิง", "x0_1", p0, min_value=0.01, step=1.0)
        c["constant1"] = ni(col1, "C₁ เงินทุน", "c1", c0, min_value=1.0, step=100.0)
        c["b1"] = ni(col1, "b₁ bias", "b1", 0.0, step=100.0)
        c["delta1"] = col1.slider("δ₁ ความชัน x < x0", 0.0, 2.0, 0.2, 0.05, key=k("d1"))
        c["long_qty"] = ni(col1, "Long Qty (y10)", "lq", 100, min_value=0)
        c["long_entry"] = ni(col1, "Long Entry", "le", p0, min_value=0.01, step=1.0)
        col2.markdown("**🟡 Shannon 2 / Short**")
        c["x0_2"] = ni(col2, "x0₂ จุดอ้างอิง", "x0_2", max(p0 * 1.5, 0.01), min_value=0.01, step=1.0)
        c["constant2"] = ni(col2, "C₂ เงินทุน", "c2", c0, min_value=1.0, step=100.0)
        c["b2"] = ni(col2, "b₂ bias", "b2", 0.0, step=100.0)
        c["delta2"] = col2.slider("δ₂ ความชัน x ≥ x0", 0.0, 2.0, 1.0, 0.05, key=k("d2"))
        c["short_qty"] = ni(col2, "Short Qty (y11)", "sq", 100, min_value=0)
        c["short_entry"] = ni(col2, "Short Entry", "se", max(p0 * 1.5, 0.01), min_value=0.01, step=1.0)
        col3.markdown("**⚔️ Options & Benchmark**")
        c["anchor"] = ni(col3, "ราคา Benchmark", "an", p0, min_value=0.01, step=1.0)
        c["ref_const"] = ni(col3, "เงินลงทุนอ้างอิง", "rc", c0, min_value=1.0, step=100.0)
        c["call_qty"] = ni(col3, "Call Qty (y8)", "cq", 100, min_value=0)
        c["call_strike"] = ni(col3, "Call Strike", "cs", p0, min_value=0.01, step=1.0)
        c["put_qty"] = ni(col3, "Put Qty (y9)", "pq", 100, min_value=0)
        c["put_strike"] = ni(col3, "Put Strike", "ps", p0, min_value=0.01, step=1.0)

    lines = st.multiselect("Lines", list(eng.PAYOFF_LINES), default=["y1", "y3", "y6"],
                           format_func=eng.PAYOFF_LINES.get, key="v2_po_lines")
    st.caption("y3 = ผลรวมของเส้นที่เลือกใน y1, y2, y4, y5, y8–y11 (ไม่รวม y6/y7)")
    chart = st.radio("Chart", ["เปรียบเทียบ", "Net vs Benchmark", "Delta Overlay"], horizontal=True,
                     key="v2_po_chart", label_visibility="collapsed")
    xs = eng.payoff_prices(p0)
    ys = eng.payoff_curves(c, xs, net_of=lines)

    import plotly.graph_objects as go

    fig = go.Figure()

    def add(key, label, **line):
        fig.add_trace(go.Scatter(x=xs, y=ys[key], name=label, line=line))
    if chart == "เปรียบเทียบ":
        for key in lines:
            add(key, eng.PAYOFF_LINES[key], color=PAYOFF_COLORS[key], width=3.5 if key == "y3" else 2.5,
                dash="dash" if key in ("y6", "y7") else "solid")
    elif chart == "Net vs Benchmark":
        add("y3", "Net (y3)", color=PAYOFF_COLORS["y3"], width=3.5)
        add("y6", "Benchmark (y6)", color=PAYOFF_COLORS["y6"], width=3, dash="dash")
    else:
        add("overlay", "Net − Benchmark", color="#ea580c", width=3.5)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=p0, line_color="#facc15", opacity=0.8, annotation_text="t")
    fig.update_layout(height=520, xaxis_title="Price (x)", yaxis_title="P/L (y)", margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


# ── ➕ Data ───────────────────────────────────────────────────────────────────────
def stage_import(raw, src: str) -> None:
    try:
        eng.normalize(raw)
    except eng.DataError as e:
        return st.error(f"❌ {e}")
    st.session_state["_v2_import"] = {"raw": raw, "src": src}


def import_export(store, d) -> None:
    if d is not None:
        st.markdown("##### 📤 Export")
        st.download_button("💾 Download JSON", data=json.dumps(d, indent=2, ensure_ascii=False),
                           file_name=f"chain_system_backup_{datetime.now():%Y-%m-%d}.json", mime="application/json")
    st.markdown("##### 📥 Import")
    url = st.text_input("🔗 URL ของไฟล์ JSON (https)", value=DEFAULT_IMPORT_URL, key="v2_import_url")
    if st.button("📥 ดึงจาก URL", key="v2_import_btn"):
        try:
            with st.spinner("กำลังดาวน์โหลด…"):
                raw = eng.fetch_json_url(url)
        except (ValueError, OSError) as e:   # ChainError / bad JSON / network
            st.error(f"❌ {e}")
        else:
            stage_import(raw, url.strip())
    up = st.file_uploader("หรืออัปโหลดไฟล์ JSON ที่ Export ไว้", type=["json"], key="v2_upload")
    if up is not None and st.session_state.get("_v2_up_id") != up.file_id:
        st.session_state["_v2_up_id"] = up.file_id
        try:
            if up.size > eng.MAX_IMPORT_BYTES:
                raise ValueError("ไฟล์ใหญ่เกินไป")
            stage_import(json.loads(up.getvalue().decode("utf-8")), up.name)
        except ValueError as e:
            st.error(f"❌ อ่านไฟล์ไม่ได้: {e}")

    imp = st.session_state.get("_v2_import")
    if imp:
        data, warns = eng.normalize(imp["raw"])
        st.warning(esc(f"⚠️ พร้อม Import จาก {imp['src']} — {len(data['tickers'])} tickers · "
                       f"Pool CF {money(data['global_pool_cf'])} · "
                       f"{sum(len(x['rounds']) for x in data['tickers'])} rounds\n\n"
                       "จะ **เขียนทับ** ข้อมูลปัจจุบัน (ย้อนกลับได้ด้วยปุ่ม Restore)"))
        for w in warns[:5]:
            st.caption(esc(f"• {w}"))
        ok, no = st.columns(2)
        if ok.button("✅ Confirm Import", type="primary", key="v2_imp_ok", width="stretch"):
            def do_import():
                store.replace(imp["raw"])
                st.session_state.pop("_v2_import", None)   # keep the staged file if the write fails
            attempt(do_import, "✅ นำเข้าข้อมูลสำเร็จ")
        if no.button("✖ ยกเลิก", key="v2_imp_no", width="stretch"):
            st.session_state.pop("_v2_import", None)
            st.rerun()


def view_data(store, d, t):
    with st.expander("➕ เพิ่ม Ticker ใหม่", expanded=not d["tickers"]):
        with st.form("v2_add_ticker", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            sym = c1.text_input("Ticker Symbol")
            price = c2.number_input("Initial Price (t)", min_value=0.01, value=100.0)
            fix_c = c3.number_input("Initial Fix_C ($)", min_value=1.0, value=10000.0, step=100.0)
            if st.form_submit_button("Add Ticker"):
                run(lambda d_: eng.add_ticker(d_, sym, price, fix_c), f"✅ Added {sym.strip().upper()}")
    with st.expander("💾 Export / Import", expanded=bool(st.session_state.get("_v2_import"))):
        import_export(store, d)
    with st.expander("↩️ Restore เวอร์ชันก่อนหน้า"):
        when = store.backup_time()
        if when:
            st.caption(f"Backup ล่าสุด: {when} — ระบบเก็บเวอร์ชันก่อนการบันทึกครั้งล่าสุดไว้ 1 ชุด · กดซ้ำเพื่อย้อนกลับ")
            if st.button("↩️ Restore", key="v2_restore"):
                attempt(store.restore_backup, "↩️ Restore เรียบร้อย")
        else:
            st.caption("ยังไม่มี backup (จะถูกสร้างเมื่อบันทึกครั้งถัดไป)")
    with st.expander("⚠️ ล้างข้อมูลทั้งหมด"):
        with st.form("v2_wipe", clear_on_submit=True):
            word = st.text_input("พิมพ์ DELETE เพื่อยืนยัน")
            if st.form_submit_button("ลบทั้งหมด", type="primary"):
                if word.strip() == "DELETE":
                    attempt(lambda: store.replace({"tickers": []}), "🗑️ ล้างข้อมูลแล้ว (Restore ย้อนกลับได้)")
                else:
                    st.error("พิมพ์ DELETE ให้ตรงก่อน")
    st.caption(f"💾 {store.path}")


VIEW_FNS = {"⚡ Engine": view_engine, "📜 History": view_history, "📊 Portfolio": view_portfolio,
            "🏛️ Treasury": view_treasury, "📐 Payoff": view_payoff, "➕ Data": view_data}


# ── page ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    store = get_store(DATA_FILE)
    if msg := st.session_state.pop("_v2_flash", None):
        st.toast(msg)
    st.header("⚡ Chain System — Main Engine · v2")
    st.caption("simple · stable · fast — ใช้ข้อมูลชุดเดียวกับหน้า Main Engine เดิม")
    try:
        d = store.read()
    except eng.DataError as e:
        st.error(f"🛑 {e}")
        st.info("ไฟล์เดิมไม่ถูกแตะต้อง — Import ข้อมูลด้านล่าง หรือเริ่มพอร์ตว่าง")
        import_export(store, None)
        if st.button("🆕 เริ่มพอร์ตว่าง"):
            attempt(lambda: store.replace({"tickers": []}), "🆕 เริ่มพอร์ตใหม่")
        return
    if store.warnings:
        with st.expander(f"⚠️ แจ้งเตือนข้อมูล ({len(store.warnings)})"):
            for w in store.warnings:
                st.write("•", w)

    with st.expander("📖 หลักการ & สมการ"):
        st.markdown("**ขาขึ้น:** กำไร Shannon + Harvest → จ่ายค่า Put Hedge → Surplus → Scale Up `fix_c` (Free Risk)  \n"
                    "**ขาลง:** Put ระเบิดกำไร → **Pool CF** → Deploy (เมื่อ Regime กลับ) + Reserve")
        st.latex(r"\text{Shannon}=c\ln\frac{P}{t}\qquad \text{Surplus}=\text{Shannon}-\text{Hedge}"
                 r"\qquad b'=b+c\ln\frac{P}{t}-c'\ln\frac{P}{t'}")

    tot = eng.totals(d)
    k = st.columns(4)
    k[0].metric("🎱 Pool CF", money(tot["pool"]), border=True)
    k[1].metric("🛡️ EV Reserve", money(tot["reserve"]), border=True)
    k[2].metric(f"⚡ Fix_C · {len(d['tickers'])} tickers", money(tot["fix_c"]), border=True)
    k[3].metric("💵 Cash (Pool + Reserve)", money(tot["cash"]), border=True)

    by_name = {x["ticker"]: x for x in d["tickers"]}
    if by_name and st.session_state.get("v2_ticker") not in by_name:   # first load → first active (fix_c > 0) ticker
        st.session_state["v2_ticker"] = next((n for n, x in by_name.items() if x["current_state"]["fix_c"] >= 1),
                                             next(iter(by_name)))
    pick, nav = st.columns([1, 3], vertical_alignment="center")
    with pick:
        name = st.selectbox("Ticker", list(by_name), key="v2_ticker", label_visibility="collapsed",
                            format_func=lambda n: chip(by_name[n]), placeholder="ยังไม่มี ticker") if by_name else None
    with nav:
        view = st.radio("View", VIEWS, horizontal=True, key="v2_view", label_visibility="collapsed")

    try:   # one broken view must never take the whole page down
        VIEW_FNS[view](store, d, by_name.get(name))
    except Exception as e:  # noqa: BLE001
        st.error(f"⚠️ {view}: {type(e).__name__}: {e}")
        with st.expander("รายละเอียด"):
            st.exception(e)


if __name__ == "__main__":
    main()
