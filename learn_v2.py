"""learn_v2 — the 🎓 Learn view of pages/app-v2.py: the modern (LEGO) equations on Chain System numbers.

Mirrors streamlit_lego's "Rebalancing 101 + Cheat Sheet + Testing Lab" tabs in one native view:
  1. the three lines to read (Rₙ · Aₙ · Eₙ) and how Chain's numbers map onto them
  2. two flow diagrams (one round · rollover) — ``st.graphviz_chart``, coloured for the active theme
  3. break-even explorer — how big a move pays for one round of hedge, on each profit basis
  4. Testing Lab — random price path → Chain Rounds → the same ledger the real data uses
Pure presentation: every number comes from ``ledger_v2`` / ``engine_v2``.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

import engine_v2 as eng
import ledger_v2 as led

CHAIN_FLOW = [
    ("A", "1. ราคาใหม่ P  (= P_fill)\\nP_acted = t  (ราคารอบก่อน)", False),
    ("B", "2. rebalance กลับเข้า Fix_c ที่ราคา P\\n(ขาย/ซื้อส่วนต่าง)", False),
    ("C", "3. ΔA = Fix_c × (P / P_acted − 1)\\nA = A + ΔA", True),
    ("D", "4. ΔR = Fix_c × ln(P / P_acted)  = Shannon\\nR = R + ΔR   ·   E = A − R  (≥ 0)", False),
    ("E", "5. Hedge = (Fix_c / t) × Hedge× × Put(K = 0.9t)\\nSurplus = Shannon − Hedge", False),
    ("G", "Scale Up\\nFix_c′ = Fix_c + Surplus", True),
    ("H", "Fix_c′ = Fix_c\\n(ไม่ลด)", False),
    ("I", "6. rollover: β += (Fix_c − Fix_c′) × ln(P / P₀)\\nP_acted = P   ·   b′ = b + ΔR", False),
]
CHAIN_EDGES = [("A", "B", ""), ("B", "C", ""), ("C", "D", ""), ("D", "E", ""), ("F", "G", "ใช่"),
               ("F", "H", "ไม่"), ("G", "I", ""), ("H", "I", "")]
ROLLOVER_FLOW = [
    ("S", "0. จำ 2 ค่า: Fix_c และ β\\nβ เริ่มที่ 0 · P₀ ตรึงไว้ ไม่เปลี่ยน", False),
    ("A", "1. ได้ทุนใหม่ Fix_c′\\nจุดหมุนคือ P_acted (ราคา act ล่าสุด)", False),
    ("C", "3. β += (Fix_c − Fix_c′) × ln(P_acted / P₀)\\nFix_c = Fix_c′", True),
    ("K", "ข้าม", False),
    ("D", "4. R = β + Fix_c × ln(P / P₀)\\nβ บวกที่ R เท่านั้น · A ไม่บวก β", False),
]
ROLLOVER_EDGES = [("S", "A", ""), ("A", "B", ""), ("B", "C", "ใช่"), ("B", "K", "ไม่"), ("C", "D", "")]


def dot(nodes, edges, extra: str, theme: str | None) -> str:
    """Graphviz source with colours for the active Streamlit theme (transparent background)."""
    dark = theme == "dark"
    fg, line = ("#e6e8ee", "#7fa6f5") if dark else ("#1f2430", "#3b6fd4")
    fill, act = ("#1c2433", "#3a2f12") if dark else ("#e8f1ff", "#fff4d6")
    out = ['digraph G {', 'bgcolor="transparent"; rankdir=TB; nodesep=0.35; ranksep=0.35;',
           f'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.22,0.12", '
           f'color="{line}", fillcolor="{fill}", fontcolor="{fg}"];',
           f'edge [color="{line}", fontcolor="{fg}", fontsize=10, fontname="Helvetica"];']
    for nid, label, hot in nodes:
        style = f', fillcolor="{act}", color="#d49a1a"' if hot else ""
        out.append(f'{nid} [label="{label}"{style}];')
    out.append(extra)
    for a, b, label in edges:
        out.append(f'{a} -> {b}' + (f' [label=" {label} "]' if label else "") + ";")
    out.append("}")
    return "\n".join(out)


def equations() -> None:
    st.markdown("**อ่านกำไรที่ Eₙ — ไม่ใช่ Aₙ และไม่ใช่ Rₙ** (หลักเดียวกับ streamlit_lego · Rebalancing 101)")
    c1, c2, c3 = st.columns(3)
    c1.markdown("##### Rₙ · เส้นอ้างอิง")
    c1.latex(r"R_n=\beta+F\,\ln\frac{P_n}{P_0}")
    c1.caption("ทุน × เทรนด์ — ขึ้นลงตามราคา ไม่ใช่ฝีมือ")
    c2.markdown("##### Aₙ · เงินสดจริงสะสม")
    c2.latex(r"\begin{aligned}\Delta A_n&=F\Big(\frac{P_{fill}}{P_{acted}}-1\Big)\\ A_n&=\textstyle\sum\Delta A\end{aligned}")
    c2.caption("ยังปนเทรนด์อยู่ — อ่านเป็นกำไรตรง ๆ ไม่ได้")
    c3.markdown("##### Eₙ · กำไร harvest จริง")
    c3.latex(r"E_n=A_n-R_n\ \ge 0")
    c3.caption("ตัดเทรนด์ทิ้ง เหลือกำไรจากความแกว่งล้วน ๆ — ไม่ลดลงเลย เพราะ x − 1 ≥ ln x")
    st.markdown("##### Chain ↔ LEGO — ตัวเลขของ Chain System คืออะไรในสมการ")
    st.markdown(
        "| Chain System | LEGO (streamlit_lego) |\n|---|---|\n"
        "| `fix_c` (c) | `Fix_c` |\n"
        "| `t` ราคารอบก่อน · `P New` | `P_acted` · `P_fill` |\n"
        "| **Shannon** = `c·ln(P/t)` | **ΔR** — ส่วนเพิ่มของ Rₙ (**ไม่ใช่** ΔA และไม่ใช่ E) |\n"
        "| **Baseline `b`** | Rₙ ณ จุดหมุน — rollover เทียบเท่ากัน (Chain ย้าย anchor `t` ทุกรอบ, LEGO ตรึง `P₀` + `β`) |\n"
        "| **Scale Up** (fix_c เปลี่ยน) | rollover: `β += (Fix_c − Fix_c′)·ln(P_acted/P₀)` |\n"
        "| — | **ΔAₙ · Aₙ · Eₙ** → v2 คำนวณจาก round log ให้ในแท็บ 🧬 Ledger |\n"
        "| Hedge · Pool CF · EV Reserve | — (มีเฉพาะ Chain) |")
    st.info("Shannon ของ Chain คือฝั่ง **R** (เส้น ln) — แท็บ 🧬 Ledger จึงแสดง A และ **E (กำไร harvest จริง)** เพิ่มเติมจากข้อมูลชุดเดิม "
            "โดยไม่แก้ตัวเลขที่บันทึกไว้ · Legacy Rollover ไม่นับ (LEGO: ไม่เดาสูตรให้หลักฐานเก่า)", icon="🧬")


def flows(theme: str | None) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### 1 รอบ — จาก Main rebalancing → Chain Round")
        st.graphviz_chart(dot(CHAIN_FLOW, CHAIN_EDGES, 'F [shape=diamond, label="Surplus > 0 ?"]; E -> F;', theme))
    with right:
        st.markdown("##### Rollover — เมื่อ Fix_c เปลี่ยน")
        st.graphviz_chart(dot(ROLLOVER_FLOW, ROLLOVER_EDGES,
                              'B [shape=diamond, label="2. ใช้ได้ ?\\nFix_c′ ≠ Fix_c · P_acted > 0"];', theme))
    st.caption("ต้นฉบับ: cheat_sheet/main_rebalancing.mmd และ rollover.mmd ใน firstnattapon/streamlit_lego")


def break_even_explorer() -> None:
    st.markdown("#### ⚖️ ขยับแค่ไหนถึงจ่ายเบี้ยประกัน 1 รอบได้")
    c1, c2 = st.columns(2)
    hedge_x = c1.slider("Hedge ×", 0.0, 3.0, 1.0, 0.25, key="learn_hx",
                        help="1.0 = พอดีตัว · 2.0 = Over-hedge")
    fix_c = c2.number_input("Fix_c (USD)", min_value=1.0, value=1000.0, step=100.0, key="learn_fc")
    rate = led.hedge_rate()
    be = led.break_even(hedge_x)
    m = st.columns(4)
    m[0].metric("เบี้ย 1 รอบ", f"${fix_c * hedge_x * rate:,.2f}", f"{hedge_x * rate:.1%} ของ Fix_c", delta_color="off")
    m[1].metric("ฐาน R (ln — Shannon ของ Chain)", f"{be['R']:+.1%}", help="ln(1+m) = เบี้ย/Fix_c")
    m[2].metric("ฐาน A (เงินสดจริง ΔA)", f"{be['A']:+.1%}", help="m = เบี้ย/Fix_c")
    m[3].metric("ฐาน E (harvest ก้าวเดียว)", f"{be['E_up']:+.0%} / {be['E_down']:+.0%}",
                help="m − ln(1+m) = เบี้ย/Fix_c — แก้สมการเป็นขาขึ้น/ขาลง")
    moves = [i / 100 for i in range(-50, 101, 2)]
    frame = pd.DataFrame({
        "เบี้ย Hedge": [fix_c * hedge_x * rate] * len(moves),
        "ΔR = Fix_c·ln(1+m)": [fix_c * math.log1p(x) for x in moves],
        "ΔA = Fix_c·m": [fix_c * x for x in moves],
        "ΔE = Fix_c·(m − ln(1+m))": [fix_c * (x - math.log1p(x)) for x in moves],
    }, index=[round(x * 100) for x in moves])
    frame.index.name = "ราคาเปลี่ยน % (P/t − 1)"
    st.line_chart(frame)
    st.caption("ฐาน R/A: ขยับราว 12–13% ก็ครอบเบี้ย 1 รอบ · ฐาน E (กำไร harvest ล้วน) ก้าวเดียวต้องขยับแรงกว่ามาก — "
               "E สะสมจากการแกว่งซ้ำหลายรอบ ไม่ใช่ก้าวเดียว (ดู Testing Lab) · ค่าคงที่โมเดล: Put 1 ปี OTM 10% σ 50% r 4%")


def testing_lab() -> None:
    st.markdown("#### 🧪 Testing Lab — สุ่มราคา → Chain Round → ledger เดียวกับข้อมูลจริง")
    if "lab_seed" not in st.session_state:
        st.session_state["lab_seed"] = 101
    r1 = st.columns(4)
    fix_c = r1[0].number_input("Fix_c เริ่มต้น", min_value=1.0, value=1500.0, step=100.0, key="lab_fc")
    p0 = r1[1].number_input("ราคาเริ่มต้น P₀", min_value=0.01, value=100.0, format="%.4f", key="lab_p0")
    vol = r1[2].number_input("ความผันผวน/ก้าว (%)", min_value=0.0, max_value=40.0, value=4.0, step=0.1, key="lab_vol")
    drift = r1[3].number_input("แนวโน้ม/ก้าว (%)", min_value=-10.0, max_value=10.0, value=0.0, step=0.01, key="lab_dr")
    r2 = st.columns(4)
    steps = r2[0].number_input("จำนวนก้าว", min_value=2, max_value=led.MAX_SIM_STEPS, value=120, step=1, key="lab_steps")
    every = r2[1].number_input("Chain Round ทุกกี่ก้าว", min_value=1, max_value=led.MAX_SIM_STEPS, value=20, step=1, key="lab_every")
    hedge_x = r2[2].number_input("Hedge ×", min_value=0.0, value=1.0, step=0.5, key="lab_hx")
    r2[3].number_input("Seed", min_value=0, step=1, key="lab_seed")
    f1, f2 = st.columns(2)
    no_hedge, no_surplus = f1.checkbox("No Hedge", key="lab_nh"), f2.checkbox("No Surplus (fix_c คงที่)", key="lab_ns")
    try:
        prices = led.simulate_prices(float(p0), float(vol) / 100, float(drift) / 100, int(steps), int(st.session_state["lab_seed"]))
        sim = led.simulate_chain(float(p0), prices, float(fix_c), float(hedge_x), int(every), no_hedge, no_surplus)
    except eng.ChainError as e:
        return st.error(f"คำนวณไม่ได้: {e}")
    S = sim["ledger"]["summary"]
    m = st.columns(4)
    m[0].metric("Chain Round", f"{S['acts']} รอบ", f"fix_c {fix_c:,.0f} → {sim['final_fix_c']:,.0f}", delta_color="off")
    m[1].metric("Aₙ เงินสดสะสม", f"{S['A']:+,.2f}")
    m[2].metric("Eₙ กำไร harvest", f"{S['E']:+,.2f}")
    m[3].metric("E − Σ Hedge", f"{S['e_minus_hedge']:+,.2f}", f"Σ Hedge {S['hedge']:,.2f}", delta_color="off")
    ser = led.series(sim["ledger"])
    st.line_chart(pd.DataFrame(ser, index=sim["steps"]).rename_axis("ก้าวที่"))
    table = pd.DataFrame(led.table(sim["ledger"]))
    st.dataframe(table, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(table)))
    st.download_button("ดาวน์โหลด CSV (Testing Lab)", data=table.to_csv(index=False), file_name="chain_lab_ledger.csv",
                       mime="text/csv", key="lab_csv")
    st.caption("แบบจำลองเพื่อการเรียนรู้ · ราคาสุ่ม (geometric random walk) · ไม่รวม spread, slippage, ภาษี · "
               "เบี้ย Hedge คิดเต็ม 1 ปีทุกรอบ จึงอย่าตั้ง Chain Round ถี่เกินจริง")


def render(theme: str | None = None) -> None:
    equations()
    st.divider()
    flows(theme)
    st.divider()
    break_even_explorer()
    st.divider()
    testing_lab()
