"""⚡ Chain System · v3 — minimal  (5 ขั้น + 1 เช็ก)

app_v2_minimal.mmd as a page: ① harvest · ② pay the insurance first · ③ is there a surplus? · ④ grow only from it ·
⑤ re-center — then one check, Ev Efficiency = ΣH ÷ ΣK ≥ 1.0x (ΣH = Shannon of the Chain Rounds, ΣK = Ev burn),
and the goal. Nothing else: no treasury, ledger, payoff or import (those stay in app-v2); the only override is the
real premium K. Same portfolio file as app / app-v2, so a round committed here shows up there; the numbers come
from the same ``engine_v2.plan_round`` the minimal loop was proven against (tests/test_minimal_loop.py).
"""
import os
import sys
import zlib
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import engine_v2 as eng  # noqa: E402

st.set_page_config(page_title="Chain System v3", page_icon="🎯", layout="centered")

DATA_FILE = os.environ.get("CHAIN_V2_DATA_FILE") or str(ROOT / "trading_data.json")
RATE = eng.bs_put(1.0, eng.HEDGE_STRIKE)    # 12.4% — the whole Black-Scholes model is this one constant
GOAL = ("🏆 **เป้าหมายสูงสุด · Self-funding Compounding**  \n"       # word for word the goal node of app_v2_minimal.mmd
        "ให้ **ความแกว่งของราคาจ่ายค่าประกันของตัวเอง**  \n"
        "แล้วเอากำไรที่เหลือ **ขยายทุน (fix_c) วนเป็นลูกโซ่ ↻ ทุกรอบ**  \n"
        "Ev Efficiency ≥ 1.0x · Surplus เป็นบวก · Eₙ ไม่ลด  \n"
        "ทุนเดิมไม่เพิ่มความเสี่ยง (Free Risk)")
money = eng.money


@st.cache_resource(show_spinner=False)
def get_store(path: str, api: int) -> eng.Store:   # ``api`` keys the cache so a new engine never reuses an old Store
    return eng.Store(path)


def esc(text: str) -> str:
    """Streamlit markdown treats ``$…$`` as LaTeX — escape dollars in captions / toasts."""
    return text.replace("$", "\\$")


def px(x: float) -> str:
    return f"${x:,.2f}" if x >= 1 or x == 0 else f"${x:,.4f}"


def efficiency(t: dict) -> tuple[float, float, int]:
    """ΣH (Shannon harvested by the Chain Rounds), ΣK (Ev burn = every premium paid) and the number of rounds."""
    rows = [r for r in t["rounds"] if r.get("action") == "Chain Round"]
    return sum(eng.num(r.get("shannon_profit")) for r in rows), eng.num(t["current_state"].get("cumulative_ev")), len(rows)


def steps(plan: dict, ratio: float, real_k: bool) -> str:
    """The five steps with this round's numbers — the flowchart, as text."""
    c, t, p = plan["c_before"], plan["p_old"], plan["p_new"]
    h, k, s, c1 = plan["shannon_profit"], plan["hedge_cost"], plan["surplus"], plan["c_after"]
    k_rule = "K = เบี้ยจริงที่ใส่" if real_k else f"K = {RATE:.1%} · c · Hedge× = {RATE:.1%} × {c:,.2f} × {ratio:g}"
    return "\n".join([
        f"① H = c · ln(P / t) = {c:,.2f} × ln({p:g} / {t:g}) = {money(h)}",
        f"② {k_rule} = {money(k)}",
        f"③ Surplus = H − K = {money(h)} − {money(k)} = {money(s)}   {'✓ เป็นบวก' if s > 0 else '✗ ไม่เป็นบวก'}",
        f"④ c′ = c + Surplus = {money(c)} + {money(s)} = {money(c1)}   (โตด้วยกำไรเท่านั้น)" if s > 0
        else f"④ c′ = c = {money(c1)}   (ไม่ลด · เสียเบี้ย {money(k)})",
        f"⑤ t′ = P = {p:g}   → รอบถัดไป",
    ])


def commit(store: eng.Store, name: str, plan: dict) -> None:
    try:
        store.update(lambda d: eng.apply_round(d, name, plan))
    except (ValueError, eng.DataError) as e:   # ChainError / StaleState are ValueErrors
        st.error(f"❌ {e}")
    except OSError as e:
        st.error(f"💾 บันทึกไฟล์ไม่สำเร็จ: {e}")
    else:
        st.session_state["_v3_flash"] = esc(f"✅ {name}: fix_c → {money(plan['c_after'])} · t → {plan['p_new']:g}")
        st.rerun()


def main() -> None:
    if msg := st.session_state.pop("_v3_flash", None):
        st.toast(msg)
    st.header("⚡ Chain System · v3 — minimal")
    st.caption("5 ขั้น + 1 เช็ก ตามแผนผัง `app_v2_minimal.mmd` · ข้อมูลชุดเดียวกับ app / app-v2 "
               "(Treasury · Ledger · Payoff · Import ยังอยู่ที่ app-v2)")
    store = get_store(DATA_FILE, eng.API_VERSION)
    try:
        d = store.read()
    except eng.DataError as e:
        st.error(f"🛑 {e}")
        return st.info("ไฟล์ข้อมูลไม่ถูกแตะต้อง — Import / Restore ที่ app-v2 → ➕ Data")
    by_name = {x["ticker"]: x for x in d["tickers"]}
    if not by_name:
        return st.info("ยังไม่มี ticker — เพิ่มที่ app-v2 → ➕ Data")

    active = next((n for n, x in by_name.items() if x["current_state"]["fix_c"] >= 1), next(iter(by_name)))
    if st.session_state.get("v3_ticker") not in by_name:
        st.session_state["v3_ticker"] = active
    name = st.selectbox("Ticker", list(by_name), key="v3_ticker")
    t = by_name[name]
    s = t["current_state"]
    col_c, col_t = st.columns(2)
    col_c.metric("c = fix_c", money(s["fix_c"]), border=True)
    col_t.metric("t = ราคารอบก่อน", px(s["price"]), border=True)

    if s["fix_c"] <= 0 or s["price"] <= 0:
        st.info("ticker นี้ไม่มีตำแหน่งให้วน (fix_c หรือราคา = 0) — ตั้งต้นที่ app-v2 → Override")
    else:
        sig = f"{name}|{s['price']}|{s['fix_c']}|{s['baseline']}"        # a new state ⇒ fresh widgets with fresh defaults
        k = lambda x: f"v3|{x}|{sig}"                                      # noqa: E731
        dec = 2 if s["price"] >= 1 else 4
        st.session_state.setdefault(k("pnew"), max(float(s["price"]), 1e-4))
        p_col, h_col, k_col = st.columns(3)
        p_new = p_col.number_input("P New", min_value=1e-4, step=10.0 ** -dec, format=f"%.{dec}f", key=k("pnew"),
                                   help="ราคาใหม่ของรอบนี้ — เทียบกับ t เพื่อหา H = c · ln(P / t)")
        ratio = h_col.number_input("Hedge ×", min_value=0.0, value=float(d["settings"]["default_hedge_ratio"]),
                                   step=0.5, key=k("hx"), help="1 = พอดีตัว · 2 = Over-hedge")
        real = k_col.number_input("เบี้ยจริง K ($)", min_value=0.0, value=None, step=10.0, format="%.2f", key=k("k"),
                                  placeholder="ว่าง = ใช้สูตร", help="ใส่ค่า Put ที่จ่ายจริง · ว่าง = 12.4% · c · Hedge×")
        try:
            plan = eng.plan_round(s, p_new, ratio, hedge=real)
        except eng.ChainError as e:
            return st.error(f"❌ {e}")
        st.code(steps(plan, ratio, real is not None), language=None, wrap_lines=True)
        plan_id = zlib.crc32(repr(sorted(plan.items())).encode())          # a click on an outdated plan finds no button
        if st.button("✅ COMMIT — บันทึกรอบนี้", type="primary", width="stretch", key=f"v3|commit|{name}|{plan_id:08x}"):
            commit(store, name, plan)

    h, kk, n = efficiency(t)
    eff = h / kk if kk > 0 else None
    st.metric("วัดผล · Ev Efficiency = ΣH ÷ ΣK", "—" if eff is None else f"{eff:.2f}x",
              None if eff is None else ("Sustainable" if eff >= 1 else "Bleeding"),
              delta_color="normal" if (eff or 0) >= 1 else "inverse", border=True)
    st.caption(esc(f"ΣH = {money(h)} (Shannon ของ {n} Chain Round) · ΣK = {money(kk)} (Ev burn = เบี้ยสะสม)"))
    reached = eff is not None and eff >= 1.0
    (st.success if reached else st.info)(("✅ ถึงเป้าหมายแล้ว\n\n" if reached else "ยังไม่ถึง 1.0x — วนรอบถัดไป ↻\n\n") + GOAL)


if __name__ == "__main__":
    main()
