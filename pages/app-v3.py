"""🎯 Chain System · v3 — a page that explains itself.

The one idea: let the swing of the price pay for its own insurance — then grow the capital only from what is left.
A round is five steps — ① harvest the swing (E) · ② pay the insurance first · ③ what is left? · ④ grow only from
that · ⑤ start the next round here — and one score, Ev Efficiency = ΣE ÷ ΣK ≥ 1.0x. The Surplus is read at E, the
LEGO equation E = A − R (the trend is cut out), so only the *swing* of the price can pay for the insurance.

Standalone: the numbers come from chain_v3.py (stdlib only) and the rounds live in this page's own file
(data/app_v3.json, or the path in APP_V3_DATA_FILE). Every symbol on the screen is explained in the glossary.
"""
import os
import sys
import zlib
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import chain_v3 as cv  # noqa: E402

st.set_page_config(page_title="Chain System v3", page_icon="🎯", layout="centered")

PATH = os.environ.get("APP_V3_DATA_FILE") or str(ROOT / "data" / "app_v3.json")
TITLE = "🎯 Chain System v3"
DEMO = ("DEMO", 1000.0, 100.0)                  # name · c · t of the one-click example …
DEMO_PATH = (200.0, 100.0)                      # … whose price doubles, then comes back to where it started
DEMO_HINT = ("ตัวอย่างรอบ 1: ราคาขึ้นไป 200 (2 เท่า) — กด “บันทึกรอบนี้” แล้วดูรอบถัดไป",
             "ตัวอย่างรอบ 2: ราคากลับมา 100 — กด “บันทึกรอบนี้” แล้วดูคะแนนด้านล่าง")
ESSENCE = "ให้ **ความแกว่งของราคา** จ่าย **ค่าประกันของตัวเอง** — ที่เหลือค่อยเอาไปขยายทุน"
FIVE = "**① เก็บ** → **② จ่ายประกันก่อน** → **③ เหลือไหม?** → **④ โตเฉพาะที่เหลือ** → **⑤ เริ่มรอบใหม่ ↻**"
LEGEND = ("**c** ทุน · **t** ราคารอบก่อน · **P** ราคาใหม่ · **E** กำไรจากความแกว่ง · **K** ค่าประกัน · **S** ส่วนที่เหลือ · "
          "**c′** ทุนใหม่")
HOW_TO = ("วิธีใช้ 3 ขั้น: **1)** เริ่มติดตาม — ตั้งชื่อ ทุน c และราคาเริ่มต้น t · **2)** ราคาเปลี่ยนเมื่อไร ใส่ราคาใหม่ P "
          "แล้วอ่านตัวเลข 5 ขั้น · **3)** กด “บันทึกรอบนี้” แล้วทำซ้ำทุกครั้งที่ราคาเปลี่ยน")
GOAL = ("🏆 **เป้าหมายสูงสุด · Self-funding Compounding**  \n"
        "ให้ **ความแกว่งของราคาจ่ายค่าประกันของตัวเอง**  \n"
        "แล้วเอากำไรที่เหลือ **ขยายทุน (fix_c) วนเป็นลูกโซ่ ↻ ทุกรอบ**  \n"
        "Ev Efficiency ≥ 1.0x · Surplus เป็นบวก · Eₙ ไม่ลด  \n"
        "ทุนเดิมไม่เพิ่มความเสี่ยง (Free Risk)")
GOAL_PLAIN = "ถ้าพูดง่ายๆ: ค่าประกันจ่ายจากกำไรของความแกว่งเอง ไม่ใช่จากกระเป๋าเรา — ทุนโตได้อย่างเดียว ไม่เคยลด"
_LOW, _HIGH = cv.breakeven_prices(1000.0, 100.0, 1.0)      # the two break-even prices at Hedge × 1, for t = 100


def pct(x: float, digits: int = 0) -> str:
    return f"{x:+.{digits}%}".replace("-", "−")


# Every symbol the page shows: (symbol, name, what it is, what it is for)
GLOSSARY = (
    ("ตัวเลขที่คุณถือและกรอก", (
        ("c", "ทุน", "เงินที่ถือไว้ในตำแหน่งนี้ (เช่น 1,000 ดอลลาร์)", "เป็นฐานของทุกสูตร และเป็นสิ่งที่ระบบพยายามทำให้โต"),
        ("t", "ราคาอ้างอิง", "ราคาที่เป็นจุดเริ่มของรอบนี้ (คือราคาของรอบก่อน)", "วัดว่าราคาขยับจากจุดเดิมไปเท่าไร"),
        ("P", "ราคาใหม่", "ราคาปัจจุบันที่คุณกรอกในรอบนี้", "เทียบกับ t เพื่อรู้ว่าราคาขึ้นหรือลง และเท่าไร"),
        ("Hedge ×", "จำนวนเท่าของประกัน", "1 = ประกันเต็มทุน c · 2 = ซื้อเกินตัว 2 เท่า · 0 = ไม่ซื้อ",
         "ปรับระดับความอุ่นใจ — ยิ่งสูง ค่าประกันยิ่งแพงตามสัดส่วน"),
    )),
    ("ตัวเลขที่ระบบคำนวณให้ทุกรอบ", (
        ("E", "กำไรจากความแกว่ง", "E = c · (x − 1 − ln x) เมื่อ x = P / t (คือ A − R ของ LEGO) — ตัดเทรนด์ทิ้ง เหลือแต่ความแกว่ง "
         "ขึ้นหรือลงก็ได้ ไม่เคยติดลบ", "คือ “รายได้” ของรอบนี้ ไว้จ่ายค่าประกัน — ยิ่งแกว่งแรงยิ่งได้มาก"),
        ("K", "ค่าประกัน", f"K = {cv.PREMIUM_RATE:.1%} · c · Hedge× — ราคาประกัน Put ต่ำกว่าตลาด 10% อายุ 1 ปี "
         "(Put = สิทธิ์ขายราคาตายตัว ไว้กันราคาตก) หรือกรอกค่าที่จ่ายจริง", "คือ “ต้นทุน” ของรอบนี้ — จ่ายก่อนเสมอ"),
        ("S", "ส่วนที่เหลือ (Surplus)", "S = E − K — กำไรจากความแกว่งหลังหักค่าประกัน",
         "บวก = ความแกว่งจ่ายค่าประกันครบและยังเหลือ · ลบ = แกว่งยังไม่พอจ่ายรอบนี้"),
        ("c′", "ทุนใหม่", "c′ = c + max(0, S) — เพิ่มเฉพาะเมื่อ S เป็นบวก", "ทุนโตจากกำไรเท่านั้น และไม่เคยลด"),
        ("t′", "ราคาอ้างอิงใหม่", "t′ = P — รอบถัดไปเริ่มนับที่ราคานี้", "แยกแต่ละรอบออกจากกัน ไม่ปนกับรอบเก่า"),
        ("P*", "ราคาคุ้มทุน", f"ราคาที่ E เท่ากับ K พอดี มี 2 ด้าน: ขึ้นไป ≈ {pct(_HIGH / 100 - 1)} หรือลงไป ≈ {pct(_LOW / 100 - 1)} "
         "(เมื่อ Hedge× 1)", "บอกก่อนบันทึกว่าราคาต้องแกว่งไปไกลแค่ไหน ค่าประกันถึงคุ้ม"),
    )),
    ("คะแนนรวมทุกรอบ", (
        ("ΣE", "กำไรสะสมจากความแกว่ง", "ผลรวม E ของทุกรอบที่บันทึก (คือ Eₙ ในสมการ LEGO)", "ตัวตั้งของ Ev Efficiency"),
        ("ΣK", "ค่าประกันสะสม", "ผลรวม K ของทุกรอบที่บันทึก", "ตัวหารของ Ev Efficiency"),
        ("Ev Efficiency", "คะแนนจ่ายตัวเอง", "ΣE ÷ ΣK — กำไรสะสมจากความแกว่งเป็นกี่เท่าของค่าประกันสะสม",
         "ตัวเดียวที่บอกว่าถึงเป้าหมายหรือยัง — 1.0x ขึ้นไป = ความแกว่งจ่ายค่าประกันสะสมได้ครบแล้ว"),
    )),
)
TERMS = ("คำในกล่องเป้าหมาย: **fix_c** = ทุน c · **Eₙ** = ΣE กำไรจากความแกว่งสะสม ไม่มีวันติดลบ (คณิตศาสตร์ x − 1 ≥ ln x) "
         "จึงไม่ลดลง · **Free Risk** = ทุนโตจากกำไรเท่านั้น ไม่เอาเงินใหม่มาเสี่ยงเพิ่ม · **A** = c · (x − 1) เงินสดจากการ rebalance · "
         "**R** = c · ln x เส้นอ้างอิงตามเทรนด์ (E = A − R) · **ln** = ลอการิทึมธรรมชาติ (ปุ่ม ln ในเครื่องคิดเลข) · **e** ≈ 2.718")
FOOTER = ("เครื่องมือคำนวณและจดบันทึกเพื่อการเรียนรู้ — ไม่ใช่คำแนะนำการลงทุน และไม่ส่งคำสั่งซื้อขายจริง · "
          f"ข้อมูลเก็บในไฟล์ {Path(PATH).name} บนเครื่องที่รันแอป")


# ── small helpers ─────────────────────────────────────────────────────────────────────
def esc(text: str) -> str:
    """Streamlit markdown reads ``$…$`` as LaTeX — escape every dollar sign."""
    return text.replace("$", "\\$")


def usd(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.2f}"


def sg(x: float) -> str:
    return f"{'−' if x < 0 else '+'}${abs(x):,.2f}"


def neg(x: float) -> str:
    """A cost: −$123.96 (just $0.00 when nothing was paid)."""
    return f"−{usd(x)}" if x > 0 else usd(x)


def mult(x: float) -> str:
    return f"{x:.2f}x".replace("-", "−")


def px(x: float) -> str:
    """A price without trailing zeros: 125 · 113.2 · 0.0042."""
    return f"{x:,.4f}".rstrip("0").rstrip(".") or "0"


def sym(s: str) -> str:
    """A symbol in bold; a literal ``*`` (as in P*) is escaped so Markdown does not read it as emphasis."""
    return "**" + s.replace("*", "\\*") + "**"


def table(head: str, rows) -> str:
    """A Markdown table; ``head`` and each row are cells separated by ' | '."""
    cols = head.count("|") + 1
    return "\n".join([f"| {head} |", "|" + "---|" * cols, *(f"| {r} |" for r in rows)])


def act(fn, *args, ok: str, **after) -> None:
    """One change to the book. Success → remember a toast (plus the session keys in ``after``) and redraw.
    Failure → say why; nothing was saved."""
    try:
        fn(PATH, *args)
    except cv.BookError as e:
        st.error(f"❌ {e}")
    except OSError as e:
        st.error(f"💾 บันทึกไฟล์ไม่สำเร็จ: {e}")
    else:
        st.session_state.update(_v3_flash=esc(ok), **after)
        st.rerun()


# ── the explanation ─────────────────────────────────────────────────────────────────
def intro() -> None:
    one = cv.plan(DEMO[1], DEMO[2], DEMO_PATH[0], 1.0)                 # the worked example is computed, never typed in
    back = cv.plan(one["c_after"], DEMO_PATH[0], DEMO_PATH[1], 1.0)
    st.title(TITLE)
    st.markdown(f"#### {ESSENCE}")
    st.info(esc(
        "🔑 **แก่นแท้:** ราคาแกว่ง → เก็บกำไรจากความแกว่ง (**E**) → จ่ายค่าประกัน (**K**) ก่อนเสมอ → เหลือเท่าไร (**S**) → "
        "ค่อยเอาไปขยายทุน (**c′**) · ถ้าไม่เหลือ ทุนเท่าเดิม **ไม่ลด**\n\n"
        f"**ตัวอย่าง** ทุน {usd(one['c'])} · ราคา {px(one['t'])} → {px(one['p'])}  \n"
        f"① เก็บได้ {sg(one['E'])}  \n② จ่ายประกัน {neg(one['K'])}  \n③ เหลือ {sg(one['S'])}  \n"
        f"④ ทุนโตเป็น {usd(one['c_after'])}\n\n"
        f"แล้วราคากลับมา {px(back['p'])} → ทุนโตอีกเป็น {usd(back['c_after'])} — **ราคาเท่าเดิม แต่ทุนโต** "
        "เพราะเก็บจากความแกว่งทั้งขึ้นและลง"))
    st.markdown(FIVE)
    st.markdown(LEGEND)
    st.caption(HOW_TO)
    with st.expander("📖 ตัวแปรทุกตัวคืออะไร · ไว้ทำไม"):
        for title, rows in GLOSSARY:
            st.markdown(f"**{title}**")
            st.markdown(esc(table("ตัวแปร | คืออะไร | ไว้ทำไม", (f"{sym(s)} | **{n}** — {what} | {why}" for s, n, what, why in rows))))
        st.caption(TERMS)


def start_form() -> None:
    n = st.session_state.get("_v3_form", 0)                           # a new number = a fresh, empty form after a start
    with st.form("v3_start"):
        name = st.text_input("ชื่อ", key=f"v3_name|{n}", max_chars=cv.MAX_NAME, placeholder="เช่น BTC · TSLA · ทองคำ")
        c = st.number_input("ทุน c — เงินที่ถือไว้ในตัวนี้", key=f"v3_c|{n}", min_value=0.01, value=1000.0, step=100.0,
                            format="%.2f")
        t = st.number_input("ราคาเริ่มต้น t — ราคาตอนนี้", key=f"v3_t|{n}", min_value=0.0, value=None, step=0.01,
                            format="%.4f", placeholder="เช่น 100")
        if st.form_submit_button("🚀 เริ่มติดตาม", width="stretch"):
            act(cv.start, name, c, t, ok=f"✅ เริ่มติดตาม {name.strip()}", _v3_select=name.strip(), _v3_form=n + 1)


# ── one round ───────────────────────────────────────────────────────────────────────
def steps_table(pl: dict) -> str:
    """The five steps with this round's numbers."""
    c, t, p, e, k, s, c1 = (pl[x] for x in ("c", "t", "p", "E", "K", "S", "c_after"))
    x = p / t
    rule = ("K = ค่าประกันที่จ่ายจริง (กรอกเอง)" if pl["real_k"]
            else f"K = {cv.PREMIUM_RATE:.1%} · c · Hedge× = {cv.PREMIUM_RATE:.1%} × {c:,.2f} × {pl['hedge_x']:g}")
    grow = f"c′ = c + S = {c:,.2f} + {s:,.2f}" if s > 0 else "c′ = c — ไม่มีส่วนเหลือ ทุนจึงไม่โต (และไม่ลด)"
    return table("ขั้น | คำนวณ | ได้", (
        f"**① E** เก็บกำไรจากความแกว่ง | x = P / t = {px(p)} / {px(t)} = {px(x)} · "
        f"E = c · (x − 1 − ln x) = {c:,.2f} × ({px(x)} − 1 − ln {px(x)}) | **{sg(e)}**",
        f"**② K** จ่ายค่าประกันก่อน | {rule} | **{neg(k)}**",
        f"**③ S** เหลือไหม? | S = E − K = {usd(e)} − {usd(k)} | **{sg(s)}** {'✓ เหลือ' if s > 0 else '✗ ไม่เหลือ'}",
        f"**④ c′** โตเฉพาะส่วนที่เหลือ | {grow} | **{usd(c)} → {usd(c1)}**",
        f"**⑤ t′** เริ่มรอบใหม่ | t′ = P · รอบถัดไปเริ่มนับที่ราคานี้ | **{px(t)} → {px(p)}**"))


def breakeven_note(c: float, t: float, hedge: float, real, dec: int) -> str:
    """Where the price must go for this round's swing to pay for its insurance — above or below t."""
    low, high = cv.breakeven_prices(c, t, hedge, real)
    if (low, high) == (t, t):
        return "💡 ไม่มีค่าประกันรอบนี้ — ราคาขยับเท่าไรก็เหลือ"
    sides = [f"**{word} {price:,.{dec}f}** ({pct(price / t - 1, 1)})" for word, price in (("สูงกว่า", high), ("ต่ำกว่า", low))
             if price is not None]
    if not sides:
        return "💡 ค่าประกันสูงเกินไป — ไม่มีราคาไหนจ่ายคืนได้"
    return (f"💡 ราคาคุ้มทุน {sym('P*')} — ราคาใหม่ P ต้อง {' หรือ '.join(sides)} ค่าประกันรอบนี้ถึงจะคุ้ม "
            "(แกว่งน้อยกว่านี้ยังไม่เหลือ)")


def run_round(name: str, pos: dict) -> None:
    c, t = cv.current(pos)
    n = len(pos["rounds"])
    st.subheader("🔁 รอบถัดไป — ราคาไปที่ไหนแล้ว?")
    st.caption(esc(f"ติดตามมาแล้ว {n} รอบ · เริ่มที่ทุน {usd(pos['c0'])} ราคา {px(pos['t0'])}"))
    col_c, col_t = st.columns(2)
    col_c.metric("c · ทุนตอนนี้", usd(c), border=True, help="เงินที่ถือไว้ในตำแหน่งนี้ — โตเมื่อรอบไหนมีส่วนเหลือ ไม่เคยลด")
    col_t.metric("t · ราคาอ้างอิง", px(t), border=True, help="ราคาที่เป็นจุดเริ่มของรอบนี้ = ราคาของรอบก่อน")

    sig = f"{name}|{n}|{c}|{t}"                                         # a new state ⇒ fresh widgets with fresh defaults

    def key(part: str) -> str:
        return f"v3|{part}|{sig}"

    dec = 2 if t >= 1 else 4
    on_demo = ((name, pos["c0"], pos["t0"]) == DEMO and n < len(DEMO_PATH)
               and [r["p"] for r in pos["rounds"]] == list(DEMO_PATH[:n]))        # the example, still on its path
    if on_demo:
        st.caption(f"👆 {DEMO_HINT[n]}")
    st.session_state.setdefault(key("p"), float(DEMO_PATH[n] if on_demo else t))
    col_p, col_h, col_k = st.columns(3)
    p = col_p.number_input("ราคาใหม่ P", min_value=0.0, step=10.0 ** -dec, format=f"%.{dec}f", key=key("p"),
                           help="ราคาปัจจุบัน — เทียบกับ t เพื่อหา E = c · (x − 1 − ln x) เมื่อ x = P / t")
    hedge = col_h.number_input("Hedge ×", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key=f"v3|h|{name}",
                               help="1 = ประกันเต็มทุน c · 2 = ซื้อเกินตัว 2 เท่า · 0 = ไม่ซื้อ")
    real = col_k.number_input("ค่าประกันที่จ่ายจริง K", min_value=0.0, value=None, step=10.0, format="%.2f", key=key("k"),
                              placeholder="ว่าง = ใช้สูตร",
                              help="ซื้อ Put จริงแล้วจ่ายเท่าไร ใส่ตรงนี้ · ว่างไว้ = ให้ระบบประมาณ K = 12.4% · c · Hedge×")
    try:
        pl = cv.plan(c, t, p, hedge, real)
    except cv.BookError as e:
        st.error(f"❌ {e}")
        return
    st.caption(breakeven_note(c, t, hedge, real, dec))
    st.markdown(esc(steps_table(pl)))
    if pl["S"] > 0:
        st.markdown(esc(f"✅ **รอบนี้ความแกว่งจ่ายค่าประกันให้แล้ว และเหลือ {sg(pl['S'])} → ทุนโตเป็น {usd(pl['c_after'])}**"))
    else:
        st.markdown(esc(f"⚠️ **รอบนี้ราคายังแกว่งไม่พอจ่ายค่าประกัน ({sg(pl['S'])}) → ทุนเท่าเดิม {usd(c)} (ไม่ลด)**"))
    plan_id = zlib.crc32(repr(sorted(pl.items())).encode())            # a click on an outdated plan finds no button
    if st.button("✅ บันทึกรอบนี้", type="primary", width="stretch", key=f"v3|commit|{name}|{plan_id:08x}"):
        act(cv.commit, name, pl, ok=f"✅ {name}: ทุน → {usd(pl['c_after'])} · t → {px(p)}")
    if n and st.button("↩️ ย้อนรอบล่าสุด (กรณีบันทึกผิด)", key="v3_undo"):
        act(cv.undo, name, ok=f"↩️ {name}: ย้อนรอบล่าสุดแล้ว")


# ── the score ───────────────────────────────────────────────────────────────────────
def score(name: str, pos: dict) -> None:
    e, k = cv.totals(pos)
    eff = cv.efficiency(pos)
    n = len(pos["rounds"])
    st.divider()
    st.subheader("📊 วัดผล — ถึงเป้าหมายหรือยัง?")
    if eff is None:
        st.progress(0.0, text="ยังวัดไม่ได้ — ต้องมีอย่างน้อย 1 รอบที่จ่ายค่าประกัน")
    else:
        st.progress(min(eff, 1.0), text=f"Ev Efficiency {mult(eff)} · เป้าหมาย 1.00x")
    m_e, m_k, m_x = st.columns(3)
    m_e.metric("ΣE · กำไรสะสม", usd(e), border=True, help="ผลรวม E (กำไรจากความแกว่ง) ของทุกรอบที่บันทึก — คือ Eₙ")
    m_k.metric("ΣK · ค่าประกันสะสม", usd(k), border=True, help="ผลรวม K ของทุกรอบที่บันทึก")
    m_x.metric("Ev Efficiency", "—" if eff is None else mult(eff), border=True,
               help="ΣE ÷ ΣK · 1.0x ขึ้นไป = ความแกว่งจ่ายค่าประกันสะสมได้ครบแล้ว")
    reached = eff is not None and eff >= 1.0
    (st.success if reached else st.info)(("✅ ถึงเป้าหมายแล้ว\n\n" if reached else "ยังไม่ถึง 1.0x — วนรอบถัดไป ↻\n\n") + GOAL)
    st.caption(GOAL_PLAIN)

    with st.expander(f"🧾 ประวัติ ({n} รอบ)"):
        if not n:
            st.caption("ยังไม่มีรอบที่บันทึก")
        else:
            st.markdown(esc(table("รอบ | เวลา | t → P | E | K | S | c′", (
                f"{i} | {str(pos['rounds'][i - 1]['at']).replace('|', '¦')} | {px(r['t'])} → {px(r['p'])} | {sg(r['E'])} | "
                f"{neg(r['K'])} | {sg(r['S'])} | {usd(r['c_after'])}"
                for i in range(n, max(0, n - 20), -1) for r in (pos["rounds"][i - 1],)))))
            if n > 20:
                st.caption(f"แสดง 20 รอบล่าสุดจากทั้งหมด {n} รอบ")
    with st.expander("🗑️ ลบตัวนี้ออกจากรายการ"):
        sure = st.checkbox(f"ยืนยัน: ลบ “{name}” พร้อมประวัติ {n} รอบ (ย้อนกลับไม่ได้)", key=f"v3_confirm|{name}")
        if st.button("ลบถาวร", key="v3_delete", disabled=not sure):
            act(cv.remove, name, ok=f"🗑️ ลบ {name} แล้ว")


def main() -> None:
    if msg := st.session_state.pop("_v3_flash", None):
        st.toast(msg)
    if pick := st.session_state.pop("_v3_select", None):
        st.session_state["v3_pos"] = pick
    try:
        book = cv.load(PATH)
    except cv.BookError as e:
        st.title(TITLE)
        st.error(f"🛑 {e}")
        st.caption("ไฟล์เดิมยังอยู่ครบ ไม่ถูกแก้หรือลบ · กด “เริ่มใหม่” เพื่อย้ายไฟล์เดิมไปเก็บไว้ข้างๆ (ไม่ลบ) แล้วเริ่มสมุดเปล่า")
        if st.button("🆕 เริ่มใหม่ (เก็บไฟล์เดิมไว้)", key="v3_reset"):
            try:
                kept = cv.quarantine(PATH)
            except OSError as err:
                st.error(f"💾 ย้ายไฟล์ไม่สำเร็จ: {err}")
            else:
                st.session_state["_v3_flash"] = esc(f"🗄️ เก็บไฟล์เดิมไว้ที่ {kept.name} แล้ว — เริ่มสมุดใหม่ได้เลย")
                st.rerun()
        return

    by_name = {pos["name"]: pos for pos in book["positions"]}
    intro()
    if not by_name:
        st.subheader("🚀 เริ่มเลย")
        if st.button(f"▶ ลองดูตัวอย่างก่อน (ทุน {DEMO[1]:,.0f} · ราคา {DEMO[2]:,.0f})", key="v3_demo", type="primary", width="stretch"):
            act(cv.start, DEMO[0], DEMO[1], DEMO[2], ok="✅ เริ่มตัวอย่าง DEMO — ราคาจะขึ้นไป 200 แล้วกลับมา 100 ลองกดบันทึกดู",
                _v3_select=DEMO[0])
        st.caption("หรือใส่ของจริงของคุณ ↓")
        start_form()
    else:
        if st.session_state.get("v3_pos") not in by_name:
            st.session_state["v3_pos"] = next(iter(by_name))
        name = st.selectbox("ตัวที่ติดตาม", list(by_name), key="v3_pos")
        run_round(name, by_name[name])
        score(name, by_name[name])
        with st.expander("➕ เริ่มติดตามตัวใหม่"):
            start_form()
    st.caption(FOOTER)


if __name__ == "__main__":
    main()
