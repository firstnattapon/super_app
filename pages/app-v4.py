"""🔗 Chain System · v4 — the whole cycle, many tickers, one shared treasury.

① pay K first → ② harvest E → ③ is the Surplus positive? — no: ③.1 roll P and pull the capital gain · yes: ④ Shield → ⑤.1 Scale Up →
⑤.2 Rollover → measure ΣE ÷ ΣK → ↻ ①. Every ticker runs its own cycle; Pool CF, EV Reserve and Ev Burn are shared by all of them.

Standalone: the numbers come from chain_v4.py (stdlib only) and everything is kept in this page's own file (data/app_v4.json, or the path
in APP_V4_DATA_FILE) as one chain of steps. The ledger is the LEGO one (R · ΔA · A · E). Every symbol on the screen is explained.
"""
import os
import sys
import zlib
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import chain_v4 as cv  # noqa: E402

st.set_page_config(page_title="Chain System v4", page_icon="🔗", layout="wide")

PATH = os.environ.get("APP_V4_DATA_FILE") or str(ROOT / "data" / "app_v4.json")
TITLE = "🔗 Chain System v4"
ESSENCE = "ประกันก่อน · ความแกว่งจ่าย · โตด้วยกำไรที่ล็อกแล้ว · เทรนด์เก็บเข้ากระเป๋า"
DEMO = ("DEMO", 1500.0, 10.0)                   # name · Fix_c · P₀ of the one-click example …
DEMO_FUND = 300.0                               # … with 300 put in EV Reserve …
DEMO_PATH = (15.0, 10.0)                        # … whose price goes up, then comes back to where it started
DEMO_HINT = ("👆 ตัวอย่างขั้น ①: กด “จ่าย K — เริ่มวงจร”",
             "👆 ตัวอย่างขั้น ②: ราคาขึ้นไป 15 — กด “บันทึกราคานี้” (S จะยังติดลบ)",
             "👆 ตัวอย่างขั้น ③.1: S ยังไม่พอ — กด “ดึง capital gain” แล้วบันทึกราคากลับ 10",
             "👆 ตัวอย่างขั้น ④–⑤: S เป็นบวกแล้ว — ติ๊ก Shield แล้วกด “ขยายทุน”")
CYCLE = """① จ่าย K ก่อน (Deep ITM LEAPS)
② เก็บ E (หุ้น 20% · LEAPS 80%)
③ Surplus = E − K บวกไหม ?
     ไม่ → ③.1 roll P ขึ้น · ดึง capital gain (เทรนด์ R) ออก · E อยู่ต่อ → กลับ ② (ไม่จ่าย K ซ้ำ)
     ใช่ → ④ Shield (Long Put · Zero-Cost · Put Ratio 2-1)
          → ⑤.1 Scale Up fix_c (Liquidity ถ้าพอจ่าย K วงจรหน้า · ไม่พอ ซื้อ Deep ITM)
          → ⑤.2 Rollover (β · roll P ขึ้น · ดึง capital gain ออก)
          → วัดผล (ΣE ÷ ΣK ≥ 1.0x) → ↻ ① (จ่าย K ของทุนใหม่ c′ ก่อนเสมอ)"""
GOAL = ("🏆 **เป้าหมายสูงสุด · Self-funding Compounding**  \n"
        "ให้ **ความแกว่งของราคาจ่ายค่าประกันของตัวเอง**  \n"
        "แล้วเอากำไรที่เหลือ **ขยายทุน (fix_c) วนเป็นลูกโซ่ ↻ ทุกรอบ**  \n"
        "Ev Efficiency ≥ 1.0x · Surplus เป็นบวก · Eₙ ไม่ลด  \n"
        "ทุนเดิมไม่เพิ่มความเสี่ยง (Free Risk)")
HOW_TO = ("วิธีใช้: **1)** เริ่มติดตาม ticker (ชื่อ · ทุน Fix_c · ราคาตั้งต้น P₀) หรือนำเข้า Chain data เดิม · **2)** ทำทีละขั้นตามที่การ์ดบอก — "
          "จ่าย K ก่อน แล้วบันทึกราคา fill ทุกครั้งที่ rebalance · **3)** S เป็นบวกเมื่อไร ขยายทุนได้ — แล้ววนกลับไปจ่าย K ของทุนใหม่")
PHASE = {"pay": "① จ่าย K", "swing": "② เก็บ E", "scale": "④–⑤ ขยายทุนได้"}
FUNDING_NAME = {"liquidity": "Liquidity", "deep_itm": "Deep ITM"}
LANE_NAME = {cv.POOL: "Pool CF", cv.RESERVE: "EV Reserve", cv.OUT: "นอกระบบ"}
ROUTES = (("นอกระบบ → Pool CF (เงินเข้า · กำไร Put / Harvest / Funding)", cv.OUT, cv.POOL),
          ("นอกระบบ → EV Reserve (เงินเข้า)", cv.OUT, cv.RESERVE),
          ("Pool CF → EV Reserve (Allocate)", cv.POOL, cv.RESERVE),
          ("EV Reserve → Pool CF", cv.RESERVE, cv.POOL),
          ("Pool CF → นอกระบบ (เงินออก)", cv.POOL, cv.OUT),
          ("EV Reserve → นอกระบบ (เงินออก · จ่าย LEAPS นอกสูตร)", cv.RESERVE, cv.OUT),
          ("Baseline b ของ ticker → Pool CF (ดึง capital gain)", "{t}", cv.POOL),
          ("Pool CF → Baseline b ของ ticker (Deploy)", cv.POOL, "{t}"))

# Every symbol the page shows: (symbol, name, what it is, what it is for)
GLOSSARY = (
    ("ตัวเลขที่คุณกรอก", (
        ("c", "ทุน Fix_c", "เงินที่ถือไว้ใน ticker นี้ที่มูลค่าคงที่ (เช่น $1,500) · แบ่งเป็นหุ้น 20% (Liquidity) + LEAPS 80% (Buying Power)",
         "เป็นฐานของทุกสูตร และเป็นสิ่งที่ระบบพยายามทำให้โต"),
        ("P₀", "ราคาตั้งต้น", "ราคาตอนเริ่มติดตาม (ช่อง t) — ตรึงไว้ไม่เปลี่ยนเลย", "จุดอ้างอิงของเส้นเทรนด์ Rₙ — Eₙ นับจาก 0 ที่นี่"),
        ("Pₙ", "ราคา fill", "ราคาที่ rebalance จริง (VWAP ของ fill) ที่คุณกรอกทุกครั้ง", "ใช้คำนวณทุกแถวของ ledger — บันทึกหลังได้ fill จริงเท่านั้น"),
        ("Hedge ×", "จำนวนเท่าของประกัน", "1 = ประกันเต็มทุน c · 2 = ซื้อเกินตัว 2 เท่า · 0 = ไม่ซื้อ", "ปรับระดับความอุ่นใจ — ค่าประกันเพิ่มตามสัดส่วน"),
        ("k", "ค่าประกันที่จ่ายจริง", "ถ้าซื้อ LEAPS จริงแล้วจ่ายเท่าไร ใส่ตรงนี้ · ว่าง = ใช้สูตร 12.4%", "เปลี่ยนเบี้ยโมเดลเป็นเบี้ยจริง"),
    )),
    ("ledger แบบ LEGO — คำนวณให้ทุกครั้งที่ act", (
        ("P_acted", "ราคา act ล่าสุด", "ราคา fill ของครั้งก่อน — ครั้งถัดไปวัดจากราคานี้ (roll P ขึ้น/ลง)", "ΔAₙ วัดจากการขยับห่างราคานี้"),
        ("Vₙ", "มูลค่าที่ถืออยู่", "จำนวนหุ้นที่ถือ × Pₙ (จำนวนหุ้น = c ÷ P_acted)", "เทียบกับ Fix_c เพื่อรู้ว่าต้องซื้อหรือขายเท่าไร"),
        ("gap", "ส่วนต่างเป้าหมาย", "gap = Vₙ − Fix_c · บวก = ถือเกินเป้า ขายออก (SELL) · ลบ = ถือขาดเป้า ซื้อเข้า (BUY) · จำนวนหุ้นที่สั่ง = ค่าสัมบูรณ์ของ gap ÷ Pₙ · ΔAₙ = gap",
         "บอกว่าต้องสั่งอะไรเท่าไรเพื่อกลับสู่ Fix_c"),
        ("Rₙ", "เส้นเทรนด์", "Rₙ = β + Fix_c × ln(Pₙ / P₀) — ขึ้นลงตามระดับราคา ไม่ใช่ฝีมือของระบบ", "ส่วนที่ต้องหักทิ้ง ไม่ให้เทรนด์ปนกับกำไรจริง"),
        ("ΔAₙ", "เงินสดต่อก้าว", "ΔAₙ = Fix_c × (Pₙ / P_acted − 1) · บวก = ได้เงินจากการขาย · ลบ = จ่ายซื้อคืน", "ตัวสะสมเป็น Aₙ"),
        ("Aₙ", "เงินสดสะสม", "Aₙ = Aₙ₋₁ + ΔAₙ — รวมทั้งเทรนด์และความแกว่ง", "ยังปนเทรนด์ จึงใช้อ่านกำไรตรงๆ ไม่ได้"),
        ("Eₙ", "กำไรจากความแกว่ง", "Eₙ = Aₙ − Rₙ — ตัดเทรนด์ทิ้ง เหลือแต่ความแกว่ง ไม่ติดลบ ไม่ลดลง · รอบ buy/sell ที่ยังไม่ปิด = กำไรค้างท่อ",
         "คือ “รายได้” ที่ไว้จ่ายประกันและขยายทุน"),
        ("β", "ความจำของ Rollover", "β += (Fix_c − Fix_c′) × ln(P_acted / P₀) ทุกครั้งที่เปลี่ยน Fix_c — บวกที่ Rₙ เท่านั้น ไม่บวกที่ Aₙ",
         "ทำให้เส้นเทรนด์ไม่กระโดดตอนขยายทุน"),
        ("b", "Baseline (capital gain ค้าง)", "ส่วนเทรนด์ของ Rₙ ที่เก็บเป็นเงินสดของ ticker นี้ ดึงเข้า Pool CF ได้ · ติดลบ = ต้องมีเงินซื้อคืนเมื่อราคากลับ",
         "เทรนด์เก็บเข้ากระเป๋า — ไม่เอาไปขยายทุน"),
    )),
    ("วงจร — ชั้น Chain ที่ต่อบน ledger", (
        ("K", "ค่าประกัน", f"K = {cv.PREMIUM_RATE:.1%} × c × Hedge× (เบี้ย Put 10% OTM อายุ 1 ปี) หรือ K จริง — จ่ายครั้งเดียวต่อวงจร ก่อนเสมอ",
         "ต้นทุนของวงจรนี้ — หักจาก EV Reserve"),
        ("E_cyc", "E ค้างท่อ", "Eₙ ที่เก็บได้ตั้งแต่จ่าย K ในวงจรนี้ — ยังไม่เข้า Pool CF จนกว่าจะปิดวงจรที่ S > 0", "คือรายได้ของวงจรนี้"),
        ("S", "Surplus", "S = E_cyc − K · บวก = ความแกว่งจ่ายประกันครบและยังเหลือ", "ประตูของวงจร: S > 0 จึงขยายทุนได้ (④–⑤)"),
        ("c′", "ทุนใหม่", "c′ = c + S — โตเฉพาะส่วนที่เหลือ ไม่เคยลด", "Free Risk — ไม่ใช้เงินต้นเพิ่ม"),
    )),
    ("กระเป๋ากลาง — ใช้ร่วมกันทุก ticker", (
        ("Pool CF", "กระเป๋าเงินสดกลาง", "capital gain ที่ดึงออก + E ที่ชำระ + เงินที่ใส่เอง", "แหล่ง Liquidity สำหรับขยายทุนและซื้อคืน"),
        ("EV Reserve", "กองทุนจ่ายประกัน", "K จ่ายออกจากที่นี่ · วงจรที่สำเร็จได้ K คืนจาก E · ติดลบได้ (จ่ายล่วงหน้า)",
         "บอกว่าประกันถูกจ่ายด้วยความแกว่ง หรือด้วยเงินนอก"),
        ("Ev Burn", "ค่าประกันที่เผาไปแล้ว", "ΣK ของทุก ticker (รวมที่ยกมาจากไฟล์เก่า)", "ต้นทุนสะสมของทั้งระบบ"),
        ("Liquidity", "สภาพคล่อง", "Pool CF + EV Reserve", "ตัดสินใจ ⑤.1: ใช้ Liquidity ถ้าหลังใช้ยังพอจ่าย K ของวงจรหน้า ไม่พอ → Deep ITM"),
    )),
    ("คะแนนรวมทุกรอบ", (
        ("ΣE", "กำไรสะสมจากความแกว่ง", "ผลรวม Eₙ ของ ticker (คือ Eₙ ใน ledger)", "ตัวตั้งของ Ev Efficiency"),
        ("ΣK", "ค่าประกันสะสม", "ผลรวม K ที่จ่ายในสมุดนี้", "ตัวหารของ Ev Efficiency"),
        ("Ev Efficiency", "คะแนนจ่ายตัวเอง", "ΣE ÷ ΣK — กำไรสะสมจากความแกว่งเป็นกี่เท่าของค่าประกันสะสม",
         "1.0x ขึ้นไป = ความแกว่งจ่ายประกันสะสมได้ครบแล้ว — เป็นป้ายสถานะ ประตูจริงคือ S > 0"),
    )),
)
BACKUP_NOTE = ("ข้อมูลทั้งหมดอยู่ในไฟล์เดียวบนเครื่องที่รันแอป — ถ้าแอปรีสตาร์ตหรือย้ายเครื่อง ข้อมูลอาจหาย · ดาวน์โหลดไฟล์สำรองไว้เป็นระยะ "
               "นำเข้ากลับได้ที่นี่ · นำเข้า Chain data เดิม (ไฟล์สำรองของแอปรุ่นก่อน / ไฟล์รายการ ticker รุ่นแรก) ได้ที่เดียวกัน เริ่มวงจรใหม่จากสถานะปัจจุบัน "
               "(นำเข้า = แทนที่ข้อมูลปัจจุบัน · ไฟล์เดิมถูกเก็บไว้ข้างๆ ไม่ลบ)")
FOOTER = ("เครื่องมือคำนวณและจดบันทึกเพื่อการเรียนรู้ — ไม่ใช่คำแนะนำการลงทุน และไม่ส่งคำสั่งซื้อขายจริง · A และ E คือโมเดล (ledger ตามสมการ LEGO) "
          "ไม่ใช่กำไรหลังค่าธรรมเนียม · Pool CF และ EV Reserve คือยอดที่สมุดนี้บันทึก · "
          f"ข้อมูลเก็บในไฟล์ {Path(PATH).name} บนเครื่องที่รันแอป — สำรองไว้เป็นระยะที่ “💾 สำรอง / กู้คืนข้อมูล”")


# ── small helpers ─────────────────────────────────────────────────────────────────────
def esc(text: str) -> str:
    """Streamlit markdown reads ``$…$`` as LaTeX — escape every dollar sign."""
    return text.replace("$", "\\$")


def _z(x: float) -> float:
    return 0.0 if abs(x) < 0.005 else x                          # never "−$0.00"


def usd(x: float) -> str:
    return f"{'−' if _z(x) < 0 else ''}${abs(_z(x)):,.2f}"


def sg(x: float) -> str:
    return f"{'−' if _z(x) < 0 else '+'}${abs(_z(x)):,.2f}"


def neg(x: float) -> str:
    """A cost: −$123.96 (just $0.00 when nothing was paid)."""
    return f"−{usd(x)}" if _z(x) > 0 else usd(x)


def mult(x: float) -> str:
    return f"{x:.2f}x".replace("-", "−")


def px(x: float) -> str:
    """A price without trailing zeros: 125 · 113.2 · 0.0042."""
    return f"{x:,.4f}".rstrip("0").rstrip(".") or "0"


def sym(s: str) -> str:
    return "**" + s.replace("*", "\\*") + "**"


def table(head: str, rows) -> str:
    """A Markdown table; ``head`` and each row are cells separated by ' | '."""
    return "\n".join([f"| {head} |", "|" + "---|" * (head.count("|") + 1), *(f"| {r} |" for r in rows)])


def work(row: dict) -> str:
    """The "ที่มา" of one row: symbol · formula with the numbers put in · result."""
    return esc(table("ตัวแปร | สูตร → แทนค่า | ได้", (f"{sym(s)} | {how} | **{res}**" for s, how, res in cv.explain(row))))


def save(steps, seen: int, ok: str, **after) -> None:
    """Save step(s) to the chain. Success → remember a toast (plus the session keys in ``after``) and redraw. Failure → say why; nothing was saved."""
    try:
        cv.commit(PATH, steps, seen)
    except cv.Stale as e:
        st.warning(f"⏱️ {e}")
    except cv.BookError as e:
        st.error(f"❌ {e}")
    except OSError as e:
        st.error(f"💾 บันทึกไฟล์ไม่สำเร็จ: {e}")
    else:
        st.session_state.update(_v4_flash=esc(ok), **after)
        st.rerun()


def act(fn, *args, ok: str, **after) -> None:
    """Any other change to the book (undo · remove · import · move a broken file aside)."""
    try:
        fn(PATH, *args)
    except cv.BookError as e:
        st.error(f"❌ {e}")
    except OSError as e:
        st.error(f"💾 บันทึกไฟล์ไม่สำเร็จ: {e}")
    else:
        st.session_state.update(_v4_flash=esc(ok), **after)
        st.rerun()


def first_active(led: dict) -> str:
    """The ticker to show first: the first one with real capital (Fix_c ≥ $1) — a dust position is not where you start — else the first."""
    return next((n for n, t in led["tickers"].items() if t["c"] >= 1), next(iter(led["tickers"])))


def lane(end: str) -> str:
    return LANE_NAME.get(end, f"Baseline {end}")


def describe(r: dict) -> str:
    k = r["do"]
    if k == "open":
        return f"เริ่ม Fix_c {usd(r['c'])} ที่ P₀ {px(r['t'])}"
    if k == "pay":
        return f"① จ่าย K {neg(r['K'])} · Hedge× {r['hedge']:g}" + (" · K จริง" if r["real"] else "")
    if k == "move":
        return f"② act {r['side']} · {px(r['pa'])} → {px(r['p'])} · E {sg(r['d_e'])}"
    if k == "scale":
        return f"④⑤ ขยายทุน {usd(r['c'])} → {usd(r['c_new'])} · S {sg(r['s'])} · {FUNDING_NAME[r['by']]}"
    return f"💸 {lane(r['src'])} → {lane(r['dst'])} {usd(r['amount'])}"


# ── the explanation ─────────────────────────────────────────────────────────────────────
def intro() -> None:
    steps = [{"do": "open", "ticker": DEMO[0], "c": DEMO[1], "t": DEMO[2]}, {"do": "cash", "src": cv.OUT, "dst": cv.RESERVE, "amount": DEMO_FUND},
             {"do": "pay", "ticker": DEMO[0]}, *({"do": "move", "ticker": DEMO[0], "p": p} for p in DEMO_PATH),
             {"do": "scale", "ticker": DEMO[0], "by": "liquidity", "shield": True}]
    rows = cv.replay(steps)["rows"]                                   # the worked example is computed, never typed in
    pay, up, back, grow = rows[2], rows[3], rows[4], rows[5]
    st.title(TITLE)
    st.markdown(f"#### {ESSENCE}")
    st.info(esc(
        "🔑 **แก่นแท้:** จ่ายประกัน (**K**) ก่อนเสมอ → ให้ความแกว่ง (**E**) จ่ายประกัน → เหลือ (**S**) จึงขยายทุน (**c′**) · "
        "เทรนด์ (**R**) ไม่ใช่กำไร — เก็บเข้ากระเป๋า ไม่เอาไปขยาย\n\n"
        f"**ตัวอย่าง** Fix_c {usd(pay['c'])} · ราคา {px(DEMO[2])} → {px(DEMO_PATH[0])} → {px(DEMO_PATH[1])}  \n"
        f"① จ่าย K {neg(pay['K'])}  \n"
        f"② ราคา {px(up['p'])}: E {sg(up['e'])} · เทรนด์ R {sg(up['r'])} → S {sg(up['s'])} (ยังไม่พอ → ดึงเทรนด์เก็บเข้ากระเป๋า)  \n"
        f"② ราคากลับ {px(back['p'])}: E สะสม {sg(back['e'])} → S {sg(back['s'])} ✓ → ขยายทุนเป็น {usd(grow['c_new'])} · "
        f"EV Reserve กลับมา {usd(grow['reserve'])} (ความแกว่งจ่ายประกันเอง)"))
    st.caption(HOW_TO)
    with st.expander("🔁 วงจร 1 หน้า"):
        st.code(CYCLE, language=None)
        st.caption("แต่ละ ticker วนวงจรของตัวเอง · ทุกขั้นที่ทำ = 1 บรรทัดในสมุดเดียว · ตัวเลขทุกตัวคำนวณย้อนจากสมุดนั้น")
    with st.expander("📖 ตัวแปรทุกตัวคืออะไร · ไว้ทำไม"):
        for title, items in GLOSSARY:
            st.markdown(f"**{title}**")
            st.markdown(esc(table("ตัวแปร | คืออะไร | ไว้ทำไม", (f"{sym(s)} | **{n}** — {what} | {why}" for s, n, what, why in items))))


def start_form(seen: int) -> None:
    n = st.session_state.get("_v4_form", 0)                         # a new number = a fresh, empty form after a start
    with st.form("v4_start"):
        name = st.text_input("ชื่อ ticker", key=f"v4_name|{n}", max_chars=cv.MAX_NAME, placeholder="เช่น TSLA · BTC · ทองคำ")
        c = st.number_input("ทุน Fix_c — เงินที่ถือไว้ในตัวนี้", key=f"v4_c|{n}", min_value=0.01, value=1000.0, step=100.0, format="%.2f")
        t = st.number_input("ราคาตั้งต้น P₀ — ราคาตอนนี้", key=f"v4_t|{n}", min_value=0.0, value=None, step=0.01, format="%.4f", placeholder="เช่น 100")
        if st.form_submit_button("🚀 เริ่มติดตาม", width="stretch"):
            save({"do": "open", "ticker": name, "c": c, "t": t}, seen, ok=f"✅ เริ่มติดตาม {name.strip()}", _v4_select=name.strip(), _v4_form=n + 1)


# ── the shared treasury and the board ─────────────────────────────────────────────────────
def shared(led: dict) -> None:
    tot = cv.totals(led)
    st.markdown("##### 🏛️ ส่วนกลาง — ใช้ร่วมกันทุก ticker")
    cols = st.columns(5)
    cols[0].metric("🎱 Pool CF", usd(tot["pool"]), border=True, help="เงินสดกลาง: capital gain ที่ดึง + E ที่ชำระ + เงินที่ใส่เอง — แหล่ง Liquidity")
    cols[1].metric("🛡️ EV Reserve", usd(tot["reserve"]), border=True, help="กองทุนจ่ายประกัน: K ออกจากที่นี่ วงจรที่สำเร็จคืน K จาก E · ติดลบได้")
    cols[2].metric("🔥 Ev Burn", usd(tot["burn"]), border=True, help="ค่าประกัน (K) ที่จ่ายไปแล้วทุก ticker รวมที่ยกมาจากไฟล์เก่า")
    cols[3].metric("📈 Ev Efficiency", "—" if tot["eff"] is None else mult(tot["eff"]), border=True,
                   help="ΣE ÷ ΣK ของทุก ticker (เฉพาะที่วัดในสมุดนี้) · เทียบข้าม ticker ได้เมื่อใช้กฎ act เดียวกัน")
    cols[4].metric("💧 Liquidity", usd(tot["liquidity"]), border=True, help="Pool CF + EV Reserve — ใช้ตัดสิน ⑤.1")
    opening = tot["burn"] - tot["sum_k"]
    st.caption(esc(f"⏳ E ค้างท่อ {usd(tot['pending'])} (ยังไม่ชำระเข้า Pool/Reserve) · 💵 เงินนอกระบบสุทธิ {sg(tot['net_in'])} · "
                   f"Fix_c รวม {usd(tot['fix_c'])} · b รวม {usd(tot['b'])}" + (f" · Ev Burn ยกมา {usd(opening)} ไม่นับใน Efficiency" if opening > 0.004 else "")))
    if tot["reserve"] < 0:
        st.warning("⚠️ EV Reserve ติดลบ = ประกันถูกจ่ายล่วงหน้าด้วยเงินนอกกองทุน — วงจรที่สำเร็จจะคืน K ให้ · หรือโอนจาก Pool CF ที่ “💸 โอนเงิน”")


def board(led: dict) -> None:
    rows = []
    for name, t in led["tickers"].items():
        eff = cv.efficiency(t)
        cyc = f"{usd(cv.e_cyc(t))} / {usd(t['K'])}" if t["open"] else "—"
        rows.append(f"**{name}** | {PHASE[cv.phase(t)]} | {usd(t['c'])} | {px(t['pa'])} | {usd(t['b'])} | {cyc} | {'—' if eff is None else mult(eff)}")
    st.markdown(esc(table("ticker | ขั้นตอนตอนนี้ | Fix_c | P_acted | b | E ค้างท่อ / K | ΣE ÷ ΣK", rows)))


# ── one cycle: the one step to take now ─────────────────────────────────────────────────
def demo_state(led: dict, t: dict):
    """(hint index, default price) while the example is still on its path, else (None, None)."""
    if t["name"] != DEMO[0] or t["scales"]:
        return None, None
    prices = [r["p"] for r in led["rows"] if r.get("ticker") == t["name"] and r["do"] == "move"]
    if prices != list(DEMO_PATH[: len(prices)]):
        return None, None
    ph = cv.phase(t)
    return {"pay": 0, "swing": 1 + min(len(prices), 1), "scale": 3}[ph], DEMO_PATH[len(prices)] if len(prices) < len(DEMO_PATH) else None


def cycle(led: dict, name: str, seen: int) -> None:
    t = led["tickers"][name]
    ph = cv.phase(t)

    def key(part: str) -> str:
        return f"v4|{part}|{name}|{seen}"                           # a new state ⇒ fresh widgets with fresh defaults

    st.subheader(f"🔁 วงจรของ {name}")
    m = st.columns(4)
    m[0].metric("c · Fix_c", usd(t["c"]), border=True, help="ทุนที่ถือที่มูลค่าคงที่ — โตเมื่อ S เป็นบวก ไม่เคยลด")
    m[1].metric("P_acted", px(t["pa"]), border=True, help="ราคา fill ของ act ล่าสุด — ครั้งถัดไปวัดจากราคานี้")
    m[2].metric("Eₙ", usd(t["E"]), border=True, help="Eₙ = Aₙ − Rₙ — กำไรจากความแกว่งสะสม ไม่ติดลบ ไม่ลดลง")
    m[3].metric("b · capital gain", usd(t["b"]), border=True, help="ส่วนเทรนด์ของ Rₙ ที่เก็บเป็นเงินสด ดึงเข้า Pool CF ได้ · ติดลบ = ต้องมีเงินซื้อคืน")
    steps = {"pay": "**① จ่าย K ← ตอนนี้** → ② เก็บ E → ③ S > 0 ? → ④–⑤ ขยายทุน",
             "swing": "① จ่าย K ✓ → **② เก็บ E ← ตอนนี้** → ③ S ยัง ≤ 0 → (③.1 แล้วกลับ ②)",
             "scale": "① จ่าย K ✓ → ② เก็บ E ✓ → ③ S > 0 ✓ → **④–⑤ ขยายทุน ← ตอนนี้**"}[ph]
    st.markdown(steps)
    hint, demo_p = demo_state(led, t)
    if hint is not None:
        st.caption(DEMO_HINT[hint])
    if ph == "pay":
        pay_card(led, t, key, seen)
    elif ph == "swing":
        swing_card(led, t, key, seen, demo_p)
        pull_card(t, key, seen)
    else:                                                           # S > 0: scaling up is the step; harvesting more first is allowed
        scale_card(led, t, key, seen)
        with st.expander("② เก็บ E ต่อก่อนขยายทุน (บันทึกราคา fill ถัดไป)"):
            swing_card(led, t, key, seen, demo_p)


def pay_card(led: dict, t: dict, key, seen: int) -> None:
    name = t["name"]
    st.markdown("#### ① จ่าย K ก่อน · Deep ITM LEAPS (Stock Replacement)")
    st.caption(esc(f"ถือ c = หุ้น {cv.STOCK_SHARE:.0%} ({usd(t['c'] * cv.STOCK_SHARE)} · Liquidity) + "
                   f"LEAPS {1 - cv.STOCK_SHARE:.0%} ({usd(t['c'] * (1 - cv.STOCK_SHARE))} · Buying Power)"))
    col_h, col_k = st.columns(2)
    hedge = col_h.number_input("Hedge ×", min_value=0.0, max_value=100.0, value=float(t["hedge"]), step=0.5, key=key("h"),
                               help="1 = ประกันเต็มทุน c · 2 = ซื้อเกินตัว 2 เท่า · 0 = ไม่ซื้อ")
    real = col_k.number_input("ค่าประกันที่จ่ายจริง K", min_value=0.0, value=None, step=10.0, format="%.2f", key=key("k"), placeholder="ว่าง = ใช้สูตร",
                              help="ซื้อ LEAPS จริงแล้วจ่ายเท่าไร ใส่ตรงนี้ · ว่างไว้ = ให้ระบบประมาณ K = 12.4% · c · Hedge×")
    ev = {"do": "pay", "ticker": name, "hedge": hedge, "k": real}
    try:
        after, row = cv.preview(led, ev)
    except cv.BookError as e:
        st.error(f"❌ {e}")
        return
    st.markdown(work(row))
    if after["reserve"] < 0:
        st.warning(esc(f"⚠️ จ่ายแล้ว EV Reserve จะติดลบ ({usd(after['reserve'])}) — ใส่เงินเข้า EV Reserve ก่อนได้ที่ “💸 โอนเงิน” "
                       "หรือจ่ายต่อเลย (วงจรที่สำเร็จจะคืน K ให้)"))
    if st.button("✅ จ่าย K — เริ่มวงจร", type="primary", width="stretch", key=key("pay")):
        save(ev, seen, ok=f"✅ {name}: จ่าย K {neg(row['K'])} แล้ว — เก็บ E ได้")


def swing_card(led: dict, t: dict, key, seen: int, demo_p) -> None:
    name = t["name"]
    st.markdown("#### ② เก็บ E · ใส่ราคา fill ใหม่")
    st.caption("บันทึกหลังส่งออเดอร์และได้ fill จริงแล้วเท่านั้น (ราคา VWAP) · A และ E คือโมเดล ไม่ใช่กำไรหลังค่าธรรมเนียม")
    if t["K"] > 0:
        st.progress(min(max(cv.e_cyc(t) / t["K"], 0.0), 1.0), text=esc(f"E ค้างท่อ {usd(cv.e_cyc(t))} / K {usd(t['K'])}"))
    dec = 2 if t["pa"] >= 1 else 4
    p = st.number_input("ราคา fill ใหม่ P", min_value=0.0, value=float(demo_p or t["pa"]), step=10.0 ** -dec, format=f"%.{dec}f", key=key("p"),
                        help="ราคาที่ rebalance จริง (VWAP) · ΔAₙ = Fix_c × (P / P_acted − 1)")
    ev = {"do": "move", "ticker": name, "p": p}
    try:
        row = cv.preview(led, ev)[1]
    except cv.BookError as e:
        st.error(f"❌ {e}")
        return
    st.markdown(work(row))
    if row["s"] > 0:
        st.markdown(esc(f"✅ **S = {sg(row['s'])} — ความแกว่งจ่ายประกันครบและยังเหลือ → ขยายทุนได้ (การ์ด ④–⑤ จะขึ้นหลังบันทึก)**"))
    else:
        st.markdown(esc(f"⚠️ **S = {sg(row['s'])} ยังไม่พอ → ③.1 ดึง capital gain แล้วเก็บ E ต่อ (ไม่จ่าย K ซ้ำ)**"))
    if st.button("✅ บันทึกราคานี้ (act)", type="primary", width="stretch", key=key("move"), disabled=row["side"] == "PASS"):
        save(ev, seen, ok=f"✅ {name}: act {row['side']} ที่ {px(p)} · Eₙ → {usd(row['e'])}")
    if row["side"] == "PASS":
        st.caption("ราคาเท่า P_acted เดิม — ไม่มีอะไรให้ rebalance")


def pull_card(t: dict, key, seen: int) -> None:
    name, b = t["name"], t["b"]
    st.markdown("#### ③.1 S ยังไม่เป็นบวก → roll P ขึ้นแล้ว · ดึง capital gain (เทรนด์ R) · E อยู่ต่อ")
    st.caption("P_acted เลื่อนไปที่ราคา fill ล่าสุดแล้ว (roll P) · ไม่จ่าย K ซ้ำ — บันทึกราคาถัดไปได้เลย")
    if b > 0.004:
        if st.button(f"💰 ดึง capital gain {usd(b)} → Pool CF", key=key("pull")):
            save({"do": "cash", "src": name, "dst": cv.POOL, "amount": b}, seen, ok=f"💰 {name}: ดึง {usd(b)} เข้า Pool CF")
    elif b < -0.004:
        st.warning(esc(f"b ติดลบ {usd(b)} = เงินที่ดึงออกไปแล้วถูกใช้ไป — ถ้าราคากลับ ต้องมีเงินซื้อคืน (เงินที่ดึงไม่ใช่กระสุนซื้อคืน) · "
                       "โอนกลับจาก Pool CF ที่ “💸 โอนเงิน” (Deploy)"))
    else:
        st.caption("ยังไม่มี capital gain ให้ดึง")


def scale_card(led: dict, t: dict, key, seen: int) -> None:
    name = t["name"]
    st.markdown(esc(f"#### ④⑤ S = {sg(cv.surplus(t))} → ขยายทุนแบบ Free Risk"))
    shield = st.checkbox("④ ตั้ง Shield แล้ว (Long Put · Zero-Cost · Put Ratio 2-1)", key=key("shield"),
                         help="ขยายทุนเพิ่มขนาดที่ขาดทุนได้ — ต้องมีโล่ก่อนเสมอ")
    st.caption("Zero-Cost ≠ ไม่เสี่ยง · Free Risk คือไม่ใช้เงินต้นเพิ่ม ไม่ได้แปลว่าส่วนที่ขยายขาดทุนไม่ได้")
    if not shield:
        st.info("ติ๊ก Shield (④) ก่อน จึงจะดูตัวอย่างและขยายทุนได้")
        return
    settle = st.checkbox("ชำระ: E ค้างท่อ → Pool CF · คืน K ให้ EV Reserve", value=True, key=key("settle"),
                         help="ความแกว่งจ่ายประกันตัวเอง — ปิดไว้ถ้าคุณบันทึกเงินจริงเอง")
    pull = st.checkbox(f"⑤.2 ดึง capital gain ({usd(t['b'])}) → Pool CF", value=t["b"] > 0.004, disabled=t["b"] <= 0.004, key=key("pull2"))
    base = {"do": "scale", "ticker": name, "shield": True, "settle": settle, "pull": pull}
    rec = cv.recommend_funding(led, base)
    by = st.radio("⑤.1 ขยาย fix_c ด้วย", list(FUNDING_NAME), index=list(FUNDING_NAME).index(rec), format_func=FUNDING_NAME.get, horizontal=True,
                  key=key("by"))
    try:
        after = cv.preview(led, {**base, "by": "liquidity"})[0]
        left, need = after["pool"] + after["reserve"], cv.next_k(after["tickers"][name])
        why = f"Liquidity หลังใช้เหลือ {usd(left)} {'≥' if left + 1e-9 >= need else '<'} K ของวงจรหน้า {usd(need)}"
    except cv.BookError:
        why = "Pool CF ไม่พอสำหรับ Liquidity"
    st.caption(esc(f"แนะนำ {FUNDING_NAME[rec]} — {why}"))
    ev = {**base, "by": by}
    try:
        row = cv.preview(led, ev)[1]
    except cv.BookError as e:
        st.error(f"❌ {e}")
        return
    st.markdown(work(row))
    if st.button("✅ ขยายทุน fix_c (④–⑤)", type="primary", width="stretch", key=key("scale")):
        save(ev, seen, ok=f"✅ {name}: ทุน → {usd(row['c_new'])} · วนกลับ ① จ่าย K ของทุนใหม่")


def measure(t: dict) -> None:
    eff = cv.efficiency(t)
    st.divider()
    st.subheader("📊 วัดผล — ถึงเป้าหมายหรือยัง?")
    if eff is None:
        st.progress(0.0, text="ยังวัดไม่ได้ — ต้องจ่าย K อย่างน้อย 1 วงจร")
    else:
        st.progress(min(max(eff, 0.0), 1.0), text=f"Ev Efficiency {mult(eff)} · เป้าหมาย 1.00x")
    m = st.columns(3)
    m[0].metric("ΣE · กำไรสะสม", usd(t["E"]), border=True, help="Eₙ ของ ticker นี้ = ผลรวม E ทุกก้าว")
    m[1].metric("ΣK · ค่าประกันสะสม", usd(t["sumK"]), border=True, help="ผลรวม K ที่จ่ายในสมุดนี้")
    m[2].metric("Ev Efficiency", "—" if eff is None else mult(eff), border=True, help="ΣE ÷ ΣK · 1.0x ขึ้นไป = ความแกว่งจ่ายประกันสะสมได้ครบแล้ว")
    reached = eff is not None and eff >= 1.0
    (st.success if reached else st.info)(("✅ ถึงเป้าหมายแล้ว — วงจรไม่หยุด: ↻ ① จ่าย K ของทุนใหม่ c′ ก่อนเสมอ\n\n" if reached
                                          else "ยังไม่ถึง 1.0x — วนรอบถัดไป ↻\n\n") + GOAL)
    st.caption("Ev Efficiency เป็นป้ายสถานะสะสมทุกวงจร ประตูของวงจรคือ S > 0 · เทียบข้าม ticker ได้เมื่อใช้กฎ act เดียวกัน"
               + (f" · ไม่รวม Ev Burn ยกมา {usd(t['burn0'])}" if t["burn0"] > 0.004 else ""))


# ── the books: ledger, treasury, cash, manage ──────────────────────────────────────────────
def ledger(led: dict, name: str) -> None:
    rows = [r for r in led["rows"] if r.get("ticker") == name]
    with st.expander(f"🧾 Ledger LEGO ของ {name} ({len(rows)} รายการ)"):
        if not rows:
            st.caption("ยังไม่มีรายการ")
            return
        shown = rows[::-1][:20]
        st.markdown(esc(table("n | รายการ | Pₙ | Fix_c | Rₙ | ΔAₙ | Aₙ | Eₙ | b", (
            f"{r['n']} | {describe(r)} | {px(r['p'] if r['do'] == 'move' else r['after']['pa'])} | {usd(r['after']['c'])} | "
            f"{usd(r['after']['R'])} | {sg(r.get('d_a', 0.0))} | {usd(r['after']['A'])} | {usd(r['after']['E'])} | {usd(r['after']['b'])}"
            for r in shown))))
        if len(rows) > 20:
            st.caption(f"แสดง 20 รายการล่าสุดจากทั้งหมด {len(rows)}")
        pick = st.selectbox("ดูที่มาของตัวเลขแถวที่", [r["n"] for r in shown], key=f"v4_why|{name}|{len(led['rows'])}")
        st.markdown(work(next(r for r in shown if r["n"] == pick)))


def treasury(led: dict) -> None:
    rows = [r for r in led["rows"] if r["do"] in ("pay", "scale", "cash")]
    with st.expander(f"🏛️ ความเคลื่อนไหวกระเป๋ากลาง ({len(rows)} รายการ)"):
        if not rows:
            st.caption("ยังไม่มีรายการ")
            return
        st.markdown(esc(table("n | เวลา | รายการ | Pool CF | EV Reserve", (
            f"{r['n']} | {r['at']} | {r['ticker'] + ' · ' if r.get('ticker') and r['do'] != 'cash' else ''}{describe(r)} | "
            f"{usd(r['pool'])} | {usd(r['reserve'])}" for r in rows[::-1][:30]))))


def cash_form(led: dict, names: list, selected: str, seen: int) -> None:
    with st.expander("💸 โอนเงินระหว่างกระเป๋า"):
        st.caption("ย้ายเงินระหว่าง นอกระบบ · Pool CF · EV Reserve · Baseline b ของ ticker — ลบที่ต้นทาง บวกที่ปลายทาง เงินไม่เกิดขึ้นเองและไม่หายไปไหน")
        label = st.selectbox("เส้นทาง", [r[0] for r in ROUTES], key=f"v4_route|{seen}")
        _, src, dst = next(r for r in ROUTES if r[0] == label)
        tick = None
        if "{t}" in (src, dst):
            tick = st.selectbox("ticker", names, index=names.index(selected), key=f"v4_cash_t|{seen}")
        src, dst = (x.replace("{t}", tick) if tick else x for x in (src, dst))
        gap = round(-led["reserve"], 2) if (src, dst) == (cv.POOL, cv.RESERVE) and led["reserve"] < 0 else 0.0     # the shortfall, ready to fill
        a = st.number_input("จำนวนเงิน", min_value=0.01, value=gap if gap >= 0.01 else 100.0, step=10.0, format="%.2f",
                            key=f"v4_cash_a|{seen}|{zlib.crc32(label.encode()):08x}")
        ev = {"do": "cash", "src": src, "dst": dst, "amount": a}
        try:
            after = cv.preview(led, ev)[0]
        except cv.BookError as e:
            st.error(f"❌ {e}")
            return
        st.caption(esc(f"Pool CF {usd(led['pool'])} → {usd(after['pool'])} · EV Reserve {usd(led['reserve'])} → {usd(after['reserve'])}"))
        if st.button("💸 โอน", key=f"v4_cash|{seen}"):
            save(ev, seen, ok=f"💸 {lane(src)} → {lane(dst)} {usd(a)}")


def manage(led: dict, name: str, seen: int) -> None:
    last = led["rows"][-1]
    if st.button("↩️ ย้อนรายการล่าสุด (กรณีบันทึกผิด)", key="v4_undo"):
        act(cv.undo, ok=f"↩️ ย้อนแล้ว: {describe(last)}")
    st.caption(esc(f"รายการล่าสุดของทั้งสมุด: n={last['n']} · {last.get('ticker') or ''} {describe(last)}"))
    used = sum(1 for r in led["rows"] if r.get("ticker") == name) - 1
    with st.expander("🗑️ ลบตัวนี้ออกจากรายการ"):
        if used > 0:
            st.caption(f"“{name}” มี {used} รายการแล้ว — ลบไม่ได้ (เงินที่ผ่านมือมันอยู่ในกระเป๋ากลาง) ใช้ “ย้อนรายการล่าสุด” ทีละรายการ")
        else:
            sure = st.checkbox(f"ยืนยัน: ลบ “{name}” (ยังไม่เคยมีรายการ)", key=f"v4_confirm|{name}")
            if st.button("ลบถาวร", key="v4_delete", disabled=not sure):
                act(cv.remove, name, ok=f"🗑️ ลบ {name} แล้ว")


# ── backup and the Chain data that already exists ─────────────────────────────────────────────
def backup_section(book) -> None:
    """Export / import the whole book as one JSON file. ``book`` is None when the data file itself is unreadable."""
    n = st.session_state.get("_v4_up", 0)                            # a new number = a cleared file chooser after an import
    held = bool(book and book["chain"])
    with st.expander("💾 สำรอง / กู้คืนข้อมูล · นำเข้า Chain data"):
        st.caption(BACKUP_NOTE)
        st.markdown("**สำรอง**")
        st.download_button("⬇️ ดาวน์โหลดไฟล์สำรอง (JSON)", data=cv.export_text(book) if held else "", file_name=cv.backup_filename(),
                           mime="application/json", disabled=not held, on_click="ignore", key="v4_export", width="stretch")
        if not held:
            st.caption("ยังไม่มีข้อมูลให้สำรอง" if book is not None else "ไฟล์ข้อมูลอ่านไม่ได้ จึงสำรองไม่ได้ (ไฟล์เดิมยังอยู่ครบ)")
        st.markdown("**กู้คืน / นำเข้า**")
        up = st.file_uploader("เลือกไฟล์ (.json) — ไฟล์สำรองของแอปนี้ หรือ Chain data เดิม", type=["json"], key=f"v4_import|{n}")
        if up is None:
            return
        data = up.getvalue()
        try:
            parsed, warns = cv.parse_backup(data)
            led = cv.fold(parsed)
        except cv.BookError as e:
            st.error(f"❌ {e}")
            return
        names = list(led["tickers"])
        st.markdown(esc(f"✅ **อ่านไฟล์ได้** — {len(names)} ticker · {len(parsed['chain'])} รายการ · Pool CF {usd(led['pool'])} · "
                        f"EV Reserve {usd(led['reserve'])}: " + " · ".join(names[:8]) + (" …" if len(names) > 8 else "")))
        for w in warns[:6]:
            st.caption(esc(f"• {w}"))
        replacing = book is None or held                              # something would be replaced → ask first
        if replacing:
            st.warning("ข้อมูลปัจจุบันจะถูกแทนที่ด้วยไฟล์นี้ — ไฟล์เดิมถูกเก็บไว้ข้างๆ (ไม่ลบ)")
        sure = (not replacing) or st.checkbox("ยืนยัน: แทนที่ข้อมูลปัจจุบันด้วยไฟล์นี้", key=f"v4_import_ok|{n}|{zlib.crc32(data):08x}")
        if st.button("⬆️ นำเข้า (แทนที่ทั้งหมด)", key="v4_import", disabled=not sure):
            act(cv.import_book, data, ok=f"✅ นำเข้าแล้ว: {len(names)} ticker", _v4_up=n + 1, _v4_select=first_active(led))


def main() -> None:
    if msg := st.session_state.pop("_v4_flash", None):
        st.toast(msg)
    if pick := st.session_state.pop("_v4_select", None):
        st.session_state["v4_pos"] = pick
    try:
        book = cv.load(PATH)
    except cv.BookError as e:
        st.title(TITLE)
        st.error(f"🛑 {e}")
        st.caption("ไฟล์เดิมยังอยู่ครบ ไม่ถูกแก้หรือลบ · กด “เริ่มใหม่” เพื่อย้ายไฟล์เดิมไปเก็บไว้ข้างๆ (ไม่ลบ) แล้วเริ่มสมุดเปล่า")
        if st.button("🆕 เริ่มใหม่ (เก็บไฟล์เดิมไว้)", key="v4_reset"):
            try:
                kept = cv.quarantine(PATH)
            except OSError as err:
                st.error(f"💾 ย้ายไฟล์ไม่สำเร็จ: {err}")
            else:
                st.session_state["_v4_flash"] = esc(f"🗄️ เก็บไฟล์เดิมไว้ที่ {kept.name} แล้ว — เริ่มสมุดใหม่ได้เลย")
                st.rerun()
        backup_section(None)                                        # … or put a backup back
        return

    led, seen = cv.fold(book), len(book["chain"])
    names = list(led["tickers"])
    intro()
    if not names:
        st.subheader("🚀 เริ่มเลย")
        if st.button(f"▶ ลองดูตัวอย่างก่อน (Fix_c {DEMO[1]:,.0f} · ราคา {DEMO[2]:,.0f})", key="v4_demo", type="primary", width="stretch"):
            save([{"do": "open", "ticker": DEMO[0], "c": DEMO[1], "t": DEMO[2]},
                  {"do": "cash", "src": cv.OUT, "dst": cv.RESERVE, "amount": DEMO_FUND}], seen,
                 ok=f"✅ เริ่มตัวอย่าง {DEMO[0]} — ใส่ EV Reserve {DEMO_FUND:,.0f} ไว้แล้ว ลองไล่ตามคำแนะนำ 👆 ได้เลย", _v4_select=DEMO[0])
        st.caption("หรือใส่ของจริงของคุณ ↓ (หรือนำเข้า Chain data เดิมด้านล่าง)")
        start_form(seen)
    else:
        shared(led)
        board(led)
        if st.session_state.get("v4_pos") not in names:
            st.session_state["v4_pos"] = first_active(led)
        name = st.selectbox("ตัวที่ทำวงจร", names, key="v4_pos")
        cycle(led, name, seen)
        measure(led["tickers"][name])
        ledger(led, name)
        treasury(led)
        cash_form(led, names, name, seen)
        manage(led, name, seen)
        with st.expander("➕ เริ่มติดตามตัวใหม่"):
            start_form(seen)
    backup_section(book)
    st.caption(FOOTER)


if __name__ == "__main__":
    main()
