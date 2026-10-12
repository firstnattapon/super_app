"""pages/app-v4.py — the page of the whole cycle (Streamlit AppTest) — skipped when streamlit is not installed."""
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import chain_v4 as cv  # noqa: E402

HAVE_ST = importlib.util.find_spec("streamlit") is not None
PAGE = ROOT / "pages" / "app-v4.py"
YOUR_CYCLE = """① จ่าย K ก่อน (Deep ITM LEAPS)
② เก็บ E (หุ้น 20% · LEAPS 80%)
③ Surplus = E − K บวกไหม ?
     ไม่ → ③.1 roll P ขึ้น · ดึง capital gain (เทรนด์ R) ออก · E อยู่ต่อ → กลับ ② (ไม่จ่าย K ซ้ำ)
     ใช่ → ④ Shield (Long Put · Zero-Cost · Put Ratio 2-1)
          → ⑤.1 Scale Up fix_c (Liquidity ถ้าพอจ่าย K วงจรหน้า · ไม่พอ ซื้อ Deep ITM)
          → ⑤.2 Rollover (β · roll P ขึ้น · ดึง capital gain ออก)
          → วัดผล (ΣE ÷ ΣK ≥ 1.0x) → ↻ ① (จ่าย K ของทุนใหม่ c′ ก่อนเสมอ)"""
SYMBOLS = {"c", "P₀", "Pₙ", "Hedge ×", "k", "P_acted", "Vₙ", "gap", "Rₙ", "ΔAₙ", "Aₙ", "Eₙ", "β", "b", "K", "E_cyc", "S", "c′",
           "Pool CF", "EV Reserve", "Ev Burn", "Liquidity", "ΣE", "ΣK", "Ev Efficiency"}       # everything the page shows
LABELS = {"ชำระ", "⑤.1", "⑤.2", "โอน", "เริ่ม"}                                              # rows of a "ที่มา" table that are steps, not symbols

OPEN = {"do": "open", "ticker": "DEMO", "c": 1500, "t": 10}


def pay(t="DEMO", hedge=1):
    return {"do": "pay", "ticker": t, "hedge": hedge}


def move(p, t="DEMO"):
    return {"do": "move", "ticker": t, "p": p}


def cash(src, dst, amount):
    return {"do": "cash", "src": src, "dst": dst, "amount": amount}


def scale(t="DEMO", **kw):
    return {"do": "scale", "ticker": t, "by": "liquidity", "shield": True, **kw}


DEMO_STEPS = [OPEN, cash(cv.OUT, cv.RESERVE, 300)]
CYCLE_STEPS = [*DEMO_STEPS, pay(), move(15), cash("DEMO", cv.POOL, 608.2), move(10), scale()]

if HAVE_ST:
    import streamlit as st_mod
    from streamlit.testing.v1 import AppTest


def unesc(md: str) -> str:
    return md.replace("\\$", "$")


def plain(md: str) -> str:
    return re.sub(r"\s*\n\s*", "\n", md.replace("**", "")).strip()


def goal_of(mmd: str) -> str:
    label = re.search(r'^\s*GOAL\("(.*)"\)\s*$', mmd, re.M).group(1)
    return re.sub(r"</?b>", "", label.replace("<br/>", "\n")).strip()


def cells(md: str) -> list:
    """The body rows of a Markdown table, each as a list of cells."""
    lines = [x for x in md.splitlines() if x.startswith("|")]
    return [[c.strip() for c in x.strip().strip("|").split("|")] for x in lines[2:]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def money(x: float) -> str:
    return f"{'−' if round(x, 2) < 0 else ''}${abs(x):,.2f}"


class FakeUpload:
    """What st.file_uploader hands back — the parts the page uses."""

    def __init__(self, data: bytes, name: str = "backup.json"):
        self.name, self.size, self.type, self._data = name, len(data), "application/json", data

    def getvalue(self) -> bytes:
        return self._data

    def read(self) -> bytes:
        return self._data


@unittest.skipUnless(HAVE_ST, "streamlit not installed")
class AppV4Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "book" / "app_v4.json"          # a folder that does not exist yet
        patcher = mock.patch.dict(os.environ, {"APP_V4_DATA_FILE": str(self.path)})
        patcher.start()
        self.addCleanup(patcher.stop)

    # helpers ---------------------------------------------------------------------------
    def build(self, *steps, path=None):
        path = path or self.path
        cv.commit(path, list(steps), len(cv.load(path)["chain"]))

    def book(self):
        return json.loads(self.path.read_text("utf-8"))

    def kinds(self):
        return [e["do"] for e in self.book()["chain"]]

    def start(self):
        return AppTest.from_file(str(PAGE), default_timeout=60).run()

    def demo(self):
        at = self.start()
        at.button(key="v4_demo").click().run()
        return at

    @staticmethod
    def widget(at, kind, part):
        return next(w for w in getattr(at, kind) if f"|{part}|" in (w.key or ""))

    @staticmethod
    def button(at, part):
        return next(b for b in at.button if (b.key or "").startswith(f"v4|{part}|"))

    @staticmethod
    def buttons(at):
        return sorted(b.key.split("|")[1] for b in at.button if (b.key or "").startswith("v4|"))

    @staticmethod
    def metric(at, label):
        return next(m.value for m in at.metric if m.label.startswith(label))

    @staticmethod
    def work(at, needle=""):
        """The "สูตร → แทนค่า" table(s) on the screen (optionally only those containing ``needle``) as {symbol: (formula, result)}."""
        rows = {}
        for m in at.markdown:
            if "ตัวแปร | สูตร → แทนค่า | ได้" in m.value and needle in unesc(m.value):
                for sym, how, res in cells(unesc(m.value)):
                    rows[sym.strip("*")] = (how, res.strip("*"))
        return rows

    @staticmethod
    def hint(at):
        return [c.value for c in at.caption if c.value.startswith("👆")]

    def assert_clean(self, at):
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])

    def goal_box(self, at):
        return next(b for b in list(at.success) + list(at.info) if "เป้าหมายสูงสุด" in b.value)

    def upload(self, data):
        """Make the page's file chooser (key v4_import|0) hold ``data`` — AppTest cannot drive st.file_uploader itself."""
        fake = FakeUpload(data)
        patcher = mock.patch.object(st_mod, "file_uploader", side_effect=lambda *a, key=None, **k: fake if key == "v4_import|0" else None)
        spy = patcher.start()
        self.addCleanup(patcher.stop)
        return spy

    def backup_of(self, *steps):
        other = Path(self._tmp.name) / "source" / "app_v4.json"
        self.build(*steps, path=other)
        return other.read_bytes()

    @staticmethod
    def checkbox_for_import(at):
        return next(w for w in at.checkbox if (w.key or "").startswith("v4_import_ok"))

    # the first screen ------------------------------------------------------------------
    def test_first_visit_says_what_it_is_at_once(self):
        at = self.start()
        self.assert_clean(at)
        self.assertFalse(self.path.exists())                                     # looking creates nothing
        self.assertEqual([t.value for t in at.title], ["🔗 Chain System v4"])
        texts = "\n".join(m.value for m in at.markdown)
        self.assertIn("ประกันก่อน · ความแกว่งจ่าย · โตด้วยกำไรที่ล็อกแล้ว · เทรนด์เก็บเข้ากระเป๋า", texts)     # the final key message
        key = unesc(at.info[0].value)
        for text in ("แก่นแท้", "−$185.94", "+$141.80", "+$608.20", "−$44.14", "+$250.00", "+$64.06", "$1,564.06", "EV Reserve กลับมา $300.00"):
            self.assertIn(text, key)                                             # the example is computed, not typed in
        self.assertEqual([b.label for b in at.button][:1], ["▶ ลองดูตัวอย่างก่อน (Fix_c 1,500 · ราคา 10)"])   # one click to see it work
        self.assertTrue(any(b.label == "🚀 เริ่มติดตาม" for b in at.button))     # or start with your own numbers
        self.assertTrue(any(e.label == "💾 สำรอง / กู้คืนข้อมูล · นำเข้า Chain data" for e in at.expander))   # or bring the data you have

    def test_the_cycle_on_the_page_is_the_one_you_wrote(self):
        at = self.start()
        self.assertEqual(next(c.value for c in at.code if "จ่าย K ก่อน" in c.value), YOUR_CYCLE)
        for step in ("① จ่าย K", "② เก็บ E", "③ Surplus", "③.1 roll P", "④ Shield", "⑤.1 Scale Up", "⑤.2 Rollover", "วัดผล", "↻ ①"):
            self.assertIn(step, YOUR_CYCLE)

    def test_the_goal_is_the_flowcharts_goal_word_for_word(self):
        at = self.demo()
        for name in ("app_v3_playbook.mmd", "app_v2_minimal.mmd", "app_v3_final.mmd"):
            mmd = (ROOT / name).read_text("utf-8")
            self.assertTrue(plain(self.goal_box(at).value).endswith(goal_of(mmd)), name)

    def test_every_variable_is_explained_with_what_it_is_and_what_it_is_for(self):
        at = self.demo()
        tables = [cells(m.value) for m in at.markdown if "ตัวแปร | คืออะไร | ไว้ทำไม" in m.value]
        self.assertEqual(len(tables), 5)                                          # you fill in · ledger · cycle · shared · score
        shown = set()
        for sym, what, why in (row for rows in tables for row in rows):
            name = re.fullmatch(r"\*\*(.+)\*\*", sym).group(1).replace("\\*", "*")
            shown.add(name)
            self.assertRegex(what, r"^\*\*[^*]+\*\* — .{8,}", name)               # **Thai name** — what it is
            self.assertGreaterEqual(len(why), 8, name)                            # what it is for
        self.assertEqual(shown, SYMBOLS)
        # every symbol the page puts on the screen is one of those
        for label in [m.label for m in at.metric]:
            self.assertTrue(any(s in label for s in SYMBOLS | {"Fix_c"}), label)
        for sym in self.work(at):
            self.assertIn(sym, SYMBOLS | LABELS, sym)

    def test_the_glossary_matches_the_formulas_of_the_logic_module(self):
        at = self.demo()
        text = unesc("\n".join(m.value for m in list(at.markdown) + list(at.caption)))
        self.assertIn(f"K = {cv.PREMIUM_RATE:.1%} × c × Hedge×", text)           # 12.4%
        for formula in ("Rₙ = β + Fix_c × ln(Pₙ / P₀)", "ΔAₙ = Fix_c × (Pₙ / P_acted − 1)", "Aₙ = Aₙ₋₁ + ΔAₙ", "Eₙ = Aₙ − Rₙ", "S = E_cyc − K",
                        "c′ = c + S", "β += (Fix_c − Fix_c′) × ln(P_acted / P₀)", "gap = Vₙ − Fix_c", "ΣE ÷ ΣK", "Pool CF + EV Reserve"):
            self.assertIn(formula, text)

    # the whole cycle, with the buttons ------------------------------------------------------
    def test_the_demo_walks_the_whole_cycle_with_the_buttons(self):
        at = self.demo()
        self.assert_clean(at)
        self.assertEqual(self.kinds(), ["open", "cash"])
        self.assertEqual(at.selectbox(key="v4_pos").value, "DEMO")
        self.assertTrue(any("DEMO" in t.value for t in at.toast))
        self.assertEqual([self.metric(at, x) for x in ("🎱 Pool CF", "🛡️ EV Reserve", "🔥 Ev Burn", "📈 Ev Efficiency", "💧 Liquidity")],
                         ["$0.00", "$300.00", "$0.00", "—", "$300.00"])
        self.assertEqual(self.buttons(at), ["pay"])                              # ① is the only thing to do
        self.assertIn("ตัวอย่างขั้น ①", self.hint(at)[0])

        # ① pay K first
        table = self.work(at, "EV Reserve")
        self.assertEqual(table["K"], ("K = 12.4% × c × Hedge× = 12.4% × 1,500.00 × 1", "185.94"))
        self.assertEqual(table["EV Reserve"], ("ลบ K → เหลือ 114.06", "Ev Burn +185.94"))
        self.button(at, "pay").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "🛡️ EV Reserve"), self.metric(at, "🔥 Ev Burn")), ("$114.06", "$185.94"))
        self.assertEqual(self.buttons(at), ["move"])
        self.assertIn("ตัวอย่างขั้น ②", self.hint(at)[0])

        # ② the price goes to 15: E is the swing, R is the trend
        self.assertEqual(self.widget(at, "number_input", "p").value, 15.0)       # the example price
        table = self.work(at, "gap")
        self.assertEqual(table["Vₙ"][1], "2,250.00")
        self.assertEqual(table["gap"][1], "+750.00 → SELL ≈ 50.0000 หุ้น")
        self.assertEqual((table["Rₙ"][1], table["ΔAₙ"][1], table["Aₙ"][1], table["Eₙ"][1], table["S"][1]),
                         ("608.20", "+750.00", "750.00", "141.80", "−44.14"))
        self.assertEqual(table["ΔAₙ"][0], "FIX_C × (Pₙ / P_acted − 1) = 1,500.00 × (15.0000 / 10.0000 − 1)")     # with the numbers put in
        self.assertTrue(any("S = −$44.14 ยังไม่พอ" in unesc(m.value) for m in at.markdown))
        self.button(at, "move").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "Eₙ"), self.metric(at, "b ·"), self.metric(at, "P_acted")), ("$141.80", "$608.20", "15"))
        self.assertEqual(self.buttons(at), ["move", "pull"])                    # ③ says no: pull the capital gain, or go on harvesting
        self.assertIn("ตัวอย่างขั้น ③.1", self.hint(at)[0])
        self.assertEqual(len(at.success), 0)

        # ③.1 pull the capital gain (the trend R) into Pool CF — E stays, K is not paid again
        self.button(at, "pull").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "b ·"), self.metric(at, "Eₙ")), ("$608.20", "$0.00", "$141.80"))
        self.assertEqual(self.widget(at, "number_input", "p").value, 10.0)       # the price comes back
        self.assertEqual(self.kinds(), ["open", "cash", "pay", "move", "cash"])

        # ② back to 10: the swing paid twice, the trend is gone
        table = self.work(at, "gap")
        self.assertEqual(table["gap"][1], "−500.00 → BUY ≈ 50.0000 หุ้น")
        self.button(at, "move").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "Eₙ"), self.metric(at, "b ·")), ("$250.00", "−$608.20"))        # b < 0: owed back to Pool CF
        self.assertEqual(self.buttons(at), ["move"])                            # ④ is locked until the Shield is confirmed
        self.assertTrue(any("ติ๊ก Shield" in i.value for i in at.info))
        self.assertIn("ตัวอย่างขั้น ④–⑤", self.hint(at)[0])
        self.assertTrue(any(unesc(m.value).startswith("#### ④⑤ S = +$64.06") for m in at.markdown))

        # ④ Shield → ⑤.1 Scale Up → ⑤.2 Rollover
        self.widget(at, "checkbox", "shield").check().run()
        self.assertEqual(self.buttons(at), ["move", "scale"])
        self.assertEqual(self.widget(at, "radio", "by").value, "liquidity")      # 908.20 left ≥ K′ 193.88
        self.assertTrue(any("แนะนำ Liquidity — Liquidity หลังใช้เหลือ $908.20 ≥ K ของวงจรหน้า $193.88" in unesc(c.value) for c in at.caption))
        table = self.work(at, "c′")
        self.assertEqual((table["S"][1], table["c′"][1], table["ชำระ"][1], table["⑤.1"][1]), ("+64.06", "1,564.06", "Pool 608.20 · Reserve 300.00", "Pool −64.06"))
        self.button(at, "scale").click().run()
        self.assert_clean(at)
        self.assertEqual([self.metric(at, x) for x in ("🎱 Pool CF", "🛡️ EV Reserve", "🔥 Ev Burn", "📈 Ev Efficiency", "💧 Liquidity")],
                         ["$608.20", "$300.00", "$185.94", "1.34x", "$908.20"])   # the swing paid for the insurance: EV Reserve is whole
        self.assertEqual((self.metric(at, "c · Fix_c"), self.metric(at, "Eₙ")), ("$1,564.06", "$250.00"))
        self.assertEqual(self.kinds(), ["open", "cash", "pay", "move", "cash", "move", "scale"])
        last = self.book()["chain"][-1]
        self.assertEqual((last["by"], last["shield"], last["settle"], last["pull"]), ("liquidity", True, True, False))   # b < 0: nothing to pull
        self.assertEqual(len(at.success), 1)
        self.assertIn("ถึงเป้าหมายแล้ว", self.goal_box(at).value)
        self.assertEqual(self.buttons(at), ["pay"])                             # ↻ ① — K of the new capital comes first, always
        self.assertEqual(self.work(at, "EV Reserve")["K"][1], "193.88")
        self.assertEqual(self.hint(at), [])                                      # the example is over
        self.assertEqual(at.selectbox(key="v4_pos").value, "DEMO")

    def test_settling_and_the_way_to_fund_it_change_the_treasury_as_the_logic_says(self):
        for by, settle in (("liquidity", True), ("deep_itm", True), ("liquidity", False), ("deep_itm", False)):
            with self.subTest(by=by, settle=settle):
                self.path.unlink(missing_ok=True)
                self.build(*DEMO_STEPS, pay(), move(15), cash("DEMO", cv.POOL, 608.2), move(10))
                at = self.start()
                self.widget(at, "checkbox", "shield").check().run()
                self.widget(at, "checkbox", "settle").set_value(settle).run()
                self.widget(at, "radio", "by").set_value(by).run()
                self.button(at, "scale").click().run()
                self.assert_clean(at)
                want = cv.replay([*DEMO_STEPS, pay(), move(15), cash("DEMO", cv.POOL, 608.2), move(10), scale(by=by, settle=settle)])
                self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "🛡️ EV Reserve")), (money(want["pool"]), money(want["reserve"])))
                self.assertEqual(self.metric(at, "c · Fix_c"), "$1,564.06")
                last = self.book()["chain"][-1]
                self.assertEqual((last["by"], last["settle"]), (by, settle))

    def test_the_shield_must_be_confirmed_before_anything_can_grow(self):
        self.build(*DEMO_STEPS, pay(), move(15), move(10))
        at = self.start()
        self.assertEqual(self.buttons(at), ["move"])                            # no way to press "scale-up" yet
        self.assertEqual(len([w for w in at.radio]), 0)                          # not even the choice of how
        self.assertFalse(any("c′" in sym for sym in self.work(at)))              # and no preview of it
        self.assertEqual(self.kinds().count("scale"), 0)
        self.widget(at, "checkbox", "shield").check().run()
        self.assertEqual(self.buttons(at), ["move", "scale"])
        self.widget(at, "checkbox", "shield").uncheck().run()
        self.assertEqual(self.buttons(at), ["move"])

    def test_scale_up_is_offered_only_when_the_surplus_is_positive(self):
        self.build(*DEMO_STEPS, pay(), move(15))
        at = self.start()
        self.assertEqual([w.key for w in at.checkbox if "|shield|" in (w.key or "")], [])        # S = −44.14: ③ says no
        self.build(move(10))
        at = self.start()
        self.assertEqual(len([w for w in at.checkbox if "|shield|" in (w.key or "")]), 1)        # S = +64.06: ④ appears

    def test_the_recommendation_follows_the_cash_that_is_left(self):
        for funded, want in ((150, "deep_itm"), (400, "liquidity")):
            with self.subTest(funded=funded):
                self.path.unlink(missing_ok=True)
                self.build(OPEN, cash(cv.OUT, cv.RESERVE, funded), pay(), move(15), move(10))
                at = self.start()
                self.widget(at, "checkbox", "shield").check().run()
                self.assertEqual(self.widget(at, "radio", "by").value, want)
                note = next(unesc(c.value) for c in at.caption if c.value.startswith("แนะนำ"))
                self.assertIn(f"แนะนำ {'Liquidity' if want == 'liquidity' else 'Deep ITM'} — Liquidity หลังใช้เหลือ ${funded:,.2f}", note)
                self.assertIn("≥" if want == "liquidity" else "<", note)
                self.assertIn("K ของวงจรหน้า $193.88", note)

    def test_the_pull_is_only_offered_for_a_gain_and_a_negative_baseline_is_a_warning(self):
        self.build(*DEMO_STEPS, pay(), move(15))
        self.assertEqual(self.buttons(self.start()), ["move", "pull"])
        self.build(cash("DEMO", cv.POOL, 608.2), move(10))                       # the gain was spent: b < 0
        at = self.start()
        self.assertEqual(self.metric(at, "b ·"), "−$608.20")
        self.build(scale())
        self.build(pay(), move(9))                                               # next cycle, ② again: b is still negative
        at = self.start()
        self.assertEqual(self.buttons(at), ["move"])
        self.assertTrue(any("เงินที่ดึงไม่ใช่กระสุนซื้อคืน" in w.value for w in at.warning))

    def test_the_pull_is_on_by_default_when_there_is_a_gain_to_pull(self):
        self.build(*DEMO_STEPS, pay(), move(15), move(30))                       # a one-way climb: b = 1,647.92 and S > 0
        at = self.start()
        self.widget(at, "checkbox", "shield").check().run()
        pull = self.widget(at, "checkbox", "pull2")
        self.assertEqual((pull.value, pull.disabled), (True, False))
        self.widget(at, "radio", "by").set_value("deep_itm").run()
        self.button(at, "scale").click().run()
        self.assert_clean(at)
        want = cv.replay([*DEMO_STEPS, pay(), move(15), move(30), scale(by="deep_itm")])
        self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "b ·")), (money(want["pool"]), "$0.00"))
        self.assertGreater(want["pool"], 1647.0)                                   # S rests in Pool CF, and so does the capital gain

    def test_hedge_multiplies_the_premium_a_real_premium_replaces_it_and_hedge_is_remembered(self):
        self.build(*DEMO_STEPS)
        at = self.start()
        self.widget(at, "number_input", "h").set_value(2.0).run()
        k2 = f"{round(cv.PREMIUM_RATE * 1500 * 2, 2):,.2f}"
        self.assertEqual(self.work(at, "EV Reserve")["K"], ("K = 12.4% × c × Hedge× = 12.4% × 1,500.00 × 2", k2))
        self.widget(at, "number_input", "k").set_value(200.0).run()              # the premium you really paid
        self.assertEqual(self.work(at, "EV Reserve")["K"], ("K จริงที่จ่าย", "200.00"))
        self.button(at, "pay").click().run()
        self.assert_clean(at)
        last = self.book()["chain"][-1]
        self.assertEqual((last["hedge"], last["k"]), (2.0, 200.0))
        self.assertEqual((self.metric(at, "🛡️ EV Reserve"), self.metric(at, "🔥 Ev Burn")), ("$100.00", "$200.00"))
        self.build(move(15), move(10), scale())                                  # the cycle closes …
        at = self.start()
        self.assertEqual(self.widget(at, "number_input", "h").value, 2.0)        # … and the next ① starts from the Hedge × you used

    def test_a_bad_price_is_explained_and_cannot_be_committed(self):
        self.build(*DEMO_STEPS, pay())
        at = self.start()
        before = self.path.read_bytes()
        self.widget(at, "number_input", "p").set_value(0.0).run()
        self.assertTrue(any("ราคา fill P ต้องมากกว่า 0" in e.value for e in at.error))
        self.assertEqual(self.buttons(at), [])                                   # nothing to press
        self.assertEqual(self.path.read_bytes(), before)
        self.widget(at, "number_input", "p").set_value(10.0).run()               # the price has not moved: nothing to rebalance
        self.assertTrue(self.button(at, "move").disabled)
        self.assertTrue(any("ไม่มีอะไรให้ rebalance" in c.value for c in at.caption))

    def test_cheap_prices_keep_four_decimals(self):
        self.build({**OPEN, "ticker": "PENNY", "c": 500, "t": 0.0042}, pay("PENNY"))
        at = self.start()
        self.assertEqual(self.metric(at, "P_acted"), "0.0042")
        self.assertEqual(self.widget(at, "number_input", "p").value, 0.0042)
        self.widget(at, "number_input", "p").set_value(0.0063).run()
        self.assertEqual(self.work(at, "gap")["Eₙ"][1], f"{500 * (1.5 - 1 - math.log(1.5)):,.2f}")

    # many tickers, one treasury --------------------------------------------------------------
    def test_two_tickers_share_one_treasury(self):
        self.build(*DEMO_STEPS, {"do": "open", "ticker": "BBB", "c": 1000, "t": 100}, pay(), move(15), cash("DEMO", cv.POOL, 608.2))
        at = self.start()
        self.assertEqual(self.metric(at, "🎱 Pool CF"), "$608.20")
        board = cells(unesc(next(m.value for m in at.markdown if "ticker | ขั้นตอนตอนนี้" in m.value)))
        self.assertEqual([(r[0], r[1]) for r in board], [("**DEMO**", "② เก็บ E"), ("**BBB**", "① จ่าย K")])
        at.selectbox(key="v4_pos").set_value("BBB").run()                       # a second ticker, its own cycle …
        self.assertEqual((self.metric(at, "c · Fix_c"), self.metric(at, "P_acted")), ("$1,000.00", "100"))
        self.assertEqual(self.metric(at, "🎱 Pool CF"), "$608.20")              # … the same Pool CF, EV Reserve, Ev Burn
        self.assertEqual(self.buttons(at), ["pay"])
        self.button(at, "pay").click().run()
        self.assertEqual(self.metric(at, "🔥 Ev Burn"), money(185.94 + 123.96))  # both tickers' insurance, one burn
        self.assertEqual(self.metric(at, "🛡️ EV Reserve"), money(300 - 185.94 - 123.96))
        self.widget(at, "number_input", "p").set_value(200.0).run()
        self.button(at, "move").click().run()                                   # BBB doubles: E 306.85, S +182.89
        self.widget(at, "checkbox", "shield").check().run()
        self.widget(at, "checkbox", "settle").set_value(False).run()            # no settlement: Liquidity must come from the shared Pool CF
        self.widget(at, "checkbox", "pull2").set_value(False).run()             # and BBB's own capital gain stays in its Baseline
        self.widget(at, "radio", "by").set_value("liquidity").run()
        self.button(at, "scale").click().run()
        self.assert_clean(at)
        self.assertEqual(self.metric(at, "🎱 Pool CF"), money(608.2 - 182.89))   # BBB spent Pool CF that DEMO's capital gain had put there
        self.assertEqual(self.metric(at, "c · Fix_c"), "$1,182.89")
        at.selectbox(key="v4_pos").set_value("DEMO").run()
        self.assertEqual(self.metric(at, "🎱 Pool CF"), money(608.2 - 182.89))   # and DEMO sees the same pool
        self.assertEqual(self.metric(at, "c · Fix_c"), "$1,500.00")
        want = cv.totals(cv.fold(cv.load(self.path)))
        self.assertEqual((self.metric(at, "🔥 Ev Burn"), self.metric(at, "💧 Liquidity")), (money(want["burn"]), money(want["liquidity"])))

    def test_the_cash_form_moves_money_and_fills_in_the_shortfall(self):
        self.build(OPEN, cash(cv.OUT, cv.POOL, 500), pay())                      # Pool 500 · EV Reserve −185.94
        at = self.start()
        self.assertTrue(any("EV Reserve ติดลบ" in w.value for w in at.warning))
        route = next(w for w in at.selectbox if (w.key or "").startswith("v4_route|"))
        route.set_value("Pool CF → EV Reserve (Allocate)").run()
        amount = next(w for w in at.number_input if (w.key or "").startswith("v4_cash_a|"))
        self.assertEqual(amount.value, 185.94)                                   # the shortfall, ready to fill
        self.assertTrue(any("Pool CF $500.00 → $314.06 · EV Reserve −$185.94 → $0.00" in unesc(c.value) for c in at.caption))
        before = self.path.read_bytes()
        next(b for b in at.button if (b.key or "").startswith("v4_cash|")).click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "🛡️ EV Reserve")), ("$314.06", "$0.00"))
        self.assertNotEqual(self.path.read_bytes(), before)
        self.assertEqual(self.book()["chain"][-1]["src"], "@pool")
        at = self.start()                                                         # (a fresh page: AppTest keeps widgets of the pass before a rerun)
        route = next(w for w in at.selectbox if (w.key or "").startswith("v4_route|"))
        route.set_value("Pool CF → นอกระบบ (เงินออก)").run()                     # more than the pool holds
        amount = next(w for w in at.number_input if (w.key or "").startswith("v4_cash_a|"))
        amount.set_value(9999.0).run()
        self.assertTrue(any("Pool CF ไม่พอ" in e.value for e in at.error))

    def test_the_ledger_table_is_the_lego_ledger_and_every_row_can_be_explained(self):
        self.build(*CYCLE_STEPS)
        at = self.start()
        self.assert_clean(at)
        rows = cells(unesc(next(m.value for m in at.markdown if "n | รายการ | Pₙ | Fix_c" in m.value)))
        self.assertEqual([r[0] for r in rows], ["7", "6", "5", "4", "3", "1"])  # newest first; row 2 is a cash step of the treasury only
        by_n = {r[0]: r for r in rows}
        self.assertEqual(by_n["4"][2:], ["15", "$1,500.00", "$608.20", "+$750.00", "$750.00", "$141.80", "$608.20"])      # Pₙ Fix_c Rₙ ΔAₙ Aₙ Eₙ b
        self.assertEqual(by_n["6"][2:], ["10", "$1,500.00", "$0.00", "−$500.00", "$250.00", "$250.00", "−$608.20"])
        self.assertEqual(by_n["7"][3], "$1,564.06")                              # the scale-up raises Fix_c, E and A do not move
        self.assertEqual((by_n["7"][5], by_n["7"][6], by_n["7"][7]), ("+$0.00", "$250.00", "$250.00"))
        pick = next(w for w in at.selectbox if (w.key or "").startswith("v4_why|"))
        pick.set_value(4).run()
        why = self.work(at, "Vₙ")
        self.assertEqual(why["Eₙ"], ("Aₙ − Rₙ = 750.00 − 608.20", "141.80"))
        self.assertEqual(why["Rₙ"][0], "β + FIX_C × ln(Pₙ / P₀) = 0.00 + 1,500.00 × ln(15.0000 / 10.0000)")
        pick.set_value(7).run()
        self.assertEqual(self.work(at, "β")["β"][1], "0.0000")                 # the Rollover of the scale-up, with its numbers
        treasury = cells(unesc(next(m.value for m in at.markdown if "n | เวลา | รายการ | Pool CF | EV Reserve" in m.value)))
        self.assertEqual([r[0] for r in treasury], ["7", "5", "3", "2"])
        self.assertEqual((treasury[0][3], treasury[0][4]), ("$608.20", "$300.00"))

    def test_history_shows_the_newest_rows_first_and_at_most_twenty(self):
        self.build(OPEN, cash(cv.OUT, cv.RESERVE, 10_000))
        price = 10.0
        for i in range(14):
            if not cv.fold(cv.load(self.path))["tickers"]["DEMO"]["open"]:
                self.build(pay())
            price *= 1.8 if i % 2 == 0 else 0.6                                  # big swings: some cycles close, some stay open
            self.build(move(round(price, 4)))
            if cv.surplus(cv.fold(cv.load(self.path))["tickers"]["DEMO"]) > 0:
                self.build(scale())
        total = len([r for r in cv.fold(cv.load(self.path))["rows"] if r.get("ticker") == "DEMO"])
        self.assertGreater(total, 20)
        at = self.start()
        self.assert_clean(at)
        shown = cells(unesc(next(m.value for m in at.markdown if "n | รายการ | Pₙ | Fix_c" in m.value)))
        self.assertEqual(len(shown), 20)
        ns = [int(r[0]) for r in shown]
        self.assertEqual(ns, sorted(ns, reverse=True))
        self.assertTrue(any(f"แสดง 20 รายการล่าสุดจากทั้งหมด {total}" in c.value for c in at.caption))

    def test_a_hand_edited_timestamp_cannot_break_the_tables_or_smuggle_markdown(self):
        self.build(*DEMO_STEPS, pay(), move(15))
        book = self.book()
        book["chain"][3]["at"] = "2026|01|01 ![x](http://evil.example/p.png)"
        self.path.write_text(json.dumps(book), encoding="utf-8")
        at = self.start()
        self.assert_clean(at)
        rows = cells(unesc(next(m.value for m in at.markdown if "n | เวลา | รายการ" in m.value)))
        self.assertEqual([len(r) for r in rows], [5] * len(rows))               # still five cells per row
        for row in rows:
            self.assertRegex(row[1], r"^[0-9: \-]*$")
        self.assertFalse(any("evil" in m.value for m in at.markdown))

    # starting, undoing, removing -----------------------------------------------------------------
    def test_the_start_form_checks_what_you_type(self):
        at = self.start()
        for name, c, t, why in (("a*b", 1000.0, 100.0, "อักขระพิเศษ"), ("", 1000.0, 100.0, "1–20"), ("OK", 1000.0, None, "ราคา t")):
            at.text_input(key="v4_name|0").set_value(name)
            at.number_input(key="v4_c|0").set_value(c)
            at.number_input(key="v4_t|0").set_value(t)
            next(b for b in at.button if b.label == "🚀 เริ่มติดตาม").click().run()
            self.assertTrue(any(why in e.value for e in at.error), (name, [e.value for e in at.error]))
            self.assertFalse(self.path.exists(), name)                           # nothing was saved
            self.assertEqual([e.value for e in at.exception], [])
        at.text_input(key="v4_name|0").set_value("ทอง")
        at.number_input(key="v4_c|0").set_value(2500.0)
        at.number_input(key="v4_t|0").set_value(31.5)
        next(b for b in at.button if b.label == "🚀 เริ่มติดตาม").click().run()
        self.assert_clean(at)
        self.assertEqual(self.book()["chain"][0]["ticker"], "ทอง")
        self.assertEqual(at.selectbox(key="v4_pos").value, "ทอง")                # the new one is selected
        self.assertTrue(any("ทอง" in t.value for t in at.toast))
        self.build({"do": "open", "ticker": "Gold", "c": 100, "t": 1})            # added by "another session"
        at = self.start()
        for dup in ("ทอง", "GOLD", "gold"):                                      # the same name, or the same letters in any case
            at.text_input(key="v4_name|0").set_value(dup)
            at.number_input(key="v4_t|0").set_value(10.0)
            next(b for b in at.button if b.label == "🚀 เริ่มติดตาม").click().run()
            self.assertTrue(any("ชื่อนี้มีอยู่แล้ว" in e.value for e in at.error), dup)
            self.assertEqual(len(self.book()["chain"]), 2, dup)

    def test_an_outdated_click_commits_nothing(self):
        at = self.demo()
        self.build(cash(cv.OUT, cv.POOL, 10))                                    # "another session" changes the book after this page drew
        before = self.path.read_bytes()
        self.button(at, "pay").click().run()                                     # its button belongs to the old state: the new page has none like it
        self.assert_clean(at)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.kinds(), ["open", "cash", "cash"])
        at = self.start()                                                        # looking again shows the truth, and then the click lands
        self.button(at, "pay").click().run()
        self.assertEqual(self.kinds(), ["open", "cash", "cash", "pay"])

    def test_undo_takes_back_the_last_step_of_the_whole_book(self):
        at = self.demo()
        self.button(at, "pay").click().run()
        self.assertEqual(self.kinds(), ["open", "cash", "pay"])
        self.assertTrue(any("② เก็บ E" in m.value for m in at.markdown))
        at.button(key="v4_undo").click().run()
        self.assert_clean(at)
        self.assertEqual(self.kinds(), ["open", "cash"])
        self.assertEqual(self.buttons(at), ["pay"])                              # ① again, as if it never happened
        self.assertEqual(self.metric(at, "🛡️ EV Reserve"), "$300.00")
        self.assertTrue(any("ย้อนแล้ว" in t.value for t in at.toast))

    def test_a_ticker_is_deleted_only_while_nobody_used_it(self):
        at = self.demo()
        at.checkbox(key="v4_confirm|DEMO").check().run()                         # nothing was done with DEMO yet
        at.button(key="v4_delete").click().run()
        self.assert_clean(at)
        self.assertEqual(self.kinds(), ["cash"])                                 # the treasury step stays: money was put in
        self.assertEqual([b.key for b in self.start().button if b.key == "v4_demo"], ["v4_demo"])
        self.path.unlink()
        at = self.demo()
        self.button(at, "pay").click().run()
        self.assertEqual([c for c in at.checkbox if (c.key or "").startswith("v4_confirm")], [])
        self.assertEqual([b for b in at.button if b.key == "v4_delete"], [])
        self.assertTrue(any("ลบไม่ได้" in c.value for c in at.caption))

    # the score and the goal ----------------------------------------------------------------
    def test_the_goal_box_follows_ev_efficiency(self):
        self.build(*DEMO_STEPS, pay(), move(15))
        at = self.start()
        self.assertEqual(len(at.success), 0)
        self.assertIn("ยังไม่ถึง 1.0x", self.goal_box(at).value)
        self.assertEqual((self.metric(at, "ΣE"), self.metric(at, "ΣK"), self.metric(at, "Ev Efficiency")), ("$141.80", "$185.94", "0.76x"))
        self.assertTrue(0 < at.get("progress")[-1].proto.value < 100)
        self.build(move(10))                                                     # the swing comes back: ΣE 250.00 ≥ ΣK 185.94
        at = self.start()
        self.assertEqual(len(at.success), 1)
        self.assertEqual(self.metric(at, "Ev Efficiency"), "1.34x")
        self.assertIn("ถึงเป้าหมายแล้ว", self.goal_box(at).value)
        self.assertEqual(at.get("progress")[-1].proto.value, 100)                # the bar is full at 1.0x and above

    def test_a_ticker_without_insurance_yet_cannot_be_measured(self):
        self.build(OPEN)
        at = self.start()
        self.assertIn("ยังวัดไม่ได้", at.get("progress")[-1].proto.text)
        self.assertEqual(self.metric(at, "Ev Efficiency"), "—")
        self.assertEqual(self.buttons(at), ["pay"])

    # backup, and the Chain data that already exists ------------------------------------------------
    def test_the_backup_section_is_there_on_every_screen(self):
        label = "💾 สำรอง / กู้คืนข้อมูล · นำเข้า Chain data"
        self.assertTrue(any(e.label == label for e in self.start().expander))             # a new install
        self.build(*DEMO_STEPS)
        self.assertTrue(any(e.label == label for e in self.start().expander))             # a book with data
        self.path.write_text("{ broken", encoding="utf-8")
        at = self.start()
        self.assertTrue(any("ไฟล์ข้อมูลไม่ใช่ JSON" in e.value for e in at.error))
        self.assertTrue(any(e.label == label for e in at.expander))                        # even when the file cannot be read

    def test_the_download_is_exactly_the_book(self):
        self.build(*CYCLE_STEPS)
        with mock.patch.object(st_mod, "download_button", return_value=False) as spy:
            at = self.start()
        self.assert_clean(at)
        kw = spy.call_args.kwargs
        self.assertIn("ดาวน์โหลดไฟล์สำรอง", spy.call_args.args[0])
        self.assertEqual(kw["data"], self.path.read_text("utf-8"))                         # byte for byte what the file holds
        self.assertEqual(kw["data"], cv.export_text(cv.load(self.path)))
        self.assertRegex(kw["file_name"], r"^app_v4_backup-\d{8}-\d{4}\.json$")
        self.assertEqual((kw["mime"], kw["disabled"]), ("application/json", False))
        self.assertEqual(cv.parse_backup(kw["data"].encode("utf-8"))[0], cv.load(self.path))   # and the page itself accepts it back

    def test_there_is_nothing_to_download_before_the_first_ticker(self):
        with mock.patch.object(st_mod, "download_button", return_value=False) as spy:
            at = self.start()
        self.assertTrue(spy.call_args.kwargs["disabled"])
        self.assertTrue(any("ยังไม่มีข้อมูลให้สำรอง" in c.value for c in at.caption))
        self.assertFalse(self.path.exists())                                                # looking creates nothing

    def test_a_backup_restores_a_fresh_install_without_asking(self):
        data = self.backup_of(*CYCLE_STEPS, {"do": "open", "ticker": "BBB", "c": 500, "t": 20})
        spy = self.upload(data)
        at = self.start()
        self.assert_clean(at)
        preview = next(unesc(m.value) for m in at.markdown if "อ่านไฟล์ได้" in m.value)
        for text in ("2 ticker · 8 รายการ", "DEMO", "BBB", "Pool CF $608.20", "EV Reserve $300.00"):
            self.assertIn(text, preview)
        self.assertFalse(self.path.exists())                                                # reading the file changes nothing
        self.assertEqual([w for w in at.checkbox if (w.key or "").startswith("v4_import_ok")], [])   # nothing to replace: no question
        self.assertEqual(len(at.warning), 0)
        at.button(key="v4_import").click().run()
        self.assert_clean(at)
        self.assertEqual(self.path.read_bytes(), data)                                      # the book is the backup, exactly
        self.assertEqual(at.selectbox(key="v4_pos").value, "DEMO")
        self.assertEqual((self.metric(at, "c · Fix_c"), self.metric(at, "🎱 Pool CF")), ("$1,564.06", "$608.20"))
        self.assertTrue(any("นำเข้าแล้ว" in t.value and "2 ticker" in t.value for t in at.toast))
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["app_v4.json"])     # nothing was replaced, nothing kept
        self.assertEqual(spy.call_args.kwargs["key"], "v4_import|1")                        # and the chooser is a fresh, empty one

    def test_restoring_over_existing_data_asks_first_and_keeps_the_old_file(self):
        self.build(*DEMO_STEPS)
        old = self.path.read_bytes()
        data = self.backup_of({**OPEN, "ticker": "NEW"})
        self.upload(data)
        at = self.start()
        self.assert_clean(at)
        self.assertEqual(len(at.warning), 1)                                                # "this replaces what you have"
        self.assertTrue(at.button(key="v4_import").disabled)                                # … until you say yes
        self.assertEqual(self.path.read_bytes(), old)
        self.checkbox_for_import(at).check().run()
        self.assertFalse(at.button(key="v4_import").disabled)
        self.assertEqual(self.path.read_bytes(), old)                                       # ticking is not importing
        at.button(key="v4_import").click().run()
        self.assert_clean(at)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual([k.read_bytes() for k in self.path.parent.glob("app_v4.json.replaced-*")], [old])   # the old book is kept
        self.assertEqual(at.selectbox(key="v4_pos").value, "NEW")

    def test_the_confirmation_belongs_to_one_file(self):
        self.build(*DEMO_STEPS)
        first, second = self.backup_of({**OPEN, "ticker": "ONE"}), self.backup_of({**OPEN, "ticker": "TWO"})
        self.upload(first)
        at = self.start()
        self.checkbox_for_import(at).check().run()
        self.assertFalse(at.button(key="v4_import").disabled)
        self.upload(second)                                                                 # another file is chosen …
        at.run()
        self.assertFalse(self.checkbox_for_import(at).value)                                # … so the yes does not carry over
        self.assertTrue(at.button(key="v4_import").disabled)

    def test_a_bad_backup_is_explained_and_nothing_changes(self):
        self.build(*DEMO_STEPS)
        before = self.path.read_bytes()
        good = json.loads(before)
        cases = (
            (b"{ not json", "ไม่ใช่ JSON"), (b"\xff\xfe\x00", "UTF-8"), (b"7", "ไม่ใช่ไฟล์ Chain"), (b"[]", "ไม่มี ticker"),
            (json.dumps({**good, "version": 5}).encode(), "ไม่ใช่สมุดของ app-v4"), (b" " * 2_000_001, "ใหญ่เกินไป"),
            (json.dumps({**good, "chain": [{"do": "move", "ticker": "A", "p": 5}]}).encode(), "ไม่พบ"),
            (json.dumps({**good, "chain": [{**good["chain"][0], "ticker": "[x](http://evil.example)"}]}).encode(), "อักขระพิเศษ"),
            (json.dumps({**good, "chain": [{**good["chain"][0], "c": -1}]}).encode(), "ทุน c"),
        )
        for data, why in cases:
            with self.subTest(why=why):
                self.upload(data)
                at = self.start()
                self.assertEqual([e.value for e in at.exception], [])
                self.assertTrue(any(why in e.value for e in at.error), (why, [e.value for e in at.error]))
                self.assertEqual([b for b in at.button if b.key == "v4_import"], [])        # nothing to press
                self.assertEqual(self.path.read_bytes(), before)
                self.assertEqual([p.name for p in self.path.parent.iterdir()], ["app_v4.json"])

    def test_a_broken_data_file_can_be_repaired_by_a_backup(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ broken", encoding="utf-8")
        self.upload(self.backup_of({**OPEN, "ticker": "BACK"}))
        at = self.start()
        self.assertTrue(any("ไฟล์ข้อมูลไม่ใช่ JSON" in e.value for e in at.error))          # the error is still what you see first
        self.assertTrue(at.button(key="v4_import").disabled)                                # something would be replaced: ask
        self.checkbox_for_import(at).check().run()
        at.button(key="v4_import").click().run()
        self.assert_clean(at)
        self.assertEqual([e["ticker"] for e in self.book()["chain"]], ["BACK"])
        self.assertEqual([k.read_text("utf-8") for k in self.path.parent.glob("app_v4.json.replaced-*")], ["{ broken"])

    def test_a_backup_from_the_page_goes_back_into_the_page(self):
        """The whole loop: use it, download, lose the file, restore — and carry on from the same numbers."""
        self.demo()
        self.build(pay(), move(15), cash("DEMO", cv.POOL, 608.2), move(10))
        with mock.patch.object(st_mod, "download_button", return_value=False) as spy:
            self.start()
        data = spy.call_args.kwargs["data"].encode("utf-8")
        self.path.unlink()                                                                  # the host restarted: the file is gone
        self.assertTrue(any(b.key == "v4_demo" for b in self.start().button))
        self.upload(data)
        at = self.start()
        at.button(key="v4_import").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "Eₙ"), self.metric(at, "b ·"), self.metric(at, "🎱 Pool CF")), ("$250.00", "−$608.20", "$608.20"))
        self.assertEqual(len(at.success), 1)                                                # the goal box is green again

    def test_the_real_chain_data_of_the_earlier_app_comes_in_and_runs(self):
        real = (ROOT / "data" / "chain_system_backup.json").read_bytes()
        self.upload(real)
        at = self.start()
        self.assert_clean(at)
        preview = next(unesc(m.value) for m in at.markdown if "อ่านไฟล์ได้" in m.value)
        for text in ("15 ticker", "Pool CF $17,758.58", "EV Reserve −$16,256.17"):
            self.assertIn(text, preview)
        self.assertTrue(any("FSUN" in c.value for c in at.caption))                         # the closed one is left out, and said so
        self.assertFalse(self.path.exists())
        at.button(key="v4_import").click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "🛡️ EV Reserve")), ("$17,758.58", "−$16,256.17"))
        self.assertTrue(any("EV Reserve ติดลบ" in w.value for w in at.warning))
        board = cells(unesc(next(m.value for m in at.markdown if "ticker | ขั้นตอนตอนนี้" in m.value)))
        self.assertEqual(len(board), 15)
        self.assertEqual({r[1] for r in board}, {"① จ่าย K"})                               # every ticker starts a new cycle with ①
        at = self.start()                                                                  # (a fresh page: AppTest keeps widgets of the pass before a rerun)
        name = at.selectbox(key="v4_pos").value
        self.assertEqual(name, "RIVN")                                                     # not NEGG (Fix_c $0.01): the first with real capital
        self.assertEqual(self.metric(at, "c · Fix_c"), "$2,300.00")
        self.assertEqual(self.buttons(at), ["pay"])
        route = next(w for w in at.selectbox if (w.key or "").startswith("v4_route|"))
        route.set_value("Pool CF → EV Reserve (Allocate)").run()                           # fill the shortfall from Pool CF in one click
        amount = next(w for w in at.number_input if (w.key or "").startswith("v4_cash_a|"))
        self.assertEqual(amount.value, 16256.17)
        next(b for b in at.button if (b.key or "").startswith("v4_cash|")).click().run()
        self.assertEqual((self.metric(at, "🎱 Pool CF"), self.metric(at, "🛡️ EV Reserve")), ("$1,502.41", "$0.00"))
        at = self.start()
        self.button(at, "pay").click().run()                                                # ① on a real ticker, from the real treasury
        self.assert_clean(at)
        self.assertEqual(at.selectbox(key="v4_pos").value, name)
        self.assertEqual(self.buttons(at), ["move"])

    def test_the_first_generation_list_comes_in_too(self):
        self.upload((ROOT / "trading_data.json").read_bytes())
        at = self.start()
        self.assert_clean(at)
        self.assertIn("16 ticker", next(unesc(m.value) for m in at.markdown if "อ่านไฟล์ได้" in m.value))

    # a broken file · isolation ---------------------------------------------------------------
    def test_a_broken_file_is_kept_and_reset_moves_it_aside(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ definitely not json", encoding="utf-8")
        at = self.start()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("ไฟล์ข้อมูลไม่ใช่ JSON" in e.value for e in at.error))
        self.assertEqual(self.path.read_text("utf-8"), "{ definitely not json")        # never overwritten
        self.assertEqual([b for b in at.button if b.key == "v4_demo"], [])               # nothing can be written over it
        at.button(key="v4_reset").click().run()
        self.assert_clean(at)
        kept = list(self.path.parent.glob("app_v4.json.broken-*"))
        self.assertEqual([k.read_text("utf-8") for k in kept], ["{ definitely not json"])  # moved, not deleted
        self.assertFalse(self.path.exists())
        self.assertTrue(any(b.key == "v4_demo" for b in at.button))                      # a fresh book, first screen again

    def test_a_book_of_another_version_is_kept_and_explained(self):
        self.path.parent.mkdir(parents=True)
        old = json.dumps({"version": 3, "positions": [{"name": "OLD", "c0": 1000, "t0": 100, "rounds": []}]})
        self.path.write_text(old, encoding="utf-8")
        at = self.start()
        self.assertTrue(any("ไม่ใช่สมุดของ app-v4" in e.value for e in at.error))
        self.assertEqual(self.path.read_text("utf-8"), old)                              # never touched
        at.button(key="v4_reset").click().run()
        self.assert_clean(at)
        self.assertEqual([k.read_text("utf-8") for k in self.path.parent.glob("app_v4.json.broken-*")], [old])
        self.assertTrue(any(b.key == "v4_demo" for b in at.button))

    def test_a_hand_edited_book_that_breaks_a_rule_names_the_step(self):
        self.path.parent.mkdir(parents=True)
        raw = {"version": 4, "chain": [{"do": "open", "ticker": "A", "c": 100, "t": 5}, {"do": "move", "ticker": "A", "p": 6}]}
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        at = self.start()
        self.assertTrue(any("รายการที่ 2: A: ต้องจ่าย K ก่อน" in e.value for e in at.error))
        self.assertEqual(json.loads(self.path.read_text("utf-8")), raw)

    def test_a_session_touches_no_other_file_of_the_repo(self):
        watched = [ROOT / "trading_data.json", ROOT / "data" / "chain_system_backup.json"]
        other = Path(self._tmp.name) / "trading_data.json"                                 # the old portfolio file, if it were pointed at
        other.write_text(watched[0].read_text("utf-8"), encoding="utf-8")
        before = {p: sha(p) for p in watched + [other]}
        listing = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("*"))
        with mock.patch.dict(os.environ, {"CHAIN_V2_DATA_FILE": str(other), "APP_V3_DATA_FILE": str(other)}):
            at = self.demo()
            self.button(at, "pay").click().run()
            self.button(at, "move").click().run()
            at.button(key="v4_undo").click().run()
            at.button(key="v4_undo").click().run()
            at.checkbox(key="v4_confirm|DEMO").check().run()
            at.button(key="v4_delete").click().run()
            self.assert_clean(at)
        self.assertEqual({p: sha(p) for p in watched + [other]}, before)
        self.assertEqual(sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("*")), listing)   # no new file in data/

    def test_the_page_loads_none_of_the_other_modules(self):
        code = (
            "import os, sys\n"
            "from streamlit.testing.v1 import AppTest\n"
            "at = AppTest.from_file(sys.argv[1], default_timeout=60).run()\n"
            "at.button(key='v4_demo').click().run()\n"
            "assert not at.exception, [e.value for e in at.exception]\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in {'engine_v2', 'ledger_v2', 'learn_v2', 'lego_hover_table',\n"
            "                                                            'chain_system', 'flywheels', 'chain_v3'})\n"
            "print('LOADED', bad)\n")
        out = subprocess.run([sys.executable, "-c", code, str(PAGE)], capture_output=True, text=True, timeout=120, cwd=self._tmp.name,
                             env={**os.environ, "PYTHONPATH": ""})
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertIn("LOADED []", out.stdout)

    # housekeeping -----------------------------------------------------------------------------
    def test_the_data_file_is_git_ignored_and_the_page_stays_small(self):
        src = PAGE.read_text("utf-8")
        self.assertIn('ROOT / "data" / "app_v4.json"', src)
        ignored = (ROOT / ".gitignore").read_text("utf-8").split()
        self.assertIn("data/app_v4.json", ignored)
        self.assertIn("data/app_v4.json.*", ignored)                                          # temp, .replaced-… and .broken-… files too
        self.assertLessEqual(len(src.splitlines()), 600)
        for heavy in ("pandas", "plotly", "numpy", "components", "dataframe"):
            self.assertNotIn(heavy, src, heavy)

    def test_ci_compiles_the_new_files(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
        self.assertIn("chain_v4.py", ci)
        self.assertIn("pages/app-v4.py", ci)


if __name__ == "__main__":
    unittest.main()
