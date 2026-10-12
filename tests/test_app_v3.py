"""pages/app-v3.py — the standalone, beginner-first page (Streamlit AppTest) — skipped when streamlit is not installed."""
import hashlib
import importlib.util
import json
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
import chain_v3 as cv  # noqa: E402

HAVE_ST = importlib.util.find_spec("streamlit") is not None
PAGE = ROOT / "pages" / "app-v3.py"
SYMBOLS = {"c", "t", "P", "Hedge ×", "E", "K", "S", "c′", "t′", "P*", "ΣE", "ΣK", "Ev Efficiency"}   # everything the page shows
ONE = cv.plan(1000, 100, 200, 1.0)                                       # the demo: the price doubles …
BACK = cv.plan(ONE["c_after"], 200, 100, 1.0)                            # … and comes back to where it was

if HAVE_ST:
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


def mult(x: float) -> str:
    return f"{x:.2f}x".replace("-", "−")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(HAVE_ST, "streamlit not installed")
class AppV3Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "book" / "app_v3.json"          # a folder that does not exist yet
        patcher = mock.patch.dict(os.environ, {"APP_V3_DATA_FILE": str(self.path)})
        patcher.start()
        self.addCleanup(patcher.stop)

    # helpers ---------------------------------------------------------------------------
    def book(self):
        return json.loads(self.path.read_text("utf-8"))

    def pos(self, name="DEMO"):
        return next(p for p in self.book()["positions"] if p["name"] == name)

    def start(self):
        return AppTest.from_file(str(PAGE), default_timeout=60).run()

    def demo(self):
        at = self.start()
        at.button(key="v3_demo").click().run()
        return at

    @staticmethod
    def widget(at, kind, part):
        return next(w for w in getattr(at, kind) if f"|{part}|" in (w.key or ""))

    @staticmethod
    def commit_button(at):
        return next(b for b in at.button if (b.key or "").startswith("v3|commit|"))

    @staticmethod
    def submit(at):
        return next(b for b in at.button if b.label == "🚀 เริ่มติดตาม")

    @staticmethod
    def steps(at):
        return unesc(next(m.value for m in at.markdown if "ขั้น | คำนวณ | ได้" in m.value))

    @staticmethod
    def metric(at, label):
        return next(m.value for m in at.metric if m.label.startswith(label))

    @staticmethod
    def breakeven(at):
        return unesc(next(c.value for c in at.caption if "ราคาคุ้มทุน" in c.value or "ไม่มีค่าประกัน" in c.value))

    def goal_box(self, at):
        return next(b for b in list(at.success) + list(at.info) if "เป้าหมายสูงสุด" in b.value)

    def assert_clean(self, at):
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])

    def set_price(self, at, price):
        self.widget(at, "number_input", "p").set_value(price).run()

    # the first screen ------------------------------------------------------------------
    def test_first_visit_says_what_it_is_at_once(self):
        at = self.start()
        self.assert_clean(at)
        self.assertFalse(self.path.exists())                                     # looking creates nothing
        self.assertEqual([t.value for t in at.title], ["🎯 Chain System v3"])
        texts = "\n".join(m.value for m in at.markdown)
        self.assertIn("ให้ **ความแกว่งของราคา** จ่าย **ค่าประกันของตัวเอง**", texts)       # the essence, in one line
        for step in ("① เก็บ", "② จ่ายประกันก่อน", "③ เหลือไหม?", "④ โตเฉพาะที่เหลือ", "⑤ เริ่มรอบใหม่"):
            self.assertIn(step, texts)
        key = unesc(at.info[0].value)
        for text in ("แก่นแท้", "เก็บกำไรจากความแกว่ง (**E**)", "+$306.85", "−$123.96", "+$182.89", "$1,182.89", "$1,264.73",
                     "ราคาเท่าเดิม แต่ทุนโต", "ไม่ลด"):
            self.assertIn(text, key)
        self.assertEqual((ONE["E"], ONE["K"], ONE["S"], ONE["c_after"], BACK["c_after"]),     # computed, not typed in
                         (306.85, 123.96, 182.89, 1182.89, 1264.73))
        self.assertEqual([b.label for b in at.button][:1], ["▶ ลองดูตัวอย่างก่อน (ทุน 1,000 · ราคา 100)"])   # one click to see it work
        self.assertTrue(any(b.label == "🚀 เริ่มติดตาม" for b in at.button))     # or start with your own numbers

    def test_every_variable_is_explained_with_what_it_is_and_what_it_is_for(self):
        at = self.demo()
        tables = [cells(m.value) for m in at.markdown if "ตัวแปร | คืออะไร | ไว้ทำไม" in m.value]
        self.assertEqual(len(tables), 3)                                          # asked → computed → score
        shown = set()
        for sym, what, why in (row for rows in tables for row in rows):
            name = re.fullmatch(r"\*\*(.+)\*\*", sym).group(1).replace("\\*", "*")
            shown.add(name)
            self.assertRegex(what, r"^\*\*[^*]+\*\* — .{8,}", name)               # **Thai name** — what it is
            self.assertGreaterEqual(len(why), 8, name)                            # what it is for
        self.assertEqual(shown, SYMBOLS)
        # every symbol the page puts on the screen is one of those
        for label in [m.label for m in at.metric]:
            self.assertIn(label.split(" · ")[0], SYMBOLS, label)
        for row in cells(next(m.value for m in at.markdown if "ขั้น | คำนวณ | ได้" in m.value)):
            self.assertIn(re.search(r"\*\*\S+ (\S+)\*\*", row[0]).group(1), SYMBOLS, row[0])
        legend = next(m.value for m in at.markdown if m.value.startswith("**c** ทุน"))
        self.assertTrue({s.replace("\\*", "*") for s in re.findall(r"\*\*(\S+)\*\*", legend)} <= SYMBOLS, legend)

    def test_the_glossary_matches_the_formulas_of_the_logic_module(self):
        at = self.demo()
        text = unesc("\n".join(m.value for m in list(at.markdown) + list(at.caption)))
        self.assertIn(f"K = {cv.PREMIUM_RATE:.1%} · c · Hedge×", text)           # 12.4%
        for formula in ("E = c · (x − 1 − ln x)", "S = E − K", "c′ = c + max(0, S)", "t′ = P", "ΣE ÷ ΣK", "E = A − R"):
            self.assertIn(formula, text)
        low, high = cv.breakeven_prices(1000, 100, 1.0)                          # "≈ +58% up or −42% down"
        self.assertIn(f"ขึ้นไป ≈ {high / 100 - 1:+.0%} หรือลงไป ≈ {low / 100 - 1:+.0%}".replace("-", "−"), text)

    # one round --------------------------------------------------------------------------
    def test_demo_shows_the_worked_example_and_saves_no_round(self):
        at = self.demo()
        self.assert_clean(at)
        self.assertEqual((self.pos()["c0"], self.pos()["t0"], self.pos()["rounds"]), (1000.0, 100.0, []))
        self.assertEqual(at.selectbox(key="v3_pos").value, "DEMO")
        self.assertTrue(any("DEMO" in t.value for t in at.toast))
        self.assertEqual(self.widget(at, "number_input", "p").value, 200.0)       # the example price
        self.assertTrue(any("ตัวอย่างรอบ 1" in c.value for c in at.caption))
        table = self.steps(at)
        for text in ("x = P / t = 200 / 100 = 2", "1,000.00 × (2 − 1 − ln 2)", "+$306.85", "−$123.96", "+$182.89",
                     "$1,000.00 → $1,182.89", "✓ เหลือ", "100 → 200", "12.4% × 1,000.00 × 1", "S = E − K = $306.85 − $123.96"):
            self.assertIn(text, table)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "100"))
        hint = self.breakeven(at)                                                  # two prices, one on each side of t
        low, high = cv.breakeven_prices(1000, 100, 1.0)
        for text in ("สูงกว่า 158.38", "+58.4%", "ต่ำกว่า 58.11", "−41.9%"):
            self.assertIn(text, hint)
        self.assertEqual((f"{low:.2f}", f"{high:.2f}"), ("58.11", "158.38"))
        self.assertTrue(any("ทุนโตเป็น $1,182.89" in unesc(m.value) for m in at.markdown))

    def test_the_demo_walks_the_example_in_two_clicks(self):
        at = self.demo()
        self.commit_button(at).click().run()
        self.assert_clean(at)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,182.89", "200"))
        self.assertEqual(self.widget(at, "number_input", "p").value, 100.0)       # round 2 of the example: back to where it began
        self.assertTrue(any("ตัวอย่างรอบ 2" in c.value for c in at.caption))
        self.assertIn("**+$228.47**", self.steps(at))                               # E of the way back
        self.commit_button(at).click().run()
        self.assert_clean(at)
        self.assertEqual([r["p"] for r in self.pos()["rounds"]], [200.0, 100.0])
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,264.73", "100"))   # the price is where it began …
        self.assertEqual((self.metric(at, "ΣE"), self.metric(at, "ΣK"), self.metric(at, "Ev Efficiency")),
                         ("$535.32", "$270.59", "1.98x"))                           # … and the capital grew 26%
        self.assertEqual(len(at.success), 1)
        self.assertIn("ถึงเป้าหมายแล้ว", self.goal_box(at).value)
        self.assertFalse(any("ตัวอย่างรอบ" in c.value for c in at.caption))        # the example is over
        self.assertEqual(self.widget(at, "number_input", "p").value, 100.0)       # then every round starts at t as usual

    def test_the_example_hint_goes_away_when_you_leave_its_path(self):
        at = self.demo()
        self.set_price(at, 150.0)
        self.commit_button(at).click().run()
        self.assertFalse(any("ตัวอย่างรอบ" in c.value for c in at.caption))
        self.assertEqual(self.widget(at, "number_input", "p").value, 150.0)       # not the example's 100: the default is t again

    def test_commit_writes_exactly_what_the_table_showed(self):
        at = self.demo()
        old_key = self.commit_button(at).key
        self.commit_button(at).click().run()
        self.assert_clean(at)
        rounds = self.pos()["rounds"]
        self.assertEqual(len(rounds), 1)
        for k in ("t", "p", "c", "E", "K", "S", "c_after", "hedge_x", "real_k"):
            self.assertEqual(rounds[0][k], ONE[k], k)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,182.89", "200"))   # ④ grew · ⑤ re-centred
        self.assertIsNone(self.widget(at, "number_input", "k").value)             # a real premium is per round
        self.assertTrue(any("DEMO" in t.value and "$1,182.89" in t.value for t in at.toast))
        with self.assertRaises(KeyError):                                          # a stale click has no button to land on
            at.button(key=old_key)
        self.assertTrue(any(e.label == "🧾 ประวัติ (1 รอบ)" for e in at.expander))

    def test_the_next_round_starts_where_this_one_ended(self):
        cv.start(self.path, "AAA", 1000, 100)
        at = self.start()
        self.assertEqual(self.widget(at, "number_input", "p").value, 100.0)       # no example here: P starts at t
        self.set_price(at, 130.0)
        self.commit_button(at).click().run()
        self.assert_clean(at)
        self.assertEqual(self.widget(at, "number_input", "p").value, 130.0)       # ⑤ t′ = P
        self.assertEqual(self.metric(at, "t ·"), "130")
        self.assertIsNone(self.widget(at, "number_input", "k").value)

    def test_a_plain_rise_does_not_pay_only_the_swing_does(self):
        at = self.demo()
        self.set_price(at, 125.0)                                                  # +25%: a trend, not much of a swing
        table = self.steps(at)
        self.assertIn("**+$26.86**", table)                                        # E = 1,000 · (1.25 − 1 − ln 1.25)
        self.assertIn("**−$97.10** ✗ ไม่เหลือ", table)
        self.assertIn("c′ = c — ไม่มีส่วนเหลือ", table)
        self.assertIn("$1,000.00 → $1,000.00", table)
        self.assertTrue(any("ทุนเท่าเดิม $1,000.00 (ไม่ลด)" in unesc(m.value) for m in at.markdown))
        self.commit_button(at).click().run()
        last = self.pos()["rounds"][-1]
        self.assertEqual((last["E"], last["c"], last["c_after"], last["K"], last["p"]), (26.86, 1000.0, 1000.0, 123.96, 125.0))
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "125"))   # c did not shrink, the premium was paid

    def test_a_halving_is_a_swing_too(self):
        at = self.demo()
        self.set_price(at, 50.0)
        table = self.steps(at)
        self.assertIn("**+$193.15**", table)                                       # E is never negative — down pays as well as up
        self.assertIn("**+$69.19** ✓ เหลือ", table)
        self.assertIn("$1,000.00 → $1,069.19", table)
        self.commit_button(at).click().run()
        self.assertEqual(self.pos()["rounds"][-1]["c_after"], 1069.19)

    def test_hedge_multiplies_the_premium_and_stays_for_the_next_round(self):
        at = self.demo()
        self.widget(at, "number_input", "h").set_value(2.0).run()
        self.assertIn("12.4% × 1,000.00 × 2", self.steps(at))
        low, high = cv.breakeven_prices(1000, 100, 2.0)
        for text in (f"สูงกว่า {high:,.2f}", f"ต่ำกว่า {low:,.2f}"):               # the insurance is dearer: the swing must be bigger
            self.assertIn(text, self.breakeven(at))
        self.commit_button(at).click().run()
        self.assertEqual((self.pos()["rounds"][-1]["K"], self.pos()["rounds"][-1]["hedge_x"]), (247.92, 2.0))
        self.assertEqual(self.widget(at, "number_input", "h").value, 2.0)         # you do not retype it every round

    def test_a_real_premium_replaces_the_formula(self):
        at = self.demo()
        self.widget(at, "number_input", "k").set_value(0.0).run()                  # K = 0: a round with no insurance bought
        self.assert_clean(at)
        self.assertIn("| **$0.00** |", self.steps(at))                              # a cost of nothing is not "−$0.00"
        self.assertIn("+$306.85", self.steps(at))
        self.assertIn("ไม่มีค่าประกันรอบนี้", self.breakeven(at))                   # then any move pays
        self.widget(at, "number_input", "k").set_value(50.0).run()
        table = self.steps(at)
        self.assertIn("ค่าประกันที่จ่ายจริง", table)
        self.assertNotIn("12.4% × 1,000.00", table)                                # the formula is no longer used
        self.assertIn("−$50.00", table)
        self.assertIn("+$256.85", table)                                           # 306.85 − 50
        self.commit_button(at).click().run()
        last = self.pos()["rounds"][-1]
        self.assertEqual((last["K"], last["S"], last["real_k"]), (50.0, 256.85, True))

    def test_a_bad_price_is_explained_and_cannot_be_committed(self):
        at = self.demo()
        self.set_price(at, 0.0)
        self.assertTrue(any("ราคาใหม่ P ต้องมากกว่า 0" in e.value for e in at.error))
        self.assertEqual([b for b in at.button if (b.key or "").startswith("v3|commit|")], [])
        self.assertEqual(self.pos()["rounds"], [])

    def test_cheap_prices_keep_four_decimals(self):
        at = self.start()
        at.text_input(key="v3_name|0").set_value("MEME")
        at.number_input(key="v3_c|0").set_value(500.0)
        at.number_input(key="v3_t|0").set_value(0.0042)
        self.submit(at).click().run()
        self.assert_clean(at)
        self.assertEqual(self.metric(at, "t ·"), "0.0042")
        at = self.start()                                                           # (a new session: AppTest keeps first-pass leftovers)
        self.set_price(at, 0.006)
        table = self.steps(at)
        self.assertIn("x = P / t = 0.006 / 0.0042 = 1.4286", table)
        self.assertIn("500.00 × (1.4286 − 1 − ln 1.4286)", table)
        low, high = cv.breakeven_prices(500, 0.0042)
        for text in (f"สูงกว่า {high:,.4f}", f"ต่ำกว่า {low:,.4f}"):
            self.assertIn(text, self.breakeven(at))

    def test_an_outdated_click_commits_nothing(self):
        at = self.demo()
        stale = self.commit_button(at)
        cv.commit(self.path, "DEMO", cv.plan(1000, 100, 110, 1.0))                # "another session" saves between render and click
        stale.click().run()
        self.assert_clean(at)
        rounds = self.pos()["rounds"]
        self.assertEqual([r["p"] for r in rounds], [110.0])                        # only the other session's round

    # undo · delete · start ---------------------------------------------------------------
    def test_undo_takes_back_the_last_round(self):
        at = self.demo()
        self.commit_button(at).click().run()
        self.commit_button(at).click().run()
        self.assertEqual([r["p"] for r in self.pos()["rounds"]], [200.0, 100.0])
        at.button(key="v3_undo").click().run()
        self.assert_clean(at)
        self.assertEqual([r["p"] for r in self.pos()["rounds"]], [200.0])
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,182.89", "200"))   # back where it was
        at.button(key="v3_undo").click().run()
        self.assertEqual(self.pos()["rounds"], [])
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "100"))
        self.assertEqual([b for b in at.button if b.key == "v3_undo"], [])         # nothing left to take back

    def test_delete_needs_the_confirmation(self):
        at = self.demo()
        self.assertTrue(at.button(key="v3_delete").disabled)
        at.checkbox(key="v3_confirm|DEMO").check().run()
        self.assertFalse(at.button(key="v3_delete").disabled)
        at.button(key="v3_delete").click().run()
        self.assert_clean(at)
        self.assertEqual(self.book()["positions"], [])
        self.assertTrue(any("ลบ DEMO" in t.value for t in at.toast))
        self.assertTrue(any(b.key == "v3_demo" for b in at.button))                # back to the first screen

    def test_deleting_the_selected_one_falls_back_to_another(self):
        cv.start(self.path, "AAA", 1000, 100)
        cv.start(self.path, "BBB", 500, 20)
        at = self.start()
        at.selectbox(key="v3_pos").set_value("BBB").run()
        at.checkbox(key="v3_confirm|BBB").check().run()
        at.button(key="v3_delete").click().run()
        self.assert_clean(at)
        self.assertEqual([p["name"] for p in self.book()["positions"]], ["AAA"])
        self.assertEqual(at.selectbox(key="v3_pos").value, "AAA")
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "100"))
        self.assertFalse(at.checkbox(key="v3_confirm|AAA").value)                      # the next one is not pre-confirmed

    def test_the_start_form_checks_what_you_type(self):
        at = self.start()
        for name, c, t, why in (("a*b", 1000.0, 100.0, "อักขระพิเศษ"), ("", 1000.0, 100.0, "1–20"), ("OK", 1000.0, None, "ราคา t")):
            at.text_input(key="v3_name|0").set_value(name)
            at.number_input(key="v3_c|0").set_value(c)
            at.number_input(key="v3_t|0").set_value(t)
            self.submit(at).click().run()
            self.assertTrue(any(why in e.value for e in at.error), (name, [e.value for e in at.error]))
            self.assertFalse(self.path.exists(), name)                              # nothing was saved
            self.assertEqual([e.value for e in at.exception], [])
        at.text_input(key="v3_name|0").set_value("ทอง")
        at.number_input(key="v3_c|0").set_value(2500.0)
        at.number_input(key="v3_t|0").set_value(31.5)
        self.submit(at).click().run()
        self.assert_clean(at)
        self.assertEqual((self.pos("ทอง")["c0"], self.pos("ทอง")["t0"]), (2500.0, 31.5))
        self.assertEqual(at.selectbox(key="v3_pos").value, "ทอง")                    # the new one is selected
        self.assertTrue(any("ทอง" in t.value for t in at.toast))
        self.assertEqual(at.text_input(key="v3_name|1").value, "")                  # and the form is empty again
        cv.start(self.path, "Gold", 100, 1)                                          # added by "another session"
        at = self.start()                                                            # a new session: the form counter restarts
        for dup in ("ทอง", "GOLD", "gold"):                                          # the same name, or the same letters in any case
            at.text_input(key="v3_name|0").set_value(dup)
            at.number_input(key="v3_t|0").set_value(10.0)
            self.submit(at).click().run()
            self.assertTrue(any("ชื่อนี้มีอยู่แล้ว" in e.value for e in at.error), dup)
            self.assertEqual([p["name"] for p in self.book()["positions"]], ["ทอง", "Gold"], dup)

    def test_several_positions_are_kept_apart(self):
        cv.start(self.path, "AAA", 1000, 100)
        cv.start(self.path, "BBB", 500, 20)
        at = self.start()
        self.assertEqual(at.selectbox(key="v3_pos").value, "AAA")
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "100"))
        at.selectbox(key="v3_pos").set_value("BBB").run()
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$500.00", "20"))
        self.set_price(at, 25.0)
        self.commit_button(at).click().run()
        self.assertEqual([len(p["rounds"]) for p in self.book()["positions"]], [0, 1])

    # the score and the goal ----------------------------------------------------------------
    def test_the_goal_box_follows_ev_efficiency(self):
        at = self.demo()
        self.assertEqual(len(at.success), 0)
        self.assertIn("ยังไม่ถึง 1.0x", self.goal_box(at).value)
        self.assertEqual((self.metric(at, "ΣE"), self.metric(at, "ΣK"), self.metric(at, "Ev Efficiency")), ("$0.00", "$0.00", "—"))
        self.assertIn("ยังวัดไม่ได้", at.get("progress")[0].proto.text)
        self.commit_button(at).click().run()                                         # the price doubles: the swing pays 2.5× the premium
        self.assert_clean(at)
        e, k = cv.totals(self.pos())
        self.assertEqual((e, k), (306.85, 123.96))
        self.assertEqual((self.metric(at, "ΣE"), self.metric(at, "ΣK")), ("$306.85", "$123.96"))
        self.assertEqual(self.metric(at, "Ev Efficiency"), mult(e / k))
        self.assertIn("ถึงเป้าหมายแล้ว", self.goal_box(at).value)
        self.assertEqual(len(at.success), 1)
        self.assertEqual(at.get("progress")[0].proto.value, 100)                    # the bar is full at 1.0x and above
        for _ in range(2):                                                           # two rounds with no move: premium paid, nothing harvested
            self.set_price(at, 200.0)
            self.commit_button(at).click().run()
        e, k = cv.totals(self.pos())
        self.assertEqual(e, 306.85)
        self.assertLess(e / k, 1.0)                                                  # the score falls back below 1.0x …
        self.assertEqual(len(at.success), 0)
        self.assertIn("ยังไม่ถึง 1.0x", self.goal_box(at).value)
        self.assertEqual(self.metric(at, "Ev Efficiency"), mult(e / k))
        self.assertTrue(0 < at.get("progress")[0].proto.value < 100)                # … and the bar shows how far it is
        self.assertEqual(self.pos()["rounds"][-1]["c_after"], self.pos()["rounds"][0]["c_after"])   # c did not shrink meanwhile

    def test_exactly_one_x_reaches_the_goal_and_less_does_not(self):
        for premium, reached in ((306.85, True), (306.86, False), (350.0, False)):     # E of the demo move is 306.85
            with self.subTest(premium=premium):
                at = self.demo()
                self.widget(at, "number_input", "k").set_value(premium).run()
                self.commit_button(at).click().run()
                self.assert_clean(at)
                e, k = cv.totals(self.pos())
                self.assertEqual(e >= k, reached)
                self.assertEqual(len(at.success), int(reached))
                self.assertIn("ถึงเป้าหมายแล้ว" if reached else "ยังไม่ถึง 1.0x", self.goal_box(at).value)
                self.path.unlink()

    def test_the_goal_is_the_flowcharts_goal_word_for_word(self):
        at = self.demo()
        mmd = (ROOT / "app_v2_minimal.mmd").read_text("utf-8")
        self.assertTrue(plain(self.goal_box(at).value).endswith(goal_of(mmd)), self.goal_box(at).value)

    def test_history_shows_the_newest_rounds_first_and_at_most_twenty(self):
        cv.start(self.path, "LONG", 1000, 100)
        price = 100.0
        for i in range(22):
            price *= 1.8 if i % 2 == 0 else 0.6                                      # big swings: some rounds grow c, some do not
            cv.commit(self.path, "LONG", cv.plan(*cv.current(cv.load(self.path)["positions"][0]), price, 1.0))
        at = self.start()
        self.assert_clean(at)
        rows = cells(unesc(next(m.value for m in at.markdown if "รอบ | เวลา | t → P" in m.value)))
        self.assertEqual([r[0] for r in rows], [str(i) for i in range(22, 2, -1)])
        self.assertTrue(any("แสดง 20 รอบล่าสุดจากทั้งหมด 22 รอบ" in c.value for c in at.caption))
        last = self.pos("LONG")["rounds"][-1]
        self.assertEqual(rows[0][-1], f"${last['c_after']:,.2f}")
        self.assertEqual(rows[0][3], f"+${last['E']:,.2f}")                           # the E column: never negative
        self.assertGreater(last["c_after"], 1000.0)                                  # the capital really grew over the run

    def test_a_pipe_in_a_hand_edited_timestamp_cannot_break_the_history_table(self):
        cv.start(self.path, "EDIT", 1000, 100)
        cv.commit(self.path, "EDIT", cv.plan(1000, 100, 125, 1.0))
        book = self.book()
        book["positions"][0]["rounds"][0]["at"] = "2026|01|01"
        self.path.write_text(json.dumps(book), encoding="utf-8")
        at = self.start()
        self.assert_clean(at)
        rows = cells(next(m.value for m in at.markdown if "รอบ | เวลา | t → P" in m.value))
        self.assertEqual([len(r) for r in rows], [7])
        self.assertIn("2026¦01¦01", rows[0][1])

    # a broken file · isolation ---------------------------------------------------------------
    def test_a_broken_file_is_kept_and_reset_moves_it_aside(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ definitely not json", encoding="utf-8")
        at = self.start()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("ไฟล์ข้อมูลไม่ใช่ JSON" in e.value for e in at.error))
        self.assertEqual(self.path.read_text("utf-8"), "{ definitely not json")        # never overwritten
        self.assertEqual([b for b in at.button if b.key == "v3_demo"], [])               # nothing can be written over it
        at.button(key="v3_reset").click().run()
        self.assert_clean(at)
        kept = list(self.path.parent.glob("app_v3.json.broken-*"))
        self.assertEqual([k.read_text("utf-8") for k in kept], ["{ definitely not json"])  # moved, not deleted
        self.assertFalse(self.path.exists())
        self.assertTrue(any(b.key == "v3_demo" for b in at.button))                      # a fresh book, first screen again

    def test_a_book_from_the_earlier_equation_is_kept_and_explained(self):
        """The first v3 read its Surplus at R (ln). Those rounds cannot be re-read at E — so they are set aside, not converted."""
        self.path.parent.mkdir(parents=True)
        old = json.dumps({"version": 1, "positions": [{"name": "OLD", "c0": 1000, "t0": 100, "rounds": [
            {"t": 100, "p": 125, "c": 1000, "H": 223.14, "K": 123.96, "S": 99.18, "c_after": 1099.18, "hedge_x": 1}]}]})
        self.path.write_text(old, encoding="utf-8")
        at = self.start()
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("สมการรุ่นก่อน" in e.value for e in at.error), [e.value for e in at.error])
        self.assertEqual(self.path.read_text("utf-8"), old)
        at.button(key="v3_reset").click().run()
        self.assert_clean(at)
        self.assertEqual([k.read_text("utf-8") for k in self.path.parent.glob("app_v3.json.broken-*")], [old])
        self.assertTrue(any(b.key == "v3_demo" for b in at.button))

    def test_a_session_touches_no_other_file_of_the_repo(self):
        watched = [ROOT / "trading_data.json", ROOT / "data" / "chain_system_backup.json"]
        other = Path(self._tmp.name) / "trading_data.json"                                 # the old portfolio file, if it were pointed at
        other.write_text(watched[0].read_text("utf-8"), encoding="utf-8")
        before = {p: sha(p) for p in watched + [other]}
        listing = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("*"))
        with mock.patch.dict(os.environ, {"CHAIN_V2_DATA_FILE": str(other)}):
            at = self.demo()
            self.commit_button(at).click().run()
            at.button(key="v3_undo").click().run()
            at.checkbox(key="v3_confirm|DEMO").check().run()
            at.button(key="v3_delete").click().run()
            self.assert_clean(at)
        self.assertEqual({p: sha(p) for p in watched + [other]}, before)
        self.assertEqual(sorted(str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("*")), listing)   # no new file in data/

    def test_the_page_loads_none_of_the_other_modules(self):
        code = (
            "import os, sys\n"
            "from streamlit.testing.v1 import AppTest\n"
            "at = AppTest.from_file(sys.argv[1], default_timeout=60).run()\n"
            "at.button(key='v3_demo').click().run()\n"
            "assert not at.exception, [e.value for e in at.exception]\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in {'engine_v2', 'ledger_v2', 'learn_v2', 'lego_hover_table',\n"
            "                                                            'chain_system', 'flywheels'})\n"
            "print('LOADED', bad)\n")
        out = subprocess.run([sys.executable, "-c", code, str(PAGE)], capture_output=True, text=True, timeout=120, cwd=self._tmp.name,
                             env={**os.environ, "PYTHONPATH": ""})
        self.assertEqual(out.returncode, 0, out.stderr[-800:])
        self.assertIn("LOADED []", out.stdout)

    # housekeeping -----------------------------------------------------------------------------
    def test_the_data_file_is_git_ignored_and_the_page_stays_small(self):
        src = PAGE.read_text("utf-8")
        self.assertIn('ROOT / "data" / "app_v3.json"', src)
        ignored = (ROOT / ".gitignore").read_text("utf-8").split()
        self.assertIn("data/app_v3.json", ignored)
        self.assertIn("data/app_v3.json.*", ignored)                                          # temp and .broken-… files too
        self.assertLessEqual(len(src.splitlines()), 340)
        for heavy in ("pandas", "plotly", "numpy", "components", "dataframe"):
            self.assertNotIn(heavy, src, heavy)


if __name__ == "__main__":
    unittest.main()
