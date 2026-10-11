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
SYMBOLS = {"c", "t", "P", "Hedge ×", "H", "K", "S", "c′", "t′", "P*", "ΣH", "ΣK", "Ev Efficiency"}   # everything the page shows

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
        ex = cv.plan(1000, 100, 125, 1.0)                                         # the example is computed, not typed in
        for text in ("แก่นแท้", "+$223.14", "−$123.96", "+$99.18", "$1,099.18", "ไม่ลด"):
            self.assertIn(text, key)
        self.assertEqual((ex["H"], ex["K"], ex["S"], ex["c_after"]), (223.14, 123.96, 99.18, 1099.18))
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
        text = unesc("\n".join(m.value for m in at.markdown))
        self.assertIn(f"K = {cv.PREMIUM_RATE:.1%} · c · Hedge×", text)           # 12.4%
        for formula in ("H = c · ln(P / t)", "S = H − K", "c′ = c + max(0, S)", "t′ = P", "P* = t · e^(K / c)", "ΣH ÷ ΣK"):
            self.assertIn(formula, text)
        self.assertIn(f"ประมาณ {cv.breakeven_price(1000, 100) / 100 - 1:+.1%} เมื่อ Hedge× 1", text)       # +13.2%

    # one round --------------------------------------------------------------------------
    def test_demo_shows_the_worked_example_and_saves_no_round(self):
        at = self.demo()
        self.assert_clean(at)
        self.assertEqual((self.pos()["c0"], self.pos()["t0"], self.pos()["rounds"]), (1000.0, 100.0, []))
        self.assertEqual(at.selectbox(key="v3_pos").value, "DEMO")
        self.assertTrue(any("DEMO" in t.value for t in at.toast))
        self.assertEqual(self.widget(at, "number_input", "p").value, 125.0)       # the example price
        table = self.steps(at)
        for text in ("ln(125 / 100)", "+$223.14", "−$123.96", "+$99.18", "$1,000.00 → $1,099.18", "✓ เหลือ", "100 → 125",
                     "12.4% × 1,000.00 × 1"):
            self.assertIn(text, table)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "100"))
        hint = next(unesc(c.value) for c in at.caption if "ราคาคุ้มทุน" in c.value)
        self.assertIn("113.20", hint)
        self.assertIn("+13.2%", hint)
        self.assertTrue(any("ทุนโตเป็น $1,099.18" in unesc(m.value) for m in at.markdown))

    def test_commit_writes_exactly_what_the_table_showed(self):
        at = self.demo()
        want = cv.plan(1000, 100, 125, 1.0)
        old_key = self.commit_button(at).key
        self.commit_button(at).click().run()
        self.assert_clean(at)
        rounds = self.pos()["rounds"]
        self.assertEqual(len(rounds), 1)
        for k in ("t", "p", "c", "H", "K", "S", "c_after", "hedge_x", "real_k"):
            self.assertEqual(rounds[0][k], want[k], k)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,099.18", "125"))   # ④ grew · ⑤ re-centred
        self.assertEqual(self.widget(at, "number_input", "p").value, 125.0)       # the next round starts at P
        self.assertIsNone(self.widget(at, "number_input", "k").value)             # a real premium is per round
        self.assertTrue(any("DEMO" in t.value and "$1,099.18" in t.value for t in at.toast))
        with self.assertRaises(KeyError):                                          # a stale click has no button to land on
            at.button(key=old_key)
        self.assertTrue(any(e.label == "🧾 ประวัติ (1 รอบ)" for e in at.expander))

    def test_a_drop_never_shrinks_c_but_the_premium_is_still_booked(self):
        at = self.demo()
        self.set_price(at, 90.0)
        table = self.steps(at)
        self.assertIn("✗ ไม่เหลือ", table)
        self.assertIn("**−$105.36**", table)                                       # H = 1,000 · ln(0.9) is a loss
        self.assertIn("**−$229.32** ✗", table)                                     # S = H − K
        self.assertIn("c′ = c — ไม่มีส่วนเหลือ", table)
        self.assertIn("$1,000.00 → $1,000.00", table)
        self.assertTrue(any("ทุนเท่าเดิม $1,000.00 (ไม่ลด)" in unesc(m.value) for m in at.markdown))
        self.commit_button(at).click().run()
        last = self.pos()["rounds"][-1]
        self.assertEqual((last["c"], last["c_after"], last["K"], last["p"]), (1000.0, 1000.0, 123.96, 90.0))
        self.assertLess(last["H"], 0)
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,000.00", "90"))

    def test_hedge_multiplies_the_premium_and_stays_for_the_next_round(self):
        at = self.demo()
        self.widget(at, "number_input", "h").set_value(2.0).run()
        self.assertIn("12.4% × 1,000.00 × 2", self.steps(at))
        self.assertIn(f"{cv.breakeven_price(1000, 100, 2.0):,.2f}", next(c.value for c in at.caption if "ราคาคุ้มทุน" in c.value))
        self.commit_button(at).click().run()
        self.assertEqual((self.pos()["rounds"][-1]["K"], self.pos()["rounds"][-1]["hedge_x"]), (247.92, 2.0))
        self.assertEqual(self.widget(at, "number_input", "h").value, 2.0)         # you do not retype it every round

    def test_a_real_premium_replaces_the_formula(self):
        at = self.demo()
        self.widget(at, "number_input", "k").set_value(0.0).run()                  # K = 0: a round with no insurance bought
        self.assert_clean(at)
        self.assertIn("| **$0.00** |", self.steps(at))                              # a cost of nothing is not "−$0.00"
        self.assertIn("+$223.14", self.steps(at))
        self.widget(at, "number_input", "k").set_value(50.0).run()
        table = self.steps(at)
        self.assertIn("ค่าประกันที่จ่ายจริง", table)
        self.assertNotIn("12.4% × 1,000.00", table)                                # the formula is no longer used
        self.assertIn("−$50.00", table)
        self.assertIn("+$173.14", table)                                           # 223.14 − 50
        self.commit_button(at).click().run()
        last = self.pos()["rounds"][-1]
        self.assertEqual((last["K"], last["S"], last["real_k"]), (50.0, 173.14, True))

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
        self.assertIn("ln(0.006 / 0.0042)", self.steps(at))
        self.assertIn(f"{cv.breakeven_price(500, 0.0042):,.4f}", next(c.value for c in at.caption if "ราคาคุ้มทุน" in c.value))

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
        self.set_price(at, 140.0)
        self.commit_button(at).click().run()
        self.assertEqual(len(self.pos()["rounds"]), 2)
        at.button(key="v3_undo").click().run()
        self.assert_clean(at)
        self.assertEqual([r["p"] for r in self.pos()["rounds"]], [125.0])
        self.assertEqual((self.metric(at, "c ·"), self.metric(at, "t ·")), ("$1,099.18", "125"))   # back where it was
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
        self.assertEqual((self.metric(at, "ΣH"), self.metric(at, "ΣK"), self.metric(at, "Ev Efficiency")), ("$0.00", "$0.00", "—"))
        self.assertIn("ยังวัดไม่ได้", at.get("progress")[0].proto.text)
        self.set_price(at, 200.0)                                                    # a doubling pays for the insurance many times
        self.commit_button(at).click().run()
        self.assert_clean(at)
        h, k = cv.totals(self.pos())
        self.assertGreaterEqual(h / k, 1.0)
        self.assertEqual((self.metric(at, "ΣH"), self.metric(at, "ΣK")), (f"${h:,.2f}", f"${k:,.2f}"))
        self.assertEqual(self.metric(at, "Ev Efficiency"), mult(h / k))
        self.assertIn("ถึงเป้าหมายแล้ว", self.goal_box(at).value)
        self.assertEqual(len(at.success), 1)
        self.assertEqual(at.get("progress")[0].proto.value, 100)                    # the bar is full at 1.0x and above
        self.set_price(at, 100.0)                                                    # then a big drop: the score can fall back
        self.commit_button(at).click().run()
        h, k = cv.totals(self.pos())
        self.assertLess(h / k, 1.0)
        self.assertEqual(len(at.success), 0)
        self.assertIn("ยังไม่ถึง 1.0x", self.goal_box(at).value)
        self.assertEqual(at.get("progress")[0].proto.value, 0)                      # never below empty (a negative score)
        self.assertEqual(self.metric(at, "Ev Efficiency"), mult(h / k))
        self.assertTrue(self.metric(at, "Ev Efficiency").startswith("−"))

    def test_exactly_one_x_reaches_the_goal_and_less_does_not(self):
        for premium, reached in ((223.14, True), (223.15, False), (250.0, False)):     # H of the demo move is 223.14
            with self.subTest(premium=premium):
                at = self.demo()
                self.widget(at, "number_input", "k").set_value(premium).run()
                self.commit_button(at).click().run()
                self.assert_clean(at)
                h, k = cv.totals(self.pos())
                self.assertEqual(h >= k, reached)
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
            price *= 1.05 if i % 2 == 0 else 0.97
            cv.commit(self.path, "LONG", cv.plan(*cv.current(cv.load(self.path)["positions"][0]), price, 1.0))
        at = self.start()
        self.assert_clean(at)
        rows = cells(unesc(next(m.value for m in at.markdown if "รอบ | เวลา | t → P" in m.value)))
        self.assertEqual([r[0] for r in rows], [str(i) for i in range(22, 2, -1)])
        self.assertTrue(any("แสดง 20 รอบล่าสุดจากทั้งหมด 22 รอบ" in c.value for c in at.caption))
        self.assertEqual(rows[0][-1], f"${self.pos('LONG')['rounds'][-1]['c_after']:,.2f}")

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
        self.assertLessEqual(len(src.splitlines()), 300)
        for heavy in ("pandas", "plotly", "numpy", "components", "dataframe"):
            self.assertNotIn(heavy, src, heavy)


if __name__ == "__main__":
    unittest.main()
