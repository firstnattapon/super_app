"""pages/app-v3.py — the minimal flowchart as a page (Streamlit AppTest) — skipped when streamlit is not installed."""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_ST = importlib.util.find_spec("streamlit") is not None
PAGE = ROOT / "pages" / "app-v3.py"
SAMPLE = ROOT / "data" / "chain_system_backup.json"

if HAVE_ST:
    from streamlit.testing.v1 import AppTest
    import engine_v2 as eng
    import ledger_v2 as led


def text_of_goal(mmd: str) -> str:
    label = re.search(r'^\s*GOAL\("(.*)"\)\s*$', mmd, re.M).group(1)
    return re.sub(r"</?b>", "", label.replace("<br/>", "\n")).strip()


def plain(md: str) -> str:
    return re.sub(r"\s*\n\s*", "\n", md.replace("**", "")).strip()


@unittest.skipUnless(HAVE_ST, "streamlit not installed")
class AppV3Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "trading_data.json"
        patcher = mock.patch.dict(os.environ, {"CHAIN_V2_DATA_FILE": str(self.path)})
        patcher.start()
        self.addCleanup(patcher.stop)

    # helpers ---------------------------------------------------------------------------
    def load_sample(self):
        shutil.copy(SAMPLE, self.path)

    def data(self):
        return json.loads(self.path.read_text("utf-8"))

    def ticker(self, name):
        return next(t for t in self.data()["tickers"] if t["ticker"] == name)

    def start(self, ticker=None):
        at = AppTest.from_file(str(PAGE), default_timeout=60).run()
        return at.selectbox(key="v3_ticker").set_value(ticker).run() if ticker else at

    @staticmethod
    def widget(at, kind, part):
        return next(w for w in getattr(at, kind) if f"|{part}|" in (w.key or ""))

    def commit_button(self, at):
        return next(b for b in at.button if (b.key or "").startswith("v3|commit|"))

    def assert_clean(self, at):
        self.assertEqual([e.value for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])

    # tests -----------------------------------------------------------------------------
    def test_the_five_steps_show_the_engines_numbers(self):
        self.load_sample()
        at = self.start()
        self.assert_clean(at)
        self.assertEqual(at.selectbox(key="v3_ticker").value, "RIVN")            # first ticker with fix_c ≥ 1
        plan = eng.plan_round(self.ticker("RIVN")["current_state"], 15.3, 2.0)
        code = at.code[0].value
        for line in ("① H = c · ln(P / t)", "② K = 12.4% · c · Hedge× = 12.4% × 2,300.00 × 2", "③ Surplus = H − K",
                     "④ c′ = c", "⑤ t′ = P"):
            self.assertIn(line, code)
        self.assertIn(f"= {eng.money(plan['hedge_cost'])}", code)
        self.assertIn("✗ ไม่เป็นบวก", code)                                        # no move ⇒ the premium is not covered
        self.assertEqual([m.label for m in at.metric][:2], ["c = fix_c", "t = ราคารอบก่อน"])

    def test_commit_writes_exactly_what_the_steps_showed(self):
        self.load_sample()
        at = self.start("LUNR")
        self.widget(at, "number_input", "pnew").set_value(25.0).run()
        plan = eng.plan_round(self.ticker("LUNR")["current_state"], 25.0, 2.0)
        self.assertGreater(plan["surplus"], 0)
        self.assertIn("✓ เป็นบวก", at.code[0].value)
        before, old_key = len(self.ticker("LUNR")["rounds"]), self.commit_button(at).key
        self.commit_button(at).click().run()
        self.assert_clean(at)
        t = self.ticker("LUNR")
        last = t["rounds"][-1]
        self.assertEqual(len(t["rounds"]), before + 1)
        self.assertEqual(last["action"], "Chain Round")
        for key in ("shannon_profit", "hedge_cost", "surplus", "c_after", "b_after", "p_new"):
            self.assertEqual(last[key], plan[key], key)
        self.assertEqual((t["current_state"]["price"], t["current_state"]["fix_c"]), (25.0, plan["c_after"]))
        self.assertGreater(plan["c_after"], plan["c_before"])                      # grew from the surplus, only
        self.assertTrue(any("LUNR" in x.value for x in at.toast))
        with self.assertRaises(KeyError):                                           # a stale click has no button to land on
            at.button(key=old_key)
        self.assertEqual(self.widget(at, "number_input", "pnew").value, 25.0)      # ⑤ re-centred: the next round starts at P

    def test_no_surplus_never_shrinks_fix_c(self):
        self.load_sample()
        at = self.start("LUNR")
        self.widget(at, "number_input", "pnew").set_value(10.0).run()              # a big drop
        plan = eng.plan_round(self.ticker("LUNR")["current_state"], 10.0, 2.0)
        self.assertLess(plan["surplus"], 0)
        self.assertEqual(plan["c_after"], plan["c_before"])
        self.assertIn("ไม่ลด", at.code[0].value)
        self.commit_button(at).click().run()
        t = self.ticker("LUNR")
        self.assertEqual(t["current_state"]["fix_c"], plan["c_before"])
        self.assertEqual(t["rounds"][-1]["hedge_cost"], plan["hedge_cost"])        # the premium is still paid (Ev burn)

    def test_a_real_premium_replaces_the_formula(self):
        self.load_sample()
        at = self.start("LUNR")
        self.widget(at, "number_input", "pnew").set_value(25.0).run()
        self.widget(at, "number_input", "k").set_value(100.0).run()
        code = at.code[0].value
        self.assertIn("K = เบี้ยจริงที่ใส่", code)
        self.assertIn("= $100.00", code)
        ev_before = self.ticker("LUNR")["current_state"]["cumulative_ev"]
        self.commit_button(at).click().run()
        t = self.ticker("LUNR")
        self.assertEqual((t["rounds"][-1]["hedge_cost"], t["rounds"][-1]["ev_change"]), (100.0, 100.0))
        self.assertAlmostEqual(t["current_state"]["cumulative_ev"], ev_before + 100.0, places=2)

    def test_click_on_an_outdated_plan_commits_nothing(self):
        self.load_sample()
        at = self.start("QXO")
        self.widget(at, "number_input", "pnew").set_value(30.0).run()
        stale = self.commit_button(at)
        other = eng.Store(self.path)                       # "another session" saves between render and click
        other.update(lambda d: eng.add_to_pool(d, 100))
        other.update(lambda d: eng.deploy(d, "QXO", 50))
        expected = self.ticker("QXO")
        stale.click().run()
        self.assert_clean(at)
        self.assertEqual(self.ticker("QXO"), expected)

    def test_a_ticker_without_a_position_cannot_be_committed(self):
        self.load_sample()
        at = self.start("FSUN")                            # closed position: fix_c = 0
        self.assert_clean(at)
        self.assertTrue(any("ไม่มีตำแหน่ง" in i.value for i in at.info))
        self.assertEqual([b for b in at.button if (b.key or "").startswith("v3|commit|")], [])
        self.assertEqual(len(at.code), 0)

    def test_ev_efficiency_is_sum_shannon_over_ev_burn(self):
        self.load_sample()
        at = self.start()
        seen = set()
        for t in self.data()["tickers"]:
            chain = [r for r in t["rounds"] if r["action"] == "Chain Round"]
            h, k = sum(r["shannon_profit"] for r in chain), t["current_state"]["cumulative_ev"]      # ΣH and Ev burn
            at = at.selectbox(key="v3_ticker").set_value(t["ticker"]).run()
            self.assert_clean(at)
            want = "—" if k <= 0 else f"{h / k:.2f}x"
            self.assertEqual(next(m.value for m in at.metric if m.label.startswith("วัดผล")), want, t["ticker"])
            seen.add(want == "—")
            if k > 0:
                self.assertEqual(bool(at.success), h / k >= 1.0, t["ticker"])      # the goal box turns green at ≥ 1.0x
        self.assertEqual(seen, {True, False})                                       # the sample covers both cases

    def test_the_goal_is_the_flowcharts_goal_word_for_word(self):
        self.load_sample()
        at = self.start()
        box = (list(at.info) + list(at.success))[-1].value
        self.assertTrue(plain(box).endswith(text_of_goal((ROOT / "app_v2_minimal.mmd").read_text("utf-8"))), box)

    def test_goal_box_turns_green_when_harvest_beats_the_premiums(self):
        self.load_sample()
        at = self.start("LUNR")                                                      # ΣH 616.60 vs Ev burn 1,643.86
        self.assertEqual(len(at.success), 0)
        self.assertTrue(any("ยังไม่ถึง" in i.value for i in at.info))
        self.widget(at, "number_input", "pnew").set_value(35.0).run()                # a doubling: H ≈ 1,941 vs K ≈ 694
        self.commit_button(at).click().run()
        self.assert_clean(at)
        self.assertTrue(any("ถึงเป้าหมายแล้ว" in s.value for s in at.success))
        h = sum(r["shannon_profit"] for r in self.ticker("LUNR")["rounds"] if r["action"] == "Chain Round")
        self.assertGreaterEqual(h / self.ticker("LUNR")["current_state"]["cumulative_ev"], 1.0)

    def test_rounds_written_here_pass_the_v2_integrity_audit(self):
        self.load_sample()
        at = self.start("LUNR")
        for price in (25.0, 24.0, 31.0):
            self.widget(at, "number_input", "pnew").set_value(price).run()
            self.commit_button(at).click().run()
            self.assert_clean(at)
        report, ok = led.integrity(self.ticker("LUNR"))
        self.assertTrue(ok, [r for r in report if r["ผ่าน"] == led.FAIL])
        v2 = AppTest.from_file(str(ROOT / "pages" / "app-v2.py"), default_timeout=60).run()   # and v2 reads the same file
        self.assertEqual([e.value for e in v2.exception], [])

    def test_empty_portfolio_and_a_corrupt_file_are_handled(self):
        at = self.start()
        self.assert_clean(at)
        self.assertTrue(any("app-v2" in i.value for i in at.info))
        self.path.write_text("{ definitely not json", encoding="utf-8")
        at = self.start()
        self.assertTrue(any("อ่านไฟล์ข้อมูลไม่ได้" in e.value for e in at.error))
        self.assertEqual(self.path.read_text("utf-8"), "{ definitely not json")            # never overwritten

    def test_the_page_stays_minimal(self):
        src = PAGE.read_text("utf-8")
        self.assertLessEqual(len(src.splitlines()), 150)
        for heavy in ("ledger_v2", "learn_v2", "lego_hover_table", "plotly", "pandas", "components"):
            self.assertNotIn(heavy, src, heavy)                                           # only engine_v2 + streamlit


if __name__ == "__main__":
    unittest.main()
