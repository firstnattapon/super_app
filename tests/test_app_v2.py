"""UI smoke tests for pages/app-v2.py (Streamlit AppTest) — skipped when streamlit is not installed."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_ST = importlib.util.find_spec("streamlit") is not None
PAGE = str(ROOT / "pages" / "app-v2.py")
SAMPLE = ROOT / "data" / "chain_system_backup.json"
VIEWS = ["⚡ Engine", "📜 History", "📊 Portfolio", "🏛️ Treasury", "📐 Payoff", "➕ Data"]

if HAVE_ST:
    from streamlit.testing.v1 import AppTest
    import engine_v2 as eng


@unittest.skipUnless(HAVE_ST, "streamlit not installed")
class AppV2Test(unittest.TestCase):
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

    def start(self):
        return AppTest.from_file(PAGE, default_timeout=60).run()

    @staticmethod
    def find(at, kind, prefix=None, label=None):
        for w in getattr(at, kind):
            if prefix is not None and (w.key or "").startswith(prefix):
                return w
            if label is not None and getattr(w, "label", None) == label:
                return w
        raise AssertionError(f"no {kind} with prefix={prefix!r} label={label!r}")

    def goto(self, at, view):
        return at.radio(key="v2_view").set_value(view).run()

    def pick_ticker(self, at, name):
        opt = next(o for o in at.selectbox(key="v2_ticker").options if f" {name} " in o)
        return at.selectbox(key="v2_ticker").set_value(opt).run()

    def assert_clean(self, at, where=""):
        self.assertEqual([e.value for e in at.exception], [], where)
        self.assertEqual([e.value for e in at.error], [], where)

    # tests -----------------------------------------------------------------------------
    def test_every_view_renders_cleanly(self):
        self.load_sample()
        at = self.start()
        self.assert_clean(at, "first load")
        for view in VIEWS:
            self.assert_clean(self.goto(at, view), view)

    def test_default_ticker_is_the_first_active_one(self):
        self.load_sample()
        at = self.start()
        # FSUN (fix_c 0) and NEGG (fix_c 0.01) are closed positions → RIVN is the first active ticker
        self.assertIn("RIVN", at.selectbox(key="v2_ticker").value)

    def test_empty_portfolio_then_add_a_ticker(self):
        at = self.start()
        self.assert_clean(at)
        self.assertTrue(any("ยังไม่มี ticker" in i.value for i in at.info))
        self.goto(at, "➕ Data")
        self.find(at, "text_input", label="Ticker Symbol").set_value("zzz")
        self.find(at, "button", label="Add Ticker").click().run()
        self.assert_clean(at)
        self.assertEqual([t["ticker"] for t in self.data()["tickers"]], ["ZZZ"])
        self.assertEqual(self.data()["tickers"][0]["current_state"]["fix_c"], 10000.0)

    def test_add_ticker_rejects_bad_symbol(self):
        at = self.start()
        self.goto(at, "➕ Data")
        self.find(at, "text_input", label="Ticker Symbol").set_value("<b>x</b>")
        self.find(at, "button", label="Add Ticker").click().run()
        self.assertTrue(any("A-Z" in e.value for e in at.error))
        self.assertFalse(self.path.exists())

    def test_commit_round_end_to_end(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "LUNR")
        before = self.ticker("LUNR")
        state = before["current_state"]
        self.find(at, "number_input", prefix="v2|pnew|").set_value(20.0).run()
        plan = eng.plan_round(state, 20.0, 2.0)
        self.assertIn(f"= ${plan['shannon_profit']:,.2f}", at.code[0].value)
        old_key = self.find(at, "button", prefix="v2|commit|").key
        self.find(at, "button", prefix="v2|commit|").click().run()
        self.assert_clean(at)
        after = self.ticker("LUNR")
        self.assertEqual(len(after["rounds"]), len(before["rounds"]) + 1)
        self.assertEqual(after["rounds"][-1]["hedge_cost"], plan["hedge_cost"])
        self.assertEqual(after["current_state"]["price"], 20.0)
        self.assertAlmostEqual(after["current_state"]["cumulative_ev"],
                               state["cumulative_ev"] + plan["hedge_cost"], places=2)
        self.assertTrue(any("Committed LUNR" in t.value for t in at.toast))
        with self.assertRaises(KeyError):   # a stale click (double click / second tab) has no button to land on
            at.button(key=old_key)

    def test_click_on_an_outdated_plan_commits_nothing(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "QXO")
        self.find(at, "number_input", prefix="v2|pnew|").set_value(30.0).run()
        stale = self.find(at, "button", prefix="v2|commit|")
        other = eng.Store(self.path)                  # "another session" saves between render and click
        other.update(lambda d: eng.add_to_pool(d, 100))
        other.update(lambda d: eng.deploy(d, "QXO", 50))
        expected = self.ticker("QXO")                 # exactly what the other session left on disk
        stale.click().run()
        self.assert_clean(at)
        self.assertEqual(self.ticker("QXO"), expected)                    # nothing committed on top of a changed state
        self.assertEqual(self.ticker("QXO")["current_state"]["price"], 25.0)

    def test_yahoo_button_fills_p_new_or_reports_failure(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "QXO")
        with mock.patch.object(eng, "fetch_price", return_value=(31.25, "Yahoo (QXO)")):
            self.find(at, "button", prefix="v2|yf|").click().run()
        self.assertEqual(self.find(at, "number_input", prefix="v2|pnew|").value, 31.25)
        self.assertTrue(any("31.25" in c.value for c in at.caption))
        at = self.pick_ticker(self.start(), "QXO")
        with mock.patch.object(eng, "fetch_price", return_value=(0.0, "error: down")):
            self.find(at, "button", prefix="v2|yf|").click().run()
        self.assertEqual(self.find(at, "number_input", prefix="v2|pnew|").value, 25.0)
        self.assertTrue(any("error: down" in c.value for c in at.caption))
        self.assert_clean(at)

    def test_extract_from_an_empty_baseline_is_rejected_cleanly(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "QXO"), "🏛️ Treasury")
        at.selectbox(key="v2_tr_act").set_value("📥 Extract Baseline → Pool CF").run()
        self.assert_clean(at)
        self.find(at, "button", label="📥 Extract Baseline → Pool CF").click().run()
        self.assertTrue(any("ต้องมากกว่า 0" in e.value for e in at.error))

    def test_all_views_work_with_an_empty_portfolio(self):
        at = self.start()
        for view in VIEWS:
            self.assert_clean(self.goto(at, view), view)
        for act in ("📥 Extract Baseline → Pool CF", "🚀 Deploy Pool CF → Baseline", "💸 EV Reserve flow (±)"):
            self.goto(at, "🏛️ Treasury")
            self.assert_clean(at.selectbox(key="v2_tr_act").set_value(act).run(), act)

    def test_override_changes_what_is_committed(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "LUNR")
        self.find(at, "number_input", prefix="v2|pnew|").set_value(20.0).run()
        self.find(at, "number_input", prefix="v2|oh|").set_value(100.0).run()
        self.assertIn("− $100.00", at.code[0].value)
        self.find(at, "button", prefix="v2|commit|").click().run()
        last = self.ticker("LUNR")["rounds"][-1]
        self.assertEqual((last["hedge_cost"], last["ev_change"]), (100.0, 100.0))
        self.assertEqual(last["surplus"], round(last["shannon_profit"] - 100.0, 2))
        self.assertEqual(last["c_after"], round(last["c_before"] + last["surplus"], 2))

    def test_invalid_state_never_crashes_the_engine_view(self):
        self.path.write_text(json.dumps({"tickers": [
            {"ticker": "BAD", "current_state": {"price": "x", "fix_c": None}, "rounds": ["junk", {"p_new": "?"}]},
            {"ticker": "ODD", "Final": "3, 100, 0"}]}), encoding="utf-8")
        at = self.start()
        self.assert_clean(at)
        for view in VIEWS:
            self.assert_clean(self.goto(at, view), view)

    def test_treasury_funding_and_validation(self):
        self.load_sample()
        at = self.goto(self.start(), "🏛️ Treasury")
        pool = self.data()["global_pool_cf"]
        self.find(at, "button", label="💰 Funding → Pool CF").click().run()               # amount 0 → rejected
        self.assertTrue(any("ต้องมากกว่า 0" in e.value for e in at.error))
        self.assertEqual(self.data()["global_pool_cf"], pool)
        self.find(at, "number_input", label="Amount ($)").set_value(500.0)
        self.find(at, "button", label="💰 Funding → Pool CF").click().run()
        self.assert_clean(at)
        self.assertEqual(self.data()["global_pool_cf"], round(pool + 500, 2))
        self.assertEqual(self.data()["treasury_history"][-1]["category"], "Funding")

    def test_deploy_moves_pool_to_baseline(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "QXO")
        self.goto(at, "🏛️ Treasury")
        at.selectbox(key="v2_tr_act").set_value("🚀 Deploy Pool CF → Baseline").run()
        self.assertIn("QXO", at.selectbox(key="v2_tr_tk").value)
        pool, base = self.data()["global_pool_cf"], self.ticker("QXO")["current_state"]["baseline"]
        self.find(at, "number_input", label="Amount ($)").set_value(1000.0)
        self.find(at, "button", label="🚀 Deploy Pool CF → Baseline").click().run()
        self.assert_clean(at)
        self.assertEqual(self.data()["global_pool_cf"], round(pool - 1000, 2))
        self.assertEqual(self.ticker("QXO")["current_state"]["baseline"], round(base + 1000, 2))
        self.assertEqual(self.ticker("QXO")["rounds"][-1]["action"], "Deploy to Baseline")

    def test_corrupt_file_shows_recovery_and_never_overwrites(self):
        self.path.write_text("{ definitely not json", encoding="utf-8")
        at = self.start()
        self.assertTrue(any("อ่านไฟล์ข้อมูลไม่ได้" in e.value for e in at.error))
        self.assertEqual(self.path.read_text("utf-8"), "{ definitely not json")
        self.find(at, "button", label="🆕 เริ่มพอร์ตว่าง").click().run()
        self.assert_clean(at)
        self.assertEqual(self.data()["tickers"], [])

    def test_legacy_list_file_is_usable(self):
        shutil.copy(ROOT / "trading_data.json", self.path)      # v1-era flat list with "Final" strings
        at = self.start()
        self.assert_clean(at)
        self.assertEqual(len(at.selectbox(key="v2_ticker").options), 16)
        self.assertEqual(json.loads(self.path.read_text("utf-8"))[0]["ticker"], "FFWM")   # reading never rewrites it

    def test_import_from_url_needs_confirmation_and_is_undoable(self):
        self.load_sample()
        raw = json.loads(SAMPLE.read_text("utf-8"))
        raw["tickers"] = raw["tickers"][:3]
        at = self.goto(self.start(), "➕ Data")
        with mock.patch.object(eng, "fetch_json_url", return_value=raw):
            self.find(at, "button", label="📥 ดึงจาก URL").click().run()
        self.assertTrue(any("3 tickers" in w.value for w in at.warning))
        self.assertEqual(len(self.data()["tickers"]), 16)                       # nothing written yet
        self.find(at, "button", label="✅ Confirm Import").click().run()
        self.assert_clean(at)
        self.assertEqual(len(self.data()["tickers"]), 3)
        at = self.goto(at, "➕ Data")
        self.find(at, "button", label="↩️ Restore").click().run()                # undo
        self.assertEqual(len(self.data()["tickers"]), 16)

    def test_import_rejects_non_portfolio_json(self):
        at = self.goto(self.start(), "➕ Data")
        with mock.patch.object(eng, "fetch_json_url", return_value={"hello": "world"}):
            self.find(at, "button", label="📥 ดึงจาก URL").click().run()
        self.assertTrue(any("tickers" in e.value for e in at.error))
        self.assertFalse(any("พร้อม Import" in w.value for w in at.warning))

    def test_wipe_requires_typing_delete(self):
        self.load_sample()
        at = self.goto(self.start(), "➕ Data")
        self.find(at, "text_input", label="พิมพ์ DELETE เพื่อยืนยัน").set_value("nope")
        self.find(at, "button", label="ลบทั้งหมด").click().run()
        self.assertEqual(len(self.data()["tickers"]), 16)
        self.find(at, "text_input", label="พิมพ์ DELETE เพื่อยืนยัน").set_value("DELETE")
        self.find(at, "button", label="ลบทั้งหมด").click().run()
        self.assertEqual(self.data()["tickers"], [])

    def test_dollars_in_markdown_are_escaped(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "📐 Payoff")
        body = (list(at.success) + list(at.info))[0].value
        self.assertIn("\\$", body)       # "$a … $b" would otherwise render as LaTeX
        self.assertNotIn("$17.50", body.replace("\\$", ""))

    def test_payoff_lines_and_charts(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "📐 Payoff")
        at.multiselect(key="v2_po_lines").set_value(["y1", "y2", "y4", "y5", "y8", "y9", "y10", "y11", "y3", "y6", "y7"]).run()
        self.assert_clean(at)
        for chart in ("Net vs Benchmark", "Delta Overlay", "เปรียบเทียบ"):
            self.assert_clean(at.radio(key="v2_po_chart").set_value(chart).run(), chart)

    def test_portfolio_sankey_toggle(self):
        self.load_sample()
        at = self.goto(self.start(), "📊 Portfolio")
        self.assert_clean(at.toggle(key="v2_sankey").set_value(True).run())


if __name__ == "__main__":
    unittest.main()
