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
VIEWS = ["engine", "ledger", "portfolio", "treasury", "payoff", "learn", "data"]

if HAVE_ST:
    from streamlit.testing.v1 import AppTest
    import engine_v2 as eng
    import ledger_v2 as led


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
        return at.segmented_control(key="v2_view").set_value(view).run()

    def pick_ticker(self, at, name):
        return at.selectbox(key="v2_ticker").set_value(name).run()

    @staticmethod
    def metrics(at):
        return {m.label: m.value for m in at.metric}

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
        self.assertEqual(at.selectbox(key="v2_ticker").value, "RIVN")

    def test_empty_portfolio_then_add_a_ticker(self):
        at = self.start()
        self.assert_clean(at)
        self.assertTrue(any("ยังไม่มี ticker" in i.value for i in at.info))
        self.goto(at, "data")
        self.find(at, "text_input", label="Ticker Symbol").set_value("zzz")
        self.find(at, "button", label="Add Ticker").click().run()
        self.assert_clean(at)
        self.assertEqual([t["ticker"] for t in self.data()["tickers"]], ["ZZZ"])
        self.assertEqual(self.data()["tickers"][0]["current_state"]["fix_c"], 10000.0)

    def test_add_ticker_rejects_bad_symbol(self):
        at = self.start()
        self.goto(at, "data")
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
        at = self.goto(self.pick_ticker(self.start(), "QXO"), "treasury")
        at.selectbox(key="v2_tr_act").set_value("📥 Extract Baseline → Pool CF").run()
        self.assert_clean(at)
        self.find(at, "button", label="📥 Extract Baseline → Pool CF").click().run()
        self.assertTrue(any("ต้องมากกว่า 0" in e.value for e in at.error))

    def test_all_views_work_with_an_empty_portfolio(self):
        at = self.start()
        for view in VIEWS:
            self.assert_clean(self.goto(at, view), view)
        for act in ("📥 Extract Baseline → Pool CF", "🚀 Deploy Pool CF → Baseline", "💸 EV Reserve flow (±)"):
            self.goto(at, "treasury")
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
        at = self.goto(self.start(), "treasury")
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
        self.goto(at, "treasury")
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
        at = self.goto(self.start(), "data")
        with mock.patch.object(eng, "fetch_json_url", return_value=raw):
            self.find(at, "button", label="📥 ดึงจาก URL").click().run()
        self.assertTrue(any("3 tickers" in w.value for w in at.warning))
        self.assertEqual(len(self.data()["tickers"]), 16)                       # nothing written yet
        self.find(at, "button", label="✅ Confirm Import").click().run()
        self.assert_clean(at)
        self.assertEqual(len(self.data()["tickers"]), 3)
        at = self.goto(at, "data")
        self.find(at, "button", label="↩️ Restore").click().run()                # undo
        self.assertEqual(len(self.data()["tickers"]), 16)

    def test_import_rejects_non_portfolio_json(self):
        at = self.goto(self.start(), "data")
        with mock.patch.object(eng, "fetch_json_url", return_value={"hello": "world"}):
            self.find(at, "button", label="📥 ดึงจาก URL").click().run()
        self.assertTrue(any("tickers" in e.value for e in at.error))
        self.assertFalse(any("พร้อม Import" in w.value for w in at.warning))

    def test_wipe_requires_typing_delete(self):
        self.load_sample()
        at = self.goto(self.start(), "data")
        self.find(at, "text_input", label="พิมพ์ DELETE เพื่อยืนยัน").set_value("nope")
        self.find(at, "button", label="ลบทั้งหมด").click().run()
        self.assertEqual(len(self.data()["tickers"]), 16)
        self.find(at, "text_input", label="พิมพ์ DELETE เพื่อยืนยัน").set_value("DELETE")
        self.find(at, "button", label="ลบทั้งหมด").click().run()
        self.assertEqual(self.data()["tickers"], [])

    def test_dollars_in_markdown_are_escaped(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "payoff")
        body = (list(at.success) + list(at.info))[0].value
        self.assertIn("\\$", body)       # "$a … $b" would otherwise render as LaTeX
        self.assertNotIn("$17.50", body.replace("\\$", ""))

    def test_payoff_lines_and_charts(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "payoff")
        at.multiselect(key="v2_po_lines").set_value(["y1", "y2", "y4", "y5", "y8", "y9", "y10", "y11", "y3", "y6", "y7"]).run()
        self.assert_clean(at)
        for chart in ("Net vs Benchmark", "Delta Overlay", "เปรียบเทียบ"):
            self.assert_clean(at.radio(key="v2_po_chart").set_value(chart).run(), chart)

    def test_portfolio_sankey_toggle(self):
        self.load_sample()
        at = self.goto(self.start(), "portfolio")
        self.assert_clean(at.toggle(key="v2_sankey").set_value(True).run())


    # ── modern views: Engine cards · Ledger · Learn · deep links ────────────────────────
    def test_engine_headline_cards_and_the_rae_caption(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "LUNR")
        self.assertEqual([m.label for m in at.metric][4:9], ["Price", "fix_c", "Baseline", "🔥 Ev burn", "Net P&L"])
        s = self.ticker("LUNR")["current_state"]
        m = self.metrics(at)
        self.assertEqual(m["Price"], "$17.50")
        self.assertEqual(m["fix_c"], f"${s['fix_c']:,.0f}")                # whole dollars from $1,000 up …
        self.assertEqual(m["Net P&L"], f"-${abs(s['net_pnl']):,.0f}")      # … so a 5-card row never truncates
        S = led.build(self.ticker("LUNR"))["summary"]
        caps = [c.value.replace("\\$", "$") for c in at.caption]
        self.assertTrue(any(f"Eₙ +{eng.money(S['E'])}" in c for c in caps), caps)

    def test_order_panel_shows_the_lego_lens(self):
        self.load_sample()
        at = self.pick_ticker(self.start(), "LUNR")
        self.find(at, "number_input", prefix="v2|pnew|").set_value(20.0).run()
        code = at.code[0].value
        self.assertIn("+$400.00", code)                       # ΔA = 2,800 × (20/17.5 − 1)
        self.assertIn("ΔE = ΔA − Shannon", code)
        self.assertIn("= $373.", code)                        # Shannon = 2,800 × ln(20/17.5) — the R side

    def test_ledger_shows_the_rae_numbers_and_a_hover_table(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "ledger")
        self.assert_clean(at)
        S = led.build(self.ticker("LUNR"))["summary"]
        m = self.metrics(at)
        self.assertEqual(m["Rₙ อ้างอิง"], eng.money(S["R"]))
        self.assertEqual(m["Aₙ เงินสดสะสม"], eng.money(S["A"]))
        self.assertEqual(m["Eₙ กำไร harvest จริง"], eng.money(S["E"]))
        self.assertGreater(S["E"], 0)
        frames = at.get("iframe")
        self.assertEqual(len(frames), 1)
        doc = frames[0].proto.srcdoc
        self.assertGreater(doc.count('data-tip="'), 9)       # every Chain Round row explains its numbers
        self.assertIn("Eₙ = Aₙ − Rₙ", doc)

    def test_ledger_falls_back_to_a_plain_table(self):
        self.load_sample()
        at = self.goto(self.pick_ticker(self.start(), "LUNR"), "ledger")
        at.toggle(key="ledger_hover").set_value(False).run()
        self.assert_clean(at)
        self.assertEqual(len(at.get("iframe")), 0)
        self.assertTrue(any({"รอบ", "สถานะ"} <= set(df.value.columns) for df in at.dataframe))
        with mock.patch("lego_hover_table.build_hover_table_html", side_effect=RuntimeError("boom")):
            at = self.goto(self.pick_ticker(self.start(), "LUNR"), "ledger")
        self.assert_clean(at)                                  # the failure is a warning, never an exception
        self.assertTrue(any("สร้างตารางแบบชี้เมาส์ไม่ได้" in w.value for w in at.warning))
        self.assertEqual(len(at.get("iframe")), 0)

    def test_ledger_escapes_hostile_text_from_the_data_file(self):
        evil = "<script>alert(1)</script>"
        d = {"tickers": []}
        eng.add_ticker(d, "AAA", 10, 1000)
        eng.apply_round(d, "AAA", eng.plan_round(d["tickers"][0]["current_state"], 11.0, 1.0))
        d["tickers"][0]["rounds"][-1]["action"] = evil
        self.path.write_text(json.dumps(d), encoding="utf-8")
        at = self.goto(self.start(), "ledger")
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("ผิดกติกา" in e.value for e in at.error))       # an unknown action is flagged ❌ …
        doc = at.get("iframe")[0].proto.srcdoc
        self.assertNotIn(evil, doc)                                         # … and never reaches the page unescaped
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", doc)

    def test_ledger_empty_states(self):
        d = {"tickers": []}
        eng.add_ticker(d, "AAA", 10, 1000)                     # a ticker with no rounds at all
        self.path.write_text(json.dumps(d), encoding="utf-8")
        at = self.goto(self.start(), "ledger")
        self.assert_clean(at)
        self.assertTrue(any("ยังไม่มี round" in i.value for i in at.info))
        self.assertEqual(len(at.get("iframe")), 0)

    def test_learn_view_equations_diagrams_and_break_even(self):
        at = self.goto(self.start(), "learn")
        self.assert_clean(at)
        self.assertEqual(len(at.get("graphviz_chart")), 2)
        self.assertGreaterEqual(len(at.latex), 3)
        m = self.metrics(at)
        self.assertEqual(m["เบี้ย 1 รอบ"], f"${1000 * led.hedge_rate():,.2f}")
        self.assertEqual(m["ฐาน R (ln — Shannon ของ Chain)"], "+13.2%")
        self.assertEqual(m["ฐาน A (เงินสดจริง ΔA)"], "+12.4%")
        self.assertEqual(m["ฐาน E (harvest ก้าวเดียว)"], "+58% / -42%")
        at.slider(key="learn_hx").set_value(2.0).run()          # twice the hedge → twice the premium
        self.assertEqual(self.metrics(at)["เบี้ย 1 รอบ"], f"${2000 * led.hedge_rate():,.2f}")
        self.assert_clean(at)

    def test_testing_lab_is_deterministic_and_never_negative(self):
        at = self.goto(self.start(), "learn")
        first = self.metrics(at)
        harvest = float(first["Eₙ กำไร harvest"].replace(",", ""))
        self.assertGreaterEqual(harvest, 0)
        self.assertEqual(self.metrics(self.goto(self.start(), "learn")), first)       # same seed → same numbers
        at.number_input(key="lab_seed").set_value(7).run()
        self.assert_clean(at)
        self.assertGreaterEqual(float(self.metrics(at)["Eₙ กำไร harvest"].replace(",", "")), 0)
        at.number_input(key="lab_seed").set_value(101).run()
        self.assertEqual(self.metrics(at), first)
        at.number_input(key="lab_every").set_value(1000).run()                         # more than the steps → a message, not a crash
        self.assertEqual([e.value for e in at.exception], [])
        self.assertTrue(any("คำนวณไม่ได้" in e.value for e in at.error))

    def test_portfolio_shows_the_harvest_total_and_column(self):
        self.load_sample()
        at = self.goto(self.start(), "portfolio")
        self.assert_clean(at)
        total = sum(led.build(t)["summary"]["E"] for t in self.data()["tickers"])
        self.assertEqual(self.metrics(at)["🧬 Σ Eₙ กำไร harvest จริง"], eng.money(total))
        frame = at.dataframe[0].value
        self.assertIn("Eₙ harvest", frame.columns)
        self.assertIn("fix_c trend", frame.columns)
        self.assertEqual(len(frame), 16)

    def test_data_view_audits_every_ticker(self):
        self.load_sample()
        at = self.goto(self.start(), "data")
        self.assert_clean(at)
        audit = next(df.value for df in at.dataframe if {"Ticker", "สถานะ"} <= set(df.value.columns))
        self.assertEqual(len(audit), 16)
        self.assertFalse(any("❌" in s for s in audit["สถานะ"]))       # the system's own rows never break the rules

    def test_deep_link_opens_the_requested_view_and_ticker(self):
        self.load_sample()
        at = AppTest.from_file(PAGE, default_timeout=60)
        at.query_params["view"], at.query_params["ticker"] = "ledger", "LUNR"
        at.run()
        self.assert_clean(at)
        self.assertEqual((at.segmented_control(key="v2_view").value, at.selectbox(key="v2_ticker").value),
                         ("ledger", "LUNR"))
        self.assertEqual(len(at.get("iframe")), 1)

    def test_bad_deep_link_values_fall_back_to_the_defaults(self):
        self.load_sample()
        at = AppTest.from_file(PAGE, default_timeout=60)
        at.query_params["view"], at.query_params["ticker"] = "nope", "ZZZZ"
        at.run()
        self.assert_clean(at)
        self.assertEqual((at.segmented_control(key="v2_view").value, at.selectbox(key="v2_ticker").value),
                         ("engine", "RIVN"))
        self.assertEqual(dict(at.query_params), {})

    def test_url_follows_the_navigation_and_omits_defaults(self):
        self.load_sample()
        at = self.start()
        self.assertEqual(dict(at.query_params), {})
        self.goto(at, "learn")
        self.assertEqual(dict(at.query_params), {"view": ["learn"]})
        self.pick_ticker(at, "LUNR")
        self.assertEqual(dict(at.query_params), {"view": ["learn"], "ticker": ["LUNR"]})
        self.goto(at, "engine")
        self.assertEqual(dict(at.query_params), {"ticker": ["LUNR"]})

    def test_stale_modules_are_reloaded_instead_of_crashing(self):
        self.load_sample()
        saved = eng.sparklines
        del eng.sparklines                    # a half-deployed process: new page, old module without the new API
        try:
            at = self.start()
            self.assert_clean(at)
            self.assertTrue(hasattr(eng, "sparklines"))     # the page reloaded the module itself
        finally:
            if not hasattr(eng, "sparklines"):
                eng.sparklines = saved


if __name__ == "__main__":
    unittest.main()
