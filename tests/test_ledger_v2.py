"""Unit tests for ledger_v2 — the LEGO-style R·A·E ledger (stdlib only)."""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine_v2 as eng  # noqa: E402
import ledger_v2 as led  # noqa: E402


def chain_round(p_old, p_new, c0, c1=None, b0=0.0, rid=1, **extra):
    c1 = c0 if c1 is None else c1
    row = {"action": "Chain Round", "p_old": p_old, "p_new": p_new, "c_before": c0, "c_after": c1,
           "b_before": b0, "b_after": round(b0 + c0 * math.log(p_new / p_old), 2), "round_id": rid,
           "shannon_profit": round(c0 * math.log(p_new / p_old), 2), "hedge_cost": 0.0,
           "surplus": round(c0 * math.log(p_new / p_old), 2), "scale_up": c1 - c0, "hedge_ratio": 0.0}
    row.update(extra)
    return row


def ticker(*rounds):
    return {"ticker": "T", "rounds": list(rounds)}


def engine_made(prices=(12, 9, 15, 15, 20), hedge_x=1.0):
    """A ticker produced by the app's own operations (plan_round → apply_round, plus Extract/Deploy)."""
    d = eng._empty()
    eng.add_ticker(d, "AAA", 10, 1000)
    for i, p in enumerate(prices):
        t = d["tickers"][0]
        eng.apply_round(d, "AAA", eng.plan_round(t["current_state"], p, hedge_x))
        if i == 2:
            eng.add_to_pool(d, 500)
            eng.deploy(d, "AAA", 200)
            eng.extract_baseline(d, "AAA", 50)
    return d["tickers"][0]


class TestLedger(unittest.TestCase):
    def test_round_trip_matches_the_rebalancing_101_example(self):
        """Fix_c = 1,500: 100 → 110 → 100 (numbers published in streamlit_lego's Rebalancing 101)."""
        rows = led.build(ticker(chain_round(100, 110, 1500, rid=1), chain_round(110, 100, 1500, b0=142.97, rid=2)))["rows"]
        r1, r2 = rows
        self.assertAlmostEqual(r1["dA"], 150.0, places=9)
        self.assertAlmostEqual(r1["R"], 1500 * math.log(1.1), places=9)               # 142.965
        self.assertAlmostEqual(r1["E"], 150 - 1500 * math.log(1.1), places=9)         # 7.035
        self.assertAlmostEqual(r2["dA"], 1500 * (100 / 110 - 1), places=9)            # −136.36
        self.assertAlmostEqual(r2["A"], 13.636363636, places=6)                        # 13.64 cash left over
        self.assertAlmostEqual(r2["R"], 0.0, places=9)                                 # price is back → R = 0
        self.assertAlmostEqual(r2["E"], r2["A"], places=9)                             # E = A = harvest
        self.assertEqual(led.build(ticker(chain_round(100, 110, 1500)))["p0"], 100)    # P₀ = first P_acted

    def test_E_is_never_negative_and_equals_the_closed_form(self):
        """pass-all ledger: E = c · Σ (x − 1 − ln x), x = Pᵢ / Pᵢ₋₁ ≥ 0 for every path."""
        for seed in range(1, 26):
            prices = led.simulate_prices(80.0, 0.06, 0.002, 150, seed)
            sim = led.simulate_chain(80.0, prices, 1500.0, 0.0, 1, no_hedge=True, no_surplus=True)
            path = [80.0, *prices]
            want = 1500 * sum((b / a - 1) - math.log(b / a) for a, b in zip(path, path[1:]))
            rows = sim["ledger"]["rows"]
            self.assertTrue(all(x["E"] >= -1e-9 for x in rows), seed)
            self.assertAlmostEqual(rows[-1]["E"], want, delta=0.01 + 1e-5 * want, msg=f"seed {seed}")
            self.assertAlmostEqual(rows[-1]["A"], 1500 * sum(b / a - 1 for a, b in zip(path, path[1:])),
                                   delta=0.05, msg=f"seed {seed}")

    def test_rollover_beta_form_equals_the_sum_form_when_fix_c_changes(self):
        """β += (Fix_c − Fix_c_new)·ln(P_acted/P₀) keeps R continuous — LEGO's cheat-sheet 2."""
        worst = 0.0
        for seed in range(1, 31):
            prices = led.simulate_prices(50.0, 0.05, 0.0, 200, seed)
            sim = led.simulate_chain(50.0, prices, 2000.0, 1.0, 10)
            self.assertGreater(sim["growth_rounds"] + 1, 0)
            for x in sim["ledger"]["rows"]:
                if x["R_beta"] is not None:
                    worst = max(worst, abs(x["R_beta"] - x["R"]))
        self.assertLess(worst, 1e-6)
        # …and it really did exercise a changing Fix_c
        sim = led.simulate_chain(50.0, led.simulate_prices(50.0, 0.05, 0.01, 200, 3), 2000.0, 0.0, 10, no_hedge=True)
        self.assertGreater(sim["final_fix_c"], 2000.0)
        self.assertTrue(any(abs(x["beta_after"]) > 1e-9 for x in sim["ledger"]["rows"]))

    def test_chain_baseline_follows_the_reference_line_until_cash_moves(self):
        """The Chain baseline b is the LEGO reference R (same rollover); Extract/Deploy are cash moves, not R."""
        t = engine_made(prices=(12, 9, 15, 15, 20), hedge_x=0.0)
        rows = led.build(t)["rows"]
        acts = [x for x in rows if x["kind"] == "act"]
        self.assertEqual(len(acts), 5)
        for x in acts[:3]:                                    # before the first Deploy/Extract: b′ == Rₙ
            self.assertAlmostEqual(x["b_after"], x["R"], delta=0.02)
        self.assertNotAlmostEqual(rows[-1]["b_after"], rows[-1]["R"], delta=1.0)   # cash moves made b ≠ R

    def test_legacy_and_flat_rows_do_not_move_the_ledger(self):
        legacy = {"action": "Legacy Rollover", "p_old": 19.0, "p_new": 40.0, "c_before": 1500, "c_after": 2000,
                  "b_before": 0, "b_after": 999.0, "round_id": 1}
        extract = {"action": "Extract Baseline", "p_old": 100, "p_new": 100, "c_before": 1500, "c_after": 1500,
                   "b_before": 142.96, "b_after": 100.0, "round_id": 3}
        deploy = {"action": "Deploy to Baseline", "p_old": 100, "p_new": 100, "c_before": 1500, "c_after": 1500,
                  "b_before": 100.0, "b_after": 300.0, "round_id": 4}
        led_ = led.build(ticker(legacy, chain_round(100, 110, 1500, rid=2), extract, deploy))
        kinds = [x["kind"] for x in led_["rows"]]
        self.assertEqual(kinds, ["legacy", "act", "frozen", "frozen"])
        self.assertEqual((led_["rows"][0]["R"], led_["rows"][0]["A"], led_["rows"][0]["E"]), (0.0, 0.0, 0.0))
        for x in led_["rows"][2:]:
            self.assertEqual((x["dA"], x["dR"]), (0.0, 0.0))
            self.assertEqual((x["R"], x["A"], x["E"]), (led_["rows"][1]["R"], led_["rows"][1]["A"], led_["rows"][1]["E"]))
        self.assertEqual(led_["p0"], 100)                       # anchor is the first *Chain Round*, not the legacy row
        self.assertEqual(led_["summary"]["legacy"], 1)
        self.assertEqual(led_["summary"]["acts"], 1)

    def test_zero_move_closed_and_garbage_rows_are_safe(self):
        rows = led.build(ticker(
            chain_round(10, 10, 1000),                                   # price did not move
            chain_round(10, 12, 0.0, 0.0, rid=2),                        # closed position (fix_c = 0)
            {"action": "Chain Round", "p_old": "x", "p_new": None, "c_before": float("nan"), "c_after": 5},
            {"action": 123}))["rows"]
        self.assertEqual([x["kind"] for x in rows], ["act", "frozen", "frozen", "frozen"])
        self.assertEqual((rows[0]["dA"], rows[0]["dR"]), (0.0, 0.0))
        self.assertFalse(rows[3]["known"])
        json.dumps(led.table(led.build(ticker(chain_round(10, 12, 100)))), allow_nan=False)

    def test_discontinuity_is_flagged_not_hidden(self):
        L = led.build(ticker(chain_round(100, 110, 1500, rid=1), chain_round(105, 100, 1500, rid=2)))   # 110 ≠ 105
        self.assertEqual(L["summary"]["breaks"], [2])
        self.assertTrue(L["rows"][0]["contiguous"])
        self.assertFalse(L["rows"][1]["contiguous"])
        clean = led.build(ticker(chain_round(100, 110, 1500, rid=1), chain_round(110, 100, 1500, b0=142.97, rid=2)))
        self.assertEqual(clean["summary"]["breaks"], [])

    def test_table_series_and_summary(self):
        t = engine_made()
        L = led.build(t)
        rows, ser = led.table(L), led.series(L)
        self.assertEqual(len(rows), len(t["rounds"]))
        self.assertEqual(list(rows[0]), led.COLUMNS)
        self.assertTrue(set(led.TIP_COLS) <= set(led.COLUMNS) and set(led.MONEY_COLS) <= set(led.COLUMNS))
        self.assertEqual({len(v) for v in ser.values()}, {len(rows)})
        s = L["summary"]
        self.assertAlmostEqual(s["e_minus_hedge"], s["E"] - s["hedge"], places=9)
        self.assertEqual(led.build({"rounds": []})["summary"]["n"], 0)


class TestExplain(unittest.TestCase):
    def setUp(self):
        self.t = ticker(chain_round(100, 110, 1500, rid=1), chain_round(110, 100, 1500, b0=142.97, rid=2))
        self.L = led.build(self.t)
        self.tips = led.explain(self.L, "AAPL")

    def test_one_text_per_row_per_column(self):
        self.assertEqual(len(self.tips), 2)
        for tip in self.tips:
            self.assertEqual(list(tip), led.TIP_COLS)
            self.assertTrue(all(isinstance(v, str) and v.count("\n") >= 2 for v in tip.values()))

    def test_title_formula_and_real_numbers(self):
        e2 = self.tips[1][led.C_E]
        self.assertTrue(e2.splitlines()[0].startswith("AAPL · รอบ 2 · "))
        self.assertIn("Eₙ = Aₙ − Rₙ", e2)
        self.assertIn("13.6364", e2)                                   # the real E of row 2
        da = self.tips[1][led.C_DA]
        self.assertIn("ΔAₙ = Fix_c × (P_fill / P_acted − 1)", da)
        self.assertIn("1,500.0000 × (100.0000 / 110.0000 − 1)", da)
        self.assertIn("(-136.3636)", self.tips[1][led.C_A].replace("−", "-"))   # negative operands in parentheses
        r = self.tips[1][led.C_R]
        self.assertIn("β + Fix_c × ln(Pₙ / P₀)", r)                      # LEGO fixed-anchor form is shown when contiguous

    def test_overrides_and_manual_edits_are_called_out(self):
        row = chain_round(100, 110, 1500, 1800.0, rid=1, shannon_profit=999.0, hedge_ratio=2.0, hedge_cost=10.0)
        tips = led.explain(led.build(ticker(row)), "X")[0]
        self.assertIn("⚠️", tips[led.C_SHANNON])                       # Shannon edited by hand
        self.assertIn("⚠️", tips[led.C_HEDGE])                         # hedge differs from the Put model
        self.assertIn("⚠️", tips[led.C_FIXC2])                         # fix_c′ ≠ fix_c + max(0, Surplus)

    def test_frozen_and_legacy_rows_explain_why(self):
        rows = [{"action": "Legacy Rollover", "p_old": 19, "p_new": 40, "c_before": 1500, "c_after": 1500,
                 "b_before": 0, "b_after": 5, "round_id": 1},
                {"action": "Extract Baseline", "p_old": 40, "p_new": 40, "c_before": 1500, "c_after": 1500,
                 "b_before": 5, "b_after": 1, "round_id": 2}]
        tips = led.explain(led.build(ticker(*rows)), "X")
        self.assertIn("Legacy Rollover", tips[0][led.C_DA])
        self.assertIn("ไม่เดาสูตรให้หลักฐานเก่า", tips[0][led.C_DA])
        self.assertIn("ไม่ใช่การ rebalance", tips[1][led.C_DA])
        self.assertIn("ดึงออก", tips[1][led.C_BASE])


class TestIntegrity(unittest.TestCase):
    def status(self, t):
        rows, ok = led.integrity(t)
        return {r["ข้อ"]: r["ผ่าน"] for r in rows}, ok

    def test_engine_made_data_passes_everything(self):
        st, ok = self.status(engine_made())
        self.assertTrue(ok)
        self.assertEqual(st["C1"], led.OK)                              # contiguous
        for cid in ("C2", "C3", "C4", "C5", "C6", "C7"):
            self.assertIn(st[cid], (led.OK, led.SKIP), cid)
        self.assertNotIn(led.FAIL, st.values())

    def test_empty_ticker(self):
        rows, ok = led.integrity({"ticker": "T", "rounds": []})
        self.assertTrue(ok)
        self.assertEqual(rows[0]["ข้อ"], "C0")

    def test_real_backup_has_no_failures_only_warnings(self):
        path = ROOT / "data" / "chain_system_backup.json"
        if not path.exists():
            self.skipTest("backup file not present")
        data, _ = eng.normalize(json.loads(path.read_text("utf-8")))
        for t in data["tickers"]:
            L = led.build(t)
            self.assertTrue(all(x["E"] >= -1e-9 for x in L["rows"]), t["ticker"])
            st, ok = self.status(t)
            self.assertTrue(ok, f"{t['ticker']}: {st}")

    def test_broken_identities_fail_but_manual_edits_only_warn(self):
        base = engine_made()
        # ❌ fix_c′ no longer equals fix_c + Scale Up
        t = copy.deepcopy(base)
        t["rounds"][1]["c_after"] += 50
        st, ok = self.status(t)
        self.assertEqual((st["C4"], ok), (led.FAIL, False))
        # ❌ Extract that *adds* money to the baseline
        t = copy.deepcopy(base)
        extract = next(r for r in t["rounds"] if r["action"] == "Extract Baseline")
        extract["b_after"] = extract["b_before"] + 10
        self.assertEqual(self.status(t)[0]["C5"], led.FAIL)
        # ❌ unknown action
        t = copy.deepcopy(base)
        t["rounds"].append({"action": "Mystery", "p_old": 1, "p_new": 1, "c_before": 1, "c_after": 1, "b_before": 0, "b_after": 0})
        st, ok = self.status(t)
        self.assertEqual((st["C8"], ok), (led.FAIL, False))
        # ⚠️ hand-edited Shannon / Hedge / continuity gap — still ok overall
        t = copy.deepcopy(base)
        t["rounds"][0]["shannon_profit"] += 5
        t["rounds"][1]["hedge_cost"] += 7
        t["rounds"][2]["p_old"] += 1
        st, ok = self.status(t)
        self.assertTrue(ok)
        self.assertEqual((st["C1"], st["C2"], st["C6"]), (led.WARN, led.WARN, led.WARN))

    def test_residuals_are_numbers_and_rows_have_the_lego_shape(self):
        rows, _ = led.integrity(engine_made())
        self.assertEqual(list(rows[0]), ["ข้อ", "สมการ/กฎ", "residual สูงสุด", "ผ่าน", "หมายเหตุ"])
        self.assertTrue(all(r["residual สูงสุด"] is None or r["residual สูงสุด"] >= 0 for r in rows))


class TestBreakEven(unittest.TestCase):
    def test_numbers_for_hedge_x1_and_x2(self):
        self.assertAlmostEqual(led.hedge_rate(), 0.12396167, places=7)
        b1, b2 = led.break_even(1.0), led.break_even(2.0)
        self.assertAlmostEqual(b1["R"], 0.13197, places=4)       # +13.2%  (ln basis, Chain's Shannon)
        self.assertAlmostEqual(b1["A"], 0.12396, places=4)       # +12.4%  (linear basis, ΔA)
        self.assertAlmostEqual(b1["E_up"], 0.58377, places=4)    # +58.4%  (harvest basis, one act)
        self.assertAlmostEqual(b1["E_down"], -0.41894, places=4)
        self.assertAlmostEqual(b2["R"], 0.28136, places=4)
        self.assertAlmostEqual(b2["A"], 0.24792, places=4)

    def test_roots_satisfy_their_equations(self):
        for h in (0.25, 0.5, 1.0, 2.0, 3.0):
            k, b = led.hedge_rate() * h, led.break_even(h)
            self.assertAlmostEqual(math.log1p(b["R"]), k, places=9)
            self.assertAlmostEqual(b["A"], k, places=12)
            for m in (b["E_up"], b["E_down"]):
                self.assertAlmostEqual(m - math.log1p(m), k, places=9)
            self.assertLess(b["E_down"], 0)
            self.assertGreater(b["E_up"], b["R"])                  # harvest alone needs a much bigger swing

    def test_no_hedge_means_zero(self):
        self.assertEqual(led.break_even(0.0), {"R": 0.0, "A": 0.0, "E_up": 0.0, "E_down": 0.0})


class TestLab(unittest.TestCase):
    def test_prices_are_deterministic_positive_and_validated(self):
        a = led.simulate_prices(100, 0.04, 0.0, 50, 7)
        self.assertEqual(a, led.simulate_prices(100, 0.04, 0.0, 50, 7))
        self.assertNotEqual(a, led.simulate_prices(100, 0.04, 0.0, 50, 8))
        self.assertEqual(len(a), 50)
        self.assertTrue(all(p > 0 and math.isfinite(p) for p in a))
        self.assertEqual(led.simulate_prices(100, 0.0, 0.0, 5, 1), [100.0] * 5)      # no vol, no drift → flat
        for bad in (dict(p0=0, vol=.04, drift=0, steps=10, seed=1), dict(p0=100, vol=-1, drift=0, steps=10, seed=1),
                    dict(p0=100, vol=.04, drift=0, steps=1, seed=1), dict(p0=100, vol=.04, drift=0, steps=10 ** 6, seed=1),
                    dict(p0=float("nan"), vol=.04, drift=0, steps=10, seed=1)):
            with self.assertRaises(eng.ChainError, msg=str(bad)):
                led.simulate_prices(**bad)

    def test_chain_simulation_uses_the_apps_own_round_math(self):
        prices = led.simulate_prices(100, 0.05, 0.0, 120, 11)
        sim = led.simulate_chain(100, prices, 1500, 1.0, 20)
        t = sim["ticker"]
        self.assertEqual(sim["steps"], [20, 40, 60, 80, 100, 120])
        self.assertEqual(len(t["rounds"]), 6)
        first = t["rounds"][0]
        self.assertEqual(first, {**first, **eng.plan_round({"fix_c": 1500, "price": 100, "baseline": 0.0},
                                                           prices[19], 1.0) | {"date": first["date"], "round_id": 1}})
        self.assertAlmostEqual(sim["ev_burn"], sum(r["hedge_cost"] for r in t["rounds"]), places=2)
        self.assertTrue(all(x["E"] >= -1e-9 for x in sim["ledger"]["rows"]))
        self.assertEqual(sim["final_fix_c"], t["current_state"]["fix_c"])
        self.assertEqual(led.simulate_chain(100, prices, 1500, 1.0, 20)["ledger"]["summary"], sim["ledger"]["summary"])

    def test_last_step_is_always_a_round_and_arguments_are_validated(self):
        prices = led.simulate_prices(100, 0.03, 0.0, 25, 2)
        self.assertEqual(led.simulate_chain(100, prices, 1000, 0.0, 10, no_hedge=True)["steps"], [10, 20, 25])
        for bad in (dict(fix_c=0, every=5), dict(fix_c=1000, every=0), dict(fix_c=1000, every=26)):
            with self.assertRaises(eng.ChainError, msg=str(bad)):
                led.simulate_chain(100, prices, hedge_x=1.0, **bad)

    def test_no_surplus_keeps_fix_c_and_hedge_burns_ev(self):
        prices = led.simulate_prices(100, 0.05, 0.01, 100, 5)
        flat = led.simulate_chain(100, prices, 2000, 1.0, 10, no_surplus=True)
        self.assertEqual(flat["final_fix_c"], 2000.0)
        self.assertGreater(flat["ev_burn"], 0)
        free = led.simulate_chain(100, prices, 2000, 0.0, 10, no_hedge=True)
        self.assertEqual(free["ev_burn"], 0.0)


if __name__ == "__main__":
    unittest.main()
