"""Unit tests for engine_v2 — stdlib only:  python -m unittest discover -s tests -v"""
import copy
import importlib.util
import json
import math
import random
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import engine_v2 as eng  # noqa: E402

# Captured from v1 ``flywheels.run_chain_round`` — v2 must stay numerically identical.
GOLDEN = [
    ({"fix_c": 1500, "price": 6.88, "baseline": 0.0}, 7.5, 2.0, {},
     dict(shannon_profit=129.43, hedge_cost=371.89, surplus=-242.46, scale_up=0.0, c_after=1500.0, b_after=129.43)),
    ({"fix_c": 1500, "price": 6.88, "baseline": 0.0}, 6.0, 2.0, {},
     dict(shannon_profit=-205.29, hedge_cost=371.89, surplus=-577.17, scale_up=0.0, c_after=1500.0, b_after=-205.29)),
    ({"fix_c": 3700, "price": 15.0, "baseline": -125.34}, 20.0, 1.0, {},
     dict(shannon_profit=1064.42, hedge_cost=458.66, surplus=605.77, scale_up=605.77, c_after=4305.77, b_after=939.08)),
    ({"fix_c": 2500, "price": 24.0, "baseline": 966.54}, 24.0, 2.0, {},
     dict(shannon_profit=0.0, hedge_cost=619.81, surplus=-619.81, scale_up=0.0, c_after=2500.0, b_after=966.54)),
    ({"fix_c": 1500, "price": 6.88, "baseline": 10.0}, 9.0, 2.0, {"no_hedge": True},
     dict(shannon_profit=402.91, hedge_cost=0.0, surplus=402.91, scale_up=402.91, c_after=1902.91, b_after=412.91)),
    ({"fix_c": 1500, "price": 6.88, "baseline": 10.0}, 9.0, 2.0, {"no_surplus": True},
     dict(shannon_profit=402.91, hedge_cost=371.89, surplus=31.02, scale_up=0.0, c_after=1500.0, b_after=412.91)),
    ({"fix_c": 0.0, "price": 6.88, "baseline": 10.0}, 9.0, 2.0, {},
     dict(shannon_profit=0.0, hedge_cost=0.0, surplus=0.0, scale_up=0.0, c_after=0.0, b_after=10.0)),
]


def make_store(tmp: str, *tickers) -> eng.Store:
    store = eng.Store(Path(tmp) / "trading_data.json")
    for sym, price, c in tickers:
        store.update(lambda d, s=sym, p=price, c_=c: eng.add_ticker(d, s, p, c_))
    return store


class TestPricing(unittest.TestCase):
    def test_bs_put_matches_v1(self):
        self.assertAlmostEqual(eng.bs_put(100, 90), 12.396167192642633, places=9)

    def test_degenerate_inputs_are_zero(self):
        self.assertEqual(eng.bs_put(0, 90), 0.0)
        self.assertEqual(eng.bs_put(100, 0), 0.0)
        self.assertEqual(eng.hedge_cost(0, 10, 2), 0.0)
        self.assertEqual(eng.hedge_cost(1000, 0, 2), 0.0)
        self.assertEqual(eng.hedge_cost(1000, 10, 0), 0.0)

    def test_num(self):
        for bad in (None, "", "abc", float("nan"), float("inf"), [1]):
            self.assertIsNone(eng.num(bad, None))
        self.assertEqual(eng.num("12.5"), 12.5)
        self.assertEqual(eng.num(None), 0.0)


class TestPlanRound(unittest.TestCase):
    def test_golden_values_from_v1(self):
        for state, p_new, ratio, flags, want in GOLDEN:
            got = eng.plan_round(state, p_new, ratio, **flags)
            for k, v in want.items():
                self.assertAlmostEqual(got[k], v, places=2, msg=f"{k} for {state} -> {p_new} {flags}")

    def test_record_is_v1_compatible(self):
        r = eng.plan_round({"fix_c": 1500, "price": 6.88, "baseline": 0.0}, 7.5, 2.0)
        for k in ("action", "p_old", "p_new", "c_before", "c_after", "shannon_profit", "hedge_cost",
                  "surplus", "scale_up", "b_before", "b_after", "hedge_ratio", "ev_change"):
            self.assertIn(k, r)
        self.assertEqual(r["action"], "Chain Round")
        self.assertEqual(r["ev_change"], r["hedge_cost"])
        json.dumps(r, allow_nan=False)

    def test_no_hedge_records_zero_ratio(self):
        r = eng.plan_round({"fix_c": 1500, "price": 6.88, "baseline": 0.0}, 9, 2.0, no_hedge=True)
        self.assertEqual((r["hedge_cost"], r["hedge_ratio"], r["ev_change"]), (0.0, 0.0, 0.0))

    def test_unpriced_ticker_just_seeds_the_price(self):
        r = eng.plan_round({"fix_c": 0.0, "price": 0.0, "baseline": 5.0}, 12.5, 2.0, c_after=2000)
        self.assertEqual((r["shannon_profit"], r["hedge_cost"], r["b_after"]), (0.0, 0.0, 5.0))
        self.assertEqual((r["p_new"], r["c_after"], r["scale_up"]), (12.5, 2000.0, 2000.0))

    def test_overrides(self):
        s = {"fix_c": 1500, "price": 6.88, "baseline": 0.0}
        r = eng.plan_round(s, 9, 2.0, hedge=100)               # real hedge was cheaper
        self.assertEqual((r["hedge_cost"], r["surplus"], r["c_after"], r["ev_change"]),
                         (100.0, 302.91, 1802.91, 100.0))
        r = eng.plan_round(s, 9, 2.0, c_after=0)               # close the position
        self.assertEqual((r["c_after"], r["scale_up"], r["surplus"]), (0.0, -1500.0, 31.02))
        r = eng.plan_round(s, 9, 2.0, b_after=250)
        self.assertEqual(r["b_after"], 250.0)

    def test_invalid_input_raises(self):
        s = {"fix_c": 1500, "price": 6.88, "baseline": 0.0}
        for kwargs in ({"p_new": 0, "hedge_ratio": 2}, {"p_new": -1, "hedge_ratio": 2},
                       {"p_new": float("nan"), "hedge_ratio": 2}, {"p_new": None, "hedge_ratio": 2},
                       {"p_new": 5, "hedge_ratio": -1}, {"p_new": 5, "hedge_ratio": float("inf")},
                       {"p_new": 5, "hedge_ratio": 2, "hedge": -5}, {"p_new": 5, "hedge_ratio": 2, "c_after": -1},
                       {"p_new": 5, "hedge_ratio": 2, "b_after": float("nan")}):
            with self.assertRaises(eng.ChainError, msg=str(kwargs)):
                eng.plan_round(s, **kwargs)

    def test_never_emits_nan_or_negative_zero(self):
        random.seed(1)
        for _ in range(500):
            s = {"fix_c": random.choice([0, random.uniform(1, 9e3)]), "price": random.uniform(0.01, 500),
                 "baseline": random.uniform(-500, 500)}
            r = eng.plan_round(s, s["price"] * random.uniform(0.1, 4), random.choice([0, 1, 2]))
            json.dumps(r, allow_nan=False)
            for v in r.values():
                if isinstance(v, float):
                    self.assertFalse(v == 0 and math.copysign(1, v) < 0, r)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "v1 module needs streamlit")
class TestParityWithV1(unittest.TestCase):
    def test_random_cases_match_flywheels(self):
        import flywheels as fw
        rnd = random.Random(42)
        keys = ("p_old", "p_new", "c_before", "c_after", "shannon_profit", "hedge_cost", "surplus",
                "scale_up", "b_before", "b_after")
        for _ in range(400):
            s = {"fix_c": round(rnd.choice([0, rnd.uniform(100, 9000)]), 2),
                 "price": round(rnd.uniform(0.2, 300), 2), "baseline": round(rnd.uniform(-3000, 3000), 2)}
            p_new = round(s["price"] * rnd.uniform(0.3, 3), 4)
            ratio, nh, ns = rnd.choice([0.0, 1.0, 2.0]), rnd.random() < .2, rnd.random() < .2
            a = fw.run_chain_round(dict(s), p_new, ratio, ignore_hedge=nh, ignore_surplus=ns)
            b = eng.plan_round(s, p_new, ratio, no_hedge=nh, no_surplus=ns)
            for k in keys:
                self.assertAlmostEqual(a[k], b[k], places=2, msg=f"{k} {s} {p_new}")


class TestOps(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name, ("AAA", 10, 1000), ("BBB", 5, 2000))

    def t(self, name="AAA"):
        return next(x for x in self.store.read()["tickers"] if x["ticker"] == name)

    def test_add_ticker_validation(self):
        for sym in ("", "bad symbol", "<img src=x>", "A" * 16, "aaa"):   # "aaa" == AAA (case-insensitive dup)
            with self.assertRaises(eng.ChainError, msg=sym):
                self.store.update(lambda d, s=sym: eng.add_ticker(d, s, 10, 1000))
        for price, c in ((0, 100), (-1, 100), (5, 0), (None, 100), (float("nan"), 100)):
            with self.assertRaises(eng.ChainError):
                self.store.update(lambda d, p=price, c_=c: eng.add_ticker(d, "ZZZ", p, c_))
        self.assertEqual([x["ticker"] for x in self.store.read()["tickers"]], ["AAA", "BBB"])

    def test_commit_round_updates_state_ev_and_net(self):
        plan = eng.plan_round(self.t()["current_state"], 12, 2.0)
        self.store.update(lambda d: eng.apply_round(d, "AAA", plan))
        t = self.t()
        s = t["current_state"]
        self.assertEqual(len(t["rounds"]), 1)
        self.assertEqual(t["rounds"][0]["round_id"], 1)
        self.assertEqual((s["price"], s["fix_c"], s["baseline"]), (12.0, plan["c_after"], plan["b_after"]))
        self.assertEqual(s["cumulative_ev"], plan["hedge_cost"])
        self.assertEqual(s["net_pnl"], -plan["hedge_cost"])     # Net stays consistent with Ev burn
        self.assertEqual(self.store.read()["tickers"][1]["current_state"]["net_pnl"], 0.0)   # BBB untouched

    def test_stale_plan_is_rejected_and_nothing_changes(self):
        plan = eng.plan_round(self.t()["current_state"], 12, 2.0)
        self.store.update(lambda d: eng.add_to_pool(d, 500))
        self.store.update(lambda d: eng.deploy(d, "AAA", 100))          # baseline moved after the preview
        before = json.dumps(self.store.read(), sort_keys=True)
        with self.assertRaises(eng.StaleState):
            self.store.update(lambda d: eng.apply_round(d, "AAA", plan))
        self.assertEqual(json.dumps(self.store.read(), sort_keys=True), before)

    def test_unknown_ticker(self):
        with self.assertRaises(eng.ChainError):
            self.store.update(lambda d: eng.apply_round(d, "NOPE", {}))
        with self.assertRaises(eng.ChainError):
            self.store.update(lambda d: eng.deploy(d, "NOPE", 1))

    def test_pool_funding_and_harvest(self):
        self.store.update(lambda d: eng.add_to_pool(d, 250.5, "Funding", "AAA"))
        self.store.update(lambda d: eng.add_to_pool(d, 100, "Harvest"))
        d = self.store.read()
        self.assertEqual(d["global_pool_cf"], 350.5)
        self.assertEqual([e["category"] for e in d["treasury_history"]], ["Funding", "Harvest"])
        self.assertEqual(d["treasury_history"][0]["note"], "Added to Pool CF [Ticker: AAA]")
        self.assertEqual(d["treasury_history"][1]["pool_cf_balance"], 350.5)
        for bad in (0, -5, None, "x", float("inf")):
            with self.assertRaises(eng.ChainError):
                self.store.update(lambda d_, b=bad: eng.add_to_pool(d_, b))
        with self.assertRaises(eng.ChainError):
            self.store.update(lambda d_: eng.add_to_pool(d_, 5, "Bogus"))

    def test_deploy_and_extract_move_money_both_ways(self):
        self.store.update(lambda d: eng.add_to_pool(d, 1000))
        self.store.update(lambda d: eng.deploy(d, "AAA", 400, "top up"))
        d = self.store.read()
        self.assertEqual(d["global_pool_cf"], 600.0)
        self.assertEqual(self.t()["current_state"]["baseline"], 400.0)
        self.assertEqual(self.t()["rounds"][-1]["action"], "Deploy to Baseline")
        self.assertTrue(self.t()["rounds"][-1]["note"].endswith("| top up"))
        self.assertEqual(d["treasury_history"][-1]["amount"], -400.0)

        self.store.update(lambda d: eng.extract_baseline(d, "AAA", 150))
        d = self.store.read()
        self.assertEqual((d["global_pool_cf"], self.t()["current_state"]["baseline"]), (750.0, 250.0))
        self.assertEqual(eng.withdrawn_baseline(self.t()), 150.0)
        self.assertEqual(d["treasury_history"][-1]["note"], "[Ticker: AAA]")
        self.assertEqual([r["round_id"] for r in self.t()["rounds"]], [1, 2])

    def test_insufficient_funds_raise_and_change_nothing(self):
        self.store.update(lambda d: eng.add_to_pool(d, 100))
        snap = json.dumps(self.store.read(), sort_keys=True)
        for fn in (lambda d: eng.deploy(d, "AAA", 100.01), lambda d: eng.allocate_ev(d, 100.01),
                   lambda d: eng.extract_baseline(d, "AAA", 1)):
            with self.assertRaises(eng.ChainError):
                self.store.update(fn)
        self.assertEqual(json.dumps(self.store.read(), sort_keys=True), snap)

    def test_allocate_and_ev_flow(self):
        self.store.update(lambda d: eng.add_to_pool(d, 1000))
        self.store.update(lambda d: eng.allocate_ev(d, 300))
        self.store.update(lambda d: eng.ev_flow(d, -120.25, "AAA"))
        self.store.update(lambda d: eng.ev_flow(d, 20))
        d = self.store.read()
        self.assertEqual((d["global_pool_cf"], d["global_ev_reserve"]), (700.0, 199.75))
        self.assertEqual([e["category"] for e in d["treasury_history"]], ["Funding", "Allocation", "Expense", "Income"])
        with self.assertRaises(eng.ChainError):
            self.store.update(lambda d_: eng.ev_flow(d_, 0))

    def test_extract_needs_positive_baseline(self):
        with self.assertRaises(eng.ChainError):
            self.store.update(lambda d: eng.extract_baseline(d, "AAA", 10))


class TestNormalize(unittest.TestCase):
    def test_bad_shape_raises(self):
        for raw in (None, 5, "x", {}, {"tickers": "no"}, {"version": 2}):
            with self.assertRaises(eng.DataError):
                eng.normalize(raw)

    def test_legacy_v1_list_uses_final_string(self):
        raw = [{"ticker": "FFWM", "Final": "6.88, 1500, 0", "Original": "ราคาอ้างอิง: 6.88, ทุนคงที่: 1500",
                "history_1": "x"},
               {"ticker": "ODD", "Original": "garbage", "initial_state": {"price": 3, "fix_c": 900}},
               {"ticker": "NONE"}]
        d, warns = eng.normalize(raw)
        s = [t["current_state"] for t in d["tickers"]]
        self.assertEqual((s[0]["price"], s[0]["fix_c"], s[0]["baseline"]), (6.88, 1500.0, 0.0))
        self.assertEqual((s[1]["price"], s[1]["fix_c"]), (3.0, 900.0))
        self.assertEqual((s[2]["price"], s[2]["fix_c"]), (0.0, 0.0))
        self.assertEqual(d["tickers"][0]["history_1"], "x")           # legacy keys preserved
        self.assertEqual(d["version"], 2)
        self.assertEqual(warns, [])

    def test_dict_without_version_is_accepted(self):
        d, _ = eng.normalize({"tickers": [{"ticker": "A", "current_state": {"price": 1, "fix_c": 2}}],
                              "global_pool_cf": "12.5", "custom": {"keep": 1}})
        self.assertEqual((d["global_pool_cf"], d["global_ev_reserve"], d["custom"]), (12.5, 0.0, {"keep": 1}))
        self.assertEqual(d["settings"]["default_hedge_ratio"], 2.0)

    def test_junk_values_are_coerced(self):
        d, _ = eng.normalize({"tickers": [{"ticker": "A", "current_state": {
            "price": "abc", "fix_c": -50, "baseline": None, "cumulative_ev": float("nan"), "strategy_tags": "x"}}],
            "global_pool_cf": float("inf"), "settings": {"default_hedge_ratio": "bad"},
            "treasury_history": [1, {"note": "ok"}, None]})
        s = d["tickers"][0]["current_state"]
        self.assertEqual((s["price"], s["fix_c"], s["baseline"], s["cumulative_ev"]), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(s["strategy_tags"], [])
        self.assertEqual(d["global_pool_cf"], 0.0)
        self.assertEqual(d["settings"]["default_hedge_ratio"], 2.0)
        self.assertEqual(d["treasury_history"], [{"note": "ok"}])
        json.dumps(d, allow_nan=False)

    def test_duplicate_and_blank_names(self):
        d, warns = eng.normalize({"tickers": [{"ticker": "A"}, {"ticker": "a"}, {"ticker": "  "}, "junk", {"ticker": "A"}]})
        self.assertEqual([t["ticker"] for t in d["tickers"]], ["A", "a#2", "TICKER3", "A#3"])
        self.assertEqual(len(warns), 4)

    def test_state_follows_last_round(self):
        rounds = [{"p_new": 5, "c_after": 100, "b_after": 1}, {"p_new": 7, "c_after": 120, "b_after": 3}]
        d, _ = eng.normalize({"tickers": [{"ticker": "A", "rounds": rounds + ["junk"],
                                           "current_state": {"price": 1, "fix_c": 1, "baseline": 1}}]})
        s = d["tickers"][0]["current_state"]
        self.assertEqual((s["price"], s["fix_c"], s["baseline"]), (7.0, 120.0, 3.0))
        # an incomplete last round does not clobber the state
        d, _ = eng.normalize({"tickers": [{"ticker": "A", "rounds": [{"p_new": 9}],
                                           "current_state": {"price": 1, "fix_c": 2, "baseline": 3}}]})
        self.assertEqual(d["tickers"][0]["current_state"]["price"], 1.0)

    def test_repairing_state_from_the_round_log_is_reported(self):
        rounds = [{"p_new": 7, "c_after": 120, "b_after": 3}]
        d, warns = eng.normalize({"tickers": [{"ticker": "A", "rounds": rounds,
                                               "current_state": {"price": 1, "fix_c": 1, "baseline": 1}}]})
        self.assertEqual(d["tickers"][0]["current_state"]["price"], 7.0)
        self.assertEqual(len(warns), 1)
        self.assertIn("A: state ไม่ตรงกับรอบล่าสุด", warns[0])
        # consistent state → silent (this is every file the app itself writes)
        d2, warns2 = eng.normalize({"tickers": [{"ticker": "A", "rounds": rounds,
                                                 "current_state": {"price": 7, "fix_c": 120, "baseline": 3}}]})
        self.assertEqual(warns2, [])
        self.assertEqual(eng.normalize(d)[1], [])                      # …and after repair it stays quiet

    def test_api_version_marker(self):
        self.assertIsInstance(eng.API_VERSION, int)

    def test_input_untouched_and_idempotent(self):
        raw = {"version": 2, "tickers": [{"ticker": "A", "rounds": [{"p_new": 5, "c_after": 1, "b_after": 2}]}]}
        frozen = copy.deepcopy(raw)
        d, _ = eng.normalize(raw)
        self.assertEqual(raw, frozen)
        self.assertEqual(eng.normalize(d)[0], d)

    def test_real_backup_file_roundtrips(self):
        path = ROOT / "data" / "chain_system_backup.json"
        if not path.exists():
            self.skipTest("backup file not present")
        raw = json.loads(path.read_text(encoding="utf-8"))
        d, warns = eng.normalize(raw)
        self.assertEqual(warns, [])
        self.assertEqual(len(d["tickers"]), len(raw["tickers"]))
        self.assertEqual(d["global_pool_cf"], raw["global_pool_cf"])
        self.assertEqual(sum(len(t["rounds"]) for t in d["tickers"]), sum(len(t.get("rounds", [])) for t in raw["tickers"]))
        self.assertEqual(eng.normalize(d)[0], d)


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "trading_data.json"

    def test_missing_file_is_an_empty_portfolio_and_nothing_is_written(self):
        store = eng.Store(self.path)
        d = store.read()
        self.assertEqual((d["tickers"], d["global_pool_cf"], d["version"]), ([], 0.0, 2))
        self.assertFalse(self.path.exists())

    def test_update_persists_and_keeps_one_backup(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        self.assertTrue(self.path.exists())
        store.update(lambda d: eng.add_to_pool(d, 50))
        self.assertEqual(json.loads(self.path.read_text("utf-8"))["global_pool_cf"], 50.0)
        self.assertEqual(json.loads(store.backup_path.read_text("utf-8"))["global_pool_cf"], 0.0)   # previous version
        self.assertEqual(eng.Store(self.path).read()["global_pool_cf"], 50.0)                        # survives restart
        self.assertEqual([p.name for p in self.dir.iterdir() if p.suffix == ".tmp"], [])

    def test_failed_update_changes_nothing(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        snap, raw = json.dumps(store.read(), sort_keys=True), self.path.read_bytes()

        def boom(d):
            eng.add_to_pool(d, 999)      # mutates the working copy first ...
            raise RuntimeError("boom")   # ... then fails
        with self.assertRaises(RuntimeError):
            store.update(boom)
        self.assertEqual(json.dumps(store.read(), sort_keys=True), snap)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_nan_is_never_persisted(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        raw = self.path.read_bytes()
        with self.assertRaises(ValueError):
            store.update(lambda d: d.update(global_pool_cf=float("nan")))
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertEqual(store.read()["global_pool_cf"], 0.0)

    def test_read_returns_cached_object_until_the_file_changes(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        a = store.read()
        self.assertIs(a, store.read())                      # no re-parse per rerun
        ext = json.loads(self.path.read_text("utf-8"))
        ext["global_pool_cf"] = 123456.78                   # another process (v1 page) edits the file
        self.path.write_text(json.dumps(ext, indent=2) + "\n\n", encoding="utf-8")
        self.assertEqual(store.read()["global_pool_cf"], 123456.78)

    def test_corrupt_main_falls_back_to_backup_and_keeps_it(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        store.update(lambda d: eng.add_to_pool(d, 70))      # main = 70, backup = 0
        self.path.write_text("{ not json", encoding="utf-8")
        fresh = eng.Store(self.path)
        self.assertEqual(fresh.read()["global_pool_cf"], 0.0)
        self.assertTrue(any("backup" in w for w in fresh.warnings))
        backup_before = fresh.backup_path.read_bytes()
        fresh.update(lambda d: eng.add_to_pool(d, 5))       # heals main without clobbering the backup with garbage
        self.assertEqual(json.loads(self.path.read_text("utf-8"))["global_pool_cf"], 5.0)
        self.assertEqual(fresh.backup_path.read_bytes(), backup_before)

    def test_unreadable_everything_blocks_writes_until_replaced(self):
        self.path.write_text("garbage", encoding="utf-8")
        store = eng.Store(self.path)
        with self.assertRaises(eng.DataError):
            store.read()
        with self.assertRaises(eng.DataError):
            store.update(lambda d: eng.add_to_pool(d, 1))
        self.assertEqual(self.path.read_text("utf-8"), "garbage")          # never overwritten silently
        store.replace({"tickers": [{"ticker": "A", "Final": "2, 100, 0"}]})   # explicit import recovers
        self.assertEqual(store.read()["tickers"][0]["current_state"]["price"], 2.0)
        self.assertFalse(store.backup_path.exists())                       # garbage was not kept as the backup

    def test_replace_and_restore_backup(self):
        store = make_store(self.dir, ("AAA", 10, 1000))
        store.update(lambda d: eng.add_to_pool(d, 9))
        store.replace({"tickers": [], "global_pool_cf": 1})
        self.assertEqual(store.read()["tickers"], [])
        store.restore_backup()                              # undo the import
        self.assertEqual((len(store.read()["tickers"]), store.read()["global_pool_cf"]), (1, 9.0))
        store.restore_backup()                              # and redo
        self.assertEqual(store.read()["global_pool_cf"], 1.0)
        with self.assertRaises(eng.DataError):
            store.replace({"nope": 1})
        with self.assertRaises(eng.ChainError):
            eng.Store(self.dir / "other.json").restore_backup()

    def test_concurrent_updates_lose_nothing(self):
        store = make_store(self.dir, ("AAA", 10, 1000))

        def work():
            for _ in range(25):
                store.update(lambda d: eng.add_to_pool(d, 1))
        threads = [threading.Thread(target=work) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(store.read()["global_pool_cf"], 200.0)
        self.assertEqual(json.loads(self.path.read_text("utf-8"))["global_pool_cf"], 200.0)
        self.assertEqual(len(store.read()["treasury_history"]), 200)


class TestViews(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = make_store(self._tmp.name, ("AAA", 10, 1000), ("BBB", 5, 2000))

    def commit(self, name, p_new, ratio=2.0):
        t = next(x for x in self.store.read()["tickers"] if x["ticker"] == name)
        plan = eng.plan_round(t["current_state"], p_new, ratio, no_hedge=(ratio == 0))
        self.store.update(lambda d: eng.apply_round(d, name, plan))
        return plan

    def test_totals(self):
        self.commit("AAA", 12)
        tot = eng.totals(self.store.read())
        self.assertAlmostEqual(tot["fix_c"], 1000 + 2000 + self.store.read()["tickers"][0]["rounds"][0]["scale_up"], 2)
        self.assertGreater(tot["burn"], 0)
        self.assertEqual(tot["rounds"], 1)
        self.assertIsNone(eng.totals(eng.normalize({"tickers": []})[0])["efficiency"])

    def test_treasury_rows_filter_and_order(self):
        self.store.update(lambda d: eng.add_to_pool(d, 100, "Funding", "AAA"))
        self.store.update(lambda d: eng.add_to_pool(d, 50, "Harvest", "BBB"))
        self.store.update(lambda d: eng.add_to_pool(d, 5))
        d = self.store.read()
        self.assertEqual([r["Amount"] for r in eng.treasury_rows(d)], [5.0, 50.0, 100.0])
        self.assertEqual([r["Amount"] for r in eng.treasury_rows(d, "AAA")], [100.0])
        self.assertEqual(len(eng.treasury_rows(d, limit=2)), 2)

    def test_round_rows_are_typed_and_newest_first(self):
        self.commit("AAA", 12)
        self.commit("AAA", 11)
        rows = eng.round_rows(self.store.read()["tickers"][0])
        self.assertEqual([r["#"] for r in rows], [2, 1])
        self.assertTrue(all(isinstance(r["P new"], float) and isinstance(r["Note"], str) for r in rows))
        self.assertEqual(len(eng.round_rows(self.store.read()["tickers"][0], limit=1)), 1)
        legacy, _ = eng.normalize({"tickers": [{"ticker": "L", "rounds": [{"action": None, "p_new": "x"}]}]})
        self.assertEqual(eng.round_rows(legacy["tickers"][0])[0]["P new"], 0.0)

    def test_sparklines_end_at_the_current_state(self):
        self.assertEqual(eng.sparklines(self.store.read()["tickers"][0]), {})
        self.commit("AAA", 12)
        self.commit("AAA", 9)
        t = self.store.read()["tickers"][0]
        sp, s = eng.sparklines(t), t["current_state"]
        self.assertTrue(all(len(v) == len(t["rounds"]) + 1 for v in sp.values()))
        self.assertEqual((sp["price"][0], sp["price"][-1]), (10.0, 9.0))
        self.assertEqual((sp["fix_c"][-1], sp["baseline"][-1]), (s["fix_c"], s["baseline"]))
        self.assertAlmostEqual(sp["ev"][-1], s["cumulative_ev"], places=9)
        self.assertAlmostEqual(sp["net"][-1], s["net_pnl"], places=9)
        self.assertEqual(sp["ev"][0], 0.0)                              # nothing burnt before the first round
        self.assertLess(sp["net"][-1], sp["net"][0] + 1e-9)            # hedge cost only ever lowers Net

    def test_capital_flow(self):
        self.commit("AAA", 14, 0.0)           # big up move, no hedge → surplus → scale up
        nodes, links = eng.capital_flow(self.store.read())
        self.assertEqual(nodes[0], "AAA")
        labels = {lbl for *_, lbl in links}
        self.assertTrue({"AAA Shannon", "Surplus", "Auto Scale Up"} <= labels)
        self.assertTrue(all(0 <= s < len(nodes) and 0 <= t < len(nodes) and v > 0 for s, t, v, _ in links))
        self.assertEqual(eng.capital_flow(self.store.read(), ["BBB"])[1], [])

    def test_payoff_curves_match_v1_formulas(self):
        c = dict(x0_1=10, constant1=1000, b1=5, delta1=0.2, x0_2=15, constant2=1000, b2=-5, delta2=1.0,
                 anchor=10, ref_const=1000, long_qty=100, long_entry=10, short_qty=100, short_entry=15,
                 call_qty=100, call_strike=10, put_qty=100, put_strike=10)
        xs = [5.0, 10.0, 20.0, 31.0]
        o = eng.payoff_curves(c, xs, net_of=("y1", "y8"))
        self.assertAlmostEqual(o["y1"][2], 1000 * math.log(2) + 5, places=9)
        self.assertAlmostEqual(o["y5"][0], 1000 * math.log(0.5) * 0.2 + 5, places=9)       # δ1 below x0
        self.assertAlmostEqual(o["y5"][2], 1000 * math.log(2) * 1.0 + 5, places=9)         # δ2 above x0
        self.assertAlmostEqual(o["y2"][0], 1000 * math.log(2 - 5 / 15) - 5, places=9)
        self.assertAlmostEqual(o["y2"][3], 1000 * math.log(1e-9) - 5, places=6)            # x ≥ 2·x0₂ guard
        self.assertEqual(o["y8"], [0.0, 0.0, 1000.0, 2100.0])
        self.assertEqual(o["y9"], [500.0, 0.0, 0.0, 0.0])
        self.assertEqual(o["y10"], [-500.0, 0.0, 1000.0, 2100.0])
        self.assertEqual(o["y11"], [1000.0, 500.0, -500.0, -1600.0])
        self.assertAlmostEqual(o["y3"][2], o["y1"][2] + o["y8"][2], places=9)               # Σ of the picked lines only
        self.assertAlmostEqual(o["overlay"][2], o["y3"][2] - o["y6"][2], places=9)
        xs = eng.payoff_prices(100)
        self.assertEqual((len(xs), xs[0], xs[-1]), (300, 10.0, 250.0))


class TestNetwork(unittest.TestCase):
    def setUp(self):
        eng._PRICE_CACHE.clear()

    def test_fetch_price_prefers_meta_then_last_close(self):
        payload = {"chart": {"result": [{"meta": {"regularMarketPrice": 12.34}}]}}
        with mock.patch.object(eng, "_http_json", return_value=payload):
            self.assertEqual(eng.fetch_price("abc"), (12.34, "Yahoo (ABC)"))
        eng._PRICE_CACHE.clear()
        payload = {"chart": {"result": [{"meta": {}, "indicators": {"quote": [{"close": [1.0, None, 7.5, None]}]}}]}}
        with mock.patch.object(eng, "_http_json", return_value=payload):
            self.assertEqual(eng.fetch_price("abc")[0], 7.5)

    def test_fetch_price_failure_falls_back_to_bk_then_reports_error(self):
        calls = []

        def fake(sym):
            calls.append(sym)
            if sym == "PTT":
                raise OSError("down")
            return 33.0
        with mock.patch.object(eng, "_chart_price", side_effect=fake):
            self.assertEqual(eng.fetch_price("ptt"), (33.0, "Yahoo (PTT.BK)"))
        self.assertEqual(calls, ["PTT", "PTT.BK"])
        eng._PRICE_CACHE.clear()
        with mock.patch.object(eng, "_chart_price", side_effect=OSError("down")):
            price, src = eng.fetch_price("LONGSYMBOL")
        self.assertEqual(price, 0.0)
        self.assertTrue(src.startswith("error:"))
        self.assertEqual(eng.fetch_price("")[0], 0.0)

    def test_fetch_price_is_cached_with_ttl(self):
        with mock.patch.object(eng, "_chart_price", return_value=5.0) as m, \
                mock.patch.object(eng.time, "monotonic", side_effect=[100.0, 120.0, 200.0]):   # one tick per call
            self.assertEqual(eng.fetch_price("AAA")[0], 5.0)                 # t=100 miss → network, expires at 160
            self.assertTrue(eng.fetch_price("AAA")[1].endswith("(cached)"))  # t=120 < 160 → cached
            self.assertEqual(eng.fetch_price("AAA")[0], 5.0)                 # t=200 > 160 → refetch
            self.assertEqual(m.call_count, 2)

    def test_fetch_json_url_only_allows_https(self):
        for url in ("http://example.com/a.json", "file:///etc/passwd", "ftp://x/y", "", "https://", "  "):
            with self.assertRaises(eng.ChainError, msg=url):
                eng.fetch_json_url(url)
        with mock.patch.object(eng, "_http_json", return_value={"tickers": []}) as m:
            self.assertEqual(eng.fetch_json_url(" https://example.com/a.json "), {"tickers": []})
            m.assert_called_once_with("https://example.com/a.json", 15)

    def test_http_json_size_cap(self):
        class Resp:
            def __init__(self, data): self.data = data
            def read(self, n): return self.data[:n]
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch.object(eng._OPENER, "open", return_value=Resp(b"x" * 50)):
            with self.assertRaises(eng.ChainError):
                eng._http_json("https://example.com", 5, limit=10)
        with mock.patch.object(eng._OPENER, "open", return_value=Resp(b'{"a": 1}')):
            self.assertEqual(eng._http_json("https://example.com", 5), {"a": 1})

    def test_redirects_must_stay_on_https(self):
        handler, req = eng._HttpsRedirects(), eng.urllib.request.Request("https://example.com/a")
        for target in ("http://example.com/b", "ftp://example.com/b", "file:///etc/passwd"):
            with self.assertRaises(eng.ChainError, msg=target):
                handler.redirect_request(req, None, 302, "Found", {}, target)
        out = handler.redirect_request(req, None, 302, "Found", {}, "https://example.com/b")
        self.assertEqual(out.full_url, "https://example.com/b")


if __name__ == "__main__":
    unittest.main()
