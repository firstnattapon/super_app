"""app_v2_minimal.mmd — the whole goal as a loop of a few lines, checked against the real engine.

The minimal system keeps five steps + one check: harvest · pay first · grow only from the surplus · re-center,
then Ev Efficiency. These tests show it is
  * sufficient — it reproduces the engine's fix_c, ΣShannon, ΣHedge and Eₙ (to cent rounding) with the
    Black-Scholes model collapsed to one constant and Baseline = ΣShannon, and the goal's clauses hold on any path;
  * minimal    — dropping any of the five steps changes the outcome.
"""
import math
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import chain_v3 as cv                        # noqa: E402
import engine_v2 as eng                       # noqa: E402
import ledger_v2 as led                       # noqa: E402
from test_system_flowchart import parse       # noqa: E402

RATE = led.hedge_rate()                       # 12.396% — Put(10% OTM, 1y, σ 50%, r 4%) per $ of fix_c per Hedge×
CENT = 0.011                                  # the engine records money to the cent, every round


def loop(p0, prices, c0, hedge_x, *, skip=None, shrink=False):
    """The minimal system (c = fix_c, t = last price). ``skip`` removes one step to show it is needed."""
    c, t = c0, p0
    run = {"c": [c0], "H": 0.0, "K": 0.0, "S": [], "E": []}
    e = 0.0
    for p in prices:
        h = 0.0 if skip == "harvest" else c * math.log(p / t)     # ① Shannon's Demon
        k = 0.0 if skip == "pay" else RATE * c * hedge_x           # ② pay the insurance first
        s = h - k                                                  # ③ Surplus
        e += c * (p / t - 1 - math.log(p / t))                     # Eₙ — a by-product, never a step
        if skip != "grow":
            c += s if shrink else max(0.0, s)                      # ④ grow only from the surplus
        if skip != "recenter":
            t = p                                                  # ⑤ re-center → next round
        run["c"].append(c)
        run["H"] += h
        run["K"] += k
        run["S"].append(s)
        run["E"].append(e)
    return run


def engine(p0, prices, c0, hedge_x, every):
    sim = led.simulate_chain(p0, prices, c0, hedge_x, every)
    return sim, [prices[i - 1] for i in sim["steps"]]              # the prices the engine actually acted on


def paths():
    for seed in range(1, 13):
        for hedge_x in (0.5, 1.0, 2.0):
            for every in (1, 7, 20):
                yield seed, hedge_x, every, led.simulate_prices(100.0, 0.05, 0.0, 120, seed)


class MinimalLoopTest(unittest.TestCase):
    def test_black_scholes_is_one_constant(self):
        for c in (1.0, 500.0, 2300.0, 1e6):
            for t in (0.01, 1.0, 17.5, 4000.0):
                for hx in (0.0, 0.5, 1.0, 2.0, 3.0):
                    self.assertAlmostEqual(eng.hedge_cost(c, t, hx), RATE * c * hx, delta=1e-6)
        self.assertEqual(f"{RATE:.1%}", "12.4%")

    def test_loop_reproduces_the_engine(self):
        for seed, hx, every, prices in paths():
            sim, picked = engine(100.0, prices, 1500.0, hx, every)
            rounds, n = sim["ticker"]["rounds"], len(picked)
            mine = loop(100.0, picked, 1500.0, hx)
            tol = CENT * n
            tag = (seed, hx, every)
            self.assertAlmostEqual(sim["final_fix_c"], mine["c"][-1], delta=tol, msg=tag)               # fix_c
            self.assertAlmostEqual(sum(r["shannon_profit"] for r in rounds), mine["H"], delta=tol, msg=tag)
            self.assertAlmostEqual(sim["ev_burn"], mine["K"], delta=tol, msg=tag)                       # Ev burn = ΣK
            self.assertAlmostEqual(sim["ticker"]["current_state"]["baseline"], mine["H"], delta=tol, msg=tag)  # b = ΣH
            self.assertAlmostEqual(sim["ledger"]["summary"]["E"], mine["E"][-1], delta=tol, msg=tag)    # Eₙ

    def test_the_goal_clauses_hold_on_every_path(self):
        for seed, hx, every, prices in paths():
            sim, picked = engine(100.0, prices, 1500.0, hx, every)
            mine, rounds = loop(100.0, picked, 1500.0, hx), sim["ticker"]["rounds"]
            self.assertTrue(all(b >= a for a, b in zip(mine["c"], mine["c"][1:])), "fix_c never shrinks")
            grown = sum(max(0.0, s) for s in mine["S"])                       # Free Risk: only surplus adds capital
            self.assertAlmostEqual(mine["c"][-1] - 1500.0, grown, places=6)
            for r in rounds:
                self.assertAlmostEqual(r["scale_up"], max(0.0, r["surplus"]), delta=CENT)
            es = [r["E"] for r in sim["ledger"]["rows"] if r["kind"] == "act"]
            self.assertTrue(es[0] >= 0 and all(b >= a - 1e-9 for a, b in zip(es, es[1:])), "Eₙ ≥ 0 and never falls")
            if mine["K"] > 0:                                                  # Ev Efficiency ≥ 1  ⇔  ΣSurplus ≥ 0
                self.assertEqual(mine["H"] / mine["K"] >= 1.0, sum(mine["S"]) >= 0.0)

    def test_no_step_can_be_dropped(self):
        prices = led.simulate_prices(100.0, 0.04, 0.0, 120, 101)              # the Testing Lab default path
        sim, picked = engine(100.0, prices, 1500.0, 1.0, 20)
        full = loop(100.0, picked, 1500.0, 1.0)
        self.assertGreater(full["c"][-1], 1500.0 + 100)                       # on this path the loop does grow
        for label, kw in (("① harvest", {"skip": "harvest"}), ("② pay first", {"skip": "pay"}),
                          ("③ only surplus (no max 0)", {"shrink": True}), ("④ grow", {"skip": "grow"}),
                          ("⑤ re-center", {"skip": "recenter"})):
            self.assertGreater(abs(loop(100.0, picked, 1500.0, 1.0, **kw)["c"][-1] - full["c"][-1]), 1.0, label)


class ChainV3AgreesTest(unittest.TestCase):
    """app-v3 runs on chain_v3.py — its own few lines, none of engine_v2 — and lands on the same numbers as the engine."""

    def test_round_by_round_to_the_cent(self):
        for seed, hx, every, prices in paths():
            sim, picked = engine(100.0, prices, 1500.0, hx, every)
            rounds = [r for r in sim["ticker"]["rounds"] if r["action"] == "Chain Round"]
            self.assertEqual(len(rounds), len(picked))
            c, t = 1500.0, 100.0
            for p, r in zip(picked, rounds):
                pl = cv.plan(c, t, p, hx)
                for mine, theirs in (("H", "shannon_profit"), ("K", "hedge_cost"), ("S", "surplus"), ("c_after", "c_after")):
                    self.assertAlmostEqual(pl[mine], r[theirs], delta=CENT, msg=(seed, hx, every, p, mine))
                c, t = r["c_after"], p                     # continue from the engine's own state: one round at a time

    def test_the_whole_path_ends_in_the_same_place(self):
        for seed, hx, every, prices in paths():
            sim, picked = engine(100.0, prices, 1500.0, hx, every)
            c, t, h, k = 1500.0, 100.0, 0.0, 0.0
            for p in picked:
                pl = cv.plan(c, t, p, hx)
                c, t, h, k = pl["c_after"], p, h + pl["H"], k + pl["K"]
            tol, tag = CENT * len(picked), (seed, hx, every)
            self.assertAlmostEqual(c, sim["final_fix_c"], delta=tol, msg=tag)                                # fix_c
            self.assertAlmostEqual(h, sum(r["shannon_profit"] for r in sim["ticker"]["rounds"]), delta=tol, msg=tag)   # ΣH
            self.assertAlmostEqual(k, sim["ev_burn"], delta=tol, msg=tag)                                    # ΣK = Ev burn


class MinimalChartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minimal = (ROOT / "app_v2_minimal.mmd").read_text(encoding="utf-8")
        cls.system = (ROOT / "app_v2_system.mmd").read_text(encoding="utf-8")

    @staticmethod
    def goal(text):
        return re.search(r'^\s*GOAL\("(.*)"\)\s*$', text, re.M).group(1)

    def test_the_goal_is_word_for_word_the_same(self):
        self.assertEqual(self.goal(self.minimal), self.goal(self.system))

    def test_it_stays_small_and_keeps_the_five_principles(self):
        nodes, subgraphs, edges, depth = parse(self.minimal)
        self.assertLessEqual(len(nodes), 8)
        self.assertEqual((len(subgraphs), depth), (0, 0))
        for tag in ("P1", "P2", "P3", "P4", "P6"):
            self.assertRegex(self.minimal, rf"\b{tag}\b")
        for dropped in ("P5", "P7", "P8", "P9", "Cash Lanes", "Baseline b"):
            self.assertNotRegex(self.minimal.split("flowchart TB", 1)[1], rf"\b{dropped}\b")
        self.assertIn(("Q", "H"), edges)                                       # the chain loop is drawn


if __name__ == "__main__":
    unittest.main()
