"""app_v2_minimal.mmd — the whole goal as a loop of a few lines, checked against the LEGO ledger.

The minimal system keeps five steps + one check: harvest the swing (E) · pay first · grow only from the surplus ·
re-center, then Ev Efficiency = ΣE ÷ ΣK. The Surplus is read at E (E = A − R, the trend cut out). These tests show it is
  * the ledger's own E — step ① is, to the cent, the Eₙ that ledger_v2 (the streamlit_lego equations) computes
    from the same prices and capital, round by round and on a path of its own;
  * sufficient — the goal's clauses hold on any path: c never shrinks, only the surplus grows it (Free Risk),
    Eₙ ≥ 0 and never falls, Ev Efficiency ≥ 1 ⇔ ΣSurplus ≥ 0;
  * minimal    — dropping any of the five steps changes the outcome;
and that chain_v3.py (the code behind pages/app-v3.py) runs exactly this loop.

Note what is *not* claimed: reading the Surplus at E makes the capital path differ from app-v2's engine (which reads it
at R, Shannon's ln) — on purpose. The engine is used here only as a source of realistic (c, t, P) rounds.
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
import chain_v3 as cv                         # noqa: E402
import engine_v2 as eng                       # noqa: E402
import ledger_v2 as led                       # noqa: E402
from test_system_flowchart import parse       # noqa: E402

RATE = led.hedge_rate()                       # 12.396% — Put(10% OTM, 1y, σ 50%, r 4%) per $ of fix_c per Hedge×
CENT = 0.011                                  # the book records money to the cent, every round
MIXED = [200.0, 100.0, 110.0, 100.0, 90.0, 100.0, 180.0, 100.0]     # big swings and small ones: some rounds grow c, some do not


def loop(p0, prices, c0, hedge_x, *, skip=None, shrink=False):
    """The minimal system (c = fix_c, t = last price). ``skip`` removes one step to show it is needed."""
    c, t = c0, p0
    run = {"c": [c0], "E": 0.0, "K": 0.0, "S": [], "dE": []}
    for p in prices:
        x = p / t
        e = 0.0 if skip == "harvest" else c * (x - 1 - math.log(x))   # ① the swing: E = A − R, never negative
        k = 0.0 if skip == "pay" else RATE * c * hedge_x                # ② pay the insurance first
        s = e - k                                                       # ③ Surplus, read at E
        if skip != "grow":
            c += s if shrink else max(0.0, s)                           # ④ grow only from the surplus
        if skip != "recenter":
            t = p                                                       # ⑤ re-center → next round
        run["c"].append(c)
        run["E"] += e
        run["K"] += k
        run["S"].append(s)
        run["dE"].append(e)
    return run


def engine(p0, prices, c0, hedge_x, every):
    """A realistic source of rounds: what the app-v2 engine did on this path (its R-reading capital, not ours)."""
    sim = led.simulate_chain(p0, prices, c0, hedge_x, every)
    return sim, [prices[i - 1] for i in sim["steps"]]              # the prices the engine actually acted on


def paths():
    """Random walks (12 seeds) and the demon's own picture, a price that zig-zags — at 3 hedges and a few cadences."""
    for hedge_x in (0.5, 1.0, 2.0):
        for every in (1, 7, 20):
            for seed in range(1, 13):
                yield seed, hedge_x, every, led.simulate_prices(100.0, 0.05, 0.0, 120, seed)
        for every in (1, 7):
            for amp in (1.8, 2.5):                                  # 100 → 100·amp → 100 → …
                yield f"zig{amp}", hedge_x, every, [100.0 * amp if i % 2 == 0 else 100.0 for i in range(24)]


class MinimalLoopTest(unittest.TestCase):
    def test_black_scholes_is_one_constant(self):
        for c in (1.0, 500.0, 2300.0, 1e6):
            for t in (0.01, 1.0, 17.5, 4000.0):
                for hx in (0.0, 0.5, 1.0, 2.0, 3.0):
                    self.assertAlmostEqual(eng.hedge_cost(c, t, hx), RATE * c * hx, delta=1e-6)
        self.assertEqual(f"{RATE:.1%}", "12.4%")

    def test_step_one_is_the_ledgers_E_on_every_round(self):
        """The (c, t, P) of every round the engine ran → chain_v3's E equals the ledger's step in Eₙ = Aₙ − Rₙ."""
        rounds_seen = 0
        for seed, hx, every, prices in paths():
            sim, _ = engine(100.0, prices, 1500.0, hx, every)
            rounds = [r for r in sim["ticker"]["rounds"] if r["action"] == "Chain Round"]
            rows = [r for r in sim["ledger"]["rows"] if r["kind"] == "act"]
            self.assertEqual(len(rows), len(rounds))
            before = 0.0
            for r, row in zip(rounds, rows):
                mine = cv.plan(r["c_before"], r["p_old"], r["p_new"], hx)["E"]
                self.assertAlmostEqual(mine, row["E"] - before, delta=CENT, msg=(seed, hx, every, r["p_new"]))
                before = row["E"]
                rounds_seen += 1
        self.assertGreater(rounds_seen, 1000)

    def test_the_ledger_agrees_with_the_loop_on_its_own_path(self):
        """Run chain_v3's loop (capital grows by *its* rule), hand its rounds to ledger_v2, compare Eₙ at the end."""
        for seed, hx, every, prices in paths():
            _, picked = engine(100.0, prices, 1500.0, hx, every)
            c, t, e_sum, rounds = 1500.0, 100.0, 0.0, []
            for p in picked:
                pl = cv.plan(c, t, p, hx)
                rounds.append({"action": "Chain Round", "p_old": t, "p_new": p, "c_before": c, "c_after": pl["c_after"]})
                c, t, e_sum = pl["c_after"], p, e_sum + pl["E"]
            lego = led.build({"rounds": rounds})["summary"]
            self.assertAlmostEqual(lego["E"], e_sum, delta=CENT * len(picked), msg=(seed, hx, every))

    def test_the_goal_clauses_hold_on_every_path(self):
        grew = 0
        for seed, hx, every, prices in paths():
            _, picked = engine(100.0, prices, 1500.0, hx, every)
            mine = loop(100.0, picked, 1500.0, hx)
            tag = (seed, hx, every)
            self.assertTrue(all(b >= a for a, b in zip(mine["c"], mine["c"][1:])), tag)       # fix_c never shrinks
            grown = sum(max(0.0, s) for s in mine["S"])                                       # Free Risk: only surplus adds capital
            self.assertAlmostEqual(mine["c"][-1] - 1500.0, grown, places=6, msg=tag)
            self.assertTrue(all(e >= 0 for e in mine["dE"]), tag)                             # E ≥ 0 (x − 1 ≥ ln x) …
            es = [sum(mine["dE"][:i]) for i in range(len(mine["dE"]) + 1)]
            self.assertTrue(es[0] >= 0 and all(b >= a for a, b in zip(es, es[1:])), tag)      # … so Eₙ never falls
            if mine["K"] > 0:                                                                 # Ev Efficiency ≥ 1  ⇔  ΣSurplus ≥ 0
                self.assertEqual(mine["E"] / mine["K"] >= 1.0, sum(mine["S"]) >= 0.0, tag)
            grew += mine["c"][-1] > 1500.0
        self.assertGreater(grew, 10)                                                          # not vacuous: capital really grew

    def test_no_step_can_be_dropped(self):
        full = loop(100.0, MIXED, 1500.0, 1.0)
        self.assertGreater(full["c"][-1], 1500.0 + 100)                       # on this path the loop does grow …
        self.assertTrue(any(s < 0 for s in full["S"]) and any(s > 0 for s in full["S"]))      # … and some rounds do not pay
        for label, kw in (("① harvest", {"skip": "harvest"}), ("② pay first", {"skip": "pay"}),
                          ("③ only surplus (no max 0)", {"shrink": True}), ("④ grow", {"skip": "grow"}),
                          ("⑤ re-center", {"skip": "recenter"})):
            self.assertGreater(abs(loop(100.0, MIXED, 1500.0, 1.0, **kw)["c"][-1] - full["c"][-1]), 1.0, label)

    def test_the_swing_pays_not_the_trend(self):
        """A steady climb to 2× harvests almost nothing; the same price going up and coming back harvests a lot."""
        steady = [100.0 * 2 ** (i / 100) for i in range(1, 101)]
        climb = loop(100.0, steady, 1000.0, 0.0)["E"]
        jump = loop(100.0, [200.0], 1000.0, 0.0)["E"]
        back = loop(100.0, [200.0, 100.0], 1000.0, 0.0)["E"]
        self.assertLess(climb, 3.0)
        self.assertAlmostEqual(jump, 306.85, places=2)
        self.assertGreater(back, 500.0)                                                      # the price ends where it began
        self.assertGreater(back, 100 * climb)

    def test_the_break_even_prices_are_the_ledgers(self):
        for hx in (0.25, 0.5, 1.0, 2.0, 3.0):
            ledger = led.break_even(hx)
            low, high = cv.breakeven_prices(1000, 100, hx)
            self.assertAlmostEqual(high / 100 - 1, ledger["E_up"], places=6, msg=hx)
            self.assertAlmostEqual(low / 100 - 1, ledger["E_down"], places=6, msg=hx)
        self.assertEqual((f"{led.break_even(1.0)['E_up']:+.0%}", f"{led.break_even(1.0)['E_down']:+.0%}"), ("+58%", "-42%"))


class ChainV3IsTheLoopTest(unittest.TestCase):
    """chain_v3.py — the code behind app-v3 — runs exactly the flowchart's loop (to the cent it keeps)."""

    def test_one_round_at_a_time(self):
        """From the *same* (c, t) the module and the flowchart's loop agree on E, K, S and c′ to the cent."""
        for seed, hx, every, prices in paths():
            _, picked = engine(100.0, prices, 1500.0, hx, every)
            c, t = 1500.0, 100.0
            for p in picked:
                pl, ref = cv.plan(c, t, p, hx), loop(t, [p], c, hx)
                tag = (seed, hx, every, p)
                self.assertAlmostEqual(pl["E"], ref["dE"][0], delta=CENT, msg=tag)
                self.assertAlmostEqual(pl["K"], RATE * c * hx, delta=CENT, msg=tag)
                self.assertAlmostEqual(pl["S"], ref["S"][0], delta=2 * CENT, msg=tag)
                self.assertAlmostEqual(pl["c_after"], ref["c"][-1], delta=2 * CENT, msg=tag)
                c, t = pl["c_after"], p                                      # the book's own state carries on

    def test_the_whole_path_ends_in_the_same_place(self):
        """Cents lost to rounding are compounded by growth, so the bound is relative: well under 0.1% of the end value."""
        for seed, hx, every, prices in paths():
            _, picked = engine(100.0, prices, 1500.0, hx, every)
            mine = loop(100.0, picked, 1500.0, hx)
            c, t, e_sum, k_sum = 1500.0, 100.0, 0.0, 0.0
            for p in picked:
                pl = cv.plan(c, t, p, hx)
                c, t, e_sum, k_sum = pl["c_after"], p, e_sum + pl["E"], k_sum + pl["K"]
            tol = CENT * len(picked) + 1e-4 * mine["c"][-1]
            self.assertAlmostEqual(c, mine["c"][-1], delta=tol, msg=(seed, hx, every))            # fix_c
            self.assertAlmostEqual(e_sum, mine["E"], delta=tol, msg=(seed, hx, every))            # ΣE
            self.assertAlmostEqual(k_sum, mine["K"], delta=tol, msg=(seed, hx, every))            # ΣK


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

    def test_it_stays_small_and_keeps_its_six_principles(self):
        nodes, subgraphs, edges, depth = parse(self.minimal)
        self.assertLessEqual(len(nodes), 8)
        self.assertEqual((len(subgraphs), depth), (0, 0))
        body = self.minimal.split("flowchart TB", 1)[1]
        for tag, name in (("P1", "Shannon's Demon"), ("P2", "Pay-first"), ("P3", "Free-Risk Scale-up"), ("P4", "Continuous Rollover"),
                          ("P6", "Ev Efficiency"), ("P8", "อ่านกำไรที่ E")):         # P8: the Surplus is read at E
            self.assertIn(f"{tag} {name}", body)
        for dropped in ("P5", "P7", "P9", "Cash Lanes", "Baseline b"):
            self.assertNotRegex(body, rf"\b{dropped}\b")
        self.assertIn(("Q", "H"), edges)                                       # the chain loop is drawn

    def test_the_chart_reads_the_surplus_at_e(self):
        body = self.minimal.split("flowchart TB", 1)[1]
        for text in ("E = c · (x − 1 − ln x)", "Surplus = E − K", "ΣE ÷ ΣK ≥ 1.0x"):
            self.assertIn(text, body)
        for old in ("ln(P / t)", "H − K", "ΣH"):                               # the earlier reading (at R) is gone from the nodes
            self.assertNotIn(old, body)


if __name__ == "__main__":
    unittest.main()
