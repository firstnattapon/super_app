"""chain_v4 — the standalone logic of app-v4 (the module is stdlib only: no Streamlit, no other module of this repo).

ledger_v2 (the LEGO equations of streamlit_lego, mirrored) is used here ONLY as an independent oracle — chain_v4 never imports it.
"""
import ast
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import chain_v4 as cv  # noqa: E402
import ledger_v2 as lego  # noqa: E402

CENT = 0.011


def move(p, t="TSLA"):
    return {"do": "move", "ticker": t, "p": p}


def pay(t="TSLA", hedge=1, k=None):
    return {"do": "pay", "ticker": t, "hedge": hedge, "k": k}


def cash(src, dst, amount):
    return {"do": "cash", "src": src, "dst": dst, "amount": amount}


def scale(t="TSLA", **kw):
    return {"do": "scale", "ticker": t, "by": "liquidity", "shield": True, **kw}


def book(*events):
    return {"version": cv.VERSION, "chain": [cv._event(e) for e in events]}


def run(*events):
    return cv.fold(book(*events))


OPEN = {"do": "open", "ticker": "TSLA", "c": 1500, "t": 10}
FUND = cash(cv.OUT, cv.RESERVE, 300)
# The user's case: fix_c 1,500 · t0 10 → t1 15 → back to 10 · Hedge ×1 · 300 in EV Reserve · the capital gain pulled out at 15
USERS_CASE = [OPEN, FUND, pay(), move(15), cash("TSLA", cv.POOL, 608.2), move(10)]


def black_scholes_put_rate() -> float:
    """An independent Black-Scholes: European put, S = 1, K = 0.9, T = 1 year, σ = 50%, r = 4%."""
    s, k, t, r, sigma = 1.0, 0.9, 1.0, 0.04, 0.5
    ncdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))            # noqa: E731
    d1 = (math.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    return k * math.exp(-r * t) * ncdf(-(d1 - sigma * math.sqrt(t))) - s * ncdf(-d1)


def grow(seed, steps=None):
    """A random legal chain over 1–3 tickers: the rules themselves decide what is legal. Returns (events, ledger)."""
    rnd = random.Random(seed)
    names = ["A", "B", "C"][: rnd.randint(1, 3)]
    events = [{"do": "open", "ticker": n, "c": rnd.choice([1000, 1500, 2300]), "t": round(rnd.uniform(5, 60), 2)} for n in names]
    events.append(cash(cv.OUT, cv.RESERVE, rnd.choice([1, 500, 2000])))
    live = cv.fold(book(*events))
    price = {n: live["tickers"][n]["p0"] for n in names}
    for _ in range(steps or rnd.randint(10, 70)):
        n = rnd.choice(names)
        t, r = live["tickers"][n], rnd.random()
        if not t["open"]:
            ev = pay(n, rnd.choice([0.0, 0.5, 1, 2]))
        elif cv.surplus(t) > 0 and r < 0.6:
            ev = scale(n, by=rnd.choice(cv.FUNDING), settle=rnd.random() < 0.8, pull=rnd.random() < 0.7)
        elif t["b"] > 1 and r < 0.72:
            ev = cash(n, cv.POOL, round(rnd.uniform(0.01, t["b"]), 2))
        elif live["pool"] > 1 and r < 0.78:
            ev = cash(cv.POOL, cv.RESERVE, round(rnd.uniform(0.01, live["pool"]), 2))
        else:
            price[n] = max(0.05, price[n] * math.exp(rnd.gauss(0, 0.2)))
            ev = move(round(price[n], 4), n)
        try:
            cv.apply(live, cv._event(ev))
        except cv.BookError:
            continue
        events.append(ev)
    return events, cv.fold(book(*events))


def lego_rounds(led, name):
    """One ticker's acts as rounds for the LEGO ledger (a scale-up is the capital change of the act it follows)."""
    rounds = []
    for r in led["rows"]:
        if r.get("ticker") != name:
            continue
        if r["do"] == "move":
            rounds.append({"action": "Chain Round", "p_old": r["pa"], "p_new": r["p"], "c_before": r["c"], "c_after": r["c"]})
        elif r["do"] == "scale":
            rounds[-1]["c_after"] = r["c_new"]
    return rounds


class UsersCaseTest(unittest.TestCase):
    """fix_c 1,500 · t0 10 → t1 15 → 10: every number is the one quoted in the plan."""

    def test_the_numbers_step_by_step(self):
        led = run(OPEN, FUND, pay())
        t = led["tickers"]["TSLA"]
        self.assertEqual((t["K"], led["reserve"], cv.totals(led)["burn"]), (185.94, 114.06, 185.94))      # ① K = 12.4% · 1,500
        self.assertEqual(cv.phase(t), "swing")

        led = run(OPEN, FUND, pay(), move(15))
        row, t = led["rows"][-1], led["tickers"]["TSLA"]
        self.assertAlmostEqual(row["v"], 2250.0)                                                            # V = Q·P = 150 × 15
        self.assertAlmostEqual(row["gap"], 750.0)                                                           # V − FIX_C: over target → SELL
        self.assertEqual((row["side"], round(row["qty"], 6)), ("SELL", 50.0))
        self.assertAlmostEqual(row["r"], 1500 * math.log(1.5))                                              # R = 608.20 (the trend)
        self.assertAlmostEqual(row["d_a"], 750.0)
        self.assertAlmostEqual(row["a"], 750.0)
        self.assertAlmostEqual(row["e"], 141.80, delta=CENT)                                                # E = A − R (the swing)
        self.assertEqual((row["e_cyc"], row["s"], t["b"]), (141.80, -44.14, 608.20))                        # ③ not yet
        self.assertEqual(cv.phase(t), "swing")

        led = run(*USERS_CASE)
        row, t = led["rows"][-1], led["tickers"]["TSLA"]
        self.assertAlmostEqual(row["v"], 1000.0)
        self.assertEqual((row["side"], round(row["qty"], 6)), ("BUY", 50.0))
        self.assertAlmostEqual(row["r"], 0.0, places=9)                                                     # the price is back: R = 0
        self.assertAlmostEqual(row["d_a"], -500.0)
        self.assertAlmostEqual(row["a"], 250.0)
        self.assertAlmostEqual(row["e"], 250.0, places=9)                                                   # E = A = 250.00 (141.80 + 108.20)
        self.assertEqual((row["e_cyc"], row["s"], cv.surplus(t), cv.phase(t)), (250.0, 64.06, 64.06, "scale"))
        self.assertEqual((led["pool"], led["reserve"], t["b"]), (608.2, 114.06, -608.2))                    # b went negative: it is owed back

    def test_scale_up_settles_the_cycle_and_the_swing_paid_the_insurance(self):
        ev = scale()
        self.assertEqual(cv.recommend_funding(run(*USERS_CASE), ev), "liquidity")                           # 908.20 left ≥ K′ 193.88
        led = run(*USERS_CASE, ev)
        t, row = led["tickers"]["TSLA"], led["rows"][-1]
        self.assertEqual((t["c"], cv.next_k(t)), (1564.06, 193.88))                                         # c′ = c + S · K′ of the new capital
        self.assertEqual(led["reserve"], 300.0)                                                             # EV Reserve is whole again
        self.assertEqual(led["pool"], 608.2)                                                                # only the trend cash remains
        self.assertAlmostEqual(t["beta"], 0.0)                                                              # scaled at P_acted = P₀
        self.assertEqual((row["s"], row["pulled"], t["open"], cv.phase(t)), (64.06, 0.0, False, "pay"))     # nothing to pull: b < 0
        self.assertAlmostEqual(cv.efficiency(t), 250.0 / 185.94)                                            # 1.34x ≥ 1.0 → the goal
        self.assertEqual(cv.totals(led)["net_in"], 300.0)                                                   # the only fresh money is the 300 start

    def test_deep_itm_keeps_the_surplus_as_liquidity_and_settle_can_be_off(self):
        led = run(*USERS_CASE, scale(by="deep_itm"))
        self.assertEqual((led["pool"], led["reserve"], led["tickers"]["TSLA"]["c"]), (672.26, 300.0, 1564.06))   # S rests in Pool CF
        led = run(*USERS_CASE, scale(settle=False))
        self.assertEqual((led["pool"], led["reserve"]), (544.14, 114.06))                                   # S out of Pool, K not paid back
        led = run(*USERS_CASE, scale(by="deep_itm", settle=False))
        self.assertEqual((led["pool"], led["reserve"]), (608.2, 114.06))

    def test_the_pull_takes_the_capital_gain_into_pool_cf(self):
        led = run(OPEN, FUND, pay(), move(15), move(10), scale())                                           # up and back: the gain is gone
        self.assertEqual((led["tickers"]["TSLA"]["b"], led["pool"], led["rows"][-1]["pulled"]), (0.0, 0.0, 0.0))
        climb = [OPEN, FUND, pay(), move(15), move(30)]                                                     # a one-way climb: a gain to pull
        led = run(*climb, scale(by="deep_itm"))
        row, t = led["rows"][-1], led["tickers"]["TSLA"]
        self.assertAlmostEqual(row["pulled"], 1500 * math.log(3), delta=CENT)                               # R = 1,647.92
        self.assertEqual(t["b"], 0.0)
        self.assertAlmostEqual(led["pool"], row["s"] + row["pulled"], delta=CENT)                           # S rests in Pool + the gain
        kept = run(*climb, scale(by="deep_itm", pull=False))
        self.assertAlmostEqual(kept["tickers"]["TSLA"]["b"], 1500 * math.log(3), delta=CENT)                # still in Baseline
        self.assertEqual(kept["pool"], kept["rows"][-1]["s"])

    def test_every_number_comes_with_its_working(self):
        led = run(*USERS_CASE, scale())
        move15 = {s: (how, res) for s, how, res in cv.explain(led["rows"][3])}
        self.assertEqual(move15["ΔAₙ"], ("FIX_C × (Pₙ / P_acted − 1) = 1,500.00 × (15.0000 / 10.0000 − 1)", "+750.00"))
        self.assertEqual(move15["Eₙ"], ("Aₙ − Rₙ = 750.00 − 608.20", "141.80"))
        self.assertEqual(move15["gap"], ("Vₙ − FIX_C = 2,250.00 − 1,500.00", "+750.00 → SELL ≈ 50.0000 หุ้น"))
        self.assertEqual(move15["S"], ("E_cyc − K = 141.80 − 185.94", "−44.14"))
        text = " ".join(f"{a} {b} {c}" for r in led["rows"] for a, b, c in cv.explain(r))
        self.assertNotIn("−0.00", text)
        self.assertNotRegex(text, r"\bnan\b|\binf\b")
        self.assertEqual([r["do"] for r in led["rows"]], ["open", "cash", "pay", "move", "cash", "move", "scale"])
        for seed in range(30):
            for r in grow(seed)[1]["rows"]:
                self.assertTrue(cv.explain(r), r["do"])                                                     # every kind of row can be explained


class LegoEquationsTest(unittest.TestCase):
    """The ledger is the LEGO ledger: the guide of streamlit_lego, ledger_v2 as an oracle, the chain-form shortcut."""

    def test_the_example_in_the_lego_guide(self):
        """Rebalancing 101, chapter 4 (Fix_c 1,500): 100 → 110 → 100. The guide prints R 142.96 · E 7.04 (rounded by hand)."""
        start = [{"do": "open", "ticker": "G", "c": 1500, "t": 100}, pay("G", 0)]
        r = run(*start, move(110, "G"))["rows"][-1]
        self.assertAlmostEqual(r["d_a"], 150.00, delta=CENT)
        self.assertAlmostEqual(r["r"], 142.96, delta=CENT)
        self.assertAlmostEqual(r["e"], 7.04, delta=CENT)
        r = run(*start, move(110, "G"), move(100, "G"))["rows"][-1]
        self.assertAlmostEqual(r["d_a"], -136.36, delta=CENT)
        self.assertAlmostEqual(r["a"], 13.64, delta=CENT)
        self.assertAlmostEqual(r["r"], 0.0, places=9)                                                        # the price is back: R = 0
        self.assertAlmostEqual(r["e"], 13.64, delta=CENT)                                                    # E = A: locked as cash

    def test_a_steady_climb_harvests_almost_nothing_but_a_swing_harvests_a_lot(self):
        """Guide chapter 4: one-way, A sits on R and E ≈ 0. The same rise in few steps harvests more (caveat 1 of the audit)."""
        got = {}
        for n in (1, 2, 10, 1000):
            steps = [move(10 * 1.5 ** (i / n), "G") for i in range(1, n + 1)]
            got[n] = run({"do": "open", "ticker": "G", "c": 1500, "t": 10}, pay("G", 0), *steps)["tickers"]["G"]
        self.assertEqual([round(got[n]["E"], 2) for n in (1, 2, 10)], [141.80, 66.04, 12.50])
        self.assertAlmostEqual(got[1000]["E"], 0.12, delta=0.01)
        self.assertAlmostEqual(got[1000]["A"], got[1000]["R"], delta=1.0)                                    # A sits on R
        self.assertGreater(got[1000]["A"], 600)

    def test_rollover_keeps_r_continuous_and_e_honest(self):
        """streamlit_lego/cheat_sheet/rollover.mmd: β += (Fix_c − Fix_c′)·ln(P_acted/P₀), added to R only."""
        start = [{"do": "open", "ticker": "G", "c": 1500, "t": 100}, pay("G", 0), move(300, "G")]
        before = run(*start)["tickers"]["G"]
        led = run(*start, scale("G", by="deep_itm"))
        t = led["tickers"]["G"]
        self.assertGreater(t["c"], 2800)                                                                     # S = 1,352.09 joined the capital
        self.assertAlmostEqual(t["beta"], (1500 - t["c"]) * math.log(3))
        self.assertAlmostEqual(t["beta"] + t["c"] * math.log(3), before["R"])                                # R at the same price: no jump
        self.assertEqual(led["rows"][-1]["after"]["E"], before["E"])                                         # E unchanged by the scale-up
        n = run(*start, scale("G", by="deep_itm"), pay("G", 0), move(330, "G"))["tickers"]["G"]
        self.assertAlmostEqual(n["E"] - t["E"], t["c"] * (1.1 - 1 - math.log(1.1)))                          # the next swing, on the new capital
        without_beta = n["A"] - n["c"] * math.log(3.3)                                                       # what E would be if β were forgotten
        self.assertLess(without_beta, 0)
        self.assertGreaterEqual(n["E"], 0)
        self.assertAlmostEqual(n["E"] - without_beta, -t["beta"])                                            # β is exactly the jump it removes

    def test_random_chains_obey_the_ledger_equations(self):
        acts = scaled = 0
        for seed in range(150):
            events, led = grow(seed)
            tol = 0.006 * (1 + len(events))
            for name, t in led["tickers"].items():
                rows = [r for r in led["rows"] if r.get("ticker") == name and r["do"] in ("move", "scale", "pay", "open")]
                es = [r["after"]["E"] for r in rows]
                self.assertTrue(all(b >= a - 1e-9 for a, b in zip(es, es[1:])), (seed, name))                # E never falls …
                self.assertGreaterEqual(min(es), -1e-9, (seed, name))                                        # … and is never negative
                for r in (x for x in rows if x["do"] == "move"):
                    self.assertAlmostEqual(r["r"], r["beta"] + r["c"] * math.log(r["p"] / r["p0"]), places=9)   # R = β + FIX_C·ln(P/P₀)
                    self.assertAlmostEqual(r["e"], r["a"] - r["r"], places=9)                                # E = A − R
                    self.assertAlmostEqual(r["d_a"], r["c"] * (r["p"] / r["pa"] - 1), places=9)              # ΔA = FIX_C·(P/P_acted − 1)
                    self.assertAlmostEqual(r["gap"], r["d_a"], places=9)                                     # the order is the cash: ΔA = gap
                chain_form = sum(r["c"] * (r["p"] / r["pa"] - 1 - math.log(r["p"] / r["pa"])) for r in rows if r["do"] == "move")
                self.assertAlmostEqual(t["E"], chain_form, places=6)                                         # the shortcut c·(x − 1 − ln x)
                rounds = lego_rounds(led, name)
                if rounds:
                    oracle = lego.build({"rounds": rounds})
                    self.assertAlmostEqual(oracle["summary"]["E"], t["E"], places=6)                         # the LEGO ledger agrees: E …
                    self.assertAlmostEqual(oracle["summary"]["A"], t["A"], places=6)                         # … A …
                    self.assertAlmostEqual(oracle["summary"]["R"], t["R"], places=6)                         # … R …
                    self.assertAlmostEqual(oracle["rows"][-1]["beta_after"], t["beta"], places=6)            # … and the rollover memory β
                self.assertAlmostEqual(t["b"] - t["b0"] + t["pulled"], t["R"], delta=tol)                    # Baseline = R − what was pulled out
                self.assertAlmostEqual(t["c"] - t["c0"], t["scaled"], delta=tol)                             # Free Risk: capital grew by Σ S only
                open_s = cv.surplus(t) if t["open"] else 0.0
                self.assertAlmostEqual(t["E"] - t["sumK"], t["scaled"] + open_s, delta=tol)                  # ΣE − ΣK = ΣS + the open cycle's S
                eff = cv.efficiency(t)
                self.assertEqual(eff is not None and eff >= 1.0, t["sumK"] > 0 and t["E"] >= t["sumK"])      # Ev Efficiency ≥ 1 ⇔ ΣE ≥ ΣK
                acts += t["moves"]
                scaled += t["scales"]
            self.assertGreaterEqual(led["pool"], 0.0)
        self.assertGreater(acts, 1500)                                                                        # not vacuous
        self.assertGreater(scaled, 80)

    def test_replay_equals_building_step_by_step(self):
        for seed in range(40):
            events, led = grow(seed)
            live = cv.fold(book())
            for e in events:
                cv.apply(live, cv._event(e))
            self.assertEqual((live["pool"], live["reserve"], live["net_in"]), (led["pool"], led["reserve"], led["net_in"]))
            self.assertEqual(live["tickers"], led["tickers"])

    def test_the_premium_constant_is_black_scholes(self):
        self.assertAlmostEqual(cv.PREMIUM_RATE, black_scholes_put_rate(), places=9)
        self.assertEqual(f"{cv.PREMIUM_RATE:.1%}", "12.4%")


class RulesTest(unittest.TestCase):
    def refuse(self, events, pattern, label=""):
        with self.assertRaisesRegex(cv.BookError, pattern, msg=label):
            run(*events)

    def test_pay_first_is_not_a_check_it_is_the_only_way_in(self):
        self.refuse([OPEN, move(11)], "ต้องจ่าย K ก่อน")
        self.refuse([OPEN, pay(), pay()], "ไม่จ่ายซ้ำ")
        self.refuse([OPEN, pay(), move(11), scale()], "Surplus ยังไม่เป็นบวก")                                  # K comes before any E
        self.refuse([OPEN, scale()], "ยังไม่ได้จ่าย K")
        led = run(OPEN, pay(), move(9), move(10), move(11))                                                    # E keeps accumulating, K is not paid again
        self.assertEqual(led["tickers"]["TSLA"]["sumK"], 185.94)

    def test_a_cycle_ends_only_by_scaling_up_and_then_k_is_paid_again(self):
        steps = [OPEN, FUND, pay(), move(15), move(10), scale()]
        self.refuse(steps + [move(11)], "ต้องจ่าย K ก่อน")                                                      # ↻ ① K of the new capital first
        led = run(*steps, pay(), move(12))
        self.assertAlmostEqual(led["tickers"]["TSLA"]["sumK"], 185.94 + 193.88)                                # K′ = 12.4% × 1,564.06
        self.assertEqual(led["rows"][-2]["K"], 193.88)

    def test_shield_is_required_and_funding_is_one_of_two(self):
        for bad in ({"do": "scale", "ticker": "A", "by": "liquidity"}, scale("A", shield=False), scale("A", shield="yes"),
                    scale("A", by="sell_the_house"), scale("A", settle=1), scale("A", pull="no")):
            with self.assertRaises(cv.BookError, msg=repr(bad)):
                cv._event(bad)

    def test_capital_never_shrinks(self):
        for seed in range(60):
            _, led = grow(seed)
            for name, t in led["tickers"].items():
                caps = [r["after"]["c"] for r in led["rows"] if r.get("ticker") == name and r.get("after")]
                self.assertTrue(all(b >= a for a, b in zip(caps, caps[1:])), (seed, name))
                self.assertGreaterEqual(t["c"], t["c0"])

    def test_the_lanes_and_who_may_talk_to_whom(self):
        base = [OPEN, FUND]
        for e in (cash(cv.OUT, cv.POOL, 5), cash(cv.OUT, cv.RESERVE, 5), cash(cv.RESERVE, cv.OUT, 5), cash(cv.RESERVE, cv.POOL, 5)):
            run(*base, e)
        self.refuse(base + [cash(cv.POOL, cv.OUT, 1)], "Pool CF ไม่พอ")
        self.refuse(base + [cash("TSLA", cv.POOL, 1)], "Baseline ของ TSLA ไม่พอ")
        self.refuse(base + [cash(cv.OUT, "TSLA", 1)], "เส้นทางนี้ไม่มีในระบบ")                                    # outside money does not enter a Baseline
        self.refuse(base + [cash("TSLA", cv.OUT, 1)], "เส้นทางนี้ไม่มีในระบบ")
        self.refuse(base + [cash("TSLA", cv.RESERVE, 1)], "เส้นทางนี้ไม่มีในระบบ")
        self.refuse(base + [cash(cv.POOL, cv.POOL, 1)], "คนละกระเป๋า")
        self.refuse(base + [cash("NOPE", cv.POOL, 1)], "ไม่พบ")
        self.refuse(base + [cash(cv.OUT, cv.POOL, 0.001)], "มากกว่า 0.00")
        for bad in (0, -5, "x", None, True, float("nan"), float("inf")):
            with self.assertRaises(cv.BookError, msg=repr(bad)):
                cv._event(cash(cv.OUT, cv.POOL, bad))

    def test_cash_moves_money_it_never_makes_or_loses_it(self):
        holds = lambda L: L["pool"] + L["reserve"] + sum(t["b"] for t in L["tickers"].values()) - L["net_in"]     # noqa: E731
        led = run(OPEN, cash(cv.OUT, cv.POOL, 100), cash(cv.OUT, cv.RESERVE, 100), pay(), move(15))             # Pool 100 · Reserve −85.94 · b 608.20
        for src, dst in [(cv.OUT, cv.POOL), (cv.OUT, cv.RESERVE), (cv.POOL, cv.OUT), (cv.RESERVE, cv.OUT), (cv.POOL, cv.RESERVE),
                         (cv.RESERVE, cv.POOL), ("TSLA", cv.POOL), (cv.POOL, "TSLA")]:
            after, row = cv.preview(led, cash(src, dst, 40))
            self.assertAlmostEqual(holds(after), holds(led), places=9, msg=(src, dst))                            # one subtraction, one addition
            self.assertEqual(row["amount"], 40.0)
        after = cv.preview(led, cash("TSLA", cv.POOL, 100))[0]                                                    # Extract …
        again = cv.preview(after, cash(cv.POOL, "TSLA", 100))[0]                                                  # … and Deploy undo each other
        self.assertEqual((again["tickers"]["TSLA"]["b"], again["pool"]), (led["tickers"]["TSLA"]["b"], led["pool"]))
        self.assertEqual(led["pool"], 100.0)                                                                      # preview changed nothing

    def test_ev_reserve_may_go_negative_the_pool_may_not(self):
        led = run(OPEN, pay())                                                                                 # insurance paid before any money was put in
        self.assertEqual(led["reserve"], -185.94)
        self.refuse([OPEN, cash(cv.POOL, cv.RESERVE, 1)], "Pool CF ไม่พอ")
        for seed in range(60):
            self.assertGreaterEqual(grow(seed)[1]["pool"], 0.0)

    def test_burn_is_the_sum_of_what_every_ticker_paid_and_efficiency_ignores_the_opening_burn(self):
        led = run({**OPEN, "burn": 500}, {"do": "open", "ticker": "B", "c": 1000, "t": 50, "burn": 100}, pay(), pay("B", 2))
        self.assertAlmostEqual(cv.totals(led)["burn"], 500 + 100 + 185.94 + 247.92)
        self.assertAlmostEqual(cv.totals(led)["sum_k"], 185.94 + 247.92)                                      # only what this book measured
        self.assertIsNone(cv.efficiency(run({**OPEN, "burn": 500})["tickers"]["TSLA"]))                        # nothing measured yet

    def test_a_real_premium_replaces_the_formula(self):
        led = run(OPEN, pay(k=190.5))
        self.assertEqual((led["tickers"]["TSLA"]["K"], led["reserve"], led["rows"][-1]["real"]), (190.5, -190.5, True))
        led = run(OPEN, pay(hedge=0))
        self.assertEqual(led["tickers"]["TSLA"]["K"], 0.0)                                                    # no insurance: the surplus is all E

    def test_names_and_numbers_are_checked(self):
        for bad in ("", "   ", "x" * 21, "a*b", "a_b", "[x](y)", "<b>", "a|b", "$x", "a#b", "a&b", "tab\tname", None, "@pool", "a@b"):
            with self.assertRaises(cv.BookError, msg=repr(bad)):
                cv._event({**OPEN, "ticker": bad})
        run({**OPEN, "ticker": "หุ้น ก."}, {**OPEN, "ticker": "BRK.B"}, {**OPEN, "ticker": "^GSPC"})
        self.refuse([OPEN, {**OPEN, "ticker": "tsla"}], "ชื่อนี้มีอยู่แล้ว")                                     # duplicates, ignoring case
        for key, bad in (("c", 0), ("c", -1), ("t", 0), ("c", "x"), ("t", float("nan")), ("c", float("inf")), ("c", True), ("b", "x"),
                         ("burn", -1)):
            with self.assertRaises(cv.BookError, msg=(key, bad)):
                cv._event({**OPEN, key: bad})
        for bad in ({"do": "pay", "ticker": "A", "hedge": -1}, {"do": "pay", "ticker": "A", "k": -1}, {"do": "move", "ticker": "A", "p": 0},
                    {"do": "move", "ticker": "A"}, {"do": "fly", "ticker": "A"}, None, [], "open", {"ticker": "A"}):
            with self.assertRaises(cv.BookError, msg=repr(bad)):
                cv._event(bad)

    def test_absurd_numbers_are_a_message_not_a_crash(self):
        for events in ([{**OPEN, "c": 1e308}, pay(), move(5e300)], [{**OPEN, "t": 1e-300}, pay(), move(1e300)],
                       [OPEN, pay(), move(1e-300)], [{**OPEN, "c": 1e308}, pay(hedge=1e308)]):
            try:
                run(*events)
            except cv.BookError:
                pass                                                                                          # a message, never OverflowError

    def test_every_kind_of_step_is_needed(self):
        """The fixed point of the design: take any one kind away and a part of the goal is gone."""
        full = [OPEN, FUND, pay(), move(15), move(10), scale()]
        self.assertGreater(run(*full)["tickers"]["TSLA"]["c"], 1500)
        self.assertEqual(run(*[e for e in full if e["do"] != "scale"])["tickers"]["TSLA"]["c"], 1500)          # no scale-up: no growth
        self.refuse([e for e in full if e["do"] != "pay"], "ต้องจ่าย K ก่อน")                                  # no pay: no E at all
        self.refuse([e for e in full if e["do"] != "move"], "Surplus ยังไม่เป็นบวก")                           # no move: no surplus to grow from
        self.refuse([e for e in full if e["do"] != "open"], "ไม่พบ")                                           # no open: no ticker
        self.assertEqual(run(OPEN, pay(), move(15))["pool"], 0.0)                                              # only cash (or a scale-up) fills Pool CF

    def test_the_recommendation_is_judged_on_what_is_left(self):
        up = [pay(), move(15), move(10)]                                                                      # S = 250 − 185.94 = 64.06 · K′ = 193.88
        cases = [(0, "deep_itm"), (150, "deep_itm"), (193, "deep_itm"), (194, "liquidity"), (400, "liquidity")]    # Reserve funded with …
        for funded, want in cases:
            led = run(OPEN, *([cash(cv.OUT, cv.RESERVE, funded)] if funded else []), *up)
            self.assertEqual(cv.recommend_funding(led, scale()), want, funded)                                # … K comes back, S is spent: funded is left
        led = run(OPEN, cash(cv.OUT, cv.RESERVE, 400), *up)
        self.assertEqual(cv.recommend_funding(led, scale(settle=False)), "deep_itm")                          # without settling there is no cash for S
        led = run(OPEN, cash(cv.OUT, cv.RESERVE, 400), cash(cv.OUT, cv.POOL, 100), *up)
        self.assertEqual(cv.recommend_funding(led, scale(settle=False)), "liquidity")                         # 100 − 64.06 + 214.06 = 250 ≥ 193.88
        self.assertEqual(cv.recommend_funding(led, scale(settle=False, by="deep_itm")), "liquidity")          # the rule ignores the funding asked for


class BookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "data" / "book.json"

    def save(self, *events):
        return cv.commit(self.path, list(events), len(cv.load(self.path)["chain"]))

    def leftovers(self):
        return sorted(p.name for p in self.path.parent.iterdir())

    def test_a_missing_file_is_an_empty_book_and_nothing_is_created(self):
        self.assertEqual(cv.load(self.path), {"version": cv.VERSION, "chain": []})
        self.assertFalse(self.path.exists())

    def test_commit_and_read_back_the_file_holds_only_what_was_done(self):
        self.save(*USERS_CASE)
        book_ = cv.load(self.path)
        self.assertEqual(len(book_["chain"]), 6)
        self.assertEqual(cv.fold(book_)["tickers"]["TSLA"]["E"], run(*USERS_CASE)["tickers"]["TSLA"]["E"])
        allowed = {"do", "ticker", "c", "t", "b", "burn", "hedge", "k", "p", "by", "shield", "settle", "pull", "src", "dst", "amount", "at"}
        for e in book_["chain"]:
            self.assertLessEqual(set(e), allowed)                                                             # inputs only — no E, no R, no c′
            self.assertRegex(e["at"], r"^\d{4}-\d\d-\d\d \d\d:\d\d$")
        raw = json.loads(self.path.read_text("utf-8"))
        self.assertEqual((raw["version"], [e["do"] for e in raw["chain"]]), (4, ["open", "cash", "pay", "move", "cash", "move"]))

    def test_several_steps_are_all_or_nothing(self):
        with self.assertRaises(cv.BookError):
            self.save(OPEN, move(11))                                                                         # the second step is not legal
        self.assertFalse(self.path.exists())
        self.save(OPEN, FUND)
        before = self.path.read_bytes()
        with self.assertRaises(cv.BookError):
            self.save(pay(), pay())
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_step_is_judged_on_the_fresh_book_and_an_outdated_preview_saves_nothing(self):
        self.save(OPEN, FUND)
        before = self.path.read_bytes()
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, pay(), 1)                                                                    # the preview saw 1 step, there are 2
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaisesRegex(cv.BookError, "ต้องจ่าย K ก่อน"):
            cv.commit(self.path, move(11), 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_undo_takes_back_the_last_step_of_any_kind(self):
        self.save(*USERS_CASE, scale())
        for n in range(7, 0, -1):
            self.assertEqual(len(cv.load(self.path)["chain"]), n)
            last = cv.undo(self.path)
            self.assertEqual(last["do"], (USERS_CASE + [scale()])[n - 1]["do"])
            self.assertEqual(cv.fold(cv.load(self.path))["pool"], run(*(USERS_CASE + [scale()])[: n - 1])["pool"])
        with self.assertRaisesRegex(cv.BookError, "ยังไม่มีรายการ"):
            cv.undo(self.path)

    def test_a_ticker_can_be_removed_only_while_nobody_used_it(self):
        self.save(OPEN, {"do": "open", "ticker": "B", "c": 100, "t": 5}, {"do": "open", "ticker": "C", "c": 100, "t": 5})
        cv.remove(self.path, "B")
        self.assertEqual(list(cv.fold(cv.load(self.path))["tickers"]), ["TSLA", "C"])
        self.save(pay())
        with self.assertRaisesRegex(cv.BookError, "ลบไม่ได้"):
            cv.remove(self.path, "TSLA")
        self.save(cash(cv.OUT, cv.POOL, 50))
        self.save(move(11))
        self.save(cash("TSLA", cv.POOL, 10))
        self.save(cash(cv.POOL, "C", 10))                                                                     # a cash step counts as use for both ends
        with self.assertRaisesRegex(cv.BookError, "ลบไม่ได้"):
            cv.remove(self.path, "C")
        with self.assertRaisesRegex(cv.BookError, "ไม่พบ"):
            cv.remove(self.path, "NOPE")

    def test_a_broken_file_is_reported_untouched_and_can_be_moved_aside(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ broken", encoding="utf-8")
        with self.assertRaises(cv.BookError):
            cv.load(self.path)
        with self.assertRaises(cv.BookError):
            self.save(OPEN)
        self.assertEqual(self.path.read_text("utf-8"), "{ broken")
        kept = cv.quarantine(self.path)
        self.assertEqual(kept.read_text("utf-8"), "{ broken")
        self.assertEqual(cv.load(self.path), {"version": cv.VERSION, "chain": []})

    def test_a_hand_edited_file_that_breaks_a_rule_names_the_step(self):
        self.path.parent.mkdir(parents=True)
        raw = {"version": 4, "chain": [{"do": "open", "ticker": "A", "c": 100, "t": 5}, {"do": "move", "ticker": "A", "p": 6}]}
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(cv.BookError, "รายการที่ 2: A: ต้องจ่าย K ก่อน"):
            cv.load(self.path)
        for version in (3, 5, None, "4"):
            self.path.write_text(json.dumps({**raw, "chain": [], "version": version}), encoding="utf-8")
            with self.assertRaises(cv.BookError, msg=version):
                cv.load(self.path)

    def test_saving_is_all_or_nothing_and_leaves_no_temp_files(self):
        self.save(OPEN)
        before = self.path.read_bytes()
        with mock.patch.object(cv.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save(FUND)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.leftovers(), ["book.json"])

    def test_the_file_is_plain_readable_json(self):
        self.save({**OPEN, "ticker": "หุ้น ก."})
        text = self.path.read_text("utf-8")
        self.assertIn("หุ้น ก.", text)                                                                        # not \u0e2b…
        self.assertEqual(json.loads(text)["chain"][0]["ticker"], "หุ้น ก.")


class BackupTest(unittest.TestCase):
    """The whole book as one JSON file — and the Chain data that already exists."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "data" / "book.json"

    def make(self, path, *events):
        cv.commit(path, list(events), len(cv.load(path)["chain"]))

    def leftovers(self):
        return sorted(p.name for p in self.path.parent.iterdir())

    def test_a_backup_round_trips_exactly(self):
        self.make(self.path, *USERS_CASE, scale())
        book_ = cv.load(self.path)
        text = cv.export_text(book_)
        self.assertEqual(json.loads(text), book_)                                                             # plain JSON …
        self.assertEqual(text, self.path.read_text("utf-8"))                                                  # … byte for byte what the file holds
        self.assertEqual(cv.parse_backup(text.encode("utf-8")), (book_, []))
        other = self.dir / "elsewhere" / "other.json"
        cv.import_book(other, text.encode("utf-8"))                                                           # into a place with nothing to replace
        self.assertEqual(cv.load(other), book_)
        self.assertEqual([p.name for p in other.parent.iterdir()], ["other.json"])                            # nothing to keep, no temp file

    def test_the_file_name_says_when(self):
        self.assertEqual(cv.backup_filename(datetime(2026, 10, 12, 9, 5)), "app_v4_backup-20261012-0905.json")
        self.assertRegex(cv.backup_filename(), r"^app_v4_backup-\d{8}-\d{4}\.json$")

    def test_a_bad_backup_is_refused_and_changes_nothing(self):
        self.make(self.path, *USERS_CASE)
        before = self.path.read_bytes()
        good = json.loads(before)

        def spoil(fn):
            data = json.loads(before)
            fn(data)
            return json.dumps(data).encode("utf-8")

        bad = {
            "empty": b"", "not json": b"{ nope", "not utf-8": b"\xff\xfe\x00", "a number": b"7", "an empty list": b"[]",
            "too big": b" " * (cv.MAX_IMPORT_BYTES + 1), "no tickers": b'{"tickers": []}', "tickers not a list": b'{"tickers": 5}',
            "newer book": json.dumps({**good, "version": 5}).encode(), "no version": json.dumps({"chain": good["chain"]}).encode(),
            "chain not a list": b'{"version": 4, "chain": 5}', "infinite": b'{"version": 4, "chain": [{"do":"open","ticker":"A","c":1e999,"t":1}]}',
            "nan": b'{"version": 4, "chain": [{"do":"open","ticker":"A","c":NaN,"t":1}]}',
            "unknown kind": spoil(lambda d: d["chain"].append({"do": "gift", "ticker": "TSLA"})),
            "move before pay": spoil(lambda d: d["chain"].pop(2)),
            "negative c": spoil(lambda d: d["chain"][0].update(c=-5)),
            "markdown name": spoil(lambda d: d["chain"][0].update(ticker="[x](http://evil)")),
            "long name": spoil(lambda d: d["chain"][0].update(ticker="x" * 21)),
            "same name twice": spoil(lambda d: d["chain"].insert(1, {**d["chain"][0], "ticker": "tsla"})),
            "pool overdraw": spoil(lambda d: d["chain"].append({"do": "cash", "src": "@pool", "dst": "@out", "amount": 1e9})),
            "scale without shield": spoil(lambda d: d["chain"].append({"do": "scale", "ticker": "TSLA", "by": "liquidity"})),
            "missing field": spoil(lambda d: d["chain"][0].pop("c")),
        }
        for label, data in bad.items():
            with self.assertRaises(cv.BookError, msg=label):
                cv.import_book(self.path, data)
            self.assertEqual(self.path.read_bytes(), before, label)                                          # untouched
            self.assertEqual(self.leftovers(), ["book.json"], label)                                         # no copy, no temp file

    def test_importing_replaces_the_book_and_keeps_the_old_one_beside_it(self):
        self.make(self.path, *USERS_CASE)
        old = self.path.read_bytes()
        src = self.dir / "src.json"
        self.make(src, {**OPEN, "ticker": "NEW"})
        new = src.read_bytes()
        cv.import_book(self.path, new)
        self.assertEqual(list(cv.fold(cv.load(self.path))["tickers"]), ["NEW"])
        self.assertEqual([k.read_bytes() for k in self.path.parent.glob("book.json.replaced-*")], [old])
        cv.import_book(self.path, old)                                                                        # twice more, within the same second …
        cv.import_book(self.path, new)
        kept = sorted(self.path.parent.glob("book.json.replaced-*"))
        self.assertEqual(sorted(k.read_bytes() for k in kept), sorted([old, new, old]))                       # … and nothing was overwritten
        self.assertFalse([n for n in self.leftovers() if n.endswith(".tmp")])

    def test_a_broken_file_is_kept_aside_by_an_import(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ broken", encoding="utf-8")
        src = self.dir / "src.json"
        self.make(src, OPEN)
        cv.import_book(self.path, src.read_bytes())
        self.assertEqual(list(cv.fold(cv.load(self.path))["tickers"]), ["TSLA"])
        self.assertEqual([k.read_text("utf-8") for k in self.path.parent.glob("book.json.replaced-*")], ["{ broken"])

    def test_moving_a_broken_file_aside_never_overwrites_an_earlier_one(self):
        self.path.parent.mkdir(parents=True)
        for i in range(3):
            self.path.write_text(f"junk {i}", encoding="utf-8")
            cv.quarantine(self.path)
        self.assertEqual(sorted(k.read_text("utf-8") for k in self.path.parent.glob("book.json.broken-*")), ["junk 0", "junk 1", "junk 2"])

    def test_timestamps_read_back_are_plain(self):
        """Only what this module writes survives: a foreign file cannot smuggle Markdown into the history table."""
        self.make(self.path, OPEN)
        data = json.loads(self.path.read_text("utf-8"))
        stamps = ["2026-10-12 09:30", "![x](http://evil.example/p.png)**|[a](b)", 20261012, None, "x" * 500]
        data["chain"] = [{**data["chain"][0], "ticker": f"T{i}", "at": s} for i, s in enumerate(stamps)]
        read = [e["at"] for e in cv.parse_backup(json.dumps(data).encode("utf-8"))[0]["chain"]]
        self.assertEqual(read[0], "2026-10-12 09:30")                                                         # a real one is unchanged
        for at in read:
            self.assertRegex(at, r"^[0-9: \-]{0,16}$")
        self.assertFalse(any("evil" in at or "[" in at for at in read))

    # ── the Chain data that already exists ──────────────────────────────────────────────────
    def test_the_real_chain_data_of_the_earlier_app_comes_in_as_it_stands(self):
        raw = json.loads((ROOT / "data" / "chain_system_backup.json").read_text("utf-8"))
        book_, warns = cv.parse_backup((ROOT / "data" / "chain_system_backup.json").read_bytes())
        led = cv.fold(book_)
        tot = cv.totals(led)
        usable = [t for t in raw["tickers"] if t["current_state"]["price"] > 0 and t["current_state"]["fix_c"] > 0]
        self.assertEqual((len(raw["tickers"]), len(usable), len(led["tickers"])), (16, 15, 15))
        self.assertEqual(len(warns), 1)
        self.assertIn("FSUN", warns[0])                                                                       # fix_c 0.00: closed, and said so
        self.assertEqual((led["pool"], led["reserve"]), (round(raw["global_pool_cf"], 2), round(raw["global_ev_reserve"], 2)))
        self.assertEqual((led["pool"], led["reserve"]), (17758.58, -16256.17))
        self.assertAlmostEqual(tot["fix_c"], sum(t["current_state"]["fix_c"] for t in usable), places=2)
        self.assertAlmostEqual(tot["burn"], sum(t["current_state"]["cumulative_ev"] for t in usable), places=2)   # Ev Burn comes with it
        for t in usable:
            mine = led["tickers"][t["ticker"]]
            s = t["current_state"]
            self.assertEqual((mine["c"], mine["p0"], mine["pa"], mine["b"], mine["burn0"]),
                             (round(s["fix_c"], 2), s["price"], s["price"], round(s["baseline"], 2), round(s["cumulative_ev"], 2)))
            self.assertEqual((mine["E"], mine["A"], mine["moves"], mine["open"], mine["sumK"]), (0.0, 0.0, 0, False, 0.0))   # starts from here
        self.assertIsNone(cv.totals(led)["eff"])                                                              # the opening burn is not measured
        self.assertEqual(led["net_in"], round(raw["global_pool_cf"] + raw["global_ev_reserve"], 2))
        # … and the book runs: ① pay K from the (negative) EV Reserve, ② an act, on a real ticker
        first = next(iter(led["tickers"]))
        after = run(*book_["chain"], pay(first), move(led["tickers"][first]["p0"] * 1.2, first))
        self.assertEqual(after["tickers"][first]["moves"], 1)

    def test_the_first_generation_list_comes_in_too(self):
        book_, warns = cv.parse_backup((ROOT / "trading_data.json").read_bytes())
        led = cv.fold(book_)
        raw = json.loads((ROOT / "trading_data.json").read_text("utf-8"))
        self.assertEqual((len(led["tickers"]), warns), (len(raw), []))
        self.assertEqual((led["pool"], led["reserve"]), (0.0, 0.0))                                           # a plain list has no treasury
        first = raw[0]
        price, fix_c, base = [float(x) for x in first["Final"].split(",")]
        mine = led["tickers"][first["ticker"]]
        self.assertEqual((mine["p0"], mine["c"], mine["b"]), (price, fix_c, base))                            # the "Final" string is the state

    def test_what_cannot_run_is_left_out_and_said(self):
        raw = {"tickers": [
            {"ticker": "OK", "current_state": {"price": 10, "fix_c": 500, "baseline": -20, "cumulative_ev": 5}},
            {"ticker": "ok", "current_state": {"price": 10, "fix_c": 500}},                                    # same name, ignoring case
            {"ticker": "ZERO", "current_state": {"price": 10, "fix_c": 0, "cumulative_ev": 80}},
            {"ticker": "NOPRICE", "current_state": {"price": 0, "fix_c": 500}},
            {"ticker": "a*b", "current_state": {"price": 1, "fix_c": 1}},
            {"ticker": "NAN", "current_state": {"price": float("nan"), "fix_c": 1}},
            {"ticker": "FINAL", "Final": "7.5, 300, 12"},                                                     # first-generation record
            {"ticker": "INITIAL", "initial_state": {"price": 3, "fix_c": 100}},
            "junk", None, {"price": 3},
        ], "global_pool_cf": -50, "global_ev_reserve": 25}
        book_, warns = cv.from_legacy(raw)
        led = cv.fold(cv._validated(book_))
        self.assertEqual(list(led["tickers"]), ["OK", "FINAL", "INITIAL"])
        self.assertEqual((led["tickers"]["OK"]["b"], led["tickers"]["OK"]["burn0"]), (-20.0, 5.0))
        self.assertEqual((led["tickers"]["FINAL"]["c"], led["tickers"]["FINAL"]["b"]), (300.0, 12.0))
        self.assertEqual((led["pool"], led["reserve"]), (0.0, 25.0))                                          # a negative Pool CF cannot exist
        text = "\n".join(warns)
        for needle in ("ชื่อซ้ำ", "ZERO", "NOPRICE", "a*b", "ไม่ใช่ object", "Pool CF ติดลบ"):
            self.assertIn(needle, text)
        with self.assertRaisesRegex(cv.BookError, "ไม่มี ticker"):
            cv.from_legacy({"tickers": [{"ticker": "Z", "current_state": {"price": 0, "fix_c": 0}}]})
        negative_reserve = cv.fold(cv._validated(cv.from_legacy({"tickers": raw["tickers"][:1], "global_ev_reserve": -300})[0]))
        self.assertEqual(negative_reserve["reserve"], -300.0)

    def test_importing_the_real_data_through_the_file_replaces_and_keeps_the_old(self):
        self.make(self.path, OPEN)
        old = self.path.read_bytes()
        book_, warns = cv.import_book(self.path, (ROOT / "data" / "chain_system_backup.json").read_bytes())
        self.assertEqual((len(cv.fold(cv.load(self.path))["tickers"]), len(warns)), (15, 1))
        self.assertEqual([k.read_bytes() for k in self.path.parent.glob("book.json.replaced-*")], [old])


class StandaloneTest(unittest.TestCase):
    """app-v4 is its own: no shared code and no shared data with any other page of this repo."""

    FILES = (ROOT / "chain_v4.py", ROOT / "pages" / "app-v4.py")
    STDLIB_OK = {"__future__", "ast", "json", "math", "os", "re", "sys", "tempfile", "threading", "datetime", "pathlib", "typing", "zlib"}
    OTHER = ("engine_v2", "ledger_v2", "learn_v2", "lego_hover_table", "chain_v3", "chain_system", "flywheels", "chain_engine")

    def test_only_streamlit_and_its_own_module_are_imported(self):
        for f in self.FILES:
            for node in ast.walk(ast.parse(f.read_text("utf-8"))):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                       [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for m in mods:
                    self.assertIn(m.split(".")[0], self.STDLIB_OK | {"streamlit", "chain_v4"}, f"{f.name}: {m}")

    def test_importing_the_logic_loads_no_other_module_of_the_repo(self):
        code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import chain_v4; "
                f"print(sorted(m for m in sys.modules if m.split('.')[0] in {set(self.OTHER)!r}))")
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "[]")

    def test_it_does_not_read_or_write_the_data_of_any_other_page(self):
        for f in self.FILES:
            text = f.read_text("utf-8")
            for word in (*self.OTHER, "trading_data", "app_v3.json", "CHAIN_V2", "APP_V3", "app-v2", "app-v3"):
                self.assertNotIn(word, text, f"{f.name} mentions {word}")
            self.assertNotRegex(text, r"\bv2\b|\bv3\b", f.name)

    def test_the_ledger_rows_never_leak_a_private_state_object(self):
        led = run(*USERS_CASE)
        led["rows"][-1]["after"]["E"] = 12345.0                                                              # a caller may scribble on a row …
        self.assertNotEqual(led["tickers"]["TSLA"]["E"], 12345.0)                                            # … but never on the ticker


if __name__ == "__main__":
    unittest.main()
