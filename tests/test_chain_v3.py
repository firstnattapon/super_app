"""chain_v3 — the standalone logic of app-v3 (stdlib only: no Streamlit, no other module of this repo)."""
import ast
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import chain_v3 as cv  # noqa: E402


def black_scholes_put_rate() -> float:
    """An independent Black-Scholes: European put, S = 1, K = 0.9, T = 1 year, σ = 50%, r = 4%."""
    s, k, t, r, sigma = 1.0, 0.9, 1.0, 0.04, 0.5
    ncdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))            # noqa: E731
    d1 = (math.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * ncdf(-d2) - s * ncdf(-d1)


class PlanTest(unittest.TestCase):
    def test_the_worked_example(self):
        p = cv.plan(1000, 100, 125, 1.0)
        self.assertEqual((p["H"], p["K"], p["S"], p["c_after"]), (223.14, 123.96, 99.18, 1099.18))
        self.assertEqual((p["t"], p["p"], p["c"], p["real_k"]), (100.0, 125.0, 1000.0, False))

    def test_no_surplus_means_no_growth_and_never_a_cut(self):
        for price in (100, 90, 50, 112):                    # no move, a drop, a crash, a rise too small to pay the premium
            p = cv.plan(1000, 100, price, 1.0)
            self.assertLessEqual(p["S"], 0, price)
            self.assertEqual(p["c_after"], 1000.0, price)

    def test_every_line_adds_up_exactly(self):
        rng = random.Random(7)
        for _ in range(500):
            c, t, hx = rng.uniform(1, 1e6), rng.uniform(0.01, 5000), rng.choice([0, 0.5, 1, 2, 3.5])
            p = cv.plan(c, t, t * math.exp(rng.uniform(-0.6, 0.9)), hx)
            self.assertEqual(p["S"], round(p["H"] - p["K"], 2))
            self.assertAlmostEqual(p["c_after"] - cv._r2(c), max(0.0, p["S"]), delta=0.0051)
            self.assertGreaterEqual(p["c_after"], cv._r2(c))

    def test_a_real_premium_replaces_the_formula(self):
        self.assertEqual(cv.plan(1000, 100, 125, 1.0, k=50)["K"], 50.0)
        self.assertEqual(cv.plan(1000, 100, 125, 1.0, k=0)["K"], 0.0)            # 0 = no insurance this round
        self.assertEqual(cv.plan(1000, 100, 125, 1.0, k=0)["S"], 223.14)
        self.assertTrue(cv.plan(1000, 100, 125, 1.0, k=50)["real_k"])
        self.assertEqual(cv.plan(1000, 100, 125, 2.0)["K"], 247.92)              # twice the hedge, twice the premium

    def test_bad_numbers_are_refused(self):
        bad = [(0, 100, 125), (-1, 100, 125), (1000, 0, 125), (1000, 100, 0), (1000, 100, -5),
               (float("nan"), 100, 125), (1000, float("inf"), 125), (1000, 100, "abc"), (True, 100, 125), (None, 100, 125)]
        for args in bad:
            with self.assertRaises(cv.BookError, msg=args):
                cv.plan(*args)
        with self.assertRaises(cv.BookError):
            cv.plan(1000, 100, 125, -1)
        with self.assertRaises(cv.BookError):
            cv.plan(1000, 100, 125, 1.0, k=-3)

    def test_absurd_numbers_are_a_message_not_a_crash(self):
        for args in ((1000, 1e300, 1e-300), (1000, 1e-300, 1e300), (1e308, 100, 1e300), (1e308, 100, 125, 1e5)):
            with self.assertRaises(cv.BookError, msg=args):
                cv.plan(*args)
        self.assertIsNone(cv.breakeven_price(1000, 100, 1e5))                    # e^(premium / c) is too big to show

    def test_the_premium_constant_is_black_scholes(self):
        self.assertAlmostEqual(cv.PREMIUM_RATE, black_scholes_put_rate(), places=6)
        self.assertEqual(f"{cv.PREMIUM_RATE:.1%}", "12.4%")

    def test_breakeven_price_is_where_the_surplus_turns_positive(self):
        for c, t, hx, k in ((1000, 100, 1.0, None), (2300, 15.3, 2.0, None), (500, 0.4, 1.0, 75.0)):
            star = cv.breakeven_price(c, t, hx, k)
            self.assertLessEqual(cv.plan(c, t, star * 0.995, hx, k)["S"], 0)
            self.assertGreater(cv.plan(c, t, star * 1.005, hx, k)["S"], 0)
        self.assertAlmostEqual(cv.breakeven_price(1000, 100, 1.0) / 100 - 1, 0.1320, places=3)   # "+13.2% at Hedge ×1"


class BookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "data" / "book.json"

    def run_round(self, name, price, hx=1.0, k=None):
        book = cv.load(self.path)
        c, t = cv.current(cv._find(book, name))
        cv.commit(self.path, name, cv.plan(c, t, price, hx, k))

    def test_a_missing_file_is_an_empty_book_and_nothing_is_created(self):
        self.assertEqual(cv.load(self.path), {"version": 1, "positions": []})
        self.assertFalse(self.path.exists())

    def test_start_commit_and_read_back(self):
        self.assertEqual(cv.start(self.path, "  DEMO ", 1000, 100), "DEMO")
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual((pos["c0"], pos["t0"], pos["rounds"]), (1000.0, 100.0, []))
        self.assertEqual(cv.current(pos), (1000.0, 100.0))
        self.assertIsNone(cv.efficiency(pos))
        self.run_round("DEMO", 125)
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1099.18, 125.0))                          # ④ c′ and ⑤ t′ = P
        self.assertEqual(cv.totals(pos), (223.14, 123.96))
        self.assertAlmostEqual(cv.efficiency(pos), 223.14 / 123.96)
        self.assertEqual(len(pos["rounds"]), 1)
        self.assertRegex(pos["rounds"][0]["at"], r"^\d{4}-\d\d-\d\d \d\d:\d\d$")

    def test_names_are_checked(self):
        for bad in ("", "   ", "x" * 21, "a*b", "a_b", "[x](y)", "<b>", "a|b", "$x", "a#b", "a&b", "tab\tname", None):
            with self.assertRaises(cv.BookError, msg=repr(bad)):
                cv.start(self.path, bad, 1000, 100)
        cv.start(self.path, "หุ้น ก.", 1000, 100)                                    # Thai names with tone marks are fine
        cv.start(self.path, "BRK.B", 1000, 100)
        with self.assertRaises(cv.BookError):
            cv.start(self.path, "brk.b", 1000, 100)                                  # duplicates, ignoring case
        self.assertEqual([p["name"] for p in cv.load(self.path)["positions"]], ["หุ้น ก.", "BRK.B"])

    def test_an_outdated_plan_saves_nothing(self):
        cv.start(self.path, "A", 1000, 100)
        old = cv.plan(1000, 100, 125, 1.0)
        cv.commit(self.path, "A", old)                                               # the first click lands …
        before = self.path.read_text("utf-8")
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", old)                                           # … a second click on the same plan does not
        self.assertEqual(self.path.read_text("utf-8"), before)
        forged = {**cv.plan(1099.18, 125, 130, 1.0), "c_after": 9e9}
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", forged)                                        # a plan that does not add up is refused
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", {"p": 1})
        self.assertEqual(self.path.read_text("utf-8"), before)

    def test_undo_returns_to_the_state_before_the_round(self):
        cv.start(self.path, "A", 1000, 100)
        self.run_round("A", 125)
        self.run_round("A", 140)
        undone = cv.undo(self.path, "A")
        self.assertEqual(undone["p"], 140.0)
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1099.18, 125.0))
        cv.undo(self.path, "A")
        self.assertEqual(cv.current(cv.load(self.path)["positions"][0]), (1000.0, 100.0))
        with self.assertRaises(cv.BookError):
            cv.undo(self.path, "A")

    def test_remove(self):
        cv.start(self.path, "A", 1000, 100)
        cv.start(self.path, "B", 500, 10)
        cv.remove(self.path, "A")
        self.assertEqual([p["name"] for p in cv.load(self.path)["positions"]], ["B"])
        with self.assertRaises(cv.BookError):
            cv.remove(self.path, "A")

    def test_a_broken_file_is_reported_untouched_and_can_be_moved_aside(self):
        self.path.parent.mkdir(parents=True)
        for junk in ("{ not json", "[]", '{"positions": 5}', '{"positions": [{"name": "A"}]}',
                     '{"positions": [{"name": "A", "c0": -1, "t0": 1, "rounds": []}]}',
                     '{"positions": [{"name": "A", "c0": 1, "t0": 1, "rounds": [{"p": 1}]}]}'):
            self.path.write_text(junk, encoding="utf-8")
            with self.assertRaises(cv.BookError, msg=junk):
                cv.load(self.path)
            with self.assertRaises(cv.BookError):
                cv.start(self.path, "A", 1000, 100)                                  # writes refuse too …
            self.assertEqual(self.path.read_text("utf-8"), junk)                     # … and the file is never overwritten
        moved = cv.quarantine(self.path)
        self.assertFalse(self.path.exists())
        self.assertEqual(moved.read_text("utf-8"), junk)                             # moved aside, not deleted
        self.assertEqual(cv.load(self.path)["positions"], [])

    def test_saving_is_all_or_nothing_and_leaves_no_temp_files(self):
        cv.start(self.path, "A", 1000, 100)
        before = self.path.read_text("utf-8")
        with self.assertRaises(ValueError):
            cv.save(self.path, {"positions": [{"x": float("nan")}]})                 # NaN can never reach the file
        self.assertEqual(self.path.read_text("utf-8"), before)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["book.json"])

    def test_the_goal_holds_on_any_path(self):
        """Free Risk · c never shrinks · Σ(H − K) = ΣS · Ev Efficiency ≥ 1 ⇔ the harvest paid for all the insurance."""
        for seed in range(30):
            rng = random.Random(seed)
            self.path.unlink(missing_ok=True)
            cv.start(self.path, "X", rng.uniform(100, 5000), rng.uniform(1, 500))
            hx = rng.choice([0.5, 1.0, 2.0])
            for _ in range(25):
                pos = cv.load(self.path)["positions"][0]
                price = cv.current(pos)[1] * math.exp(rng.uniform(-0.25, 0.35))
                self.run_round("X", price, hx)
            pos = cv.load(self.path)["positions"][0]
            rounds, c0 = pos["rounds"], pos["c0"]
            cs = [c0] + [r["c_after"] for r in rounds]
            self.assertTrue(all(b >= a for a, b in zip(cs, cs[1:])), "c never shrinks")
            self.assertAlmostEqual(cs[-1] - c0, sum(max(0.0, r["S"]) for r in rounds), delta=0.0051 * len(rounds))   # only surplus adds
            self.assertTrue(all(r["S"] == round(r["H"] - r["K"], 2) for r in rounds))
            h, k = cv.totals(pos)
            self.assertAlmostEqual(h - k, sum(r["S"] for r in rounds), delta=0.0051 * len(rounds))
            if k > 0:
                self.assertEqual(cv.efficiency(pos) >= 1.0, h - k >= 0, seed)
            self.assertEqual(cv.current(pos)[1], rounds[-1]["p"])                    # ⑤ t′ = P

    def test_the_file_is_plain_readable_json(self):
        cv.start(self.path, "A", 1000, 100)
        self.run_round("A", 125)
        data = json.loads(self.path.read_text("utf-8"))
        self.assertEqual(sorted(data), ["positions", "version"])
        self.assertEqual(sorted(data["positions"][0]["rounds"][0]),
                         sorted(["at", "t", "p", "c", "H", "K", "S", "c_after", "hedge_x", "real_k"]))


class StandaloneTest(unittest.TestCase):
    """app-v3 is separate from app-v2 — no shared code, no shared data, no mention."""

    FILES = (ROOT / "chain_v3.py", ROOT / "pages" / "app-v3.py")
    STDLIB_OK = {"__future__", "ast", "json", "math", "os", "re", "sys", "tempfile", "threading", "datetime", "pathlib", "typing",
                 "zlib"}
    FORBIDDEN = ("engine_v2", "ledger_v2", "learn_v2", "lego_hover_table", "app-v2", "app_v2", "trading_data",
                 "CHAIN_V2", "chain_system", "flywheels")

    def test_only_streamlit_and_its_own_module_are_imported(self):
        for f in self.FILES:
            for node in ast.walk(ast.parse(f.read_text("utf-8"))):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                       [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for m in mods:
                    self.assertIn(m.split(".")[0], self.STDLIB_OK | {"streamlit", "chain_v3"}, f"{f.name}: {m}")

    def test_nothing_of_app_v2_is_mentioned(self):
        for f in self.FILES:
            text = f.read_text("utf-8")
            for word in self.FORBIDDEN:
                self.assertNotIn(word, text, f"{f.name} mentions {word}")
            self.assertNotRegex(text, r"\bv2\b", f.name)


if __name__ == "__main__":
    unittest.main()
