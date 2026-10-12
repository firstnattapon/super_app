"""chain_v3 — the standalone logic of app-v3 (stdlib only: no Streamlit, no other module of this repo)."""
import ast
import json
import math
import random
import sys
import tempfile
import unittest
from datetime import datetime
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
        p = cv.plan(1000, 100, 200, 1.0)                                              # the price doubles
        self.assertEqual((p["E"], p["K"], p["S"], p["c_after"]), (306.85, 123.96, 182.89, 1182.89))
        self.assertEqual((p["t"], p["p"], p["c"], p["real_k"]), (100.0, 200.0, 1000.0, False))
        back = cv.plan(p["c_after"], 200, 100, 1.0)                                   # … and comes back to where it was
        self.assertEqual((back["E"], back["K"], back["S"], back["c_after"]), (228.47, 146.63, 81.84, 1264.73))

    def test_the_harvest_is_the_ledgers_E_equation(self):
        """E = A − R with A = c·(x − 1) the cash of one rebalance and R = c·ln x the reference line (LEGO)."""
        rng = random.Random(11)
        for _ in range(500):
            c, t = rng.uniform(1, 1e6), rng.uniform(0.01, 5000)
            p = t * math.exp(rng.uniform(-1.5, 1.5))
            a, r = c * (p / t - 1), c * math.log(p / t)
            self.assertAlmostEqual(cv.plan(c, t, p)["E"], a - r, delta=0.0051 + 1e-9 * abs(a))
        self.assertEqual(cv.plan(1000, 100, 100)["E"], 0.0)                           # no swing, nothing to harvest
        self.assertAlmostEqual(cv.plan(1e6, 100, 100.001)["E"], 1e6 * 1e-5 ** 2 / 2, delta=0.01)   # a tiny move keeps its digits

    def test_the_swing_pays_in_both_directions_and_is_never_negative(self):
        for price in (10, 50, 90, 99.9, 100, 100.1, 110, 125, 200, 1000):
            self.assertGreaterEqual(cv.plan(1000, 100, price)["E"], 0.0, price)
        self.assertGreater(cv.plan(1000, 100, 50)["E"], 0)                            # a halving is a swing too
        self.assertGreater(cv.plan(1000, 100, 50)["S"], 0)                            # … and it more than pays for the insurance

    def test_a_plain_rise_does_not_pay_only_the_swing_does(self):
        for price in (100, 90, 112, 125, 150, 158):                                   # all inside the two break-even prices
            p = cv.plan(1000, 100, price, 1.0)
            self.assertLessEqual(p["S"], 0, price)
            self.assertEqual(p["c_after"], 1000.0, price)                             # c never shrinks, it just does not grow
            self.assertEqual(p["K"], 123.96, price)                                   # and the premium is still paid
        self.assertEqual(cv.plan(1000, 100, 125)["E"], 26.86)

    def test_every_line_adds_up_exactly(self):
        rng = random.Random(7)
        for _ in range(500):
            c, t, hx = rng.uniform(1, 1e6), rng.uniform(0.01, 5000), rng.choice([0, 0.5, 1, 2, 3.5])
            p = cv.plan(c, t, t * math.exp(rng.uniform(-1.2, 1.2)), hx)
            self.assertEqual(p["S"], round(p["E"] - p["K"], 2))
            self.assertAlmostEqual(p["c_after"] - cv._r2(c), max(0.0, p["S"]), delta=0.0051)
            self.assertGreaterEqual(p["c_after"], cv._r2(c))
            self.assertGreaterEqual(p["E"], 0.0)

    def test_a_real_premium_replaces_the_formula(self):
        self.assertEqual(cv.plan(1000, 100, 200, 1.0, k=50)["K"], 50.0)
        self.assertEqual(cv.plan(1000, 100, 200, 1.0, k=50)["S"], 256.85)
        self.assertEqual(cv.plan(1000, 100, 200, 1.0, k=0)["K"], 0.0)                 # 0 = no insurance this round
        self.assertEqual(cv.plan(1000, 100, 200, 1.0, k=0)["S"], 306.85)
        self.assertTrue(cv.plan(1000, 100, 200, 1.0, k=50)["real_k"])
        self.assertEqual(cv.plan(1000, 100, 200, 2.0)["K"], 247.92)                   # twice the hedge, twice the premium

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
        low, high = cv.breakeven_prices(1000, 100, 1e5)                               # a premium 12,000× the capital
        self.assertIsNone(low)                                                       # no price below t can pay it
        self.assertGreater(high, 1e6)
        self.assertEqual(cv.breakeven_prices(1, 100, 1.0, 1e308), (None, None))      # too extreme to print either side

    def test_the_premium_constant_is_black_scholes(self):
        self.assertAlmostEqual(cv.PREMIUM_RATE, black_scholes_put_rate(), places=6)
        self.assertEqual(f"{cv.PREMIUM_RATE:.1%}", "12.4%")

    def test_breakeven_prices_are_where_the_surplus_turns_positive(self):
        for c, t, hx, k in ((1000, 100, 1.0, None), (2300, 15.3, 2.0, None), (500, 0.4, 1.0, 75.0), (800, 3000, 0.5, None)):
            low, high = cv.breakeven_prices(c, t, hx, k)
            self.assertLess(low, t)
            self.assertGreater(high, t)
            self.assertLessEqual(cv.plan(c, t, low * 1.005, hx, k)["S"], 0)           # between the two: not enough
            self.assertLessEqual(cv.plan(c, t, high * 0.995, hx, k)["S"], 0)
            self.assertGreater(cv.plan(c, t, low * 0.995, hx, k)["S"], 0)             # beyond either: the swing pays
            self.assertGreater(cv.plan(c, t, high * 1.005, hx, k)["S"], 0)
        low, high = cv.breakeven_prices(1000, 100, 1.0)
        self.assertAlmostEqual(high / 100 - 1, 0.584, places=3)                       # "+58% up, −42% down at Hedge ×1"
        self.assertAlmostEqual(low / 100 - 1, -0.419, places=3)
        self.assertEqual(cv.breakeven_prices(1000, 100, 1.0, 0), (100, 100))          # no premium: any move pays
        self.assertEqual(cv.breakeven_prices(1000, 100, 0.0), (100, 100))


class BookTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "data" / "book.json"

    def run_round(self, name, price, hx=1.0, k=None):
        book = cv.load(self.path)
        c, t = cv.current(cv._find(book, name))
        cv.commit(self.path, name, cv.plan(c, t, price, hx, k))

    @staticmethod
    def one_round(**override):
        """A book with one good round (the demo's first), with some fields spoiled."""
        rnd = {**{k: v for k, v in cv.plan(1000, 100, 200, 1.0).items() if k != "real_k"}, **override}
        return json.dumps({"version": 2, "positions": [{"name": "A", "c0": 1000, "t0": 100, "rounds": [rnd]}]})

    def test_a_missing_file_is_an_empty_book_and_nothing_is_created(self):
        self.assertEqual(cv.load(self.path), {"version": cv.VERSION, "positions": []})
        self.assertFalse(self.path.exists())

    def test_start_commit_and_read_back(self):
        self.assertEqual(cv.start(self.path, "  DEMO ", 1000, 100), "DEMO")
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual((pos["c0"], pos["t0"], pos["rounds"]), (1000.0, 100.0, []))
        self.assertEqual(cv.current(pos), (1000.0, 100.0))
        self.assertIsNone(cv.efficiency(pos))
        self.run_round("DEMO", 200)
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1182.89, 200.0))                          # ④ c′ and ⑤ t′ = P
        self.assertEqual(cv.totals(pos), (306.85, 123.96))                           # (ΣE, ΣK)
        self.assertAlmostEqual(cv.efficiency(pos), 306.85 / 123.96)
        self.assertEqual(len(pos["rounds"]), 1)
        self.assertRegex(pos["rounds"][0]["at"], r"^\d{4}-\d\d-\d\d \d\d:\d\d$")
        self.run_round("DEMO", 100)                                                  # back to the start: the swing paid twice
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1264.73, 100.0))
        self.assertEqual(cv.totals(pos), (535.32, 270.59))
        self.assertAlmostEqual(cv.efficiency(pos), 535.32 / 270.59)

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
        old = cv.plan(1000, 100, 200, 1.0)
        cv.commit(self.path, "A", old)                                               # the first click lands …
        before = self.path.read_text("utf-8")
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", old)                                           # … a second click on the same plan does not
        self.assertEqual(self.path.read_text("utf-8"), before)
        forged = {**cv.plan(1182.89, 200, 100, 1.0), "c_after": 9e9}
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", forged)                                        # a plan that does not add up is refused
        with self.assertRaises(cv.Stale):
            cv.commit(self.path, "A", {"p": 1})
        self.assertEqual(self.path.read_text("utf-8"), before)

    def test_undo_returns_to_the_state_before_the_round(self):
        cv.start(self.path, "A", 1000, 100)
        self.run_round("A", 200)
        self.run_round("A", 100)
        undone = cv.undo(self.path, "A")
        self.assertEqual(undone["p"], 100.0)
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1182.89, 200.0))
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
        for junk in ("{ not json", "[]", '{"positions": []}', '{"version": 2, "positions": 5}',
                     '{"version": 2, "positions": [{"name": "A"}]}',
                     '{"version": 2, "positions": [{"name": "A", "c0": -1, "t0": 1, "rounds": []}]}',
                     '{"version": 2, "positions": [{"name": "A", "c0": 1, "t0": 1, "rounds": [{"p": 1}]}]}',
                     self.one_round(E=-1.0),                                           # E can never be negative
                     self.one_round(K=-1.0),
                     self.one_round(c_after=900.0),                                    # c never shrinks
                     self.one_round(p=0.0),
                     self.one_round(hedge_x=-1.0)):
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

    def test_a_good_hand_written_book_loads(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(self.one_round(), encoding="utf-8")
        pos = cv.load(self.path)["positions"][0]
        self.assertEqual(cv.current(pos), (1182.89, 200.0))
        self.assertEqual(cv.totals(pos), (306.85, 123.96))

    def test_an_earlier_book_is_refused_with_a_clear_message_and_kept(self):
        """Its Surplus was read at R (ln): the rounds cannot be re-read at E, so they are never reinterpreted."""
        self.path.parent.mkdir(parents=True)
        old = json.dumps({"version": 1, "positions": [{"name": "OLD", "c0": 1000, "t0": 100, "rounds": [
            {"t": 100, "p": 125, "c": 1000, "H": 223.14, "K": 123.96, "S": 99.18, "c_after": 1099.18, "hedge_x": 1}]}]})
        self.path.write_text(old, encoding="utf-8")
        with self.assertRaisesRegex(cv.BookError, "รุ่นก่อน"):
            cv.load(self.path)
        with self.assertRaises(cv.BookError):
            cv.start(self.path, "NEW", 1000, 100)                                   # nothing is written over it
        self.assertEqual(self.path.read_text("utf-8"), old)
        self.assertEqual(cv.quarantine(self.path).read_text("utf-8"), old)          # "start fresh" moves it aside
        cv.start(self.path, "NEW", 1000, 100)
        self.assertEqual(cv.load(self.path)["version"], cv.VERSION)

    def test_saving_is_all_or_nothing_and_leaves_no_temp_files(self):
        cv.start(self.path, "A", 1000, 100)
        before = self.path.read_text("utf-8")
        with self.assertRaises(ValueError):
            cv.save(self.path, {"positions": [{"x": float("nan")}]})                 # NaN can never reach the file
        self.assertEqual(self.path.read_text("utf-8"), before)
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["book.json"])

    def test_the_goal_holds_on_any_path(self):
        """Free Risk · c never shrinks · Eₙ never falls · Σ(E − K) = ΣS · Ev Efficiency ≥ 1 ⇔ the swing paid for all the insurance."""
        grew = 0
        for seed in range(30):
            rng = random.Random(seed)
            self.path.unlink(missing_ok=True)
            cv.start(self.path, "X", rng.uniform(100, 5000), rng.uniform(1, 500))
            hx = rng.choice([0.5, 1.0, 2.0])
            for _ in range(25):
                pos = cv.load(self.path)["positions"][0]
                price = cv.current(pos)[1] * math.exp(rng.uniform(-1.2, 1.2))            # big swings, so the surplus is often positive
                self.run_round("X", price, hx)
            pos = cv.load(self.path)["positions"][0]
            rounds, c0 = pos["rounds"], pos["c0"]
            cs = [c0] + [r["c_after"] for r in rounds]
            self.assertTrue(all(b >= a for a, b in zip(cs, cs[1:])), "c never shrinks")
            self.assertAlmostEqual(cs[-1] - c0, sum(max(0.0, r["S"]) for r in rounds), delta=0.0051 * len(rounds))   # only surplus adds
            self.assertTrue(all(r["S"] == round(r["E"] - r["K"], 2) for r in rounds))
            self.assertTrue(all(r["E"] >= 0 for r in rounds), "E is never negative")
            es = [sum(r["E"] for r in rounds[:i]) for i in range(len(rounds) + 1)]
            self.assertTrue(all(b >= a for a, b in zip(es, es[1:])), "Eₙ never falls")
            e, k = cv.totals(pos)
            self.assertAlmostEqual(e - k, sum(r["S"] for r in rounds), delta=0.0051 * len(rounds))
            if k > 0:
                self.assertEqual(cv.efficiency(pos) >= 1.0, e - k >= 0, seed)
            self.assertEqual(cv.current(pos)[1], rounds[-1]["p"])                    # ⑤ t′ = P
            grew += cs[-1] > c0
        self.assertGreater(grew, 10)                                                 # not a vacuous test: capital really grew

    def test_the_file_is_plain_readable_json(self):
        cv.start(self.path, "A", 1000, 100)
        self.run_round("A", 200)
        data = json.loads(self.path.read_text("utf-8"))
        self.assertEqual(sorted(data), ["positions", "version"])
        self.assertEqual(data["version"], 2)
        self.assertEqual(sorted(data["positions"][0]["rounds"][0]),
                         sorted(["at", "t", "p", "c", "E", "K", "S", "c_after", "hedge_x", "real_k"]))


class BackupTest(unittest.TestCase):
    """The whole book as one JSON file: download it, and put it back (for hosts whose disk does not last)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "data" / "book.json"

    def make_book(self, path):
        cv.start(path, "หุ้น ก.", 500, 20)
        cv.start(path, "DEMO", 1000, 100)
        cv.commit(path, "DEMO", cv.plan(1000, 100, 200, 1.0))
        cv.commit(path, "DEMO", cv.plan(1182.89, 200, 100, 1.0))                   # the last thing done is a commit

    def leftovers(self):
        return sorted(p.name for p in self.path.parent.iterdir())

    def test_a_backup_round_trips_exactly(self):
        self.make_book(self.path)
        book = cv.load(self.path)
        text = cv.export_text(book)
        self.assertEqual(json.loads(text), book)                                    # plain JSON …
        self.assertEqual(text, self.path.read_text("utf-8"))                        # … byte for byte what the file holds
        self.assertEqual(cv.parse_backup(text.encode("utf-8")), book)
        other = self.dir / "elsewhere" / "other.json"
        cv.import_book(other, text.encode("utf-8"))                                 # into a place with nothing to replace
        self.assertEqual(cv.load(other), book)
        self.assertEqual(other.read_text("utf-8"), text)
        self.assertEqual([p.name for p in other.parent.iterdir()], ["other.json"])  # nothing to keep, no temp file

    def test_the_file_name_says_when(self):
        self.assertEqual(cv.backup_filename(datetime(2026, 10, 12, 9, 5)), "app_v3_backup-20261012-0905.json")
        self.assertRegex(cv.backup_filename(), r"^app_v3_backup-\d{8}-\d{4}\.json$")

    def test_a_bad_backup_is_refused_and_changes_nothing(self):
        self.make_book(self.path)
        before = self.path.read_bytes()
        good = json.loads(before)

        def spoil(fn):
            book = json.loads(before)
            fn(book)
            return json.dumps(book).encode("utf-8")

        first = lambda b: b["positions"][1]                                          # noqa: E731  (DEMO — the one with rounds)
        bad = {
            "empty": b"", "not json": b"{ nope", "not utf-8": b"\xff\xfe\x00", "a list": b"[]", "a number": b"7",
            "too big": b" " * (cv.MAX_IMPORT_BYTES + 1), "no version": json.dumps({"positions": []}).encode(),
            "earlier book": json.dumps({**good, "version": 1}).encode(), "newer book": json.dumps({**good, "version": 3}).encode(),
            "infinite": b'{"version": 2, "positions": [{"name": "A", "c0": 1e999, "t0": 1, "rounds": []}]}',
            "nan": b'{"version": 2, "positions": [{"name": "A", "c0": NaN, "t0": 1, "rounds": []}]}',
            "negative E": spoil(lambda b: first(b)["rounds"][0].update(E=-1.0)),
            "c shrinks": spoil(lambda b: first(b)["rounds"][0].update(c_after=1.0)),
            "negative c0": spoil(lambda b: first(b).update(c0=-5)),
            "markdown name": spoil(lambda b: first(b).update(name="[x](http://evil)")),
            "long name": spoil(lambda b: first(b).update(name="x" * 21)),
            "same name twice": spoil(lambda b: b["positions"].append({**first(b), "name": "demo"})),
            "missing field": spoil(lambda b: first(b)["rounds"][0].pop("K")),
        }
        for label, data in bad.items():
            with self.assertRaises(cv.BookError, msg=label):
                cv.import_book(self.path, data)
            self.assertEqual(self.path.read_bytes(), before, label)                  # untouched
            self.assertEqual(self.leftovers(), ["book.json"], label)                 # no copy, no temp file
        with self.assertRaisesRegex(cv.BookError, "รุ่นก่อน"):
            cv.parse_backup(bad["earlier book"])                                     # the reason is said plainly

    def test_importing_replaces_the_book_and_keeps_the_old_one_beside_it(self):
        self.make_book(self.path)
        old = self.path.read_bytes()
        src = self.dir / "src.json"
        cv.start(src, "NEW", 100, 1)
        new = src.read_bytes()
        cv.import_book(self.path, new)
        self.assertEqual([p["name"] for p in cv.load(self.path)["positions"]], ["NEW"])
        self.assertEqual([k.read_bytes() for k in self.path.parent.glob("book.json.replaced-*")], [old])
        cv.import_book(self.path, old)                                               # twice more, within the same second …
        cv.import_book(self.path, new)
        kept = sorted(self.path.parent.glob("book.json.replaced-*"))
        self.assertEqual(sorted(k.read_bytes() for k in kept), sorted([old, new, old]))   # … and nothing was overwritten
        self.assertFalse([n for n in self.leftovers() if n.endswith(".tmp")])

    def test_a_broken_file_is_kept_aside_by_an_import(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ broken", encoding="utf-8")
        src = self.dir / "src.json"
        cv.start(src, "OK", 100, 1)
        cv.import_book(self.path, src.read_bytes())
        self.assertEqual([p["name"] for p in cv.load(self.path)["positions"]], ["OK"])
        self.assertEqual([k.read_text("utf-8") for k in self.path.parent.glob("book.json.replaced-*")], ["{ broken"])

    def test_moving_a_broken_file_aside_never_overwrites_an_earlier_one(self):
        self.path.parent.mkdir(parents=True)
        for i in range(3):
            self.path.write_text(f"junk {i}", encoding="utf-8")
            cv.quarantine(self.path)
        self.assertEqual(sorted(k.read_text("utf-8") for k in self.path.parent.glob("book.json.broken-*")),
                         ["junk 0", "junk 1", "junk 2"])

    def test_timestamps_read_back_are_plain(self):
        """Only what this module writes survives: a foreign file cannot smuggle Markdown into the history table."""
        self.make_book(self.path)
        book = json.loads(self.path.read_text("utf-8"))
        stamps = ["2026-10-12 09:30", "![x](http://evil.example/p.png)**|[a](b)", 20261012, None, "x" * 500]
        demo = book["positions"][1]                                                  # the one with rounds
        demo["rounds"] = [{**demo["rounds"][0], "at": s} for s in stamps]
        read = [r["at"] for r in cv.parse_backup(json.dumps(book).encode("utf-8"))["positions"][1]["rounds"]]
        self.assertEqual(read[0], "2026-10-12 09:30")                                # a real one is unchanged
        for at in read:
            self.assertRegex(at, r"^[0-9: \-]{0,16}$")
        self.assertFalse(any("evil" in at or "[" in at for at in read))


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
