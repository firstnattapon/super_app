"""app_v4_system.mmd and app_v4_system.md — the picture of app-v4 and what it says (stdlib only; no Streamlit, no Mermaid).

The chart is hand-written Mermaid and the Markdown is hand-written Thai. These tests keep them honest: the loop is the
one you wrote (its words are read from the page's source, not retyped here), the equations are the page's, the events are
the five kinds the core knows, what the boxes say about the money is what the steps really do (run on chain_v4), the worked
example is computed, the table of cash routes is what ``cash`` allows, the Markdown carries the very same picture, and every
test, file and label it cites exists.
"""
import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from test_system_flowchart import parse       # noqa: E402
import chain_v4 as cv                         # noqa: E402

MMD = (ROOT / "app_v4_system.mmd").read_text(encoding="utf-8")
MD = (ROOT / "app_v4_system.md").read_text(encoding="utf-8")
PAGE = (ROOT / "pages" / "app-v4.py").read_text(encoding="utf-8")
HEADER, BODY = MMD.split("flowchart TB", 1)
OTHER_CHARTS = ("app_v2_minimal.mmd", "app_v2_system.mmd", "app_v3_playbook.mmd", "app_v3_final.mmd")
PRINCIPLES = ("Shannon's Demon", "Pay-first", "Free-Risk Scale-up", "Continuous Rollover", "Cash Lanes", "Ev Efficiency",
              "Atomic + stale-proof", "อ่านกำไรที่ E", "Human-in-the-loop")
CYC = {"PAY", "SWING", "D", "TREND", "SHIELD", "SCALE", "LIQ", "DEEP", "ROLL", "MEASURE"}
TREAS = {"POOL", "RESERVE", "BURN", "LIQD"}
DATA = {"CHAIN", "GUARD"}
LOOP = {("PAY", "SWING"), ("SWING", "D"), ("D", "SHIELD"), ("D", "TREND"), ("TREND", "SWING"), ("SHIELD", "SCALE"), ("SCALE", "LIQ"),
        ("SCALE", "DEEP"), ("LIQ", "ROLL"), ("DEEP", "ROLL"), ("ROLL", "MEASURE"), ("MEASURE", "GOAL"), ("MEASURE", "PAY")}


# ── reading the files ─────────────────────────────────────────────────────────────────────
def page_const(name: str):
    """A module-level constant of the page, read from its source (importing the page would start Streamlit)."""
    for node in ast.parse(PAGE).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def label(node: str) -> str:
    """The text of one node, tags stripped and <br/> as a space (every label is one line of the file)."""
    line = next(ln for ln in MMD.splitlines() if re.match(rf"^\s*{node}\s*[\[({{]", ln))
    return re.sub(r"\s+", " ", re.sub(r"</?[bi]>", "", line).replace("<br/>", " "))


def edge_label(a: str, b: str, arrow: str = "-->"):
    """The text on the edge a → b, or None when the edge carries none (or does not exist)."""
    m = re.search(rf"^\s*{a}\s*{re.escape(arrow)}\|([^|]*)\|\s*{b}\s*$", BODY, re.M)
    return m.group(1) if m else None


def groups() -> dict:
    """{subgraph: its nodes}; "" holds the nodes that sit outside every subgraph."""
    out, cur = {"": set()}, ""
    for ln in BODY.splitlines():
        if (m := re.match(r"^\s*subgraph\s+(\w+)", ln)):
            cur = m.group(1)
            out[cur] = set()
        elif ln.strip() == "end":
            cur = ""
        elif (m := re.match(r"^\s*(\w+)\s*[\[({]", ln)):
            out[cur].add(m.group(1))
    return out


def goal_of(text: str) -> str:
    box = re.search(r'^\s*GOAL\("(.*)"\)\s*$', text, re.M).group(1)
    return re.sub(r"</?b>", "", box.replace("<br/>", "\n")).strip()


def strip_comments(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("%%")).strip("\n")


def fenced(lang: str) -> list:
    return re.findall(rf"^```{lang}\n(.*?)\n```$", MD, re.M | re.S)


def sections() -> list:
    return MD.split("\n## ")


def section(number: int) -> str:
    return next(s for s in sections() if re.match(rf"\S+ {number}\. ", s))


def f2(x: float) -> str:
    return f"{0.0 if round(x, 2) == 0 else x:,.2f}".replace("-", "−")


def sg(x: float) -> str:
    return ("−" if round(x, 2) < 0 else "+") + f"{abs(x):,.2f}"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# ── the picture ───────────────────────────────────────────────────────────────────────────
class ChartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes, cls.subgraphs, cls.edges, cls.depth = parse(MMD)
        cls.into = lambda s, n: {a for a, b in s.edges if b == n}        # noqa: E731
        cls.out = lambda s, n: {b for a, b in s.edges if a == n}         # noqa: E731

    def test_it_is_one_page_of_three_groups_and_every_edge_has_both_ends(self):
        self.assertTrue(MMD.startswith("---\ntitle:"))
        self.assertEqual(self.nodes, {"KEY", "GOAL"} | CYC | TREAS | DATA)
        self.assertEqual(self.subgraphs, {"CYC", "TREAS", "DATA"})
        self.assertEqual(self.depth, 0)
        self.assertEqual(sorted({x for e in self.edges for x in e} - self.nodes - self.subgraphs), [])
        g = groups()
        self.assertEqual((g[""], g["CYC"], g["TREAS"], g["DATA"]), ({"KEY", "GOAL"}, CYC, TREAS, DATA))

    def test_the_loop_is_the_one_you_wrote(self):
        self.assertEqual(set(self.edges), LOOP)
        self.assertEqual(len(self.edges), len(LOOP))                                    # no edge drawn twice
        self.assertEqual((self.out("D"), self.into("D")), ({"SHIELD", "TREND"}, {"SWING"}))      # ③ has two ways out
        self.assertEqual((edge_label("D", "SHIELD"), edge_label("D", "TREND")), ("ใช่", "ไม่"))
        self.assertEqual(self.into("PAY"), {"MEASURE"})                                  # K is paid again only when a new cycle starts …
        self.assertEqual(self.into("SWING"), {"PAY", "TREND"})                           # … and "no" at ③ goes back to harvesting, not to K
        self.assertEqual((self.into("SHIELD"), self.into("SCALE")), ({"D"}, {"SHIELD"}))    # no scale-up without the shield
        self.assertEqual(self.into("ROLL"), {"LIQ", "DEEP"})                             # and no way past the Rollover
        self.assertEqual((edge_label("SCALE", "LIQ"), edge_label("SCALE", "DEEP")), ("พอ", "ไม่พอ"))
        self.assertEqual((self.into("MEASURE"), self.out("MEASURE"), self.into("GOAL")), ({"ROLL"}, {"GOAL", "PAY"}, {"MEASURE"}))
        self.assertEqual(edge_label("MEASURE", "GOAL"), "ถึงเป้า")
        self.assertIsNotNone(edge_label("MEASURE", "PAY", "-.->"))                       # the loop closes with a dotted edge, whether or not the goal is reached

    def test_the_money_and_the_data_are_not_wired_to_one_step(self):
        """Every step touches them, so the picture states it in each box instead of with one arrow; the hidden links only place the rows."""
        for edge in self.edges:
            self.assertFalse(set(edge) & (TREAS | DATA), edge)
        hidden = re.findall(r"^\s*([\w &]+?)\s*~~~\s*([\w &]+?)\s*$", BODY, re.M)
        self.assertEqual({(tuple(a.split(" & ")), tuple(b.split(" & "))) for a, b in hidden},
                         {(("KEY",), ("PAY",)), (("GOAL",), ("POOL", "RESERVE", "BURN", "LIQD")), (("POOL", "RESERVE", "BURN", "LIQD"), ("CHAIN", "GUARD"))})

    def test_the_words_are_yours(self):
        cycle = page_const("CYCLE")
        words = {"PAY": ["① จ่าย K ก่อน (Deep ITM LEAPS)"], "SWING": ["② เก็บ E (หุ้น 20% · LEAPS 80%)"], "D": ["③ Surplus = E − K บวกไหม ?"],
                 "TREND": ["③.1 roll P ขึ้น · ดึง capital gain (เทรนด์ R) ออก · E อยู่ต่อ"],
                 "SHIELD": ["④ Shield (Long Put · Zero-Cost · Put Ratio 2-1)"],
                 "SCALE": ["⑤.1 Scale Up fix_c", "Liquidity ถ้าพอจ่าย K วงจรหน้า · ไม่พอ ซื้อ Deep ITM"],
                 "ROLL": ["⑤.2 Rollover (β · roll P ขึ้น · ดึง capital gain ออก)"], "MEASURE": ["วัดผล (ΣE ÷ ΣK ≥ 1.0x)"]}
        for node, phrases in words.items():
            for phrase in phrases:
                self.assertIn(phrase, cycle, phrase)                                     # really what you wrote (the page's CYCLE) …
                self.assertIn(phrase, label(node), node)                                 # … and really in that box
        for text, phrases in ((edge_label("MEASURE", "PAY", "-.->"), ["จ่าย K ของทุนใหม่ c′ ก่อนเสมอ"]),
                              (edge_label("TREND", "SWING"), ["กลับ ②", "ไม่จ่าย K ซ้ำ"])):
            for phrase in phrases:
                self.assertIn(phrase, text)
                self.assertIn(phrase, cycle, phrase)                                     # the words on the two loop edges are yours too
        self.assertIn(page_const("ESSENCE"), label("KEY"))
        self.assertNotIn("2:1", BODY)                                                    # Put Ratio is written as given: 2-1

    def test_the_goal_is_word_for_word_the_same_as_the_other_charts_and_the_page(self):
        for other in OTHER_CHARTS:
            self.assertEqual(goal_of(MMD), goal_of((ROOT / other).read_text(encoding="utf-8")), other)
        page_goal = re.sub(r"\s*\n\s*", "\n", page_const("GOAL").replace("**", "").replace("  \n", "\n")).strip()
        self.assertEqual(goal_of(MMD), page_goal)

    def test_the_equations_are_the_pages(self):
        formulas = {"SWING": ["Rₙ = β + Fix_c × ln(Pₙ / P₀)", "ΔAₙ = Fix_c × (Pₙ / P_acted − 1)", "Aₙ = Aₙ₋₁ + ΔAₙ", "Eₙ = Aₙ − Rₙ",
                              "gap = Vₙ − Fix_c"],
                    "D": ["S = E_cyc − K"], "SCALE": ["c′ = c + S"], "ROLL": ["β += (Fix_c − Fix_c′) × ln(P_acted / P₀)"],
                    "MEASURE": ["ΣE ÷ ΣK"], "LIQD": ["Pool CF + EV Reserve"]}
        for node, items in formulas.items():
            for formula in items:
                self.assertIn(formula, label(node), node)
                self.assertIn(formula, norm(PAGE), formula)                              # the glossary of the page says the very same thing
        self.assertIn("ΣE ÷ ΣK ≥ 1.0x", page_const("CYCLE"))
        self.assertIn(f"K = {cv.PREMIUM_RATE:.1%} · c · Hedge×", label("PAY"))          # 12.4% is the core's constant
        shares = [int(x) for x in re.findall(r"(\d+)%", label("SWING"))]
        self.assertEqual(shares, [round(cv.STOCK_SHARE * 100), round((1 - cv.STOCK_SHARE) * 100)])
        self.assertEqual(sum(shares), 100)                                               # 20% + 80% is a whole
        self.assertNotIn("Rₙ − Aₙ", BODY)                                                # never the other way round: that would be negative

    def test_the_events_are_the_five_kinds_the_core_knows(self):
        used = re.findall(r"event (\w+)", " ".join(label(n) for n in ("PAY", "SWING", "TREND", "SCALE")))
        self.assertEqual(used, ["pay", "move", "cash", "scale"])
        title = next(ln for ln in BODY.splitlines() if ln.lstrip().startswith("subgraph CYC"))
        self.assertIn("event open", title)                                               # the start of a ticker
        self.assertEqual(set(used) | {"open"}, set(cv.KINDS))
        listed = re.search(r"open(?: · \w+)+", label("CHAIN")).group(0)
        self.assertEqual(listed.split(" · "), list(cv.KINDS))                            # exactly these five, in the core's order

    def test_every_principle_is_on_the_chart_with_its_name(self):
        for i, name in enumerate(PRINCIPLES, 1):
            self.assertIn(f"P{i} {name}", BODY, f"P{i}")
        self.assertEqual(sorted(set(re.findall(r"\bP\d\b", BODY))), [f"P{i}" for i in range(1, 10)])

    def test_the_example_in_the_header_is_the_computed_one(self):
        numbers = example_numbers()
        for key in ("k", "reserve_paid", "e1", "s1", "pulled", "e2", "s2", "c_new", "reserve_end", "k_next"):
            self.assertIn(numbers[key], HEADER, key)
        self.assertIn(numbers["eff"], HEADER)
        for name in ("app_v4_system.md", "pages/app-v4.py", "chain_v4.py", "tests/test_v4_charts.py", *OTHER_CHARTS):
            self.assertIn(name, HEADER, name)                                            # the files it points to
            self.assertTrue((ROOT / name).exists(), name)


# ── what the boxes say about the money is what the steps do ───────────────────────────────
DEMO = [{"do": "open", "ticker": "DEMO", "c": 1500, "t": 10}, {"do": "cash", "src": cv.OUT, "dst": cv.RESERVE, "amount": 300},
        {"do": "pay", "ticker": "DEMO", "hedge": 1}, {"do": "move", "ticker": "DEMO", "p": 15},
        {"do": "cash", "src": "DEMO", "dst": cv.POOL, "amount": 608.2}, {"do": "move", "ticker": "DEMO", "p": 10},
        {"do": "scale", "ticker": "DEMO", "by": "liquidity", "shield": True, "settle": True, "pull": True}]


def example_numbers() -> dict:
    """The worked example, computed by the core and written the way the documents write it."""
    led = cv.replay(DEMO)
    pay, (m1, m2), scale = led["rows"][2], [r for r in led["rows"] if r["do"] == "move"], led["rows"][-1]
    t = led["tickers"]["DEMO"]
    return {"k": f2(pay["K"]), "reserve_paid": f2(pay["reserve"]), "v1": f2(m1["v"]), "gap1": sg(m1["gap"]), "qty1": f"{m1['side']} {m1['qty']:g} หุ้น",
            "r1": f2(m1["r"]), "da1": sg(m1["d_a"]), "a1": f2(m1["a"]), "e1": f2(m1["e"]), "s1": sg(m1["s"]), "pulled": f2(m1["b"]),
            "v2": f2(m2["v"]), "gap2": sg(m2["gap"]), "qty2": f"{m2['side']} {m2['qty']:g} หุ้น", "r2": f2(m2["r"]), "da2": sg(m2["d_a"]),
            "a2": f2(m2["a"]), "e2": f2(m2["e"]), "s2": sg(m2["s"]), "b2": sg(m2["b"]), "c_new": f2(scale["c_new"]),
            "beta": f2(scale["beta"] + scale["d_beta"]), "pool_end": f2(scale["pool"]), "reserve_end": f2(scale["reserve"]),
            "settle_pool": f"Pool CF {sg(scale['s'])}", "settle_reserve": f"EV Reserve {sg(scale['K'])}",
            "k_next": f2(cv.next_k(t)), "eff": f"{cv.efficiency(t):.2f}x", "e_total": f2(t["E"])}


class MoneyTest(unittest.TestCase):
    def test_k_is_paid_from_ev_reserve_and_burned(self):
        before, after = cv.replay(DEMO[:2]), cv.replay(DEMO[:3])
        k = after["rows"][2]["K"]
        self.assertAlmostEqual(before["reserve"] - after["reserve"], k)                  # EV Reserve −K
        self.assertAlmostEqual(cv.totals(after)["burn"] - cv.totals(before)["burn"], k)   # Ev Burn +K
        for text in ("EV Reserve −K", "Ev Burn +K"):
            self.assertIn(text, label("PAY"))
        self.assertIn("+ K ตอนจ่าย (①)", label("BURN"))
        self.assertIn("− K ตอนจ่าย (①)", label("RESERVE"))

    def test_the_ev_burn_is_the_sum_of_k_over_every_ticker_plus_what_was_carried_in(self):
        led = cv.replay([{"do": "open", "ticker": "A", "c": 1000, "t": 10, "burn": 100}, {"do": "open", "ticker": "B", "c": 2000, "t": 5},
                         {"do": "pay", "ticker": "A", "hedge": 1}, {"do": "pay", "ticker": "B", "hedge": 2}])
        k_sum = sum(r["K"] for r in led["rows"] if r["do"] == "pay")
        self.assertAlmostEqual(cv.totals(led)["burn"], 100 + k_sum, places=2)
        self.assertIn("ΣK ทุก ticker", label("BURN"))
        self.assertIn("ที่ยกมา", label("BURN"))

    def test_liquidity_is_pool_plus_reserve_pool_cannot_go_negative_and_reserve_can(self):
        led = cv.replay(DEMO[:3])
        tot = cv.totals(led)
        self.assertAlmostEqual(tot["liquidity"], tot["pool"] + tot["reserve"])
        with self.assertRaises(cv.BookError):
            cv.preview(led, {"do": "cash", "src": cv.POOL, "dst": cv.OUT, "amount": 1})  # the pool is empty: it does not go below zero
        self.assertLess(cv.replay([DEMO[0], {"do": "pay", "ticker": "DEMO", "hedge": 1}])["reserve"], 0)   # the reserve may
        self.assertIn("ติดลบไม่ได้", label("POOL"))
        self.assertIn("ติดลบได้", label("RESERVE"))
        self.assertIn("Pool CF + EV Reserve", label("LIQD"))

    def test_pulling_the_capital_gain_moves_the_baseline_into_the_pool(self):
        before, after = cv.replay(DEMO[:4]), cv.replay(DEMO[:5])
        b = before["tickers"]["DEMO"]["b"]
        self.assertGreater(b, 0)
        self.assertAlmostEqual(after["pool"] - before["pool"], b)                       # Baseline b → Pool CF
        self.assertAlmostEqual(after["tickers"]["DEMO"]["b"], 0.0)
        move_row, cash_row = after["rows"][-2], after["rows"][-1]
        for key in ("c", "pa", "A", "R", "E", "beta"):
            self.assertEqual(move_row["after"][key], cash_row["after"][key], key)        # a flat row: A · R · E do not move
        self.assertEqual(cash_row["after"]["b"], 0.0)
        self.assertIn("Baseline b → Pool CF", label("TREND"))
        self.assertIn("P_acted เลื่อนไปที่ราคา fill แล้วตอน ②", label("TREND"))
        self.assertEqual(before["tickers"]["DEMO"]["pa"], 15.0)                          # "roll P ขึ้น" has already happened at ②

    def test_scale_up_settles_pays_k_back_and_funds_by_liquidity_or_rests_in_the_pool(self):
        led = cv.replay(DEMO[:-1])
        step = DEMO[-1]
        t = led["tickers"]["DEMO"]
        s, k = cv.surplus(t), t["K"]
        liquidity = cv.preview(led, {**step, "by": "liquidity"})[0]
        deep = cv.preview(led, {**step, "by": "deep_itm"})[0]
        self.assertGreater(s, 0)
        self.assertAlmostEqual(deep["pool"] - led["pool"], s)                            # ⑤.2 ชำระ Pool CF +S · and ข. S rests in the pool
        self.assertAlmostEqual(deep["reserve"] - led["reserve"], k)                      # EV Reserve +K
        self.assertAlmostEqual(deep["pool"] - liquidity["pool"], s)                      # ก. Liquidity: Pool CF −S
        self.assertAlmostEqual(liquidity["reserve"], led["reserve"] + k)
        self.assertAlmostEqual(liquidity["reserve"], 300.0)                              # a cycle that succeeded gives the reserve back
        for text in ("ชำระ Pool CF +S · EV Reserve +K", "ดึง Baseline b → Pool CF"):
            self.assertIn(text, label("ROLL"))
        self.assertIn("Pool CF −S", label("LIQ"))
        self.assertIn("S พักใน Pool CF", label("DEEP"))
        self.assertIn("เบี้ยที่เพิ่ม = K ของ ① หน้า", label("DEEP"))

    def test_scale_up_pulls_the_baseline_when_there_is_a_gain(self):
        led = cv.replay([DEMO[0], {"do": "pay", "ticker": "DEMO", "hedge": 0}, DEMO[3]])    # no insurance: S > 0 and the trend part is a gain
        t = led["tickers"]["DEMO"]
        self.assertGreater(cv.surplus(t), 0)
        self.assertGreater(t["b"], 0)
        pulled, plain = cv.preview(led, {**DEMO[-1], "pull": True}), cv.preview(led, {**DEMO[-1], "pull": False})
        self.assertAlmostEqual(pulled[0]["pool"] - plain[0]["pool"], t["b"])
        self.assertAlmostEqual(pulled[0]["tickers"]["DEMO"]["b"], 0.0)

    def test_the_loop_ends_with_the_next_k_of_the_new_capital(self):
        led = cv.replay(DEMO)
        t = led["tickers"]["DEMO"]
        self.assertEqual(cv.phase(t), "pay")                                             # ↻ ①: nothing open, the next step is paying K
        self.assertAlmostEqual(cv.next_k(t), round(cv.PREMIUM_RATE * t["c"], 2))         # K of the new capital c′, not of the old one
        self.assertGreater(t["c"], 1500.0)


class GuardTest(unittest.TestCase):
    """The box "กติกาเดียว (apply)" and the question of ⑤.1 say what the core does — here they are run."""

    def test_one_set_of_rules_judges_a_step_whether_it_is_previewed_saved_or_loaded(self):
        for text in ("ตอนโหลด = พรีวิว = บันทึก", "move ก่อน pay ไม่ได้", "scale ต้อง S เป็นบวก + Shield"):
            self.assertIn(text, label("GUARD"))
        bad = {"do": "move", "ticker": "DEMO", "p": 15}                                  # a move before the pay
        with self.assertRaises(cv.BookError):
            cv.preview(cv.replay([DEMO[0]]), bad)                                        # previewed …
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.json"
            cv.commit(path, [DEMO[0]], 0)
            before = path.read_bytes()
            with self.assertRaises(cv.BookError):
                cv.commit(path, [bad], 1)                                                # … saved …
            self.assertEqual(path.read_bytes(), before)                                  # and nothing was written
            with self.assertRaises(cv.Stale):
                cv.commit(path, [DEMO[1]], 0)                                            # an outdated plan is refused (P7)
            raw = json.loads(before)
            raw["chain"].append(bad)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(cv.BookError) as ctx:
                cv.load(path)                                                            # … loaded: the same refusal, naming the step
            self.assertIn("รายการที่ 2", str(ctx.exception))
        with self.assertRaises(cv.BookError):
            cv.replay([DEMO[0], DEMO[2], DEMO[2]])                                       # one K per cycle: no second pay while it is open

    def test_scale_up_needs_a_positive_surplus_and_the_shield_and_the_capital_grows_by_s(self):
        with self.assertRaises(cv.BookError):
            cv.preview(cv.replay(DEMO[:4]), DEMO[-1])                                    # S is still negative at 15: ③ says no
        led = cv.replay(DEMO[:6])                                                        # S is positive at 10
        with self.assertRaises(cv.BookError):
            cv.preview(led, {**DEMO[-1], "shield": False})                               # no shield, no scale-up
        row = cv.preview(led, DEMO[-1])[1]
        self.assertEqual(row["c_new"], round(row["c"] + row["s"], 2))                    # c′ = c + S
        self.assertGreaterEqual(row["c_new"], row["c"])

    def test_the_choice_of_funding_is_decided_by_what_is_left_for_the_next_k(self):
        """⑤.1: Liquidity when the cash left (Pool CF + EV Reserve) still pays K′, else Deep ITM."""
        self.assertEqual(cv.recommend_funding(cv.replay(DEMO[:6]), DEMO[-1]), "liquidity")           # 908.20 left ≥ K′ 193.88
        poor = cv.replay([DEMO[0], {"do": "pay", "ticker": "DEMO", "hedge": 0.5}, DEMO[3]])         # a thin treasury, S > 0
        self.assertGreater(cv.surplus(poor["tickers"]["DEMO"]), 0)
        self.assertEqual(cv.recommend_funding(poor, {**DEMO[-1], "pull": False}), "deep_itm")        # nothing left to pay K′ with
        for text in ("Liquidity ถ้าพอจ่าย K วงจรหน้า · ไม่พอ ซื้อ Deep ITM", "หลังชำระและหลังใช้"):
            self.assertIn(text, label("SCALE"))


class CashRoutesTest(unittest.TestCase):
    def test_the_table_of_eight_routes_is_what_cash_allows(self):
        table = section(6).split("| จาก → ไป | ใช้เมื่อ |", 1)[1].split("\n\n", 1)[0]
        names = {"นอกระบบ": cv.OUT, "Pool CF": cv.POOL, "EV Reserve": cv.RESERVE, "Baseline b ของ ticker": "X"}
        written = set()
        for src, dst in re.findall(r"^\| (.+?) → (.+?) \|", table, re.M):
            written.add((names[src], names[dst]))
        self.assertEqual(len(written), 8)
        led = cv.replay([{"do": "open", "ticker": "X", "c": 1000, "t": 10, "b": 50}, {"do": "open", "ticker": "Y", "c": 1000, "t": 10, "b": 50},
                         {"do": "cash", "src": cv.OUT, "dst": cv.POOL, "amount": 100}, {"do": "cash", "src": cv.OUT, "dst": cv.RESERVE, "amount": 100}])
        allowed = set()
        for src in (cv.OUT, cv.POOL, cv.RESERVE, "X", "Y"):
            for dst in (cv.OUT, cv.POOL, cv.RESERVE, "X", "Y"):
                if src == dst:
                    continue
                try:
                    cv.preview(led, {"do": "cash", "src": src, "dst": dst, "amount": 1})
                except cv.BookError:
                    continue
                allowed.add(tuple("X" if x in ("X", "Y") else x for x in (src, dst)))
        self.assertEqual(written, allowed)


class ChainDataTest(unittest.TestCase):
    def test_the_chain_data_example_is_a_real_book_and_it_is_the_pages_demo(self):
        blocks = fenced("json")
        self.assertEqual(len(blocks), 1)
        book = json.loads(blocks[0])
        self.assertEqual(book["version"], cv.VERSION)
        steps = book["chain"]
        self.assertEqual({e["do"] for e in steps}, set(cv.KINDS))                        # all five kinds
        led = cv.replay(steps)                                                           # judged by the same rules as a file
        name, c, t0 = page_const("DEMO")
        self.assertEqual(steps[0], {"do": "open", "ticker": name, "c": int(c), "t": int(t0)})
        self.assertEqual(steps[1]["amount"], page_const("DEMO_FUND"))
        self.assertEqual([e["p"] for e in steps if e["do"] == "move"], list(page_const("DEMO_PATH")))
        self.assertEqual(steps, DEMO)                                                    # and the very chain this file computes the example from
        self.assertGreater(led["tickers"][name]["c"], c)

    def test_the_real_data_numbers_in_the_markdown_are_what_the_import_gives(self):
        raw = json.loads((ROOT / "data" / "chain_system_backup.json").read_text(encoding="utf-8"))
        book, warnings = cv.from_legacy(raw)
        tot = cv.totals(cv.fold(book))
        self.assertEqual((len(raw["tickers"]), tot["tickers"], len(warnings)), (16, 15, 1))
        self.assertIn("FSUN", warnings[0])
        self.assertIn(f"{len(raw['tickers'])} ticker → นำเข้า {tot['tickers']} (ข้าม FSUN)", MD)
        for text in (f"Pool CF {f2(tot['pool'])}", f"EV Reserve {f2(tot['reserve'])}", f"Ev Burn {f2(tot['burn'])}", f"Fix_c รวม {f2(tot['fix_c'])}"):
            self.assertIn(text, MD)


# ── the Markdown ──────────────────────────────────────────────────────────────────────────
class MarkdownTest(unittest.TestCase):
    def test_it_carries_the_very_same_picture(self):
        charts = fenced("mermaid")
        self.assertEqual(len(charts), 1)
        self.assertEqual(charts[0], strip_comments(MMD))                                 # the header comments stay in the .mmd; the drawing is identical
        self.assertIn(f"> **{page_const('ESSENCE')}**", MD)
        self.assertIn(page_const("CYCLE"), MD)                                           # your cycle, as the page shows it

    def test_the_worked_example_is_computed_not_typed(self):
        n = example_numbers()
        part = section(4)
        name, c, t0 = page_const("DEMO")
        p1, p2 = page_const("DEMO_PATH")
        self.assertIn(f"Fix_c {c:,.0f} · P₀ {t0:g} → {p1:g} → {p2:g} · Hedge ×1 · ใส่ EV Reserve {page_const('DEMO_FUND'):g}", part.splitlines()[0])
        for key in ("k", "reserve_paid", "v1", "gap1", "qty1", "r1", "da1", "a1", "e1", "s1", "v2", "gap2", "qty2", "r2", "da2", "a2", "e2", "s2", "b2",
                    "c_new", "beta", "pool_end", "reserve_end", "settle_pool", "settle_reserve", "k_next", "eff"):
            self.assertIn(n[key], part, key)
        self.assertIn(f"Ev Efficiency {n['e_total']} ÷ {n['k']} = {n['eff']}", part)
        self.assertIn(f"K′ = {cv.PREMIUM_RATE:.1%} × {n['c_new']} = {n['k_next']}", part)
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in part.splitlines() if ln.startswith("| ") and "---" not in ln][1:]
        self.assertEqual(len(rows), 7)                                                   # six steps and the ↻ ① line
        for row, steps in zip(rows, (2, 3, 4, 5, 6, 7)):                                 # the treasury and the baseline after each step are the core's
            led = cv.replay(DEMO[:steps])
            tot, t = cv.totals(led), led["tickers"]["DEMO"]
            self.assertEqual(row[5:9], [f2(tot["pool"]), f2(tot["reserve"]), f2(tot["burn"]), f2(t["b"])], row[0])
            if steps in (3, 4, 5, 6):                                                    # while the cycle is open: what has been harvested, and what is left
                self.assertEqual(row[3:5], [f2(cv.e_cyc(t)), sg(cv.surplus(t))], row[0])

    def test_the_four_requirements_point_at_groups_that_exist(self):
        titles = " ".join(ln for ln in BODY.splitlines() if ln.lstrip().startswith("subgraph"))
        for group in ("วงจรของแต่ละ ticker", "Chain data", "กระเป๋ากลาง"):
            self.assertIn(group, titles)
            self.assertIn(f'กลุ่ม "{group}"', MD)
        for ui in ("🔁 วงจรของ", "📊 วัดผล", "💾 สำรอง / กู้คืนข้อมูล · นำเข้า Chain data", "🏛️ ความเคลื่อนไหวกระเป๋ากลาง",
                   "💸 โอนเงินระหว่างกระเป๋า", "➕ เริ่มติดตามตัวใหม่"):
            self.assertIn(ui, MD, ui)
            self.assertIn(ui, PAGE, ui)                                                  # the page really has what the document points at

    def test_the_principles_and_the_six_prohibitions_are_all_there(self):
        rows = re.findall(r"^\| P(\d) \| (.+?) \|", MD, re.M)
        self.assertEqual([(int(i), name) for i, name in rows], list(enumerate(PRINCIPLES, 1)))
        banned = re.findall(r"^\| ([1-6]) (.+?) \| \*\*(.+?)\*\*", section(8), re.M)
        self.assertEqual([int(i) for i, _, _ in banned], [1, 2, 3, 4, 5, 6])
        self.assertEqual({kind for _, _, kind in banned}, {"โครงสร้าง", "โครงสร้าง (ค่าเริ่มต้น)", "ข้อความ + ช่องติ๊ก"})
        self.assertIn("Zero-Cost ≠ ไม่เสี่ยง", PAGE)                                      # the text prohibition 6 relies on is on the page

    def test_every_test_it_cites_exists(self):
        cites = re.findall(r"`tests/(test_\w+\.py)::(\w+)::(test_\w+)`", MD)
        self.assertGreaterEqual(len(cites), 20)
        for file, cls, name in cites:
            tree = ast.parse((HERE / file).read_text(encoding="utf-8"))
            klass = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
            self.assertIsNotNone(klass, f"{file}::{cls}")
            self.assertIn(name, {f.name for f in klass.body if isinstance(f, ast.FunctionDef)}, f"{file}::{cls}::{name}")

    def test_every_link_points_at_a_file(self):
        targets = set(re.findall(r"\]\(([^)#]+)\)", MD))
        self.assertTrue({"app_v4_system.mmd", "pages/app-v4.py", "chain_v4.py", "tests/test_chain_v4.py", "tests/test_app_v4.py",
                         "tests/test_v4_charts.py", "app_v3_final.md", "app_v3_playbook.mmd"} <= targets)
        for target in targets:
            self.assertTrue((ROOT / target).exists(), target)

    def test_the_counted_lists_match_their_headings(self):
        nine = section(9)
        count = len(re.findall(r"^\d+\. ", nine, re.M))
        self.assertIn(f"ข้อแม้ {count} ข้อ", nine.splitlines()[0])
        ten = section(10)
        marks = re.findall(r"^\d+\. (✅|❓) ", ten, re.M)
        self.assertEqual(len(marks), 4)
        self.assertIn(f"ข้อค้างเดิม 4 ข้อ — ปิดแล้ว {marks.count('✅')} · ยังเปิด {marks.count('❓')}", ten.splitlines()[0])
        self.assertIn("Put Ratio 2-1", ten)                                              # the one that stays open

    def test_the_cut_table_names_what_the_core_says_it_cut(self):
        for cut in ("state ซ้ำ", "Extract / Deploy / Allocate / Funding / EV flow", "`cash` ชนิดเดียว"):
            self.assertIn(cut, section(7))
        self.assertIn("one ``cash`` step", cv.__doc__)                                   # the same cut, said in the module's own docstring
        self.assertIn("a ``move`` before a ``pay`` is simply not a legal step", cv.__doc__)


if __name__ == "__main__":
    unittest.main()
