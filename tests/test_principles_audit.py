"""Audit of the nine principles (P1–P9) of the Chain System v3 — every claim is *recomputed*, not re-read.

One test per principle, named after the claim. Each says what holds (✅) and, where it holds only under a condition,
what the condition is (⚠️). `app_v3_final.md` quotes these results and `app_v3_final.mmd` draws them; the second class at
the bottom keeps those two files, the code and these numbers from drifting apart (stdlib only; no Streamlit, no Mermaid).
"""
import math
import random
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from test_system_flowchart import parse       # noqa: E402
import chain_v3 as cv                         # noqa: E402
import ledger_v2 as led                       # noqa: E402

C = 1500.0
P3_PATHS, P6_PATHS = 300, 2000                # random paths behind the P3 and P6 checks (app_v3_final.md quotes both)
K1 = cv.plan(C, 10.0, 15.0, 1.0)["K"]         # 185.94 — the premium of the worked example (Hedge ×1)

# where each principle is evidenced: (test file, test name) — the audit's own test first, then the older tests behind it
EVIDENCE = {
    "P1": [("test_principles_audit.py", "test_p1_the_harvest_is_never_negative_and_it_is_the_swing"),
           ("test_chain_v3.py", "test_the_harvest_is_the_ledgers_E_equation")],
    "P2": [("test_principles_audit.py", "test_p2_the_premium_is_black_scholes_and_the_surplus_is_read_after_it"),
           ("test_chain_v3.py", "test_the_premium_constant_is_black_scholes")],
    "P3": [("test_principles_audit.py", "test_p3_free_risk_is_where_the_money_comes_from_not_how_big_the_position_gets"),
           ("test_chain_v3.py", "test_the_goal_holds_on_any_path")],
    "P4": [("test_principles_audit.py", "test_p4_rollover_keeps_r_continuous_and_e_honest_when_the_capital_changes"),
           ("test_playbook_flowchart.py", "test_step_5_scale_up_uses_the_rollover_of_streamlit_lego")],
    "P5": [("test_principles_audit.py", "test_p5_cash_lanes_move_the_baseline_not_the_ledger"),
           ("test_playbook_flowchart.py", "test_the_capital_gain_example_matches_chain_v3_and_the_ledger")],
    "P6": [("test_principles_audit.py", "test_p6_ev_efficiency_is_the_surplus_question_said_as_a_ratio"),
           ("test_minimal_loop.py", "test_the_goal_clauses_hold_on_every_path")],
    "P7": [("test_chain_v3.py", "test_an_outdated_plan_saves_nothing"),
           ("test_chain_v3.py", "test_saving_is_all_or_nothing_and_leaves_no_temp_files")],
    "P8": [("test_principles_audit.py", "test_p8_the_trend_is_cut_out_r_follows_the_endpoints_e_follows_the_swing"),
           ("test_chain_v3.py", "test_a_plain_rise_does_not_pay_only_the_swing_does")],
    "P9": [("test_app_v3.py", "test_commit_writes_exactly_what_the_table_showed"),
           ("test_app_v3.py", "test_an_outdated_click_commits_nothing")],
}


def one_way(n: int, c: float = C, rise: float = 1.5) -> float:
    """E of a one-way rise by the factor `rise`, cut into n equal (geometric) acts."""
    xn = rise ** (1.0 / n)
    return n * c * (xn - 1.0 - math.log(xn))


def added_pnl(size: float, p_from: float, p_to: float, acts: int) -> float:
    """What a position of constant value `size`, added at p_from, has made (units · price + cash) after a fall to p_to in `acts` steps."""
    units, cash, p = size / p_from, -size, p_from
    x = (p_to / p_from) ** (1.0 / acts)
    for _ in range(acts):
        p *= x
        cash += units * p - size              # rebalance back to the value `size`: sell the excess (buy when it is negative)
        units = size / p
    return units * p + cash


def chain_round(p_old: float, p_new: float, b_before: float, c: float = C, k: float = K1) -> dict:
    """A Chain Round as the round log keeps it (Hedge ×1, no scale up)."""
    shannon = round(c * math.log(p_new / p_old), 2)
    return {"action": "Chain Round", "p_old": p_old, "p_new": p_new, "c_before": c, "c_after": c, "shannon_profit": shannon,
            "harvest_profit": 0.0, "hedge_cost": k, "surplus": round(shannon - k, 2), "scale_up": 0.0,
            "b_before": b_before, "b_after": round(b_before + shannon, 2), "hedge_ratio": 1.0, "sigma": 0.5, "ev_change": 0.0, "note": ""}


def extract_round(p: float, b_before: float, b_after: float = 0.0, c: float = C) -> dict:
    """Extract Baseline: only the baseline moves (P and c stay)."""
    return {"action": "Extract Baseline", "p_old": p, "p_new": p, "c_before": c, "c_after": c, "shannon_profit": 0.0,
            "harvest_profit": 0.0, "hedge_cost": 0.0, "surplus": 0.0, "scale_up": 0.0, "b_before": b_before, "b_after": b_after,
            "hedge_ratio": 1.0, "sigma": 0.5, "ev_change": 0.0, "note": ""}


class PrinciplesAuditTest(unittest.TestCase):
    def test_p1_the_harvest_is_never_negative_and_it_is_the_swing(self):
        moves = [10 ** (k / 100) for k in range(-300, 301)]                           # a move by 0.001× … 1000×
        self.assertGreaterEqual(min(C * (x - 1.0 - math.log(x)) for x in moves), -1e-9)    # ✅ E ≥ 0 for every move (x − 1 ≥ ln x)
        self.assertEqual(C * (1.0 - 1.0 - math.log(1.0)), 0.0)                         # ✅ and 0 only when nothing moved
        sizes = [round(one_way(n), 2) for n in (1, 2, 10, 1000)]                       # ⚠️ its size depends on how often you act:
        self.assertEqual(sizes, [141.80, 66.04, 12.50, 0.12])                          #    the same ×1.5 rise in 1, 2, 10, 1000 acts
        self.assertAlmostEqual(one_way(1000), C * math.log(1.5) ** 2 / 2000, delta=0.01)   # ≈ c·(ln x)² / 2n → 0: the swing, not the rise
        there_and_back = cv.plan(C, 10.0, 15.0)["E"] + cv.plan(C, 15.0, 10.0)["E"]     # a swing pays whatever the schedule …
        self.assertAlmostEqual(there_and_back, 250.0, places=2)                        # … 10 → 15 → 10 ends with R = 0 and E = 250

    def test_p2_the_premium_is_black_scholes_and_the_surplus_is_read_after_it(self):
        n = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))                       # noqa: E731
        s0, k, t, r, sg = 1.0, 0.9, 1.0, 0.04, 0.5                                     # a put 10% out of the money, 1 year, σ 50%, r 4%
        d1 = (math.log(s0 / k) + (r + sg * sg / 2.0) * t) / (sg * math.sqrt(t))
        d2 = d1 - sg * math.sqrt(t)
        put = k * math.exp(-r * t) * n(-d2) - s0 * n(-d1)
        self.assertAlmostEqual(put, cv.PREMIUM_RATE, places=9)                         # ✅ the constant is an independent Black-Scholes put
        double = cv.plan(C, 10.0, 15.0, 2.0)
        self.assertAlmostEqual(double["K"], 2 * cv.PREMIUM_RATE * C, delta=0.005)      # ✅ K ∝ c · Hedge×
        self.assertAlmostEqual(double["S"], double["E"] - double["K"], places=9)       # ✅ Surplus = E − K: the insurance is taken first
        real = cv.plan(C, 10.0, 15.0, 1.0, k=120.0)                                    # ⚠️ it is a model premium, not the LEAPS' own:
        self.assertEqual((real["K"], real["real_k"]), (120.0, True))                   #    the real premium replaces it (k)

    def test_p3_free_risk_is_where_the_money_comes_from_not_how_big_the_position_gets(self):
        rng = random.Random(3)
        for _ in range(P3_PATHS):                                                      # ✅ c never shrinks and grows by exactly the surplus
            c, t = C, 100.0
            for _ in range(rng.randint(1, 10)):
                p = t * math.exp(rng.gauss(0.0, 0.15))
                pl = cv.plan(c, t, p)
                self.assertAlmostEqual(pl["c_after"], c + max(0.0, pl["S"]), places=9)
                self.assertGreaterEqual(pl["c_after"], c)
                c, t = pl["c_after"], p
        # ⚠️ but a bigger c is a bigger loss per 1% of price: the part added at 15 (size 100) after a fall to 5 …
        self.assertAlmostEqual(added_pnl(100.0, 15.0, 5.0, 1), -66.67, places=2)       #    within its own size in one step,
        slow = added_pnl(100.0, 15.0, 5.0, 10000)
        self.assertAlmostEqual(slow, 100.0 * math.log(5.0 / 15.0), delta=0.01)         #    −109.86 when the fall is gradual …
        self.assertLess(slow, -100.0)                                                  #    … more than the money added: the shield comes first
        self.assertAlmostEqual(added_pnl(100.0, 15.0, 1.5, 10000), 100.0 * math.log(0.1), delta=0.05)   # −230.23 on a −90% fall

    def test_p4_rollover_keeps_r_continuous_and_e_honest_when_the_capital_changes(self):
        c0, c1, p0, pa = 1500.0, 2500.0, 100.0, 110.0                                  # Fix_c → Fix_c′ at the pivot P_acted; anchor P₀
        ln = math.log(pa / p0)
        a_cash = c0 * (pa / p0 - 1.0)
        beta = (c0 - c1) * ln                                                          # LEGO: β += (Fix_c − Fix_c′) · ln(P_acted / P₀)
        self.assertAlmostEqual(beta + c1 * ln, c0 * ln, places=9)                      # ✅ R does not jump at the pivot
        self.assertAlmostEqual(a_cash - (beta + c1 * ln), 7.03, places=2)              # ✅ E is the same before and after the scale up
        self.assertAlmostEqual(a_cash - c1 * ln, -88.28, places=2)                     # ⚠️ without β it falls to −88.28: E ≥ 0 breaks
        scale_up = dict(chain_round(p0, pa, 0.0), c_after=c1, scale_up=c1 - c0)
        follow = chain_round(pa, 100.0, scale_up["b_after"], c=c1)
        rows = led.build({"rounds": [scale_up, follow]})["rows"]
        self.assertTrue(all(abs(x["R_beta"] - x["R"]) < 1e-9 for x in rows))           # ✅ the ledger: β + Fix_c·ln(P/P₀) = the accumulated R

    def test_p5_cash_lanes_move_the_baseline_not_the_ledger(self):
        first = chain_round(10.0, 15.0, 0.0)
        extract = extract_round(15.0, first["b_after"])                                # roll P up to 15, pull the baseline (= R) out
        banked = [first, extract, chain_round(15.0, 10.0, 0.0)]
        kept = [first, chain_round(15.0, 10.0, first["b_after"])]
        for rounds in ([first], [first, extract]):
            s = led.build({"rounds": rounds})["summary"]
            self.assertEqual((round(s["A"], 2), round(s["R"], 2), round(s["E"], 2)), (750.00, 608.20, 141.80))   # ✅ A·R·E untouched
        ledger = led.build({"rounds": banked})
        self.assertAlmostEqual(ledger["summary"]["E"], led.build({"rounds": kept})["summary"]["E"], places=9)    # ✅ E = 250 either way
        self.assertEqual(ledger["summary"]["breaks"], [])                              # ✅ the chain stays unbroken
        self.assertTrue(led.integrity({"rounds": banked}, ledger)[1])                  # ✅ and the ledger's own audit passes
        # ⚠️ the cash that left is no longer the demon's buy-back reserve: back at 10 it needs 500 (a jump) to 608 (a slide)
        self.assertAlmostEqual(C * (1.0 - 10.0 / 15.0), 500.0, places=9)
        self.assertAlmostEqual(C * math.log(15.0 / 10.0), 608.20, places=2)

    def test_p6_ev_efficiency_is_the_surplus_question_said_as_a_ratio(self):
        rng = random.Random(7)
        paths = 0
        for _ in range(P6_PATHS):
            c, t, e, k, s = C, 100.0, 0.0, 0.0, 0.0
            for _ in range(rng.randint(1, 12)):
                p = t * math.exp(rng.gauss(0.0, 0.12))
                pl = cv.plan(c, t, p)
                e, k, s, c, t = e + pl["E"], k + pl["K"], s + pl["S"], pl["c_after"], p
            if k > 0:
                paths += 1
                self.assertEqual(e / k >= 1.0, s >= -1e-9)                             # ✅ ΣE ÷ ΣK ≥ 1.0x  ⇔  ΣSurplus ≥ 0
        self.assertGreater(paths, P6_PATHS * 0.95)
        # ⚠️ so after one cycle the two questions are one (E > K at ③ already makes the ratio > 1): the check is a label for the
        #    whole chain's status, counted over all cycles including the premium of a bought Deep ITM, not a second gate
        e, k = 141.80 + 108.20, K1
        self.assertGreater(e / k, 1.0)

    def test_p7_and_p9_are_software_principles_and_their_evidence_is_in_the_tests(self):
        for principle in ("P7", "P9"):
            self.assertTrue(EVIDENCE[principle])                                       # ✅ atomic + stale-proof · preview → the person commits
        self.test_every_evidence_test_exists()

    def test_p8_the_trend_is_cut_out_r_follows_the_endpoints_e_follows_the_swing(self):
        mid = 10.0 * math.sqrt(1.5)
        direct, two = [(10.0, 15.0)], [(10.0, mid), (mid, 15.0)]
        r = lambda path: sum(C * math.log(b / a) for a, b in path)                     # noqa: E731
        e = lambda path: sum(cv.plan(C, a, b)["E"] for a, b in path)                   # noqa: E731
        self.assertAlmostEqual(r(direct), r(two), places=9)                            # ✅ R = c·ln(P / t₀): the same for both paths …
        self.assertAlmostEqual(r(direct), 608.20, places=2)
        self.assertEqual((round(e(direct), 2), round(e(two), 2)), (141.80, 66.04))     # … E is not: it follows the swing, not the rise

    def test_every_evidence_test_exists(self):
        for principle, items in EVIDENCE.items():
            for name_file, name in items:
                source = (HERE / name_file).read_text(encoding="utf-8")
                self.assertRegex(source, rf"def {name}\(", f"{principle}: {name_file}::{name}")


KEY_LINE = "ประกันก่อน · ความแกว่งจ่าย · โตด้วยกำไรที่ล็อกแล้ว · เทรนด์เก็บเข้ากระเป๋า"
NAMES = {"P1": "Shannon's Demon", "P2": "Pay-first", "P3": "Free-Risk Scale-up", "P4": "Continuous Rollover", "P5": "Cash Lanes",
         "P6": "Ev Efficiency", "P7": "Atomic + stale-proof", "P8": "อ่านกำไรที่ E", "P9": "Human-in-the-loop"}


def fmt(x: float, signed: bool = False) -> str:
    """Two decimals as the documents write them: a real minus sign, an explicit plus when asked."""
    s = f"{abs(x):.2f}"
    return "−" + s if x < 0 else ("+" if signed else "") + s


def label(text: str, node: str) -> str:
    line = next(ln for ln in text.splitlines() if re.match(rf"^\s*{node}\s*[\[({{]", ln))
    return re.sub(r"</?[bi]>", "", line).replace("<br/>", " ")


class FinalKeyMessageTest(unittest.TestCase):
    """app_v3_final.md and app_v3_final.mmd say what the audit above found — in the same words and with the same numbers."""

    @classmethod
    def setUpClass(cls):
        cls.md = (ROOT / "app_v3_final.md").read_text(encoding="utf-8")
        cls.mmd = (ROOT / "app_v3_final.mmd").read_text(encoding="utf-8")
        cls.body = cls.mmd.split("flowchart TB", 1)[1]
        cls.nodes, cls.subgraphs, cls.edges, cls.depth = parse(cls.mmd)
        cls.rows = dict(re.findall(r"^\| (P\d) \|(.*)$", cls.md, re.M))

    def row(self, start: str) -> str:
        return next(ln for ln in self.md.splitlines() if ln.startswith(f"| {start}"))

    def section(self, marker: str) -> str:
        return next(s for s in self.md.split("\n## ") if s.startswith(marker))

    def test_both_files_say_the_same_key_message(self):
        self.assertIn(f"> **{KEY_LINE}**", self.md)
        self.assertIn(KEY_LINE, self.mmd.split("\n---\n", 1)[0])                    # the chart's title …
        self.assertIn(f"<b>{KEY_LINE}</b>", label_raw(self.mmd, "KEY"))              # … and its first box

    def test_the_chart_is_small_flat_and_the_chain_closes(self):
        self.assertEqual(self.nodes, {"KEY", "PAY", "SWING", "D", "TREND", "GROW", "MEASURE", "GOAL", "AUDIT"})
        self.assertEqual((len(self.subgraphs), self.depth), (0, 0))
        self.assertEqual(sorted({x for e in self.edges for x in e} - self.nodes), [])
        into = lambda n: {a for a, b in self.edges if b == n}                       # noqa: E731
        out = lambda n: {b for a, b in self.edges if a == n}                        # noqa: E731
        self.assertEqual((into("PAY"), into("SWING"), out("D")), ({"KEY", "MEASURE"}, {"PAY", "TREND"}, {"GROW", "TREND"}))
        self.assertEqual((into("TREND"), out("TREND")), ({"D"}, {"SWING"}))          # the trend is banked, then back to harvesting E
        self.assertEqual((into("GROW"), out("GROW")), ({"D"}, {"MEASURE"}))          # growth is always measured …
        self.assertEqual((into("MEASURE"), out("MEASURE")), ({"GROW"}, {"GOAL", "PAY"}))   # … the measure reaches the goal or goes on
        self.assertEqual((into("GOAL"), into("AUDIT")), ({"MEASURE"}, {"GOAL"}))     # and the audit hangs on the goal
        self.assertRegex(self.body, r"MEASURE\s*-\.->\|วนต่อ ↻ K ของ c′\|\s*PAY")  # the chain closes whether or not the goal is reached
        playbook = (ROOT / "app_v3_playbook.mmd").read_text(encoding="utf-8")
        self.assertIn("Q -.->|ถึงเป้า ก็วนต่อ ↻ K ของ c′| K", playbook)               # … exactly as the playbook now says

    def test_all_nine_principles_are_on_the_chart_and_the_playbook_uses_a_subset(self):
        tags = lambda text: set(re.findall(r"\bP\d\b", text))                       # noqa: E731
        self.assertEqual(tags(self.body), set(NAMES))                                # P1 … P9, none missing
        for tag, name in NAMES.items():
            self.assertIn(f"{tag} {name}", self.body)
        playbook = (ROOT / "app_v3_playbook.mmd").read_text(encoding="utf-8").split("flowchart TB", 1)[1]
        self.assertEqual(tags(playbook), set(NAMES) - {"P7", "P9"})                  # the playbook is the strategy: no software principles
        self.assertEqual(set(self.rows), set(NAMES))                                 # the md has one row per principle

    def test_the_goal_is_word_for_word_the_same_as_the_other_charts(self):
        goal = lambda t: re.search(r'^\s*GOAL\("(.*)"\)\s*$', t, re.M).group(1)    # noqa: E731
        for other in ("app_v2_minimal.mmd", "app_v2_system.mmd", "app_v3_playbook.mmd"):
            self.assertEqual(goal(self.mmd), goal((ROOT / other).read_text(encoding="utf-8")), other)

    def test_the_audit_box_counts_what_the_md_lists(self):
        box = label(self.mmd, "AUDIT")
        caveats = re.findall(r"^\d+\. ", self.section("⚠️ 5."), re.M)
        open_items = re.findall(r"^\d+\. ", self.section("❓ 6."), re.M)
        fixed = re.findall(r"^\d+\. ", self.section("🛠 7."), re.M)
        self.assertEqual((len(self.rows), len(EVIDENCE)), (9, 9))
        self.assertIn(f"✅ {len(self.rows)} จาก 9 หลักการตรวจแล้ว", box)
        self.assertIn(f"⚠️ ข้อแม้ {len(caveats)}", box)
        self.assertIn(f"❓ รอตัดสิน {len(open_items)}", box)
        self.assertIn(f"🛠 แก้แล้ว {len(fixed)}", box)
        self.assertIn(f"## ⚠️ 5. ข้อแม้ {len(caveats)} ข้อ", self.md)
        self.assertIn(f"## ❓ 6. ยังไม่ปิด {len(open_items)} ข้อ", self.md)
        self.assertIn(f"## 🛠 7. สิ่งที่ audit แก้ให้แล้ว {len(fixed)} ข้อ", self.md)
        flagged = [n for n in self.nodes - {"AUDIT"} if "⚠️" in label(self.mmd, n)]    # the box that counts them does not count itself
        self.assertEqual(len(flagged), len(caveats))                                 # every caveat has its ⚠️ on one box of the chart

    def test_each_principle_cites_the_tests_that_back_it(self):
        for principle, row in self.rows.items():
            cited = set(re.findall(r"tests/(test_\w+\.py)::(test_\w+)", row))
            self.assertEqual(cited, set(EVIDENCE[principle]), principle)

    def test_the_numbers_in_the_md_are_the_computed_ones(self):
        plan = cv.plan(C, 10.0, 15.0)
        back = cv.plan(C, 15.0, 10.0)
        a_cash, trend = C * (15.0 / 10.0 - 1.0), C * math.log(1.5)
        e, k, s = plan["E"], plan["K"], plan["S"]
        both = e + back["E"]
        self.assertIn(f"**{fmt(a_cash)}**", self.row("capital gain"))
        self.assertIn(f"{fmt(trend)} ({round(trend / a_cash * 100)}%)", self.row("├ เทรนด์ R"))
        self.assertIn(f"{fmt(e)} ({round(e / a_cash * 100)}%)", self.row("└ ความแกว่ง E"))
        self.assertIn(f"{fmt(k)} | E จ่ายได้ {round(e / k * 100)}%", self.row("ค่าประกัน K"))
        self.assertIn(f"**{fmt(s)}**", self.row("Surplus = E − K"))
        self.assertIn(f"ดึง R {fmt(trend)} ออก · E {fmt(e)} อยู่ต่อ", self.row("Surplus = E − K"))
        swing = self.row("ถ้าแกว่งกลับ 10")
        self.assertIn(f"E เพิ่ม {fmt(back['E'], True)}", swing)
        self.assertIn(f"ΣE = {fmt(both)} → Surplus **{fmt(both - k, True)}**", swing)
        # P1 / P8: the act schedule, the swing there and back, and R against E on two paths with the same ends
        self.assertIn(f"ได้ E {fmt(both)}", self.rows["P1"])
        sizes = " · ".join(f"{label_} {fmt(one_way(n))}" for label_, n in (("2 ก้าว", 2), ("10 ก้าว", 10), ("1,000 ก้าว", 1000)))
        self.assertIn(f"1 ก้าว E = {fmt(one_way(1))} · {sizes}", self.section("⚠️ 5."))
        mid = 10.0 * math.sqrt(1.5)
        two = cv.plan(C, 10.0, mid)["E"] + cv.plan(C, mid, 15.0)["E"]
        self.assertIn(f"({fmt(trend)} ทั้ง 1 ก้าวและ 2 ก้าว)", self.rows["P8"])
        self.assertIn(f"({fmt(e)} กับ {fmt(two)})", self.rows["P8"])
        # P2: the constant · P3: the paths and what the added size loses · P4: Rollover · P5: the untouched ledger and the buy-back
        self.assertIn(f"ค่าคงที่ {cv.PREMIUM_RATE * 100:.1f}%", self.rows["P2"])
        self.assertIn(f"({P3_PATHS} เส้นทางสุ่ม)", self.rows["P3"])
        self.assertIn(f"({P6_PATHS:,} เส้นทางสุ่ม", self.rows["P6"])
        step = (f"ขาดทุน {fmt(added_pnl(100.0, 15.0, 5.0, 1))} ถ้าลงก้าวเดียว แต่ {fmt(added_pnl(100.0, 15.0, 5.0, 10000))} "
                f"ถ้าไหลลงทีละน้อย (เกิน S) และ {fmt(added_pnl(100.0, 15.0, 1.5, 10000))} ถ้าลง 90%")
        self.assertIn(step, self.section("⚠️ 5."))
        c0, c1, p0, pa = 1500.0, 2500.0, 100.0, 110.0
        ln, a_small = math.log(pa / p0), c0 * (pa / p0 - 1.0)
        self.assertIn(f"E เพี้ยนจาก {fmt(a_small - c0 * ln)} เป็น {fmt(a_small - c1 * ln)} (100 → 110 · 1,500 → 2,500)", self.rows["P4"])
        self.assertIn(f"({fmt(a_cash)} · {fmt(trend)} · {fmt(e)})", self.rows["P5"])
        self.assertIn(f"ต้องซื้อคืน {round(C * (1 - 10 / 15))} (กระโดดกลับ) ถึง {round(trend)} (ไหลกลับทีละน้อย)", self.section("⚠️ 5."))
        self.assertIn(f"เบี้ยโมเดล {cv.PREMIUM_RATE * 100:.1f}%", label(self.mmd, "PAY"))

    def test_the_md_links_only_to_files_that_exist(self):
        targets = set(re.findall(r"\]\(([^)#]+)\)", self.md))
        self.assertTrue({"app_v3_final.mmd", "app_v3_playbook.mmd", "app_v2_minimal.mmd", "chain_v3.py", "ledger_v2.py",
                         "tests/test_principles_audit.py"} <= targets)
        for target in targets:
            self.assertTrue((ROOT / target).exists(), target)

    def test_what_the_md_says_it_fixed_is_in_the_playbook(self):
        playbook = (ROOT / "app_v3_playbook.mmd").read_text(encoding="utf-8")
        self.assertIn("เพิ่มเส้นประ วัดผล → ① (ถึงเป้า ก็วนต่อ)", self.section("🛠 7."))
        for text in ("วงจรปิดเสมอ", "\"วัดผล\" เป็นป้ายสถานะ", "เป็นคำถามเดียวกันคนละหน่วย"):
            self.assertIn(text, playbook.split("flowchart TB", 1)[0])


def label_raw(text: str, node: str) -> str:
    """The node's line as written (tags kept)."""
    return next(ln for ln in text.splitlines() if re.match(rf"^\s*{node}\s*[\[({{]", ln))


if __name__ == "__main__":
    unittest.main()
