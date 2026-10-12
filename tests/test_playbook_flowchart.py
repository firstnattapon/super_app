"""app_v3_playbook.mmd — the minimal loop said with the instruments (stdlib only; no Mermaid needed).

The flowchart is hand-written Mermaid. These tests guard what is easy to break silently: a step that lost its instrument,
the 20% / 80% split that stopped adding up to a whole, the order "pay K first, then harvest E", the way out of the
question "is the Surplus positive?" (shield, then scale up fix_c with Liquidity or a Deep ITM, then the check), the
Rollover that step 5 uses (the equation of streamlit_lego, checked against the project's ledger), the goal text, and
Mermaid's one parser trap for comments.
"""
import math
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from test_system_flowchart import parse       # noqa: E402


def label(text: str, node: str) -> str:
    """The label of one node, tags stripped (Mermaid labels are one line in this file)."""
    line = next(ln for ln in text.splitlines() if re.match(rf"^\s*{node}\s*[\[({{]", ln))
    return re.sub(r"</?[bi]>", "", line).replace("<br/>", " ")


class PlaybookChartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "app_v3_playbook.mmd").read_text(encoding="utf-8")
        cls.body = cls.text.split("flowchart TB", 1)[1]
        cls.nodes, cls.subgraphs, cls.edges, cls.depth = parse(cls.text)

    def test_it_is_small_flat_and_every_edge_has_both_ends(self):
        self.assertTrue(self.text.startswith("---\ntitle:"))
        self.assertEqual(self.nodes, {"K", "E", "D", "A", "S", "L", "I", "RO", "Q", "GOAL"})
        self.assertEqual((len(self.subgraphs), self.depth), (0, 0))
        self.assertEqual(sorted({x for e in self.edges for x in e} - self.nodes), [])

    def test_pay_first_then_harvest_then_ask(self):
        self.assertIn(("K", "E"), self.edges)
        self.assertIn(("E", "D"), self.edges)
        into = lambda n: {a for a, b in self.edges if b == n}                   # noqa: E731
        self.assertEqual(into("E"), {"K", "D"})                                  # E follows K, or D saying "keep harvesting"
        self.assertEqual(into("K"), {"Q"})                                       # K is paid again only when a new cycle starts
        self.assertEqual(into("D"), {"E"})

    def test_no_surplus_means_keep_harvesting_not_paying_again(self):
        self.assertIn(("D", "E"), self.edges)
        self.assertNotIn(("D", "K"), self.edges)
        line = next(ln for ln in self.body.splitlines() if re.match(r"^\s*D\s*-->\|ไม่", ln))
        self.assertIn("เก็บ E ต่อ", line)

    def test_a_positive_surplus_goes_shield_then_scale_up_then_rollover_then_the_check(self):
        for edge in (("D", "A"), ("A", "S"), ("S", "L"), ("S", "I"), ("L", "RO"), ("I", "RO"), ("RO", "Q"), ("Q", "GOAL"), ("Q", "K")):
            self.assertIn(edge, self.edges)
        into = lambda n: {a for a, b in self.edges if b == n}                   # noqa: E731
        self.assertEqual((into("A"), into("S"), into("RO"), into("Q")), ({"D"}, {"A"}, {"L", "I"}, {"RO"}))
        self.assertNotIn(("L", "Q"), self.edges)                                 # no way to the check that skips the Rollover
        self.assertNotIn(("I", "Q"), self.edges)
        self.assertNotIn("Long Call", self.body)                                 # the earlier "or buy Long Put + Long Call" is gone

    def test_each_step_names_its_instrument(self):
        step = {n: label(self.text, n) for n in "KEDASLI"}
        for text in ("จ่ายค่าประกัน (K) ก่อนเสมอ", "Deep ITM LEAPS", "Stock Replacement"):
            self.assertIn(text, step["K"])
        for text in ("เก็บกำไรจากความแกว่ง", "E = A − R", "หุ้น 20%", "Liquidity", "Buying Power (LEAPS) 80%", "Stock Replacement"):
            self.assertIn(text, step["E"])
        self.assertIn("Surplus = E − K", step["D"])
        for text in ("Black Swan Shields", "Long Put", "Zero-Cost Hedging", "Strategy Active", "Put Ratio 2-1"):
            self.assertIn(text, step["A"])
        self.assertNotIn("2:1", self.body)                                       # written as given: 2-1
        for text in ("Scale Up fix_c", "Zero-Cost", "ดูความเหมาะสม"):
            self.assertIn(text, step["S"])
        self.assertIn("Liquidity", step["L"])                                    # scale up with the Liquidity itself …
        self.assertIn("Deep ITM", step["I"])                                     # … or by buying a Deep ITM

    def test_the_choice_between_liquidity_and_a_deep_itm_has_a_rule(self):
        """⑤ asks one question — is there enough Liquidity left to pay the next K? — enough: Liquidity; not enough: Deep ITM."""
        edge_label = lambda a, b: re.search(rf"^\s*{a}\s*-->\|([^|]*)\|\s*{b}\s*$", self.body, re.M).group(1)   # noqa: E731
        self.assertEqual(edge_label("S", "L"), "พอ")
        self.assertEqual(edge_label("S", "I"), "ไม่พอ")
        step = {n: label(self.text, n) for n in "SLI"}
        self.assertIn("Liquidity หลังใช้ ยังเหลือพอจ่าย K ของวงจรหน้า", step["S"])
        self.assertIn("ไม่มีเบี้ยใหม่", step["L"])                               # why the Liquidity is the plain way
        self.assertIn("เบี้ยที่จ่ายเพิ่ม = K ใหม่", step["I"])                     # what the Deep ITM costs …
        self.assertIn("ΣK", step["I"])                                           # … and that Ev Efficiency has to count it
        header = self.text.split("flowchart TB", 1)[0]
        self.assertIn("เกณฑ์เลือก ก./ข. (ข้อเสนอ", header)                        # the rule is marked as a proposal,
        self.assertIn("ไม่ใช่กฎที่ระบุมา", header)                                 # not as something the playbook's author said

    def test_the_split_is_a_whole(self):
        shares = [int(x) for x in re.findall(r"(\d+)%", label(self.text, "E"))]
        self.assertEqual(shares, [20, 80])
        self.assertEqual(sum(shares), 100)

    def test_e_is_a_minus_r_and_never_the_other_way_round(self):
        self.assertIn("E = A − R", self.body)
        self.assertNotIn("R − A", self.body)                                     # the nodes: R − A would be negative, E ≥ 0 comes from A − R
        self.assertIn("ถ้าเขียน R − A จะติดลบเสมอ", self.text)                    # and the header says why

    def test_the_principles_it_uses_carry_their_names(self):
        for tag, name in (("P1", "Shannon's Demon"), ("P2", "Pay-first"), ("P3", "Free-Risk Scale-up"), ("P4", "Continuous Rollover"),
                          ("P6", "Ev Efficiency"), ("P8", "อ่านกำไรที่ E")):
            self.assertIn(f"{tag} {name}", self.body)
        for dropped in ("P5", "P7", "P9"):
            self.assertNotRegex(self.body, rf"\b{dropped}\b")
        tags = lambda text: set(re.findall(r"\bP\d\b", text.split("flowchart TB", 1)[1]))                         # noqa: E731
        minimal = (ROOT / "app_v2_minimal.mmd").read_text(encoding="utf-8")
        self.assertEqual(tags(self.text), tags(minimal))                         # the same six principles as the minimal chart

    def test_step_5_scale_up_uses_the_rollover_of_streamlit_lego(self):
        """⑤.2 carries P4 and LEGO's equation; the project's ledger agrees that R stays continuous with it — and that it breaks without."""
        eq = "β += (Fix_c − Fix_c′) · ln(P_acted / P₀)"
        step = label(self.text, "RO")
        for text in ("⑤.2 Rollover", "P4 Continuous Rollover", "ได้ทุนใหม่ Fix_c′ แล้ว", "จุดหมุนคือ P_acted", eq,
                     "β บวกที่ R เท่านั้น ไม่บวกที่ A", "R ต่อเนื่อง"):
            self.assertIn(text, step)
        scale_up = label(self.text, "S")
        for text in ("⑤.1 Scale Up fix_c", "ทุนใหม่ Fix_c′ = Fix_c + Surplus"):     # 5.1 gives 5.2 its Fix_c′: new capital first, then the rollover
            self.assertIn(text, scale_up)
        header = self.text.split("flowchart TB", 1)[0]
        self.assertIn("firstnattapon/streamlit_lego (cheat_sheet/rollover.mmd)", header)     # the source it follows
        self.assertIn("ข้าม", header)                                                       # ③ "no" = LEGO's "skip": no capital change, no rollover
        norm = lambda s: re.sub(r"\s+", " ", s.replace("Fix_c_new", "Fix_c′"))              # noqa: E731
        self.assertIn(norm(eq), norm((ROOT / "ledger_v2.py").read_text(encoding="utf-8")))  # the same line as the code behind app-v2

        # evaluate the chart's own equation on the worked example of the header and check what it is for
        c0, c1, p0, pa = 1500.0, 2500.0, 100.0, 110.0                               # Fix_c → Fix_c′ at the pivot P_acted, anchor P₀
        expr = eq.split("+=", 1)[1].replace("−", "-").replace("·", "*").replace("ln", "math.log")
        expr = expr.replace("Fix_c′", "c1").replace("Fix_c", "c0").replace("P_acted", "pa").replace("P₀", "p0")
        step_beta = eval(expr, {"__builtins__": {}, "math": math}, {"c0": c0, "c1": c1, "pa": pa, "p0": p0})   # noqa: S307
        ln = math.log(pa / p0)
        a_cash = c0 * (pa / p0 - 1.0)                                               # A after one act from P₀ to P_acted
        r_before, r_after = c0 * ln, step_beta + c1 * ln
        self.assertAlmostEqual(r_after, r_before, places=9)                         # R does not jump at the pivot …
        e_with, e_without = a_cash - r_after, a_cash - c1 * ln
        self.assertAlmostEqual(e_with, a_cash - r_before, places=9)                 # … so E is the same before and after the scale-up
        self.assertLess(e_without, 0.0)                                             # without it R jumps and E ≥ 0 breaks
        for number in (f"{a_cash - r_before:.2f}", f"{e_without:.2f}".replace("-", "−")):
            self.assertIn(number, header)                                           # the header's worked numbers are the computed ones

        sys.path.insert(0, str(ROOT))
        import ledger_v2 as led                                                     # the project's own accounting of it
        rounds = [{"action": "Chain Round", "p_old": p0, "p_new": pa, "c_before": c0, "c_after": c1},
                  {"action": "Chain Round", "p_old": pa, "p_new": p0, "c_before": c1, "c_after": c1}]
        rows = led.build({"rounds": rounds})["rows"]
        for row in rows:
            self.assertAlmostEqual(row["R_beta"], row["R"], places=9)               # R = β + Fix_c · ln(P / P₀) holds across the scale-up
        self.assertAlmostEqual(rows[0]["beta_after"], step_beta, places=9)           # and its β is the chart's step
        bare = rows[1]["fix_c"] * math.log(rows[1]["p_fill"] / rows[1]["p0"])        # the same R without any β
        self.assertGreater(abs(bare - rows[1]["R"]), 1.0)                           # is far off: the rollover does real work

    def test_the_goal_is_word_for_word_the_same_as_the_other_charts(self):
        goal = lambda t: re.search(r'^\s*GOAL\("(.*)"\)\s*$', t, re.M).group(1)    # noqa: E731
        for other in ("app_v2_minimal.mmd", "app_v2_system.mmd"):
            self.assertEqual(goal(self.text), goal((ROOT / other).read_text(encoding="utf-8")), other)

    def test_the_equation_matches_the_code_behind_app_v3(self):
        """The chart's E and Surplus are the ones chain_v3 computes (so the chart cannot drift from the page)."""
        sys.path.insert(0, str(ROOT))
        import chain_v3 as cv
        doc = cv.__doc__
        for text in ("E  = c · (x − 1 − ln x)", "S  = E − K", "Ev Efficiency = ΣE ÷ ΣK"):
            self.assertIn(text, doc)
        self.assertIn("E = A − R", self.text)
        self.assertIn("ΣE ÷ ΣK ≥ 1.0x", self.body)


class MermaidCommentTrapTest(unittest.TestCase):
    def test_no_chart_has_a_bare_comment_line(self):
        """Mermaid removes a comment only when something follows the %% — a bare %% line breaks the whole parse."""
        files = sorted(ROOT.glob("*.mmd"))
        self.assertGreaterEqual(len(files), 3)
        for f in files:
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                self.assertNotEqual(line.strip(), "%%", f"{f.name}:{n}")


if __name__ == "__main__":
    unittest.main()
