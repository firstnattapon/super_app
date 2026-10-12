"""app_v3_playbook.mmd — the minimal loop said with the instruments (stdlib only; no Mermaid needed).

The flowchart is hand-written Mermaid. These tests guard what is easy to break silently: a step that lost its instrument,
the 20% / 80% split that stopped adding up to a whole, the order "pay K first, then harvest E", the way out of the
question "is the Surplus positive?" (shield, then scale up fix_c with Liquidity or a Deep ITM, then the check), the goal
text, and Mermaid's one parser trap for comments.
"""
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
        self.assertEqual(self.nodes, {"K", "E", "D", "A", "S", "L", "I", "Q", "GOAL"})
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

    def test_a_positive_surplus_goes_shield_then_scale_up_then_the_check(self):
        for edge in (("D", "A"), ("A", "S"), ("S", "L"), ("S", "I"), ("L", "Q"), ("I", "Q"), ("Q", "GOAL"), ("Q", "K")):
            self.assertIn(edge, self.edges)
        into = lambda n: {a for a, b in self.edges if b == n}                   # noqa: E731
        self.assertEqual((into("A"), into("S"), into("Q")), ({"D"}, {"A"}, {"L", "I"}))
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

    def test_the_split_is_a_whole(self):
        shares = [int(x) for x in re.findall(r"(\d+)%", label(self.text, "E"))]
        self.assertEqual(shares, [20, 80])
        self.assertEqual(sum(shares), 100)

    def test_e_is_a_minus_r_and_never_the_other_way_round(self):
        self.assertIn("E = A − R", self.body)
        self.assertNotIn("R − A", self.body)                                     # the nodes: R − A would be negative, E ≥ 0 comes from A − R
        self.assertIn("ถ้าเขียน R − A จะติดลบเสมอ", self.text)                    # and the header says why

    def test_the_principles_it_uses_carry_their_names(self):
        for tag, name in (("P1", "Shannon's Demon"), ("P2", "Pay-first"), ("P3", "Free-Risk Scale-up"), ("P6", "Ev Efficiency"),
                          ("P8", "อ่านกำไรที่ E")):
            self.assertIn(f"{tag} {name}", self.body)
        for dropped in ("P4", "P5", "P7", "P9"):
            self.assertNotRegex(self.body, rf"\b{dropped}\b")

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
