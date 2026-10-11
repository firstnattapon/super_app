"""app_v2_system.mmd — structural checks (stdlib only; no Mermaid needed).

The big system flowchart is hand-written Mermaid. These tests guard what is easy to break silently:
unbalanced subgraphs, a node nobody defined, a principle that fell out, a goal nothing reaches, and a
*cycle* — Mermaid's layout turns chaotic with a back-edge (the chain loop is drawn as a terminal
"ลูกโซ่ต่อ ⬆ กลับไป INPUT" node instead of an arrow for that reason).
"""
import re
import unittest
from collections import defaultdict, deque
from pathlib import Path

FILE = Path(__file__).resolve().parent.parent / "app_v2_system.mmd"
ID = r"[A-Za-z_][A-Za-z0-9_]*"
ARROW = r"(?:<-->|-->|==>|-\.->)(?:\|[^|]*\|)?"
EDGE_LINE = re.compile(rf"^\s*({ID})((?:\s*{ARROW}\s*{ID})+)\s*$")
EDGE_HOP = re.compile(rf"{ARROW}\s*({ID})")
NODE_DEF = re.compile(rf"^\s*({ID})\s*(?:\[\(|\[\[|\[|\(\(|\(|\{{)")
SUBGRAPH = re.compile(rf"^\s*subgraph\s+({ID})\b")
PRINCIPLES = ("Shannon's Demon", "Pay-first", "Free-Risk Scale-up", "Continuous Rollover", "Cash Lanes",
              "Ev Efficiency", "Atomic + stale-proof", "อ่านกำไรที่ E", "Human-in-the-loop")


def parse(text: str):
    body = text.split("\n---\n", 1)[1] if text.lstrip().startswith("---") else text
    lines = [ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    nodes, subgraphs, edges, depth = set(), set(), [], 0
    for ln in lines:
        if (m := SUBGRAPH.match(ln)):
            subgraphs.add(m.group(1))
            depth += 1
        elif ln.strip() == "end":
            depth -= 1
            assert depth >= 0, "`end` without a `subgraph`"
        elif (m := EDGE_LINE.match(ln)):
            chain = [m.group(1)] + EDGE_HOP.findall(m.group(2))
            edges += list(zip(chain, chain[1:]))
        elif (m := NODE_DEF.match(ln)):
            nodes.add(m.group(1))
    return nodes, subgraphs, edges, depth


class SystemFlowchartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = FILE.read_text(encoding="utf-8")
        cls.nodes, cls.subgraphs, cls.edges, cls.depth = parse(cls.text)
        cls.defined = cls.nodes | cls.subgraphs

    def test_header_and_balance(self):
        self.assertTrue(self.text.startswith("---\ntitle:"))
        self.assertIn("\nflowchart TB\n", self.text)
        self.assertEqual(self.depth, 0, "every subgraph needs its `end`")
        self.assertGreaterEqual(len(self.subgraphs), 8)

    def test_every_edge_endpoint_is_defined(self):
        used = {x for e in self.edges for x in e}
        self.assertEqual(sorted(used - self.defined), [])

    def test_nine_principles_and_the_goal_are_on_the_chart(self):
        for p in PRINCIPLES:
            self.assertIn(p, self.text, p)
        for tag in (f"P{i}" for i in range(1, 10)):
            self.assertRegex(self.text, rf"\b{tag}\b")
        self.assertIn("GOAL", self.defined)
        self.assertIn("เป้าหมายสูงสุด", self.text)

    def test_goal_is_reachable_from_the_user_and_there_are_no_cycles(self):
        out, indeg = defaultdict(list), defaultdict(int)
        for a, b in self.edges:
            out[a].append(b)
            indeg[b] += 1
        seen, queue = {"USER"}, deque(["USER"])
        while queue:
            for nxt in out[queue.popleft()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        self.assertIn("GOAL", seen, "nothing leads from the user to the goal")
        verts = self.defined | {x for e in self.edges for x in e}
        ready = deque(v for v in verts if indeg[v] == 0)
        done = 0
        while ready:                                    # Kahn: a leftover vertex means a cycle
            done += 1
            for nxt in out[ready.popleft()]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        self.assertEqual(done, len(verts), "the flowchart has a cycle — draw loops as terminal nodes")


if __name__ == "__main__":
    unittest.main()
