"""lego_hover_table — the escaped HTML table behind the Ledger's "ชี้เมาส์ดูที่มา" tooltips.

The module is vendored from firstnattapon/streamlit_lego; these tests pin the guarantees the Ledger view relies on
(everything escaped · tooltips only where there is text · position-based tips · height clamps). Needs pandas.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HAVE_PANDAS = importlib.util.find_spec("pandas") is not None

if HAVE_PANDAS:
    import numpy as np
    import pandas as pd
    import lego_hover_table as hover


@unittest.skipUnless(HAVE_PANDAS, "pandas not installed")
class HoverTableTest(unittest.TestCase):
    def build(self, frame, tips, tip_cols=("v",), money_cols=("v",), **kw):
        return hover.build_hover_table_html(frame, tips, tip_cols, money_cols, **kw)

    def test_height_follows_the_rows_within_clamps(self):
        self.assertEqual(hover.hover_table_height(0), hover.MIN_HEIGHT)
        self.assertEqual(hover.hover_table_height(5), hover.MIN_HEIGHT)       # short table → room for a tall tooltip
        self.assertEqual(hover.hover_table_height(10), 46 + 30 * 10)
        self.assertEqual(hover.hover_table_height(10_000), hover.MAX_HEIGHT)  # long table → scrolls inside the frame

    def test_every_piece_of_text_is_escaped(self):
        hostile = '"><img src=x onerror=alert(1)>'
        frame = pd.DataFrame({"<b>col</b>": ["<i>x</i>"], "v": [1.0]})
        doc = self.build(frame, [{"v": f"{hostile}\n<script>alert(2)</script>"}])
        for raw in ("<img", "<script>alert", "<b>col", "<i>x"):
            self.assertNotIn(raw, doc, raw)
        self.assertIn("&lt;b&gt;col&lt;/b&gt;", doc)
        self.assertIn("&lt;i&gt;x&lt;/i&gt;", doc)
        self.assertIn("&quot;&gt;&lt;img src=x onerror=alert(1)&gt;", doc)     # the attribute can't be closed early
        self.assertNotIn("innerHTML", doc)                                      # the script only ever uses textContent

    def test_tooltips_only_on_tip_columns_with_text(self):
        frame = pd.DataFrame({"a": [1, 2], "v": [1.5, 2.5], "w": [3.5, 4.5]})
        doc = self.build(frame, [{"v": "title\nbody"}, {}], tip_cols=("v", "w"), money_cols=("v", "w"))
        self.assertEqual(doc.count("data-tip="), 1)             # row 0 / column v only
        self.assertEqual(doc.count('tabindex="0"'), 1)          # keyboard users can focus the same cells
        self.assertEqual(doc.count('<th class="tip"'), 2)       # the ⓘ marker sits on tip columns' headers
        self.assertIn('data-tip="title\nbody"', doc)

    def test_tips_are_matched_by_row_position_not_index_label(self):
        frame = pd.DataFrame({"v": [1.0, 2.0]}, index=[10, 20])
        doc = self.build(frame, [{"v": "first"}, {"v": "second"}])
        self.assertLess(doc.index('data-tip="first"'), doc.index('data-tip="second"'))

    def test_money_columns_use_two_decimals_and_gaps_stay_empty(self):
        frame = pd.DataFrame({"v": [1234.5, np.nan, None], "t": ["x", None, "z"]})
        doc = self.build(frame, [{}, {}, {}], tip_cols=(), money_cols=("v",))
        self.assertIn(">1234.50</td>", doc)
        self.assertEqual(doc.count('<td class="num"></td>'), 2)   # NaN / None → blank cell, never the word "nan"
        self.assertNotIn("nan", doc.lower().replace("<!doctype", ""))
        self.assertIn("<td></td>", doc)                            # the missing text cell

    def test_theme_attribute_is_whitelisted(self):
        frame = pd.DataFrame({"v": [1.0]})

        def root(doc):                                              # the <html …> tag (the CSS mentions data-theme too)
            return doc.split("<head>")[0]

        self.assertIn('data-theme="dark"', root(self.build(frame, [{}], theme="dark")))
        self.assertIn('data-theme="light"', root(self.build(frame, [{}], theme="light")))
        for odd in (None, "", "purple", '"><script>'):
            self.assertNotIn("data-theme", root(self.build(frame, [{}], theme=odd)), odd)

    def test_empty_frame_still_renders_a_table(self):
        doc = self.build(pd.DataFrame({"v": []}), [])
        self.assertIn("<table>", doc)
        self.assertIn("<th", doc)


if __name__ == "__main__":
    unittest.main()
