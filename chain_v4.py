"""chain_v4 — Chain System v4 as ONE chain of steps. Standalone: stdlib only, its own data file.

The book is a single ordered list of what you *did*. Nothing else is stored: every number is re-derived by replaying
the chain — the way the LEGO dashboard re-derives its ledger from the decisions (``recompute_gated_ledger``) — so a
file can never disagree with itself, and the same rules judge a step whether it is loaded, previewed or saved.

    open   ticker, c, t [, b, burn]            start tracking: Fix_c = c held at P₀ = t   (LEGO genesis: R₀ = ΔA₀ = A₀ = E₀ = 0)
    pay    ticker, hedge [, k]                 ① pay K first — EV Reserve −K, Ev Burn +K — and open the cycle
    move   ticker, p                           ② one act at the fill price p — one row of the LEGO ledger
    scale  ticker, by, shield [, settle, pull] ④ Shield → ⑤.1 Scale Up → ⑤.2 Rollover — and closes the cycle
    cash   src, dst, amount                    money between lanes: @out · @pool · @reserve · a ticker's Baseline b

One ticker's ledger is the streamlit_lego main equations (lego_dash_core E3–E6, cheat_sheet/main_rebalancing.mmd, rollover.mmd):

    Rₙ  = β + FIX_C·ln(Pₙ / P₀)                never frozen
    ΔAₙ = FIX_C·(Pₙ / P_acted − 1)             Aₙ = Aₙ₋₁ + ΔAₙ            P_acted ← Pₙ
    Eₙ  = Aₙ − Rₙ                              ≥ 0 and never falls  (x − 1 ≥ ln x on every step)
    Vₙ  = (FIX_C / P_acted)·Pₙ                 gap = Vₙ − FIX_C  →  SELL if gap > 0, BUY if gap < 0, qty = |gap| / Pₙ   (ΔAₙ = gap)
    rollover  β += (Fix_c − Fix_c′)·ln(P_acted / P₀)     added to R only, never to A

The Chain layer on top: K = 12.4%·c·Hedge× (or the real premium) · E_cyc = Eₙ − Eₙ when K was paid · S = E_cyc − K ·
c′ = c + S · Ev Efficiency = ΣE ÷ ΣK.  ``scale`` also *settles* the cycle: the swing pays for the insurance —
E_cyc goes to Pool CF, K goes back to EV Reserve, S becomes the new capital (Liquidity) or rests in Pool CF (Deep ITM).

What was cut to get here, and why the goal's numbers do not change (tests/test_chain_v4.py proves each one):
  * no stored state (c, t, b, ΣE, ΣK, phase, β) — replayed from the chain, so a file cannot contradict itself;
  * one list for every ticker and the shared treasury — one undo, one backup, no ``[Ticker: X]`` tags to parse;
  * one ``cash`` step (src → dst: subtract at the source, add at the destination, money cannot appear from nowhere)
    instead of Extract / Deploy / Allocate / Funding / EV flow;
  * ``pay`` posts EV Reserve and Ev Burn together, ``scale`` pays K back — the two accounts cannot drift apart;
  * "pay K first" is not a check: a ``move`` before a ``pay`` is simply not a legal step.
The A and E here are the *model* ledger, not fee-inclusive broker P&L; Pool CF and EV Reserve are the balances this book records.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# Put 10% out of the money, 1 year, σ 50%, r 4% (Black-Scholes) per $1 of c per 1× hedge — proportional to c, so the
# whole pricing model is this one number. A real premium can always be typed over it (``k``).
PREMIUM_RATE = 0.1239616719
STOCK_SHARE = 0.20                                  # ② holds the exposure as stock 20% (Liquidity) + LEAPS 80% (Buying Power)
VERSION = 4
MAX_NAME = 20
MAX_IMPORT_BYTES = 2_000_000                        # a backup of thousands of steps is far smaller
POOL, RESERVE, OUT = "@pool", "@reserve", "@out"    # lane names (a ticker name stands for that ticker's Baseline b)
LANES = (POOL, RESERVE, OUT)
KINDS = ("open", "pay", "move", "scale", "cash")
FUNDING = ("liquidity", "deep_itm")
_BAD_NAME = re.compile(r"[\x00-\x1f`*_\[\]<>\\|$#&@]")      # nothing Markdown / HTML could read as formatting, and no "@" (lane names)
_EDGES = {"pool": {"out", "reserve", "base"}, "reserve": {"out", "pool"}, "out": {"pool", "reserve"}, "base": {"pool"}}
_EPS = 1e-9
_LOCK = threading.RLock()


class BookError(ValueError):
    """Something the user can fix: a bad number, a bad name, a step the rules refuse, an unreadable file."""


class Stale(BookError):
    """The book changed after the step was previewed — nothing was saved."""


# ── numbers and names ─────────────────────────────────────────────────────────────────────
def _r2(x: float) -> float:
    return round(x, 2) + 0.0                                    # + 0.0: never "-0.0"


def _num(x: Any, label: str, *, positive: bool = False, minimum: Optional[float] = None) -> float:
    if isinstance(x, bool):
        raise BookError(f"{label} ต้องเป็นตัวเลข")
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise BookError(f"{label} ต้องเป็นตัวเลข") from None
    if not math.isfinite(v):
        raise BookError(f"{label} ต้องเป็นตัวเลขที่มีค่า")
    if positive and v <= 0:
        raise BookError(f"{label} ต้องมากกว่า 0")
    if minimum is not None and v < minimum:
        raise BookError(f"{label} ต้องไม่น้อยกว่า {minimum:g}")
    return v


def _name(raw: Any) -> str:
    name = str(raw or "").strip()
    if not 1 <= len(name) <= MAX_NAME or _BAD_NAME.search(name):
        raise BookError(f"ชื่อต้องยาว 1–{MAX_NAME} ตัวอักษร และไม่มีอักขระพิเศษ ` * _ [ ] < > | $ # & @")
    return name


def _stamp(x: Any) -> str:
    """A step's timestamp as read back: only digits, '-', ':' and ' ' (all this module ever writes) — anything else, such as
    Markdown from a hand-edited or foreign file, is dropped."""
    return re.sub(r"[^0-9: \-]", "", str(x))[:16]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _finite(*xs: float) -> None:
    if not all(map(math.isfinite, xs)):                         # absurdly large inputs overflow to inf
        raise BookError("ตัวเลขใหญ่หรือเล็กเกินไป — ตรวจทุน c และราคาอีกครั้ง")


# ── one step, as written in the file ──────────────────────────────────────────────────────
def _flag(raw: dict, key: str, default: bool) -> bool:
    v = raw.get(key, default)
    if not isinstance(v, bool):
        raise BookError(f"{key} ต้องเป็น true หรือ false")
    return v


def _end(x: Any) -> str:
    """One end of a ``cash`` step: a lane, or a ticker name (= that ticker's Baseline b)."""
    return x if x in LANES else _name(x)


def _event(raw: Any) -> dict:
    """A raw step → the step as this module keeps it (known keys only, checked). Raises BookError."""
    if not isinstance(raw, dict):
        raise BookError("รายการต้องเป็น object")
    kind = raw.get("do")
    if kind not in KINDS:
        raise BookError(f"ไม่รู้จักชนิดรายการ “{str(kind)[:20]}”")
    if kind == "cash":
        e: dict = {"do": kind, "src": _end(raw.get("src")), "dst": _end(raw.get("dst")),
                   "amount": _num(raw.get("amount"), "จำนวนเงิน", positive=True)}
    else:
        e = {"do": kind, "ticker": _name(raw.get("ticker"))}
        if kind == "open":
            e.update(c=_num(raw.get("c"), "ทุน c", positive=True), t=_num(raw.get("t"), "ราคา t", positive=True),
                     b=_num(raw.get("b", 0.0), "Baseline b"), burn=_num(raw.get("burn", 0.0), "Ev burn ยกมา", minimum=0.0))
        elif kind == "pay":
            k = raw.get("k")
            e.update(hedge=_num(raw.get("hedge", 1.0), "Hedge ×", minimum=0.0),
                     k=None if k is None else _num(k, "ค่าประกันที่จ่ายจริง K", minimum=0.0))
        elif kind == "move":
            e.update(p=_num(raw.get("p"), "ราคา fill P", positive=True))
        else:                                                   # scale
            if raw.get("by") not in FUNDING:
                raise BookError("by ต้องเป็น liquidity หรือ deep_itm")
            if raw.get("shield") is not True:
                raise BookError("ต้องตั้ง Shield ก่อนขยายทุน (④) — shield ต้องเป็น true")
            e.update(by=raw["by"], shield=True, settle=_flag(raw, "settle", True), pull=_flag(raw, "pull", True))
    e["at"] = _stamp(raw.get("at", ""))
    return e


# ── the ledger: the rules and the equations live here, once ───────────────────────────────
def _ledger() -> dict:
    return {"pool": 0.0, "reserve": 0.0, "net_in": 0.0, "tickers": {}, "rows": []}


def _clone(led: dict) -> dict:
    return {"pool": led["pool"], "reserve": led["reserve"], "net_in": led["net_in"],
            "tickers": {n: dict(t) for n, t in led["tickers"].items()}, "rows": []}


def _ticker(led: dict, name: str) -> dict:
    try:
        return led["tickers"][name]
    except KeyError:
        raise BookError(f"ไม่พบ “{name}”") from None


def _after(t: dict) -> dict:
    return {"c": t["c"], "pa": t["pa"], "A": t["A"], "R": t["R"], "E": t["E"], "b": t["b"], "beta": t["beta"]}


def apply(led: dict, ev: dict) -> dict:
    """Run one (checked) step on ``led`` — changes it — and return the computed row. Raises BookError when the rules refuse."""
    kind = ev["do"]
    if kind == "open":
        return _open(led, ev)
    if kind == "cash":
        return _cash(led, ev)
    t = _ticker(led, ev["ticker"])
    return {"pay": _pay, "move": _move, "scale": _scale}[kind](led, t, ev)


def _open(led: dict, ev: dict) -> dict:
    name = ev["ticker"]
    if any(n.lower() == name.lower() for n in led["tickers"]):
        raise BookError("ชื่อนี้มีอยู่แล้ว")
    c, b, burn = _r2(ev["c"]), _r2(ev["b"]), _r2(ev["burn"])
    if c <= 0:
        raise BookError("ทุน c ต้องมากกว่า 0.00")
    led["tickers"][name] = {"name": name, "c": c, "c0": c, "p0": ev["t"], "pa": ev["t"], "beta": 0.0, "A": 0.0, "R": 0.0, "E": 0.0,
                            "b": b, "b0": b, "burn0": burn, "hedge": 1.0, "open": False, "K": 0.0, "Epay": 0.0, "sumK": 0.0,
                            "scaled": 0.0, "pulled": 0.0, "moves": 0, "scales": 0}
    return {"do": "open", "ticker": name, "c": c, "t": ev["t"], "b": b, "burn": burn, "after": _after(led["tickers"][name])}


def _pay(led: dict, t: dict, ev: dict) -> dict:
    if t["open"]:
        raise BookError(f"{t['name']}: วงจรนี้จ่าย K ไปแล้ว — ไม่จ่ายซ้ำ (เก็บ E ต่อ หรือขยายทุนเมื่อ S > 0)")
    h = ev["hedge"]
    k = _r2(PREMIUM_RATE * t["c"] * h) if ev["k"] is None else _r2(ev["k"])
    _finite(k)
    led["reserve"] = _r2(led["reserve"] - k)                    # pay first: EV Reserve −K, Ev Burn +K — one step, two accounts
    t.update(open=True, K=k, Epay=t["E"], hedge=h, sumK=_r2(t["sumK"] + k))
    return {"do": "pay", "ticker": t["name"], "c": t["c"], "hedge": h, "real": ev["k"] is not None, "K": k,
            "pool": led["pool"], "reserve": led["reserve"], "after": _after(t)}


def _move(led: dict, t: dict, ev: dict) -> dict:
    name = t["name"]
    if not t["open"]:
        raise BookError(f"{name}: ต้องจ่าย K ก่อน (① จ่ายประกันก่อนเสมอ) จึงจะบันทึกราคาได้")
    p, c, pa, p0 = ev["p"], t["c"], t["pa"], t["p0"]
    try:
        r = t["beta"] + c * math.log(p / p0)                    # Rₙ = β + FIX_C·ln(Pₙ/P₀)
        d_a = c * (p / pa - 1.0)                                # ΔAₙ = FIX_C·(Pₙ/P_acted − 1)
    except (ValueError, OverflowError, ZeroDivisionError):      # p / p0 underflowed to 0
        raise BookError("ตัวเลขใหญ่หรือเล็กเกินไป — ตรวจทุน c และราคาอีกครั้ง") from None
    a_prev = t["A"]
    a = a_prev + d_a                                            # Aₙ = Aₙ₋₁ + ΔAₙ
    e = a - r                                                   # Eₙ = Aₙ − Rₙ
    _finite(r, d_a, a, e)
    q = c / pa                                                  # shares held since the last act (value FIX_C at P_acted)
    v = q * p
    gap = v - c                                                 # Vₙ − FIX_C (the code and flowchart of streamlit_lego): SELL if > 0, BUY if < 0; = ΔAₙ
    d_r = r - t["R"]
    e_cyc = _r2(e - t["Epay"])
    row = {"do": "move", "ticker": name, "c": c, "p0": p0, "pa": pa, "p": p, "beta": t["beta"], "q": q, "v": v, "gap": gap,
           "side": "SELL" if gap > _EPS else "BUY" if gap < -_EPS else "PASS", "qty": abs(gap) / p,
           "a_prev": a_prev, "d_a": d_a, "a": a, "r": r, "e": e, "d_e": e - t["E"], "d_r": d_r,
           "e_pay": t["Epay"], "e_cyc": e_cyc, "K": t["K"], "s": _r2(e_cyc - t["K"])}
    t.update(A=a, R=r, E=e, pa=p, b=_r2(t["b"] + d_r), moves=t["moves"] + 1)   # P_acted ← Pₙ; Baseline takes the trend part
    row["b"] = t["b"]
    row["after"] = _after(t)
    return row


def _scale(led: dict, t: dict, ev: dict) -> dict:
    name = t["name"]
    if not t["open"]:
        raise BookError(f"{name}: ยังไม่ได้จ่าย K — ไม่มีวงจรให้ขยายทุน")
    e_cyc = _r2(t["E"] - t["Epay"])
    k = t["K"]
    s = _r2(e_cyc - k)
    if s <= 0:
        raise BookError(f"{name}: Surplus ยังไม่เป็นบวก (S = E − K = {_f(s)}) — เก็บ E ต่อก่อน (③.1) ยังขยายทุนไม่ได้")
    c_old, c_new = t["c"], _r2(t["c"] + s)                      # Free Risk: only the surplus adds capital
    d_beta = (c_old - c_new) * math.log(t["pa"] / t["p0"])      # rollover: β += (Fix_c − Fix_c′)·ln(P_acted/P₀)
    pool, reserve = led["pool"], led["reserve"]
    if ev["settle"]:                                            # the swing pays for the insurance:
        pool = _r2(pool + s)                                    #   E_cyc → Pool CF, K back to EV Reserve (Pool +E_cyc −K, Reserve +K)
        reserve = _r2(reserve + k)
    if ev["by"] == "liquidity":
        if pool + _EPS < s:
            raise BookError(f"{name}: Pool CF ไม่พอสำหรับ Liquidity (มี {pool:,.2f} ต้องใช้ {s:,.2f}) — เลือก Deep ITM หรือเปิดชำระ")
        pool = _r2(pool - s)
    pulled, b = 0.0, t["b"]
    if ev["pull"] and b > 0:                                    # ⑤.2 pull the capital gain (the trend part) out → Pool CF
        pulled = _r2(b)
        pool, b = _r2(pool + pulled), _r2(b - pulled)
    led["pool"], led["reserve"] = pool, reserve
    row = {"do": "scale", "ticker": name, "c": c_old, "c_new": c_new, "p0": t["p0"], "pa": t["pa"], "beta": t["beta"],
           "d_beta": d_beta, "e_cyc": e_cyc, "K": k, "s": s, "by": ev["by"], "settle": ev["settle"], "pull": ev["pull"],
           "pulled": pulled, "pool": pool, "reserve": reserve}
    t.update(c=c_new, beta=t["beta"] + d_beta, open=False, K=0.0, b=b, scaled=_r2(t["scaled"] + s),
             pulled=_r2(t["pulled"] + pulled), scales=t["scales"] + 1)
    row["after"] = _after(t)
    return row


def _kind(end: str) -> str:
    return {POOL: "pool", RESERVE: "reserve", OUT: "out"}.get(end, "base")


def _cash(led: dict, ev: dict) -> dict:
    src, dst, a = ev["src"], ev["dst"], _r2(ev["amount"])
    if a <= 0:
        raise BookError("จำนวนเงินต้องมากกว่า 0.00")
    if src == dst:
        raise BookError("ต้นทางกับปลายทางต้องคนละกระเป๋า")
    ks, kd = _kind(src), _kind(dst)
    if kd not in _EDGES[ks]:
        raise BookError("เส้นทางนี้ไม่มีในระบบ — Baseline เชื่อมได้เฉพาะ Pool CF · เงินนอกระบบเชื่อมได้เฉพาะ Pool CF / EV Reserve")
    ticks = [_ticker(led, x) for x in (src, dst) if _kind(x) == "base"]
    if ks == "pool" and led["pool"] + _EPS < a:
        raise BookError(f"Pool CF ไม่พอ (มี {led['pool']:,.2f})")
    if ks == "base" and ticks[0]["b"] + _EPS < a:
        raise BookError(f"Baseline ของ {src} ไม่พอ (มี {ticks[0]['b']:,.2f}) — ดึงได้เฉพาะ capital gain ที่มีอยู่จริง")
    for end, sign in ((src, -1), (dst, 1)):                     # subtract at the source, add at the destination — nothing else
        kind = _kind(end)
        if kind == "base":
            t = led["tickers"][end]
            t["b"] = _r2(t["b"] + sign * a)
            t["pulled"] = _r2(t["pulled"] - sign * a)
        elif kind == "out":
            led["net_in"] = _r2(led["net_in"] - sign * a)       # money that came in from outside (+) or went out (−)
        else:
            led[kind] = _r2(led[kind] + sign * a)
    tick = ticks[0] if ticks else None
    return {"do": "cash", "ticker": tick["name"] if tick else None, "src": src, "dst": dst, "amount": a,
            "pool": led["pool"], "reserve": led["reserve"], "after": _after(tick) if tick else None}


def fold(book: dict) -> dict:
    """Replay the whole chain. Returns the ledger ``{pool, reserve, net_in, tickers, rows}``; BookError names the step that breaks a rule."""
    led = _ledger()
    for n, ev in enumerate(book["chain"], 1):
        try:
            row = apply(led, ev)
        except BookError as e:
            raise BookError(f"รายการที่ {n}: {e}") from None
        row.update(n=n, at=ev.get("at", ""))
        led["rows"].append(row)
    return led


def preview(led: dict, ev: dict) -> tuple[dict, dict]:
    """(ledger after, row) for a step that is not saved — ``led`` is untouched. BookError when the rules refuse the step."""
    after = _clone(led)
    return after, apply(after, _event(ev))


def replay(steps: list) -> dict:
    """The ledger after a list of raw steps, judged by the same rules as a file — for worked examples; nothing is saved."""
    return fold({"version": VERSION, "chain": [_event(e) for e in steps]})


# ── what the page reads ───────────────────────────────────────────────────────────────────
def e_cyc(t: dict) -> float:
    """E harvested since K was paid (0 when no cycle is open) — "กำไรค้างท่อ" until the cycle closes."""
    return _r2(t["E"] - t["Epay"]) if t["open"] else 0.0


def surplus(t: dict) -> Optional[float]:
    """S = E_cyc − K (None when no cycle is open)."""
    return _r2(e_cyc(t) - t["K"]) if t["open"] else None


def phase(t: dict) -> str:
    """'pay' (① no cycle open) · 'swing' (②/③.1 open, S ≤ 0) · 'scale' (③ S > 0: ④ ⑤ may begin)."""
    if not t["open"]:
        return "pay"
    return "scale" if surplus(t) > 0 else "swing"


def efficiency(t: dict) -> Optional[float]:
    """Ev Efficiency = ΣE ÷ ΣK of what this book measured (None until some insurance was paid)."""
    return t["E"] / t["sumK"] if t["sumK"] > 0 else None


def next_k(t: dict, hedge: Optional[float] = None) -> float:
    """The model premium K for this ticker's current capital (``hedge`` None → the last Hedge × used)."""
    return _r2(PREMIUM_RATE * t["c"] * (t["hedge"] if hedge is None else hedge))


def totals(led: dict) -> dict:
    ts = list(led["tickers"].values())
    sum_e, sum_k = sum(t["E"] for t in ts), _r2(sum(t["sumK"] for t in ts))
    return {"pool": led["pool"], "reserve": led["reserve"], "liquidity": _r2(led["pool"] + led["reserve"]),
            "burn": _r2(sum(t["burn0"] + t["sumK"] for t in ts)), "net_in": led["net_in"], "fix_c": _r2(sum(t["c"] for t in ts)),
            "b": _r2(sum(t["b"] for t in ts)), "pending": _r2(sum(e_cyc(t) for t in ts)), "sum_e": sum_e, "sum_k": sum_k,
            "eff": sum_e / sum_k if sum_k > 0 else None, "tickers": len(ts)}


def recommend_funding(led: dict, ev: dict) -> str:
    """⑤.1 in one line: 'liquidity' when the cash left after using it still pays next cycle's K′, else 'deep_itm'.
    ``ev`` is the ``scale`` step; the rule is judged on the real result of the step, so settlement is counted."""
    try:
        after, _ = preview(led, {**ev, "by": "liquidity"})
    except BookError:
        return "deep_itm"
    return "liquidity" if after["pool"] + after["reserve"] + _EPS >= next_k(after["tickers"][ev["ticker"]]) else "deep_itm"


def _f(x: float, d: int = 2) -> str:
    """A number for reading: thousands separators, a real minus sign, never "−0.00"."""
    return f"{0.0 if round(x, d) == 0 else x:,.{d}f}".replace("-", "−")


def _sg(x: float) -> str:
    return ("−" if round(x, 2) < 0 else "+") + f"{abs(x):,.2f}"


def _op(x: float) -> str:
    return f"({_f(x)})" if x < 0 else _f(x)


def explain(row: dict) -> list[tuple[str, str, str]]:
    """The "ที่มา" of a row's numbers: (symbol, the formula with the numbers put in, the result) — the same wording as the LEGO tooltips."""
    k = row["do"]
    if k == "move":
        return [("Vₙ", f"Q × Pₙ = {_f(row['q'], 4)} × {_f(row['p'], 4)}  (Q = FIX_C ÷ P_acted = {_f(row['c'])} ÷ {_f(row['pa'], 4)})",
                 _f(row["v"])),
                ("gap", f"Vₙ − FIX_C = {_f(row['v'])} − {_f(row['c'])}", f"{_sg(row['gap'])} → {row['side']} ≈ {row['qty']:,.4f} หุ้น"),
                ("Rₙ", f"β + FIX_C × ln(Pₙ / P₀) = {_f(row['beta'])} + {_f(row['c'])} × ln({_f(row['p'], 4)} / {_f(row['p0'], 4)})",
                 _f(row["r"])),
                ("ΔAₙ", f"FIX_C × (Pₙ / P_acted − 1) = {_f(row['c'])} × ({_f(row['p'], 4)} / {_f(row['pa'], 4)} − 1)", _sg(row["d_a"])),
                ("Aₙ", f"Aₙ₋₁ + ΔAₙ = {_f(row['a_prev'])} + {_op(row['d_a'])}", _f(row["a"])),
                ("Eₙ", f"Aₙ − Rₙ = {_f(row['a'])} − {_op(row['r'])}", _f(row["e"])),
                ("P_acted", "← Pₙ  (act แล้ว เลื่อนมาที่ราคา fill = roll P ขึ้น/ลง)", f"{_f(row['pa'], 4)} → {_f(row['p'], 4)}"),
                ("E_cyc", f"Eₙ − Eₙ ตอนจ่าย K = {_f(row['e'])} − {_f(row['e_pay'])}", _f(row["e_cyc"])),
                ("S", f"E_cyc − K = {_f(row['e_cyc'])} − {_f(row['K'])}", _sg(row["s"]))]
    if k == "pay":
        how = "K จริงที่จ่าย" if row["real"] else f"K = {PREMIUM_RATE:.1%} × c × Hedge× = {PREMIUM_RATE:.1%} × {_f(row['c'])} × {row['hedge']:g}"
        return [("K", how, _f(row["K"])), ("EV Reserve", f"ลบ K → เหลือ {_f(row['reserve'])}", f"Ev Burn +{_f(row['K'])}")]
    if k == "scale":
        lines = [("E_cyc", "Eₙ − Eₙ ตอนจ่าย K", _f(row["e_cyc"])),
                 ("S", f"E_cyc − K = {_f(row['e_cyc'])} − {_f(row['K'])}", _sg(row["s"])),
                 ("c′", f"c + S = {_f(row['c'])} + {_f(row['s'])}", _f(row["c_new"])),
                 ("β", f"β + (Fix_c − Fix_c′) × ln(P_acted / P₀) = {_f(row['beta'], 4)} + ({_f(row['c'])} − {_f(row['c_new'])}) × "
                       f"ln({_f(row['pa'], 4)} / {_f(row['p0'], 4)})", _f(row["beta"] + row["d_beta"], 4))]
        if row["settle"]:
            lines.append(("ชำระ", f"E_cyc → Pool CF · K {_f(row['K'])} คืน EV Reserve", f"Pool {_f(row['pool'])} · Reserve {_f(row['reserve'])}"))
        if row["by"] == "liquidity":
            lines.append(("⑤.1", "Liquidity: ซื้อเพิ่มด้วย S จาก Pool CF", f"Pool −{_f(row['s'])}"))
        else:
            lines.append(("⑤.1", "Deep ITM: S พักใน Pool CF เป็นสภาพคล่อง · เบี้ยส่วนเพิ่มไปอยู่ใน K′ ของ ① หน้า", f"Pool {_f(row['pool'])}"))
        if row["pulled"]:
            lines.append(("⑤.2", "ดึง capital gain (Baseline b) → Pool CF · E อยู่ต่อ", f"+{_f(row['pulled'])}"))
        return lines
    if k == "cash":
        return [("โอน", f"{row['src']} → {row['dst']}", _f(row["amount"]))]
    return [("เริ่ม", f"Fix_c = {_f(row['c'])} ที่ P₀ = {_f(row['t'], 4)}", "R₀ = ΔA₀ = A₀ = E₀ = 0")]


# ── the book: its own JSON file ───────────────────────────────────────────────────────────
def _validated(raw: Any) -> dict:
    """Any book-shaped value → the book as this module keeps it, or BookError. Replays the chain, so the rules judge every step."""
    try:
        if raw["version"] != VERSION:
            raise BookError("ไฟล์นี้ไม่ใช่สมุดของ app-v4 (version ไม่ตรง) ใช้ต่อไม่ได้")
        chain = raw["chain"]
        if not isinstance(chain, list):
            raise BookError("chain ต้องเป็นรายการ")
        out = []
        for n, ev in enumerate(chain, 1):
            try:
                out.append(_event(ev))
            except BookError as e:
                raise BookError(f"รายการที่ {n}: {e}") from None
        book = {"version": VERSION, "chain": out}
    except BookError:
        raise
    except (KeyError, TypeError, AttributeError) as e:
        raise BookError(f"โครงสร้างไฟล์ไม่ถูกต้อง ({type(e).__name__})") from None
    fold(book)
    return book


def load(path: Any) -> dict:
    """The book ``{'version', 'chain'}``; a missing file is an empty book; a broken one raises BookError."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": VERSION, "chain": []}
    except OSError as e:
        raise BookError(f"อ่านไฟล์ข้อมูลไม่ได้: {e}") from e
    try:
        return _validated(json.loads(raw))
    except ValueError as e:                                       # not JSON (BookError is a ValueError too)
        if isinstance(e, BookError):
            raise
        raise BookError("ไฟล์ข้อมูลไม่ใช่ JSON ที่ถูกต้อง") from e


def save(path: Any, book: dict) -> None:
    """All-or-nothing: write a temp file, fsync, then replace — a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(book, ensure_ascii=False, indent=2, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _edit(path: Any, fn: Callable[[dict], Any]) -> Any:
    with _LOCK:                                                   # read → change → write, one writer at a time
        book = load(path)
        result = fn(book)                                         # raises → nothing is written
        save(path, book)
        return result


def commit(path: Any, steps: Any, seen: int) -> list:
    """Save one step (or several, all or none) — but only if the chain is still exactly as long as when it was previewed.
    The rules judge each step on the real, freshly read book. Returns the saved steps."""
    raws = [steps] if isinstance(steps, dict) else list(steps)

    def go(book: dict) -> list:
        if len(book["chain"]) != seen:
            raise Stale("ข้อมูลเปลี่ยนไปแล้วหลังดูตัวอย่าง — ตรวจตัวเลขอีกครั้งแล้วกดบันทึกใหม่")
        led = fold(book)
        saved = []
        for raw in raws:
            e = _event(raw)
            apply(led, e)
            saved.append({**e, "at": _now()})
        book["chain"].extend(saved)
        return saved
    return _edit(path, go)


def undo(path: Any) -> dict:
    """Take back the last step of the chain (any ticker, any kind) — what came before it is still a valid chain."""
    def go(book: dict) -> dict:
        if not book["chain"]:
            raise BookError("ยังไม่มีรายการให้ย้อน")
        return book["chain"].pop()
    return _edit(path, go)


def remove(path: Any, name: str) -> None:
    """Drop a ticker that was started but never used. A ticker with steps stays — its history is part of the shared balances."""
    def go(book: dict) -> None:
        mine = [e for e in book["chain"] if e["do"] != "cash" and e["ticker"] == name or e["do"] == "cash" and name in (e["src"], e["dst"])]
        if not any(e["do"] == "open" for e in mine):
            raise BookError(f"ไม่พบ “{name}”")
        if len(mine) > 1:
            raise BookError(f"“{name}” มี {len(mine) - 1} รายการแล้ว — ลบไม่ได้ (ใช้ “ย้อนรายการล่าสุด” ทีละรายการ)")
        book["chain"] = [e for e in book["chain"] if e is not mine[0]]
    _edit(path, go)


def _aside(path: Path, tag: str) -> Path:
    """A sibling name that is not taken yet — two moves in the same second never overwrite each other."""
    base = path.with_name(f"{path.name}.{tag}-{datetime.now():%Y%m%d%H%M%S}")
    dest, i = base, 1
    while dest.exists():
        i += 1
        dest = base.with_name(f"{base.name}-{i}")
    return dest


def quarantine(path: Any) -> Path:
    """Move an unreadable data file aside (never delete it) so a fresh book can start."""
    path = Path(path)
    with _LOCK:
        dest = _aside(path, "broken")
        os.replace(path, dest)
    return dest


# ── backup, and the Chain data that already exists ────────────────────────────────────────
def export_text(book: dict) -> str:
    """The text of a backup file: the book exactly as it is kept on disk."""
    return json.dumps(book, ensure_ascii=False, indent=2, allow_nan=False)


def backup_filename(now: Optional[datetime] = None) -> str:
    return f"app_v4_backup-{now or datetime.now():%Y%m%d-%H%M}.json"


def _fin(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _legacy_state(t: dict) -> tuple:
    """(price, fix_c, baseline, ev burn) of one ticker of the earlier Chain data: ``current_state``, else the "Final" string
    ("price, fix_c, baseline"), else ``initial_state``."""
    cs = t.get("current_state")
    if isinstance(cs, dict):
        v = [_fin(cs.get(k)) or 0.0 for k in ("price", "fix_c", "baseline", "cumulative_ev")]
        return v[0], v[1], v[2], v[3]
    final = t.get("Final")
    if isinstance(final, str):
        try:
            parts = [float(x) for x in final.replace("−", "-").split(",")]
        except ValueError:
            parts = []
        if len(parts) >= 2 and all(map(math.isfinite, parts)):
            return parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0, 0.0
    ist = t.get("initial_state")
    if isinstance(ist, dict):
        return _fin(ist.get("price")) or 0.0, _fin(ist.get("fix_c")) or 0.0, 0.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


def from_legacy(raw: Any) -> tuple[dict, list]:
    """The Chain data that already exists (a list of tickers, or ``{tickers, global_pool_cf, global_ev_reserve}``) → a book
    that starts from where it stands: one ``open`` per ticker that can run (price and fix_c > 0) plus the opening Pool CF and
    EV Reserve. Returns (book, warnings); the history of the old file stays in the old file."""
    items = raw.get("tickers") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise BookError("ไม่ใช่ไฟล์ Chain — ต้องเป็นไฟล์สำรองของ app-v4 หรือ Chain data เดิม (มีรายการ ticker)")
    stamp, chain, warns, seen = _now(), [], [], set()
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            warns.append(f"ข้ามรายการที่ {i}: ไม่ใช่ object")
            continue
        label = str(item.get("ticker") or "").strip()
        try:
            name = _name(label)
        except BookError:
            warns.append(f"ข้ามรายการที่ {i} “{label[:MAX_NAME]}”: ชื่อใช้ไม่ได้")
            continue
        if name.lower() in seen:
            warns.append(f"ข้าม {name}: ชื่อซ้ำ")
            continue
        price, fix_c, base, burn = _legacy_state(item)
        if not (price > 0 and fix_c > 0):
            warns.append(f"ข้าม {name}: ราคา/ทุน ≤ 0 (ปิดแล้ว) — Ev burn {burn:,.2f} ไม่ถูกนำเข้า")
            continue
        seen.add(name.lower())
        chain.append({"do": "open", "ticker": name, "c": _r2(fix_c), "t": price, "b": _r2(base), "burn": _r2(max(0.0, burn)), "at": stamp})
    if not chain:
        raise BookError("ไม่มี ticker ที่นำเข้าได้ (ต้องมีราคาและ fix_c มากกว่า 0)")
    if isinstance(raw, dict):
        for key, lane in (("global_pool_cf", POOL), ("global_ev_reserve", RESERVE)):
            v = _fin(raw.get(key)) or 0.0
            if _r2(v) == 0:
                continue
            if v < 0 and lane == POOL:
                warns.append(f"Pool CF ติดลบ ({v:,.2f}) — ไม่นำเข้า")
                continue
            src, dst = (OUT, lane) if v > 0 else (lane, OUT)
            chain.append({"do": "cash", "src": src, "dst": dst, "amount": abs(v), "at": stamp})
    return {"version": VERSION, "chain": chain}, warns


def parse_backup(data: bytes) -> tuple[dict, list]:
    """A file's bytes → (a validated book, warnings): a backup of this app, or the Chain data that already exists."""
    if len(data) > MAX_IMPORT_BYTES:
        raise BookError("ไฟล์ใหญ่เกินไป — ไม่ใช่ไฟล์สำรองของแอปนี้")
    try:
        raw = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise BookError("ไฟล์ไม่ใช่ข้อความ UTF-8 — ไม่ใช่ไฟล์สำรองของแอปนี้") from None
    except ValueError:
        raise BookError("ไฟล์ไม่ใช่ JSON ที่ถูกต้อง") from None
    if isinstance(raw, dict) and "chain" in raw:
        return _validated(raw), []
    book, warns = from_legacy(raw)
    return _validated(book), warns


def import_book(path: Any, data: bytes) -> tuple[dict, list]:
    """Replace the whole book with a backup. The file it replaces is kept beside it (never deleted); a file that does not
    validate changes nothing at all."""
    book, warns = parse_backup(data)
    path = Path(path)
    with _LOCK:
        if path.exists():
            _aside(path, "replaced").write_bytes(path.read_bytes())
        save(path, book)
    return book, warns
