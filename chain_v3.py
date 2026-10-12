"""chain_v3 — Chain System v3 in a few functions. Standalone: stdlib only, its own data file.

One position is two numbers, (c, t): c = the capital you hold at constant value, t = the reference price.
When the price moves to P (x = P / t), one round is five steps:

    ① E  = c · (x − 1 − ln x)     harvest the swing           (LEGO: E = A − R — the trend is cut out, E ≥ 0)
    ② K  = 12.4% · c · Hedge×     pay the insurance first     (Put premium — or the real premium)
    ③ S  = E − K                  what is left?               (Surplus, read at E)
    ④ c′ = c + max(0, S)          grow only from the surplus  (Free Risk — c never shrinks)
    ⑤ t′ = P                      re-center: the next round starts here

and one check: Ev Efficiency = ΣE ÷ ΣK ≥ 1.0 — the swing has paid for all the insurance so far.

E is the equation of the LEGO ledger: A = c·(x − 1) is the cash of one rebalance, R = c·ln x the reference line, and
E = A − R is what is left once the trend is removed — never negative, because x − 1 ≥ ln x. Reading the Surplus at E
(not at R) means only the *swing* of the price can pay for the insurance, never a plain rise.

Money is kept to the cent *before* it is combined (S = E − K exactly, c′ = c + S exactly), so every line on the
screen adds up. Rounds are the only source of truth: c, t, ΣE, ΣK are all read back from them.
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

# Put 10% out of the money, 1 year, σ 50%, r 4% (Black-Scholes) per $1 of c per 1× hedge — the premium is
# proportional to c, so the whole pricing model collapses to this one number.
PREMIUM_RATE = 0.1239616719
MAX_NAME = 20
VERSION = 2                                                      # 1 = the earlier book, whose Surplus was read at R (ln)
_BAD_NAME = re.compile(r"[\x00-\x1f`*_\[\]<>\\|$#&]")     # nothing that Markdown / HTML could read as formatting
_ROUND_KEYS = ("t", "p", "c", "E", "K", "S", "c_after", "hedge_x")
_ROUND_RULES = {"t": {"positive": True}, "p": {"positive": True}, "c": {"positive": True}, "E": {"minimum": 0.0},
                "K": {"minimum": 0.0}, "S": {}, "c_after": {"positive": True}, "hedge_x": {"minimum": 0.0}}   # E ≥ 0, c never shrinks
_LOCK = threading.RLock()


class BookError(ValueError):
    """Something the user can fix: a bad number, a bad name, an unreadable file."""


class Stale(BookError):
    """The position changed after the plan was made — nothing was saved."""


# ── numbers ───────────────────────────────────────────────────────────────────────
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


def plan(c: Any, t: Any, p: Any, hedge_x: Any = 1.0, k: Any = None) -> dict:
    """One round, computed but not saved. ``k`` = the real premium (None → the formula)."""
    c, t, p = _num(c, "ทุน c", positive=True), _num(t, "ราคารอบก่อน t", positive=True), _num(p, "ราคาใหม่ P", positive=True)
    hedge_x = _num(hedge_x, "Hedge ×", minimum=0.0)
    real = None if k is None else _num(k, "ค่าประกันที่จ่ายจริง K", minimum=0.0)
    try:
        e = _r2(c * _swing(p / t))                              # ① harvest the swing
    except (ValueError, OverflowError):                         # p / t underflowed to 0 or overflowed
        raise BookError("ตัวเลขใหญ่หรือเล็กเกินไป — ตรวจทุน c และราคาอีกครั้ง") from None
    cost = _r2(PREMIUM_RATE * c * hedge_x) if real is None else _r2(real)   # ② pay first
    s = _r2(e - cost)                                           # ③ what is left
    c_after = _r2(c + max(0.0, s))                              # ④ grow, never shrink
    if not all(map(math.isfinite, (e, cost, s, c_after))):      # absurdly large inputs overflow to inf
        raise BookError("ตัวเลขใหญ่เกินไป — ตรวจทุน c ราคา และ Hedge × อีกครั้ง")
    return {"t": t, "p": p, "c": c, "E": e, "K": cost, "S": s, "c_after": c_after,
            "hedge_x": hedge_x, "real_k": real is not None}                                        # ⑤ t′ = p


def _swing(x: float) -> float:
    """x − 1 − ln x: the harvest per $1 of c when the price moves by the factor x. Zero at x = 1, positive on
    both sides (that is x − 1 ≥ ln x). Near 1 it is written with log1p so a tiny move does not lose its digits."""
    u = x - 1.0
    return u - math.log1p(u) if abs(u) < 0.5 else u - math.log(x)


def breakeven_prices(c: float, t: float, hedge_x: float = 1.0, k: Optional[float] = None) -> tuple:
    """(low, high): the two prices where the swing just pays for the insurance (E = K). The Surplus is positive
    below ``low`` and above ``high`` (E grows with the size of the move, up or down). A side that no price can
    reach (or that is too extreme to print) is None; with no premium both are t."""
    cost = PREMIUM_RATE * c * hedge_x if k is None else k
    r = cost / c                                                 # the premium per $1 of c
    if r <= 0:
        return t, t
    lo, hi = math.exp(-(1.0 + r)), 1.0                           # below 1: _swing(lo) = r + lo > r, _swing(1) = 0 < r
    low = None
    if lo > 0:
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _swing(mid) > r else (lo, mid)
        low = t * (lo + hi) / 2
    lo, hi = 1.0, 2.0                                            # above 1: grow hi until _swing(hi) ≥ r, then bisect
    while _swing(hi) < r and hi < 1e300:
        hi *= 2
    high = None
    if _swing(hi) >= r:
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if _swing(mid) >= r else (mid, hi)
        high = t * (lo + hi) / 2
    ok = lambda v: v if v is not None and 0 < v < math.inf else None       # noqa: E731
    return ok(low), ok(high)


# ── one position, read back from its rounds ──────────────────────────────────────────
def current(pos: dict) -> tuple[float, float]:
    """(c, t) now: the last round's (c′, t′ = P), or the starting values."""
    r = pos["rounds"]
    return (r[-1]["c_after"], r[-1]["p"]) if r else (pos["c0"], pos["t0"])


def totals(pos: dict) -> tuple[float, float]:
    """(ΣE, ΣK) over every round — ΣE is the ledger's Eₙ."""
    return _r2(sum(r["E"] for r in pos["rounds"])), _r2(sum(r["K"] for r in pos["rounds"]))


def efficiency(pos: dict) -> Optional[float]:
    """Ev Efficiency = ΣE ÷ ΣK (None until some insurance has been paid)."""
    e, k = totals(pos)
    return e / k if k > 0 else None


# ── the book: its own JSON file ────────────────────────────────────────────────────
def _name(raw: Any) -> str:
    name = str(raw or "").strip()
    if not 1 <= len(name) <= MAX_NAME or _BAD_NAME.search(name):
        raise BookError(f"ชื่อต้องยาว 1–{MAX_NAME} ตัวอักษร และไม่มีอักขระพิเศษ ` * _ [ ] < > | $ # &")
    return name


def _validated(book: Any) -> dict:
    try:
        if book["version"] != VERSION:                            # an earlier book read its Surplus at R — its rounds do not carry over
            raise BookError("ไฟล์นี้บันทึกด้วยสมการรุ่นก่อน (Surplus ฐาน R) ใช้ต่อไม่ได้ — กด “เริ่มใหม่” "
                            "(ไฟล์เดิมถูกเก็บไว้ ไม่ถูกลบ)")
        out = {"version": VERSION, "positions": []}
        seen = set()
        for p in book["positions"]:
            name = _name(p["name"])
            if name.lower() in seen:
                raise BookError("มีชื่อซ้ำในไฟล์")
            seen.add(name.lower())
            pos = {"name": name, "c0": _num(p["c0"], "c0", positive=True), "t0": _num(p["t0"], "t0", positive=True), "rounds": []}
            for r in p["rounds"]:
                rec = {k: _num(r[k], k, **_ROUND_RULES[k]) for k in _ROUND_KEYS}
                if rec["c_after"] < rec["c"]:
                    raise BookError("ทุนหลังรอบน้อยกว่าก่อนรอบ — ไฟล์ไม่ถูกต้อง")
                rec.update(real_k=bool(r.get("real_k", False)), at=str(r.get("at", "")))
                pos["rounds"].append(rec)
            out["positions"].append(pos)
        return out
    except BookError:
        raise
    except (KeyError, TypeError, AttributeError) as e:
        raise BookError(f"โครงสร้างไฟล์ไม่ถูกต้อง ({type(e).__name__})") from None


def load(path: Any) -> dict:
    """The book ({'positions': [...]}); a missing file is an empty book; a broken one raises BookError."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": VERSION, "positions": []}
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


def _find(book: dict, name: str) -> dict:
    for pos in book["positions"]:
        if pos["name"] == name:
            return pos
    raise BookError(f"ไม่พบ “{name}”")


def start(path: Any, name: Any, c: Any, t: Any) -> str:
    """Begin tracking a position: the capital c you hold and the price t it starts from."""
    def go(book: dict) -> str:
        n = _name(name)
        if any(p["name"].lower() == n.lower() for p in book["positions"]):
            raise BookError("ชื่อนี้มีอยู่แล้ว")
        book["positions"].append({"name": n, "c0": _r2(_num(c, "ทุน c", positive=True)),
                                  "t0": _num(t, "ราคา t", positive=True), "rounds": []})
        return n
    return _edit(path, go)


def commit(path: Any, name: str, pl: dict) -> None:
    """Save a planned round — but only if the position is still exactly where the plan started."""
    def go(book: dict) -> None:
        pos = _find(book, name)
        c, t = current(pos)
        try:
            fresh = plan(c, t, pl["p"], pl["hedge_x"], pl["K"] if pl["real_k"] else None)
        except (KeyError, TypeError):
            raise Stale("แผนไม่ถูกต้อง — คำนวณใหม่") from None
        if fresh != pl:
            raise Stale("ข้อมูลเปลี่ยนไปแล้วหลังคำนวณ — ตรวจตัวเลขอีกครั้งแล้วกดบันทึกใหม่")
        pos["rounds"].append({"at": datetime.now().strftime("%Y-%m-%d %H:%M"), **{k: pl[k] for k in _ROUND_KEYS},
                              "real_k": pl["real_k"]})
    _edit(path, go)


def undo(path: Any, name: str) -> dict:
    """Take back the last round (c and t return to what they were before it)."""
    def go(book: dict) -> dict:
        pos = _find(book, name)
        if not pos["rounds"]:
            raise BookError("ยังไม่มีรอบให้ย้อน")
        return pos["rounds"].pop()
    return _edit(path, go)


def remove(path: Any, name: str) -> None:
    def go(book: dict) -> None:
        _find(book, name)
        book["positions"] = [p for p in book["positions"] if p["name"] != name]
    _edit(path, go)


def quarantine(path: Any) -> Path:
    """Move an unreadable data file aside (never delete it) so a fresh book can start."""
    path = Path(path)
    dest = path.with_name(f"{path.name}.broken-{datetime.now():%Y%m%d%H%M%S}")
    with _LOCK:
        os.replace(path, dest)
    return dest
