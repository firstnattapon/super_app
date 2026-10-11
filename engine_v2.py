"""engine_v2 — core of the Chain System v2 page (``pages/app-v2.py``).

simple · stable · fast
  * pure Python (no Streamlit / numpy / pandas) so it imports instantly and is unit-testable
  * all-or-nothing writes: ``Store.update`` edits a deep copy, writes it atomically
    (tmp file + fsync + os.replace) and only then publishes it in memory
  * same ``trading_data.json`` schema as v1 (``flywheels.py``), so both pages share one portfolio
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

# ── Hedge model (same constants as v1 ``flywheels.run_chain_round``) ─────────
HEDGE_STRIKE = 0.9   # Put strike = 90% of the current price (OTM 10%)
HEDGE_SIGMA = 0.5
HEDGE_T = 1.0
HEDGE_RATE = 0.04

DEFAULT_SETTINGS = {"surplus_mode": "auto_compound", "default_sigma": 0.5, "default_hedge_ratio": 2.0}
STATE_FIELDS = ("price", "fix_c", "baseline", "cumulative_ev", "pool_cf_net",
                "surplus_iv", "lock_pnl", "net_pnl")
SYMBOL_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-^=]{0,14}$")
MAX_IMPORT_BYTES = 8 * 1024 * 1024


class ChainError(ValueError):
    """Invalid input or operation — the message is safe to show in the UI."""


class StaleState(ChainError):
    """The ticker changed after the plan was computed."""


class DataError(RuntimeError):
    """The data file is unreadable (and so is its backup)."""


# ── Small helpers ─────────────────────────────────────────────────────────────
def num(x: Any, default: Any = 0.0) -> Any:
    """``float(x)`` when it is a finite number, otherwise ``default`` (None / junk / NaN / inf)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _r2(x: float) -> float:
    """Round to cents and never produce ``-0.0``."""
    return round(x, 2) + 0.0


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def money(x: float) -> str:
    v = _r2(x)
    return f"{'-' if v < 0 else ''}${abs(v):,.2f}"


def _positive(amount: Any, label: str = "จำนวนเงิน") -> float:
    a = num(amount, None)
    if a is None or a <= 0:
        raise ChainError(f"{label}ต้องมากกว่า 0")
    return a


# ── Pricing & the round plan ──────────────────────────────────────────────────
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(s: float, k: float, t: float = HEDGE_T, r: float = HEDGE_RATE,
           sigma: float = HEDGE_SIGMA) -> float:
    """Black-Scholes European put."""
    if min(s, k, t, sigma) <= 0:
        return 0.0
    srt = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma ** 2) * t) / srt
    return k * math.exp(-r * t) * _ncdf(-(d1 - srt)) - s * _ncdf(-d1)


def hedge_cost(fix_c: float, price: float, ratio: float) -> float:
    """Premium for ``ratio`` × (fix_c / price) shares of 10%-OTM 1y puts."""
    if fix_c <= 0 or price <= 0 or ratio <= 0:
        return 0.0
    return (fix_c / price) * ratio * bs_put(price, price * HEDGE_STRIKE)


def _override(label: str, v: Any, lo: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    x = num(v, None)
    if x is None or (lo is not None and x < lo):
        raise ChainError(f"{label} ไม่ถูกต้อง")
    return x


def plan_round(state: dict, p_new: Any, hedge_ratio: Any, *, no_hedge: bool = False,
               no_surplus: bool = False, hedge: Any = None, c_after: Any = None,
               b_after: Any = None) -> dict:
    """Compute one Chain Round (pure — nothing is saved).

        shannon = c·ln(P/t)      hedge   = ratio × (c/t) × Put(K = 0.9t)
        surplus = shannon − hedge
        c'      = c + max(0, surplus)           (Scale Up — Free Risk)
        b'      = b + shannon                   (t' = P re-centres, so the c'·ln(P/t') term is 0)

    ``hedge`` / ``c_after`` / ``b_after`` replace the model with the real numbers (override).
    A ticker without a price (t = 0) just seeds its price: Shannon and the model hedge are 0.
    """
    p_old = num(state.get("price"))
    c = max(0.0, num(state.get("fix_c")))
    b = num(state.get("baseline"))

    p_new = num(p_new, None)
    if p_new is None or p_new <= 0:
        raise ChainError("P New ต้องเป็นตัวเลขมากกว่า 0")
    ratio = num(hedge_ratio, None)
    if ratio is None or ratio < 0:
        raise ChainError("Hedge × ต้องเป็นตัวเลข ≥ 0")
    hedge, c_after, b_after = (_override(*a) for a in (("Hedge cost", hedge, 0.0),
                                                       ("fix_c after", c_after, 0.0),
                                                       ("Baseline after", b_after, None)))

    live = p_old > 0
    shannon = c * math.log(p_new / p_old) if live else 0.0
    h = hedge if hedge is not None else (0.0 if (no_hedge or not live) else hedge_cost(c, p_old, ratio))
    surplus = shannon - h
    if c_after is not None:
        c_new = c_after
    else:
        c_new = c if no_surplus else c + max(0.0, surplus)
    b_new = b + shannon if b_after is None else b_after

    c0, c1 = _r2(c), _r2(c_new)
    return {
        "action": "Chain Round",
        "p_old": round(p_old, 4), "p_new": round(p_new, 4),
        "c_before": c0, "c_after": c1,
        "shannon_profit": _r2(shannon), "hedge_cost": _r2(h), "surplus": _r2(surplus),
        "scale_up": _r2(c1 - c0),
        "b_before": _r2(b), "b_after": _r2(b_new),
        "hedge_ratio": 0.0 if no_hedge else ratio, "sigma": HEDGE_SIGMA,
        "ev_change": _r2(h),
    }


# ── Portfolio operations (each mutates the working copy ``d`` given by Store.update) ──
def _find(d: dict, name: str) -> dict:
    for t in d["tickers"]:
        if t["ticker"] == name:
            return t
    raise ChainError(f"ไม่พบ ticker “{name}”")


def _tag(text: str, ticker: Optional[str]) -> str:
    return f"{text} [Ticker: {ticker}]" if ticker and ticker != "None" else text


def _log(d: dict, category: str, amount: float, note: str = "") -> None:
    d["treasury_history"].append({
        "date": _now(), "category": category, "amount": _r2(amount),
        "pool_cf_balance": _r2(d["global_pool_cf"]), "ev_reserve_balance": _r2(d["global_ev_reserve"]),
        "note": note,
    })


def _append_round(t: dict, rec: dict) -> None:
    """Append a round and move current_state to its *after* values (the log is the source of truth)."""
    s = t["current_state"]
    rec = {**rec, "date": _now(), "round_id": len(t["rounds"]) + 1}
    t["rounds"].append(rec)
    ev_old = s["cumulative_ev"]
    ev_new = max(0.0, ev_old + num(rec.get("ev_change")))
    s.update(price=rec["p_new"], fix_c=rec["c_after"], baseline=rec["b_after"],
             cumulative_ev=_r2(ev_new), net_pnl=_r2(s["net_pnl"] - (ev_new - ev_old)))


def _flat_round(action: str, s: dict, b_after: float, note: str) -> dict:
    """A round that only moves the baseline (Extract / Deploy)."""
    p, c = round(s["price"], 4), _r2(s["fix_c"])
    return {"action": action, "p_old": p, "p_new": p, "c_before": c, "c_after": c,
            "shannon_profit": 0.0, "harvest_profit": 0.0, "hedge_cost": 0.0, "surplus": 0.0,
            "scale_up": 0.0, "b_before": _r2(s["baseline"]), "b_after": _r2(b_after),
            "note": note, "hedge_ratio": 0.0, "sigma": 0.0, "ev_change": 0.0}


def apply_round(d: dict, name: str, plan: dict) -> None:
    """Commit a plan. Refuses if the ticker moved since the plan was computed (stale preview)."""
    t = _find(d, name)
    s = t["current_state"]
    fresh = (math.isclose(s["price"], plan["p_old"], abs_tol=6e-5)
             and math.isclose(s["fix_c"], plan["c_before"], abs_tol=5.1e-3)
             and math.isclose(s["baseline"], plan["b_before"], abs_tol=5.1e-3))
    if not fresh:
        raise StaleState(f"{name} เปลี่ยนไปแล้วหลังคำนวณ — ตรวจตัวเลขอีกครั้งแล้วกด Commit ใหม่")
    _append_round(t, plan)


def add_to_pool(d: dict, amount: Any, kind: str = "Funding", ticker: Optional[str] = None) -> None:
    """Funding / Harvest → Pool CF."""
    if kind not in ("Funding", "Harvest"):
        raise ChainError("ประเภทไม่ถูกต้อง")
    a = _positive(amount)
    d["global_pool_cf"] = _r2(d["global_pool_cf"] + a)
    _log(d, kind, a, _tag("Harvest Profit" if kind == "Harvest" else "Added to Pool CF", ticker))


def extract_baseline(d: dict, name: str, amount: Any) -> None:
    """Ticker baseline → Pool CF."""
    t = _find(d, name)
    s = t["current_state"]
    a = _positive(amount)
    if a > s["baseline"] + 1e-9:
        raise ChainError(f"Baseline ไม่พอ (มี {money(s['baseline'])})")
    d["global_pool_cf"] = _r2(d["global_pool_cf"] + a)
    _append_round(t, _flat_round("Extract Baseline", s, s["baseline"] - a, f"Extracted ${a:,.2f} to Pool CF"))
    _log(d, "Baseline Harvest", a, f"[Ticker: {name}]")


def deploy(d: dict, name: str, amount: Any, note: str = "") -> None:
    """Pool CF → ticker baseline."""
    t = _find(d, name)
    s = t["current_state"]
    a = _positive(amount)
    if a > d["global_pool_cf"] + 1e-9:
        raise ChainError(f"Pool CF ไม่พอ (มี {money(d['global_pool_cf'])})")
    suffix = f" | {note.strip()}" if note and note.strip() else ""
    d["global_pool_cf"] = _r2(d["global_pool_cf"] - a)
    _append_round(t, _flat_round("Deploy to Baseline", s, s["baseline"] + a,
                                 f"Deploy from Pool CF → Baseline{suffix}"))
    _log(d, "Deploy", -a, _tag("Deploy to Baseline", name) + suffix)


def allocate_ev(d: dict, amount: Any) -> None:
    """Pool CF → EV Reserve."""
    a = _positive(amount)
    if a > d["global_pool_cf"] + 1e-9:
        raise ChainError(f"Pool CF ไม่พอ (มี {money(d['global_pool_cf'])})")
    d["global_pool_cf"] = _r2(d["global_pool_cf"] - a)
    d["global_ev_reserve"] = _r2(d["global_ev_reserve"] + a)
    _log(d, "Allocation", a, "Pool CF -> EV Reserve")


def ev_flow(d: dict, amount: Any, ticker: Optional[str] = None) -> None:
    """Manual EV Reserve adjustment: positive = income, negative = expense (e.g. LEAPS paid)."""
    a = num(amount, None)
    if a is None or a == 0:
        raise ChainError("ระบุจำนวนเงิน (บวก = Income, ลบ = Expense)")
    d["global_ev_reserve"] = _r2(d["global_ev_reserve"] + a)
    _log(d, "Income" if a > 0 else "Expense", a, _tag("Manual Adjustment", ticker))


def add_ticker(d: dict, symbol: str, price: Any, fix_c: Any) -> None:
    sym = (symbol or "").strip().upper()
    if not SYMBOL_RE.match(sym):
        raise ChainError("Ticker ใช้ได้เฉพาะ A-Z 0-9 . - ^ = (สูงสุด 15 ตัว)")
    if any(t["ticker"].upper() == sym for t in d["tickers"]):
        raise ChainError("Ticker นี้มีอยู่แล้วในพอร์ต")
    p, c = _positive(price, "ราคาเริ่มต้น"), _positive(fix_c, "fix_c")
    rec = f"{p}, {c}, 0.0"
    d["tickers"].append({
        "ticker": sym, "Final": rec, "Original": rec,
        "initial_state": {"price": p, "fix_c": c},
        # baseline starts at 0 (never the price) so Extract can't withdraw money that never existed
        "current_state": {"price": p, "fix_c": c, "baseline": 0.0, "cumulative_ev": 0.0,
                          "pool_cf_net": 0.0, "surplus_iv": 0.0, "lock_pnl": 0.0,
                          "net_pnl": 0.0, "strategy_tags": []},
        "rounds": [], "beta_momory": "0.0, 0.0, 0.0",
    })


# ── Read-only views of the data ───────────────────────────────────────────────
def withdrawn_baseline(t: dict) -> float:
    return sum(num(r.get("b_before")) - num(r.get("b_after")) for r in t["rounds"]
               if r.get("action") == "Extract Baseline" and num(r.get("b_before")) > num(r.get("b_after")))


def totals(d: dict) -> dict:
    states = [t["current_state"] for t in d["tickers"]]
    s = {k: sum(x[k] for x in states) for k in ("fix_c", "cumulative_ev", "lock_pnl", "surplus_iv", "net_pnl")}
    gross = s["lock_pnl"] + s["surplus_iv"]
    return {"pool": d["global_pool_cf"], "reserve": d["global_ev_reserve"],
            "cash": d["global_pool_cf"] + d["global_ev_reserve"],
            "fix_c": s["fix_c"], "burn": s["cumulative_ev"], "lock": s["lock_pnl"],
            "iv": s["surplus_iv"], "net": s["net_pnl"], "gross": gross,
            "efficiency": gross / s["cumulative_ev"] if s["cumulative_ev"] > 0 else None,
            "rounds": sum(len(t["rounds"]) for t in d["tickers"])}


_NOTE_TICKER = re.compile(r"\[Ticker:\s*([^\]]+)\]")


def treasury_rows(d: dict, ticker: Optional[str] = None, limit: int = 300) -> list[dict]:
    """Newest first; optionally only entries tagged ``[Ticker: X]``."""
    rows = []
    for e in reversed(d["treasury_history"]):
        m = _NOTE_TICKER.search(str(e.get("note", "")))
        if ticker and (not m or m.group(1).strip() != ticker):
            continue
        rows.append({"Date": str(e.get("date", "")), "Action": str(e.get("category", "")),
                     "Amount": num(e.get("amount")), "Pool CF": num(e.get("pool_cf_balance")),
                     "EV Reserve": num(e.get("ev_reserve_balance")), "Note": str(e.get("note", ""))})
        if len(rows) >= limit:
            break
    return rows


def round_rows(t: dict, limit: Optional[int] = None) -> list[dict]:
    """Rounds of one ticker, newest first, as flat typed rows."""
    rows = []
    for i, r in enumerate(t["rounds"], 1):
        rows.append({"#": int(num(r.get("round_id"), i)), "Date": str(r.get("date", "")),
                     "Action": str(r.get("action", "")),
                     "P old": num(r.get("p_old")), "P new": num(r.get("p_new")),
                     "fix_c old": num(r.get("c_before")), "fix_c new": num(r.get("c_after")),
                     "b old": num(r.get("b_before")), "b new": num(r.get("b_after")),
                     "Shannon": num(r.get("shannon_profit")), "Hedge": num(r.get("hedge_cost")),
                     "Surplus": num(r.get("surplus")), "Note": str(r.get("note", ""))})
    rows.reverse()
    return rows[:limit] if limit else rows


def portfolio_rows(d: dict) -> list[dict]:
    rows = []
    for t in d["tickers"]:
        s = t["current_state"]
        rows.append({"Ticker": t["ticker"], "Price": s["price"], "fix_c": s["fix_c"],
                     "Baseline": s["baseline"], "Ev burn": s["cumulative_ev"],
                     "Lock P&L": s["lock_pnl"], "Surplus IV": s["surplus_iv"],
                     "Net P&L": s["net_pnl"], "Rounds": len(t["rounds"])})
    return rows


def capital_flow(d: dict, names: Optional[list[str]] = None) -> tuple[list[str], list[tuple]]:
    """Aggregated Sankey data: (node labels, [(source, target, value, label)])."""
    keys = ("shannon_profit", "harvest_profit", "hedge_cost", "scale_up")
    per = [(t["ticker"], {k: sum(num(r.get(k)) for r in t["rounds"]) for k in keys}, t["current_state"]["cumulative_ev"])
           for t in d["tickers"] if not names or t["ticker"] in names]
    sums = {k: sum(p[k] for _, p, _ in per) for k in keys}
    burn = sum(ev for _, _, ev in per)
    shown = [(n_, p) for n_, p, _ in per if p["shannon_profit"] > 0.001 or p["harvest_profit"] > 0.001]
    nodes, links = [n_ for n_, _ in shown], []
    SH, HA, HG, SU, SC, PL, EV = range(len(nodes), len(nodes) + 7)
    nodes += ["Shannon", "Harvest", "Hedge 🛡️", "Net Surplus", "Scale Up 🚀",
              f"Pool CF 🎱 {money(d['global_pool_cf'])}", "Ev Burn 🔥"]

    def link(s, t_, v, label):
        if v > 0.001:
            links.append((s, t_, round(v, 2), label))

    for i, (n_, p) in enumerate(shown):
        link(i, SH, p["shannon_profit"], f"{n_} Shannon")
        link(i, HA, p["harvest_profit"], f"{n_} Harvest")
    net = sums["shannon_profit"] - sums["hedge_cost"]
    link(SH, HG, min(sums["hedge_cost"], sums["shannon_profit"]), "Hedge premium")
    link(SH, SU, net, "Surplus")
    link(HA, PL, sums["harvest_profit"], "Harvest → Pool CF")
    link(SU, SC, min(sums["scale_up"], max(0.0, net)), "Auto Scale Up")
    link(SU, PL, net - sums["scale_up"], "Overflow → Pool CF")
    link(SC, EV, min(burn, sums["scale_up"]), "Ev (time decay)")
    return nodes, links


# ── Payoff Profile simulator (v1's 11-line log model; pure Python lists) ─────────
PAYOFF_LINES = {
    "y1": "y1 · Shannon 1", "y2": "y2 · Shannon 2", "y3": "y3 · Net P/L (Σ)",
    "y4": "y4 · Piecewise y2", "y5": "y5 · Piecewise y1", "y6": "y6 · Benchmark",
    "y7": "y7 · Ref y2", "y8": "y8 · Call intrinsic", "y9": "y9 · Put intrinsic",
    "y10": "y10 · P/L Long", "y11": "y11 · P/L Short",
}
NET_COMPONENTS = ("y1", "y2", "y4", "y5", "y8", "y9", "y10", "y11")


def payoff_prices(p: float, n: int = 300) -> list[float]:
    lo = max(0.01, p * 0.1)
    hi = max(lo + 0.1, p * 2.5)
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def _mirror_ln(x: float, anchor: float) -> float:
    r = x / anchor
    return math.log(2 - r if r < 2 else 1e-9)


def payoff_curves(c: dict, xs: list[float], net_of: Any = NET_COMPONENTS) -> dict[str, list[float]]:
    """y1..y11 plus ``overlay`` (= Net − y6). ``net_of`` chooses which lines are summed into y3."""
    d1, d2 = c["delta1"], c["delta2"]
    y1r = [c["constant1"] * math.log(x / c["x0_1"]) for x in xs]
    y2r = [c["constant2"] * _mirror_ln(x, c["x0_2"]) for x in xs]
    out = {
        "y1": [v * d2 + c["b1"] for v in y1r],
        "y2": [v * d2 + c["b2"] for v in y2r],
        "y4": [v * (d2 if x >= c["x0_2"] else d1) + c["b2"] for v, x in zip(y2r, xs)],
        "y5": [v * (d2 if x >= c["x0_1"] else d1) + c["b1"] for v, x in zip(y1r, xs)],
        "y6": [c["ref_const"] * math.log(x / c["anchor"]) * d2 for x in xs],
        "y7": [c["ref_const"] * _mirror_ln(x, c["anchor"]) * d2 for x in xs],
        "y8": [max(0.0, x - c["call_strike"]) * c["call_qty"] for x in xs],
        "y9": [max(0.0, c["put_strike"] - x) * c["put_qty"] for x in xs],
        "y10": [(x - c["long_entry"]) * c["long_qty"] for x in xs],
        "y11": [(c["short_entry"] - x) * c["short_qty"] for x in xs],
    }
    pick = [k for k in NET_COMPONENTS if k in net_of]
    out["y3"] = [sum(out[k][i] for k in pick) for i in range(len(xs))]
    out["overlay"] = [a - b for a, b in zip(out["y3"], out["y6"])]
    return out


# ── Normalising any supported file shape into the V2 schema ─────────────────────
def _empty() -> dict:
    return {"version": 2, "tickers": [], "global_pool_cf": 0.0, "global_ev_reserve": 0.0,
            "settings": dict(DEFAULT_SETTINGS), "treasury_history": []}


def _parse_triplet(s: Any) -> Optional[tuple]:
    """'price, fix_c, baseline' → floats (v1 'Final' / 'Original' strings); None if not numeric."""
    if not isinstance(s, str):
        return None
    try:
        parts = [float(x) for x in s.replace("−", "-").split(",")]
    except ValueError:
        return None
    return (parts[0], parts[1], parts[2] if len(parts) > 2 else 0.0) if len(parts) >= 2 else None


def _legacy_state(t: dict) -> tuple:
    for key in ("Final", "Original"):
        v = _parse_triplet(t.get(key))
        if v and all(math.isfinite(x) for x in v):
            return v
    ist = t.get("initial_state")
    if isinstance(ist, dict):
        return (num(ist.get("price")), num(ist.get("fix_c")), 0.0)
    return (0.0, 0.0, 0.0)


def _normalize_ticker(t: dict) -> None:
    rounds = t.get("rounds")
    rounds = [r for r in rounds if isinstance(r, dict)] if isinstance(rounds, list) else []
    t["rounds"] = rounds
    cs = t.get("current_state")
    if not isinstance(cs, dict):   # legacy V1 record: derive from "Final" / initial_state
        cs = dict(zip(("price", "fix_c", "baseline"), _legacy_state(t)))
    for k in STATE_FIELDS:
        cs[k] = num(cs.get(k))
    cs["price"], cs["fix_c"] = max(0.0, cs["price"]), max(0.0, cs["fix_c"])
    if not isinstance(cs.get("strategy_tags"), list):
        cs["strategy_tags"] = []
    if rounds:   # the round log is the source of truth (same rule as v1 auto-repair)
        p, c, b = (num(rounds[-1].get(k), None) for k in ("p_new", "c_after", "b_after"))
        if p is not None and c is not None and b is not None and p > 0:
            cs.update(price=p, fix_c=max(0.0, c), baseline=b)
    t["current_state"] = cs


def normalize(raw: Any) -> tuple[dict, list[str]]:
    """Coerce a V1 list / V2 dict into a clean V2 dict. Returns ``(data, warnings)``; input is untouched."""
    if isinstance(raw, list):
        raw = {"tickers": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("tickers"), list):
        raise DataError("โครงสร้างข้อมูลไม่ถูกต้อง — ต้องเป็น JSON ที่มีคีย์ 'tickers' (list)")
    warns: list[str] = []
    d = {k: copy.deepcopy(v) for k, v in raw.items() if k != "tickers"}
    d["version"] = 2
    d["global_pool_cf"] = num(raw.get("global_pool_cf"))
    d["global_ev_reserve"] = num(raw.get("global_ev_reserve"))
    settings = dict(DEFAULT_SETTINGS)
    if isinstance(raw.get("settings"), dict):
        settings.update(raw["settings"])
    if num(settings.get("default_hedge_ratio"), None) is None or settings["default_hedge_ratio"] < 0:
        settings["default_hedge_ratio"] = DEFAULT_SETTINGS["default_hedge_ratio"]
    d["settings"] = settings
    hist = raw.get("treasury_history")
    d["treasury_history"] = [copy.deepcopy(e) for e in hist if isinstance(e, dict)] if isinstance(hist, list) else []

    seen: set[str] = set()
    d["tickers"] = []
    for i, item in enumerate(raw["tickers"], 1):
        if not isinstance(item, dict):
            warns.append(f"ข้ามรายการที่ {i}: ไม่ใช่ object")
            continue
        t = copy.deepcopy(item)
        name = str(t.get("ticker") or "").strip() or f"TICKER{i}"
        base, k = name, 1
        while name.upper() in seen:
            k += 1
            name = f"{base}#{k}"
        if name != str(item.get("ticker") or "").strip():
            warns.append(f"ชื่อ ticker ซ้ำ/ว่างในรายการที่ {i} → ใช้ “{name}”")
        seen.add(name.upper())
        t["ticker"] = name
        _normalize_ticker(t)
        d["tickers"].append(t)
    return d, warns


# ── Persistence ─────────────────────────────────────────────────────────────────
class Store:
    """Thread-safe, mtime-aware JSON store. Reads are free; every write is all-or-nothing."""

    def __init__(self, path: Any):
        self.path = Path(path)
        self.backup_path = self.path.with_name(self.path.stem + ".backup" + self.path.suffix)
        self.warnings: list[str] = []
        self._lock = threading.RLock()
        self._data: Optional[dict] = None
        self._sig: Optional[tuple] = None
        self._error: Optional[str] = None
        self._main_ok = True   # False while the main file is corrupt (so its backup is never overwritten)

    # -- reading
    def _stat(self) -> Optional[tuple]:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _load(self) -> None:
        self._sig = self._stat()
        self._error, self._main_ok, self.warnings = None, True, []
        if self._sig is None:
            self._data = _empty()
            return
        try:
            self._data, self.warnings = normalize(json.loads(self.path.read_text(encoding="utf-8")))
            return
        except (OSError, ValueError, DataError) as e:   # unreadable / not JSON / wrong shape
            first = str(e)
        self._main_ok = False
        try:
            self._data, warns = normalize(json.loads(self.backup_path.read_text(encoding="utf-8")))
            self.warnings = [f"ไฟล์หลักอ่านไม่ได้ ({first}) — โหลดจาก backup แทน"] + warns
        except (OSError, ValueError, DataError):
            self._data = None
            self._error = f"อ่านไฟล์ข้อมูลไม่ได้ ({first}) และไม่มี backup ที่ใช้ได้ — Import ข้อมูลใหม่ที่แท็บ ➕ Data"

    def _current(self) -> dict:
        if self._data is None and self._error is None or self._stat() != self._sig:
            self._load()
        if self._error:
            raise DataError(self._error)
        return self._data  # type: ignore[return-value]

    def read(self) -> dict:
        """Current snapshot. Treat it as read-only — change data only through ``update``."""
        with self._lock:
            return self._current()

    # -- writing
    def _write(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)   # fails before touching files
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            if self._main_ok and self.path.exists():
                try:
                    shutil.copy2(self.path, self.backup_path)
                except OSError:
                    pass   # a missing backup must never block saving
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._data, self._sig, self._main_ok, self._error = data, self._stat(), True, None

    def update(self, fn: Callable[[dict], Any]) -> Any:
        """Run ``fn(working_copy)``; on success save atomically, on any exception change nothing."""
        with self._lock:
            work = copy.deepcopy(self._current())
            result = fn(work)
            self._write(work)
            return result

    def replace(self, raw: Any) -> list[str]:
        """Replace the whole portfolio (import / restore / reset). Works even when the file is unreadable."""
        with self._lock:
            data, warns = normalize(raw)
            self._write(data)
            self.warnings = warns
            return warns

    def restore_backup(self) -> list[str]:
        """Swap back to the previous saved version (the current one becomes the new backup)."""
        with self._lock:
            try:
                raw = json.loads(self.backup_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise ChainError(f"ไม่มี backup ที่ใช้ได้ ({e})") from e
            return self.replace(raw)

    def backup_time(self) -> Optional[str]:
        try:
            return datetime.fromtimestamp(self.backup_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            return None


# ── Network helpers (stdlib only — no yfinance/curl_cffi native crashes) ─────────
_UA = {"User-Agent": "Mozilla/5.0 (compatible; ChainSystem/2.0)", "Accept": "application/json"}
_PRICE_CACHE: dict[str, tuple[float, float, str]] = {}
_PRICE_LOCK = threading.Lock()
PRICE_TTL_OK, PRICE_TTL_ERR = 60.0, 15.0


class _HttpsRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise ChainError("ถูก redirect ไป URL ที่ไม่ใช่ https — บล็อกแล้ว")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirects)


def _http_json(url: str, timeout: float, limit: int = MAX_IMPORT_BYTES) -> Any:
    with _OPENER.open(urllib.request.Request(url, headers=_UA), timeout=timeout) as resp:
        raw = resp.read(limit + 1)
    if len(raw) > limit:
        raise ChainError("ไฟล์ใหญ่เกินไป")
    return json.loads(raw.decode("utf-8"))


def _chart_price(symbol: str) -> float:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(symbol, safe='.-^=')}?range=5d&interval=1d")
    results = ((_http_json(url, 6, 1 << 20).get("chart") or {}).get("result")) or []
    if not results:
        return 0.0
    price = num((results[0].get("meta") or {}).get("regularMarketPrice"))
    if price > 0:
        return price
    quotes = (results[0].get("indicators") or {}).get("quote") or [{}]
    for close in reversed(quotes[0].get("close") or []):
        if num(close) > 0:
            return num(close)
    return 0.0


def fetch_price(ticker: str) -> tuple[float, str]:
    """Latest Yahoo price → ``(price, source)``; ``price == 0`` means failure. Cached (60 s ok / 15 s error)."""
    sym = (ticker or "").strip().upper()
    if not sym:
        return 0.0, "error: ไม่มี ticker"
    now = time.monotonic()
    with _PRICE_LOCK:
        hit = _PRICE_CACHE.get(sym)
        if hit and hit[0] > now:
            return hit[1], hit[2] + " (cached)"
    candidates = [sym] + ([f"{sym}.BK"] if "." not in sym and len(sym) <= 5 else [])
    price, source = 0.0, f"error: ไม่พบราคา {sym}"
    for cand in candidates:
        try:
            price = _chart_price(cand)
        except Exception as e:   # best-effort network call: any failure just reports an error
            price, source = 0.0, f"error: {type(e).__name__}"
            continue
        if price > 0:
            source = f"Yahoo ({cand})"
            break
    with _PRICE_LOCK:
        if len(_PRICE_CACHE) > 256:
            _PRICE_CACHE.clear()
        _PRICE_CACHE[sym] = (now + (PRICE_TTL_OK if price > 0 else PRICE_TTL_ERR), price, source)
    return price, source


def fetch_json_url(url: str, timeout: float = 15) -> Any:
    """Download a JSON document. https only (no file:// or plain http), size-capped."""
    u = urllib.parse.urlparse((url or "").strip())
    if u.scheme != "https" or not u.netloc:
        raise ChainError("รองรับเฉพาะ URL แบบ https://")
    return _http_json(u.geturl(), timeout)
