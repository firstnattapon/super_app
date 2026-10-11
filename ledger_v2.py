"""ledger_v2 — LEGO-style R·A·E ledger for Chain System tickers (pure Python, read-only).

Same equations as firstnattapon/streamlit_lego (lego_dash_core.py, cheat_sheet/*.mmd):

    Rₙ  = Fix_c · ln(Pₙ / P₀)             reference line (trend × capital)
                                           with rollover:  Rₙ = β + Fix_c · ln(Pₙ / P₀)
    ΔAₙ = Fix_c · (P_fill / P_acted − 1)  cash of one rebalance act
    Aₙ  = Aₙ₋₁ + ΔAₙ                      cumulative actual cash
    Eₙ  = Aₙ − Rₙ  ≥ 0                    harvested excess   (because x − 1 ≥ ln x)
    β  += (Fix_c − Fix_c_new) · ln(P_acted / P₀)    rollover when Fix_c changes

A Chain Round is one *act*: Fix_c = c_before, P_acted = p_old (the re-centred t), P_fill = p_new.
The ledger is *derived* from the round log and never stored, so the Chain accounting
(Shannon = ΔR, Hedge, Surplus, Scale Up, Baseline) is untouched. Like the LEGO dashboard it
audits instead of guessing: legacy rows are used as recorded and flagged, never "fixed".
"""
from __future__ import annotations

import math
import random
from typing import Any, Optional

import engine_v2 as eng

num = eng.num
ACT_ACTIONS = ("Chain Round",)                 # the only rows with a real fill price per step
LEGACY_ACTION = "Legacy Rollover"              # imported R-domain history — no per-step fills, never recomputed
FLAT_ACTIONS = ("Extract Baseline", "Deploy to Baseline")
KNOWN_ACTIONS = (*ACT_ACTIONS, LEGACY_ACTION, *FLAT_ACTIONS)
MAX_SIM_STEPS = 2000
CENT = 0.011                       # tolerance for values recorded to 2 decimals

# Display contract (Thai names + subscripts, same style as the LEGO 17-column ledger)
C_N, C_DATE, C_ACTION = "รอบ", "เวลา", "สถานะ"
C_PACTED, C_PRICE, C_FIXC = "P_acted (USD)", "ราคา Pₙ (USD)", "Fix_c (USD)"
C_SHANNON, C_HEDGE, C_SURPLUS = "Shannon ΔR (USD)", "Hedge (USD)", "Surplus (USD)"
C_FIXC2, C_BASE = "fix_c′ (USD)", "Baseline b′ (USD)"
C_R, C_DA, C_A, C_E = "Rₙ อ้างอิง (USD)", "ΔAₙ ต่อสเต็ป (USD)", "Aₙ สะสม (USD)", "Eₙ ส่วนเกินสะสม (USD)"
COLUMNS = [C_N, C_DATE, C_ACTION, C_PACTED, C_PRICE, C_FIXC, C_SHANNON, C_HEDGE, C_SURPLUS,
           C_FIXC2, C_BASE, C_R, C_DA, C_A, C_E]
TIP_COLS = [C_SHANNON, C_HEDGE, C_SURPLUS, C_FIXC2, C_BASE, C_R, C_DA, C_A, C_E]
MONEY_COLS = [C_FIXC, *TIP_COLS]


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def _f(x: float, digits: int = 4) -> str:
    return f"{x:,.{digits}f}"


def _op(x: float, digits: int = 4) -> str:
    """Operand after + − ×: negatives in parentheses (same as the LEGO tooltips)."""
    text = _f(x, digits)
    return f"({text})" if text.startswith("-") else text


# ── Ledger ──────────────────────────────────────────────────────────────────────
def build(t: dict) -> dict:
    """Derive the R·A·E ledger from ``t["rounds"]``. Returns ``{"rows", "p0", "summary"}``."""
    rows: list[dict] = []
    p0: Optional[float] = None
    beta = A = R = 0.0
    prev: Optional[dict] = None
    for n, r in enumerate(t["rounds"], 1):
        action = str(r.get("action", ""))
        p_old, p_new = num(r.get("p_old")), num(r.get("p_new"))
        c0, c1 = max(0.0, num(r.get("c_before"))), max(0.0, num(r.get("c_after")))
        b0, b1 = num(r.get("b_before")), num(r.get("b_after"))
        legacy = action == LEGACY_ACTION
        act = action in ACT_ACTIONS and p_old > 0 and p_new > 0 and c0 > 0
        contiguous = prev is None or (_close(prev["p"], p_old, 6e-5) and _close(prev["c"], c0, 5.1e-3)
                                      and _close(prev["b"], b0, 5.1e-3))
        if act and p0 is None:
            p0 = p_old                                       # E starts from 0 at the first P_acted
        dR = c0 * math.log(p_new / p_old) if act else 0.0
        dA = c0 * (p_new / p_old - 1.0) if act else 0.0
        A_prev, R_prev, beta_before = A, R, beta
        A, R = A + dA, R + dR
        r_beta = beta + c0 * math.log(p_new / p0) if (act and p0) else None   # LEGO fixed-anchor form
        pivot = p_new if p_new > 0 else p_old
        if p0 and pivot > 0 and not _close(c1, c0, 5e-3):    # rollover when Fix_c changes
            beta += (c0 - c1) * math.log(pivot / p0)
        rows.append({
            "n": n, "round_id": int(num(r.get("round_id"), n)), "date": str(r.get("date", "")),
            "action": action, "kind": "act" if act else ("legacy" if legacy else "frozen"),
            "p_acted": p_old, "p_fill": p_new, "fix_c": c0, "fix_c_after": c1,
            "shannon": num(r.get("shannon_profit")), "hedge": num(r.get("hedge_cost")),
            "surplus": num(r.get("surplus")), "scale_up": num(r.get("scale_up")),
            "hedge_ratio": num(r.get("hedge_ratio")), "b_before": b0, "b_after": b1,
            "dR": dR, "dA": dA, "R": R, "A": A, "E": A - R,
            "A_prev": A_prev, "R_prev": R_prev, "beta": beta_before, "beta_after": beta,
            "R_beta": r_beta, "p0": p0, "contiguous": contiguous,
            "legacy": legacy, "known": action in KNOWN_ACTIONS,
        })
        prev = {"p": p_new, "c": c1, "b": b1}
    hedge_sum = sum(x["hedge"] for x in rows)
    last = rows[-1] if rows else None
    summary = {"n": len(rows), "acts": sum(x["kind"] == "act" for x in rows),
               "legacy": sum(x["legacy"] for x in rows), "p0": p0,
               "R": last["R"] if last else 0.0, "A": last["A"] if last else 0.0,
               "E": last["E"] if last else 0.0, "hedge": hedge_sum,
               "e_minus_hedge": (last["E"] if last else 0.0) - hedge_sum,
               "breaks": [x["n"] for x in rows if not x["contiguous"]]}
    return {"rows": rows, "p0": p0, "summary": summary}


def table(ledger: dict) -> list[dict]:
    """Flat rows in display order (full precision — the UI rounds)."""
    return [{C_N: x["round_id"], C_DATE: x["date"], C_ACTION: x["action"],
             C_PACTED: x["p_acted"], C_PRICE: x["p_fill"], C_FIXC: x["fix_c"],
             C_SHANNON: x["shannon"], C_HEDGE: x["hedge"], C_SURPLUS: x["surplus"],
             C_FIXC2: x["fix_c_after"], C_BASE: x["b_after"],
             C_R: x["R"], C_DA: x["dA"], C_A: x["A"], C_E: x["E"]} for x in ledger["rows"]]


def series(ledger: dict) -> dict[str, list[float]]:
    """R / A / E per round — for the chart."""
    rows = ledger["rows"]
    return {"Rₙ อ้างอิง": [x["R"] for x in rows], "Aₙ สะสม": [x["A"] for x in rows],
            "Eₙ ส่วนเกิน": [x["E"] for x in rows]}


# ── "ที่มาของตัวเลข" — tooltip text with the real numbers (same layout as streamlit_lego) ──
def _model_hedge(x: dict) -> tuple[float, float]:
    """(put premium per share, model hedge) for the row's recorded inputs."""
    if x["p_acted"] <= 0:
        return 0.0, 0.0
    put = eng.bs_put(x["p_acted"], x["p_acted"] * eng.HEDGE_STRIKE)
    return put, (x["fix_c"] / x["p_acted"]) * x["hedge_ratio"] * put if x["fix_c"] > 0 else 0.0


def _title(x: dict, sym: str, label: str) -> str:
    return f"{sym} · รอบ {x['round_id']} · {label}"


def _stored(x: dict, sym: str, name: str, value: float, why: str) -> str:
    return "\n".join([_title(x, sym, "ค่าที่บันทึกไว้"), f"{name} = {_f(value)}", why])


def _why_frozen(x: dict) -> str:
    if x["legacy"]:
        return ("Legacy Rollover = ประวัติเก่าที่นำเข้า (เส้นอ้างอิงล้วน) ไม่มีราคา fill ของแต่ละก้าว "
                "→ ไม่นับใน ΔA/ΔR (LEGO: ไม่เดาสูตรให้หลักฐานเก่า)")
    if x["action"] == "Extract Baseline":
        return "Extract Baseline ย้ายเงินจาก Baseline → Pool CF ไม่ใช่การ rebalance"
    if x["action"] == "Deploy to Baseline":
        return "Deploy ย้ายเงินจาก Pool CF → Baseline ไม่ใช่การ rebalance"
    if not x["known"]:
        return f"action {x['action']!r} ไม่รู้จัก — dashboard ไม่เดาสูตรให้"
    return "ราคาหรือ fix_c เป็น 0 — ไม่มีอะไรให้ rebalance"


def explain(ledger: dict, symbol: str) -> list[dict[str, str]]:
    """Per-row tooltip text for every column in ``TIP_COLS`` (first line = title, rest = formula)."""
    out = []
    for x in ledger["rows"]:
        act, sym = x["kind"] == "act", symbol
        c0, po, pf = x["fix_c"], x["p_acted"], x["p_fill"]
        tips: dict[str, str] = {}

        # ΔR — Shannon
        if act:
            calc = c0 * math.log(pf / po)
            lines = [_title(x, sym, "Shannon = ΔR (ส่วนเพิ่มของเส้นอ้างอิง)"),
                     "Shannon = Fix_c × ln(P_fill / P_acted)",
                     f"        = {_f(c0)} × ln({_f(pf)} / {_f(po)})",
                     f"        = {_f(c0)} × {_op(math.log(pf / po), 6)}",
                     f"        = {_f(calc)}",
                     "P_acted = ราคารอบก่อน (t) · P_fill = P New ของรอบนี้"]
            if not _close(x["shannon"], calc, CENT + c0 * 5e-5 * (1 / po + 1 / pf)):
                lines.append(f"⚠️ บันทึกไว้ {_f(x['shannon'], 2)} ต่างจากสูตร — แก้มือ/ข้อมูลเก่า")
            tips[C_SHANNON] = "\n".join(lines)
        else:
            tips[C_SHANNON] = _stored(x, sym, "Shannon", x["shannon"], _why_frozen(x))

        # Hedge
        if act:
            put, model = _model_hedge(x)
            lines = [_title(x, sym, "Hedge (เบี้ย Put → Ev burn)"),
                     "Hedge = (Fix_c / t) × Hedge× × Put(S=t, K=0.9t)",
                     f"      = ({_f(c0)} / {_f(po)}) × {_f(x['hedge_ratio'])} × {_f(put)}",
                     f"      = {_f(model)}",
                     "Put 1 ปี OTM 10% (σ 50%, r 4%) ต่อหุ้น"]
            if x["hedge_ratio"] <= 0:
                lines.append("Hedge× = 0 → ไม่ได้ซื้อประกัน")
            if not _close(x["hedge"], model, CENT + abs(model) * 1e-3):
                lines.append(f"⚠️ บันทึกไว้ {_f(x['hedge'], 2)} ต่างจากโมเดล — ใช้เบี้ยจริง/Override")
            tips[C_HEDGE] = "\n".join(lines)
        else:
            tips[C_HEDGE] = _stored(x, sym, "Hedge", x["hedge"], _why_frozen(x))

        # Surplus
        if act:
            lines = [_title(x, sym, "Surplus (กำไรหลังจ่ายประกัน)"), "Surplus = Shannon − Hedge",
                     f"        = {_f(x['shannon'])} − {_op(x['hedge'])}",
                     f"        = {_f(x['shannon'] - x['hedge'])}",
                     "> 0 → ไปขยาย fix_c (Scale Up) · ≤ 0 → fix_c ไม่ลด"]
            if not _close(x["surplus"], x["shannon"] - x["hedge"], CENT):
                lines.append(f"⚠️ บันทึกไว้ {_f(x['surplus'], 2)} ต่างจากสูตร")
            tips[C_SURPLUS] = "\n".join(lines)
        else:
            tips[C_SURPLUS] = _stored(x, sym, "Surplus", x["surplus"], _why_frozen(x))

        # fix_c′
        grow = max(0.0, x["surplus"])
        lines = [_title(x, sym, "fix_c′ (Scale Up)"), "fix_c′ = fix_c + max(0, Surplus)",
                 f"       = {_f(x['fix_c'], 2)} + {_f(grow, 2)}", f"       = {_f(x['fix_c'] + grow, 2)}",
                 f"บันทึกไว้ {_f(x['fix_c_after'], 2)} (scale_up {_op(x['scale_up'], 2)})"]
        if not _close(x["fix_c_after"], x["fix_c"] + grow, CENT):
            lines.append("⚠️ ต่างจากสูตร — Override / No Surplus / ปิด position / ข้อมูลเก่า")
        tips[C_FIXC2] = "\n".join(lines)

        # b′
        if x["action"] == "Chain Round":
            lines = [_title(x, sym, "Baseline b′ (rollover ต่อเนื่อง)"),
                     "b′ = b + Shannon − fix_c′ × ln(P / t′)      (t′ = P ⇒ พจน์หลัง = 0)",
                     f"   = {_op(x['b_before'], 2)} + {_op(x['shannon'], 2)}",
                     f"   = {_f(x['b_before'] + x['shannon'], 2)}", f"บันทึกไว้ {_f(x['b_after'], 2)}"]
            if not _close(x["b_after"], x["b_before"] + x["shannon"], CENT):
                lines.append("⚠️ ต่างจากสูตร — Override Baseline หรือข้อมูลเก่า")
        elif x["action"] in FLAT_ACTIONS:
            moved = x["b_after"] - x["b_before"]
            verb = "ดึงออก (→ Pool CF)" if x["action"] == "Extract Baseline" else "เติมเข้า (← Pool CF)"
            lines = [_title(x, sym, f"Baseline b′ — {verb}"), "b′ = b ± จำนวนเงินที่ย้าย",
                     f"   = {_op(x['b_before'], 2)} {'+' if moved >= 0 else '−'} {_f(abs(moved), 2)}",
                     f"   = {_f(x['b_after'], 2)}"]
        else:
            lines = [_title(x, sym, "Baseline b′"), f"b′ = {_f(x['b_after'], 2)}", "ค่าที่บันทึกไว้ — " + _why_frozen(x)]
        tips[C_BASE] = "\n".join(lines)

        # Rₙ
        if act:
            lines = [_title(x, sym, "Rₙ เส้นอ้างอิง (สะสมต่อรอบ)"), "Rₙ = Rₙ₋₁ + ΔR",
                     f"   = {_op(x['R_prev'])} + {_op(x['dR'])}", f"   = {_f(x['R'])}"]
            if x["R_beta"] is not None and x["contiguous"]:
                lines += ["รูปแบบ LEGO:  Rₙ = β + Fix_c × ln(Pₙ / P₀)",
                          f"   = {_op(x['beta'])} + {_f(c0)} × ln({_f(pf)} / {_f(x['p0'])})",
                          f"   = {_f(x['R_beta'])}",
                          f"P₀ = {_f(x['p0'])} = P_acted ของรอบแรก · β = {_f(x['beta'])} (สะสมจาก rollover ตอน Fix_c เปลี่ยน)"]
            else:
                lines.append("ลูกโซ่ขาดตอนก่อนรอบนี้ → ใช้ผลรวมรายรอบ (รูปแบบ P₀/β ไม่ตรง)")
            tips[C_R] = "\n".join(lines)
        else:
            tips[C_R] = "\n".join([_title(x, sym, "Rₙ ไม่ขยับ"), f"Rₙ = Rₙ₋₁ = {_f(x['R'])}", _why_frozen(x)])

        # ΔAₙ / Aₙ
        if act:
            tips[C_DA] = "\n".join([_title(x, sym, "act — rebalance จริง"), "ΔAₙ = Fix_c × (P_fill / P_acted − 1)",
                                    f"    = {_f(c0)} × ({_f(pf)} / {_f(po)} − 1)", f"    = {_f(x['dA'])}",
                                    "หลัง act นี้ P_acted เลื่อนเป็น P_fill",
                                    f"ΔE รอบนี้ = ΔA − ΔR = {_f(x['dA'])} − {_op(x['dR'])} = {_f(x['dA'] - x['dR'])}"])
            tips[C_A] = "\n".join([_title(x, sym, "สะสมต่อจากรอบก่อน"), "Aₙ = Aₙ₋₁ + ΔAₙ",
                                   f"   = {_op(x['A_prev'])} + {_op(x['dA'])}", f"   = {_f(x['A'])}"])
        else:
            tips[C_DA] = "\n".join([_title(x, sym, "แช่แข็ง — ไม่ rebalance"), "ΔAₙ = 0", _why_frozen(x)])
            tips[C_A] = "\n".join([_title(x, sym, "ค้างไว้"), "Aₙ = Aₙ₋₁ (ΔAₙ = 0)", f"   = {_f(x['A'])}"])

        # Eₙ
        tips[C_E] = "\n".join([_title(x, sym, "ส่วนเกินสะสม = กำไร harvest จริง"), "Eₙ = Aₙ − Rₙ",
                               f"   = {_f(x['A'])} − {_op(x['R'])}", f"   = {_f(x['E'])}",
                               "บวก = เงินสดจริง (Aₙ) เกินเส้นอ้างอิง (Rₙ)",
                               "Eₙ ≥ 0 เสมอ เพราะ x − 1 ≥ ln x · เริ่มนับจาก 0 ที่ P₀"])
        out.append(tips)
    return out


# ── Integrity — the Chain counterpart of LEGO's E1–E8 ─────────────────────────────
OK, WARN, FAIL, SKIP = "✅", "⚠️", "❌", "—"


def _shannon_tol(x: dict) -> float:
    """Tolerance for Shannon: cents + rounding of the prices recorded to 4 decimals."""
    return CENT + x["fix_c"] * 5e-5 * (1 / x["p_acted"] + 1 / x["p_fill"])


def _checked(legacy_n: int, checked: int) -> str:
    return f"ตรวจ {checked} รอบ Chain Round" + (f" · ข้าม {legacy_n} แถว Legacy Rollover (semantics เก่า)" if legacy_n else "")


def integrity(t: dict, ledger: Optional[dict] = None) -> tuple[list[dict], bool]:
    """Audit the round log against the equations. Returns ``(rows, ok)``; ``ok`` = no ❌.

    ❌ = an identity that Chain's own operations always satisfy is broken;
    ⚠️ = differs from the formula (manual Override, legacy import, or a gap in the log).
    """
    ledger = ledger or build(t)
    rows = ledger["rows"]
    chain = [x for x in rows if x["kind"] == "act"]
    legacy_n = sum(x["legacy"] for x in rows)
    unknown_n = sum(not x["known"] for x in rows)
    checks: list[dict] = []

    def add(cid: str, rule: str, residual: Optional[float], status: str, note: str = "") -> None:
        checks.append({"ข้อ": cid, "สมการ/กฎ": rule, "residual สูงสุด": residual, "ผ่าน": status, "หมายเหตุ": note})

    if not rows:
        add("C0", "มีรอบอย่างน้อย 1 รอบ", None, SKIP, "ยังไม่มี round — ไม่มีอะไรให้ตรวจ")
        return checks, True

    # C1 continuity
    breaks = ledger["summary"]["breaks"]
    prev = None
    gap = 0.0
    for x in rows:
        if prev is not None:
            gap = max(gap, abs(prev["p_fill"] - x["p_acted"]), abs(prev["fix_c_after"] - x["fix_c"]),
                      abs(prev["b_after"] - x["b_before"]))
        prev = x
    add("C1", "ต่อเนื่อง: p_oldₙ₊₁ = p_newₙ · c_beforeₙ₊₁ = c_afterₙ · b_beforeₙ₊₁ = b_afterₙ", gap,
        OK if not breaks else WARN,
        "" if not breaks else f"ขาดตอน {len(breaks)} จุด (รอบ {', '.join(str(rows[i - 1]['round_id']) for i in breaks[:6])}"
                              f"{'…' if len(breaks) > 6 else ''}) — ข้อมูล legacy / แก้มือ")

    # C2 Shannon, C3 Surplus (the v1 page let users hand-edit both → a mismatch is ⚠️, not ❌)
    d2 = [(abs(x["shannon"] - x["dR"]), _shannon_tol(x)) for x in chain]
    bad2 = sum(d > tol for d, tol in d2)
    add("C2", "Shannon = Fix_c × ln(P_fill / P_acted)", max((d for d, _ in d2), default=None),
        SKIP if not d2 else (OK if not bad2 else WARN),
        (f"{bad2} รอบต่างจากสูตร (แก้มือ/ข้อมูลเก่า) · " if bad2 else "") + _checked(legacy_n, len(chain)))
    d3 = [abs(x["surplus"] - (x["shannon"] - x["hedge"])) for x in chain]
    bad3 = sum(d > CENT for d in d3)
    add("C3", "Surplus = Shannon − Hedge", max(d3, default=None),
        SKIP if not d3 else (OK if not bad3 else WARN),
        (f"{bad3} รอบต่างจากสูตร (แก้มือ/ข้อมูลเก่า) · " if bad3 else "") + _checked(legacy_n, len(chain)))

    # C4 fix_c
    r4 = max((abs(x["fix_c_after"] - (x["fix_c"] + x["scale_up"])) for x in rows if x["known"]), default=None)
    manual = sum(not _close(x["scale_up"], max(0.0, x["surplus"]), CENT) for x in chain)
    add("C4", "fix_c′ = fix_c + Scale Up  (Scale Up = max(0, Surplus))", r4,
        SKIP if r4 is None else (OK if r4 <= CENT else FAIL),
        f"{manual} รอบ Scale Up ≠ max(0, Surplus) (Override / No Surplus / ปิด position)" if manual else "")

    # C5 baseline
    r5c = [abs(x["b_after"] - (x["b_before"] + x["shannon"])) for x in chain]
    flat = [x for x in rows if x["action"] in FLAT_ACTIONS]
    flat_bad = sum(not (_close(x["fix_c_after"], x["fix_c"], CENT) and _close(x["p_fill"], x["p_acted"], 6e-5)
                        and ((x["b_after"] <= x["b_before"] + CENT) if x["action"] == "Extract Baseline"
                             else (x["b_after"] >= x["b_before"] - CENT))) for x in flat)
    r5 = max(r5c, default=None)
    status5 = FAIL if flat_bad else (SKIP if r5 is None and not flat else (OK if (r5 or 0) <= CENT else WARN))
    note5 = f"Extract/Deploy ผิดกติกา {flat_bad} แถว · " if flat_bad else ""
    if r5 is not None and r5 > CENT:
        note5 += f"{sum(v > CENT for v in r5c)} รอบ b′ ≠ b + Shannon (Override Baseline / ข้อมูลเก่า)"
    add("C5", "b′ = b + Shannon (Chain Round) · Extract: b′ ≤ b · Deploy: b′ ≥ b (fix_c, P คงที่)", r5, status5, note5.strip(" ·"))

    # C6 hedge vs model
    h = [abs(x["hedge"] - _model_hedge(x)[1]) for x in chain if x["hedge_ratio"] > 0]
    bad_h = sum(v > CENT + 1e-3 * max(_model_hedge(x)[1], 0) for v, x in zip(h, [y for y in chain if y["hedge_ratio"] > 0]))
    add("C6", "Hedge = (Fix_c / t) × Hedge× × Put(K = 0.9t)", max(h, default=None),
        SKIP if not h else (OK if not bad_h else WARN),
        f"{bad_h} รอบต่างจากโมเดล (ใช้เบี้ยจริง/Override)" if bad_h else "")

    # C7 R·A·E
    e_min = min((x["E"] for x in rows), default=0.0)
    beta_gap = max((abs(x["R_beta"] - x["R"]) for x in rows if x["R_beta"] is not None and x["contiguous"]),
                   default=None)
    status7 = FAIL if e_min < -1e-6 else (WARN if (beta_gap or 0) > 1e-6 * max(1.0, abs(ledger["summary"]["R"])) + 1e-6 else OK)
    note7 = "Eₙ < 0 — เป็นไปไม่ได้ทางคณิตศาสตร์ (x − 1 ≥ ln x)" if e_min < -1e-6 else ""
    if status7 == WARN:
        note7 = "รูปแบบ P₀/β ต่างจากผลรวมรายรอบ — ลูกโซ่ไม่ต่อเนื่อง (ดู C1)"
    add("C7", "Aₙ = Aₙ₋₁ + ΔAₙ · Eₙ = Aₙ − Rₙ ≥ 0 · Rₙ = β + Fix_c × ln(Pₙ/P₀)", beta_gap, status7, note7)

    if unknown_n:
        add("C8", "action ต้องเป็น Chain Round / Legacy Rollover / Extract / Deploy", None, FAIL,
            f"{unknown_n} แถว action ไม่รู้จัก — fail closed (ไม่นับเป็น act)")
    return checks, all(c["ผ่าน"] != FAIL for c in checks)


# ── Break-even — when does a move pay for the hedge? ─────────────────────────────
def hedge_rate() -> float:
    """Model put premium as a fraction of the hedged value (≈ 12.4%)."""
    return eng.bs_put(1.0, eng.HEDGE_STRIKE)


def break_even(hedge_x: float) -> dict[str, Optional[float]]:
    """Price move m (= P/t − 1) at which each profit basis equals one round of hedge.

    R (ln):      ln(1 + m)            = rate·h   → m = e^(rate·h) − 1
    A (linear):  m                    = rate·h
    E (harvest): m − ln(1 + m)        = rate·h   → solved for m > 0 and m < 0
    """
    k = hedge_rate() * max(0.0, hedge_x)
    out: dict[str, Optional[float]] = {"R": math.expm1(k), "A": k, "E_up": None, "E_down": None}
    if k <= 0:
        return {"R": 0.0, "A": 0.0, "E_up": 0.0, "E_down": 0.0}
    f = lambda m: m - math.log1p(m)                                          # noqa: E731
    lo, hi = 0.0, 1.0
    while f(hi) < k and hi < 1e9:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < k else (lo, mid)
    out["E_up"] = (lo + hi) / 2
    lo, hi = -1.0 + 1e-12, 0.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) < k else (mid, hi)
    out["E_down"] = (lo + hi) / 2
    return out


# ── Testing Lab — random price path → Chain Rounds → the same ledger ────────────────
def simulate_prices(p0: float, vol: float, drift: float, steps: int, seed: int) -> list[float]:
    """Geometric random walk per step (deterministic): Pᵢ = Pᵢ₋₁ · exp((drift − vol²/2) + vol · Z)."""
    if not all(math.isfinite(float(x)) for x in (p0, vol, drift)):
        raise eng.ChainError("p0, vol, drift ต้องเป็นตัวเลข")
    if p0 <= 0:
        raise eng.ChainError("P₀ ต้องมากกว่า 0")
    if vol < 0:
        raise eng.ChainError("ความผันผวนต้อง ≥ 0")
    if not 2 <= steps <= MAX_SIM_STEPS:
        raise eng.ChainError(f"จำนวนก้าวต้องอยู่ระหว่าง 2 ถึง {MAX_SIM_STEPS}")
    rng, price, path = random.Random(int(seed)), float(p0), []
    for _ in range(int(steps)):
        price = max(price * math.exp((drift - 0.5 * vol * vol) + vol * rng.gauss(0.0, 1.0)), p0 * 1e-8)
        path.append(price)
    return path


def simulate_chain(p0: float, prices: list[float], fix_c: float, hedge_x: float, every: int,
                   no_hedge: bool = False, no_surplus: bool = False) -> dict:
    """Run a Chain Round every ``every`` steps (and at the last step) with the app's own ``plan_round``."""
    if fix_c <= 0:
        raise eng.ChainError("Fix_c ต้องมากกว่า 0")
    if not 1 <= every <= max(1, len(prices)):
        raise eng.ChainError("ทำ Chain Round ทุกกี่ก้าว ต้องอยู่ระหว่าง 1 ถึงจำนวนก้าว")
    t: dict[str, Any] = {"ticker": "LAB", "rounds": [],
                         "current_state": {"price": float(p0), "fix_c": float(fix_c), "baseline": 0.0,
                                           "cumulative_ev": 0.0, "net_pnl": 0.0}}
    steps_at = [i for i in range(every, len(prices) + 1, every)]
    if steps_at[-1:] != [len(prices)]:
        steps_at.append(len(prices))
    for i in steps_at:
        plan = eng.plan_round(t["current_state"], prices[i - 1], hedge_x, no_hedge=no_hedge, no_surplus=no_surplus)
        eng._append_round(t, plan)
    ledger = build(t)
    return {"ticker": t, "ledger": ledger, "steps": steps_at, "final_fix_c": t["current_state"]["fix_c"],
            "ev_burn": t["current_state"]["cumulative_ev"],
            "growth_rounds": sum(r["scale_up"] > 0 for r in t["rounds"])}
