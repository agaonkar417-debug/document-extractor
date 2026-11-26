"""Field-level accuracy scoring: compare an extraction against ground truth.

Every leaf in the expected JSON is one point. Lists of objects are matched
greedily to the best predicted item; extra predicted items cost points, so a
model can't win by over-extracting.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def _empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = re.sub(r"[,$€£\s]", "", v)
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def scalar_eq(exp: Any, got: Any) -> bool:
    if _empty(exp) or _empty(got):
        return _empty(exp) and _empty(got)
    en, gn = _num(exp), _num(got)
    if en is not None and gn is not None:
        return abs(en - gn) <= max(0.01, abs(en) * 0.001)
    e, g = _norm(str(exp)), _norm(str(got))
    if e == g:
        return True
    # Phone numbers: compare the last 10 digits.
    ed, gd = re.sub(r"\D", "", str(exp)), re.sub(r"\D", "", str(got))
    if len(ed) >= 7 and len(ed) >= len(e.replace(" ", "")) - 2:
        return ed[-10:] == gd[-10:]
    # Long free text (addresses, descriptions): allow small wording differences.
    if len(e.split()) >= 4:
        return SequenceMatcher(None, e, g).ratio() >= 0.9
    return False


def leaves(v: Any) -> int:
    if isinstance(v, dict):
        return sum(leaves(x) for x in v.values()) or 1
    if isinstance(v, list):
        return sum(leaves(x) for x in v) or 1
    return 1


def score(exp: Any, got: Any) -> tuple[float, int]:
    """Return (points earned, points possible)."""
    if isinstance(exp, dict):
        got = got if isinstance(got, dict) else {}
        earned = possible = 0
        for k, v in exp.items():
            e, p = score(v, got.get(k))
            earned += e
            possible += p
        return earned, possible

    if isinstance(exp, list):
        got = got if isinstance(got, list) else []
        if not exp:
            return (1, 1) if not got else (0, 1)

        if not any(isinstance(x, (dict, list)) for x in exp):
            # List of scalars (e.g. skills): matched items, extras penalised.
            pool = list(got)
            hits = 0
            for e in exp:
                for i, g in enumerate(pool):
                    if scalar_eq(e, g):
                        hits += 1
                        pool.pop(i)
                        break
            return hits, len(exp) + len(pool)

        # List of objects (line items, jobs): greedy best-pair matching.
        pairs = sorted(
            ((score(e, g)[0], i, j) for i, e in enumerate(exp) for j, g in enumerate(got)),
            reverse=True,
        )
        used_e, used_g, earned = set(), set(), 0.0
        for s, i, j in pairs:
            if i in used_e or j in used_g or s == 0:
                continue
            used_e.add(i)
            used_g.add(j)
            earned += s
        possible = sum(leaves(e) for e in exp)
        # Each extra (hallucinated) item costs as much as a real one is worth.
        possible +=max(0, len(got) - len(exp)) * leaves(exp[0])
        return earned, possible

    return (1 if scalar_eq(exp, got) else 0), 1


def field_errors(exp: Any, got: Any, path: str = "") -> list[str]:
    """Human-readable list of scalar mismatches, for the per-document report."""
    out: list[str] = []
    if isinstance(exp, dict):
        got = got if isinstance(got, dict) else {}
        for k, v in exp.items():
            out += field_errors(v, got.get(k), f"{path}.{k}" if path else k)
    elif isinstance(exp, list):
        e_s, e_p = score(exp, got)
        if e_s < e_p:
            n_got = len(got) if isinstance(got, list) else 0
            out.append(f"{path}: {e_s:g}/{e_p} points ({len(exp)} expected, {n_got} extracted)")
    elif not scalar_eq(exp, got):
        out.append(f"{path}: expected {exp!r}, got {got!r}")
    return out
