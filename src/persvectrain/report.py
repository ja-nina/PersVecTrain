"""`report`: one pass/fail table over several runs' eval.json and the pick of the best run."""
from __future__ import annotations

_MARGIN_CS = ("c=+1", "c=-1")


def is_pending(ev: dict) -> bool:
    """Free generation or the persona judge not run yet (`freegen` / `persona` still to do): the run's pass flags
    are not final. An eval.json without a persona_judge key (an old run) is not pending for it."""
    return (ev.get("free_gen") or {}).get("status") == "pending" or \
        (ev.get("persona_judge") or {}).get("status") == "pending"


def n_passed(ev: dict) -> int:
    return sum(1 for t in ev["traits"].values() if t["pass"]["passed"])


def mean_trained_margin(ev: dict) -> float:
    """Mean over traits of the mean of the trained persona margin at c=+1 and c=-1."""
    per_trait = []
    for t in ev["traits"].values():
        ms = [t["kinds"]["trained"][c]["margin_mean"] for c in _MARGIN_CS]
        ms = [m for m in ms if m is not None]
        if ms:
            per_trait.append(sum(ms) / len(ms))
    return sum(per_trait) / len(per_trait) if per_trait else float("-inf")


def trait_margin(trait_eval: dict) -> float:
    """Mean of one trait's trained persona margin at c=+1 and c=-1 (-inf without any)."""
    ms = [trait_eval["kinds"]["trained"][c]["margin_mean"] for c in _MARGIN_CS]
    ms = [m for m in ms if m is not None]
    return sum(ms) / len(ms) if ms else float("-inf")


def pick_per_trait(evals: dict[str, dict]) -> dict[str, str | None]:
    """Per trait (union over runs, first-seen order): the non-pending run where the trait passes with the largest
    mean trained margin at c=+-1 (ties: first listed); None where it passes in no run."""
    traits: list[str] = []
    for ev in evals.values():
        traits += [t for t in ev["traits"] if t not in traits]
    out: dict[str, str | None] = {}
    for t in traits:
        best, best_m = None, None
        for name, ev in evals.items():
            if is_pending(ev) or t not in ev["traits"] or not ev["traits"][t]["pass"]["passed"]:
                continue
            m = trait_margin(ev["traits"][t])
            if best_m is None or m > best_m:
                best, best_m = name, m
        out[t] = best
    return out


def pick_run(evals: dict[str, dict]) -> str | None:
    """The run passing the most traits (ties: larger mean trained margin, then first listed); None if none pass.
    Runs with pending free generation are never picked."""
    best, best_key = None, None
    for name, ev in evals.items():
        if is_pending(ev):
            continue
        n = n_passed(ev)
        if n == 0:
            continue
        key = (n, mean_trained_margin(ev))
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


def _cell(trait_eval: dict, pending: bool = False) -> str:
    p = trait_eval["pass"]
    if pending:  # a teacher-forced check failing already is final: free generation cannot make it pass
        return "PENDING (FAIL: " + ", ".join(p["failed"]) + ")" if p.get("failed") else "PENDING"
    return "PASS" if p["passed"] else "FAIL (" + ", ".join(p["failed"]) + ")"


def format_table(evals: dict[str, dict]) -> str:
    """Rows: runs; columns: traits (union, in first-seen order); cells PASS or FAIL (failed checks), PENDING
    while free generation is pending (PENDING (FAIL: ...) when a teacher-forced check already fails)."""
    traits: list[str] = []
    for ev in evals.values():
        traits += [t for t in ev["traits"] if t not in traits]
    cells = [[name] + [_cell(ev["traits"][t], is_pending(ev)) if t in ev["traits"] else "-" for t in traits]
             for name, ev in evals.items()]
    rows = [["run"] + traits] + cells
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows)
