"""`analysis`: every paired comparison of the trained vector with its controls, in one long table, plus the rotation.

Reads only (train.json, eval.json, bench.json and the vectors); the notebook notebooks/analysis.ipynb draws on it.

One row per (metric, run, trait, c, against): diff = paired mean over items, oriented so that diff > 0 always means
"trained is better than `against`", with its SE and z = diff / SE. Metrics:
  bench_acc        500 bench questions, correct_trained - correct_X
  bench_len        bench, length shift |tok_X - tok_unsteered| - |tok_trained - tok_unsteered| (smaller shift = better)
  bench_cut        bench, share of answers cut at max_tokens (no answer = wrong): cut_X - cut_trained
  bench_acc_uncut  bench accuracy over the questions where neither answer was cut (is a loss reasoning or length?)
  fg_acc, fg_cut, fg_acc_uncut   the same on the 100 held SuperGPQA questions (eval.json free_gen)
  judge_trait      persona judge 0-100: c=+1 score_trained - score_X (more trait), c=-1 score_X - score_trained
                   (suppresses more)
  judge_coherence  persona judge 0-100, coherence_trained - coherence_X
  tf_margin        teacher-forced held persona margin (eval.json, every c), margin_trained - margin_X
  tf_kl            teacher-forced KL on held reference answers, kl_X - kl_trained (smaller KL = better)
against: source_full ("source"), source_at_trained_norm ("srcN", the source direction at the trained length),
unsteered, trained_random (where the metric has them).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from persvecfast.util import read_json

from persvectrain.evaluate import mean_se
from persvectrain.output import run_root

SOURCE, SRCN, UNSTEERED, RANDOM = "source_full", "source_at_trained_norm", "unsteered", "trained_random"
SHORT = {SOURCE: "source", SRCN: "srcN", UNSTEERED: "unsteered", RANDOM: "random"}
METRICS = ("bench_acc", "bench_acc_uncut", "bench_cut", "bench_len", "fg_acc", "fg_acc_uncut", "fg_cut",
           "judge_trait", "judge_coherence", "tf_margin", "tf_kl")


# ---------------------------------------------------------------- loading


def discover_runs(out: Path) -> list[str]:
    """Run names under <out>/trained with a finished train.json and an eval.json, sorted."""
    base = Path(out) / "trained"
    if not base.is_dir():
        return []
    runs = []
    for d in sorted(base.iterdir()):
        if (d / "train.json").is_file() and (d / "eval.json").is_file():
            if read_json(d / "train.json").get("finished"):
                runs.append(d.name)
    return runs


def load_run(out: Path, run: str) -> dict:
    """{"name", "config", "train", "eval", "bench" (None if not benched)} of one run."""
    root = run_root(Path(out), run)
    meta = read_json(root / "train.json")
    bench = root / "bench.json"
    return {"name": run, "config": meta["config"], "train": meta, "eval": read_json(root / "eval.json"),
            "bench": read_json(bench) if bench.is_file() else None}


def label(run: dict) -> dict:
    """The settings that tell runs apart (for tables and plots)."""
    c = run["config"]
    return {"run": run["name"], "lambda_kl": c.get("lambda_kl"), "fix_norm": c.get("fix_norm", 0),
            "seed": c.get("seed"), "mode": c.get("mode")}


# ---------------------------------------------------------------- paired differences


def paired(xa: list, xb: list) -> tuple[float | None, float | None, int]:
    """Mean, SE and n of xa - xb over items where both are not None."""
    if len(xa) != len(xb):
        raise ValueError(f"not paired: {len(xa)} vs {len(xb)} items")
    d = [float(a) - float(b) for a, b in zip(xa, xb) if a is not None and b is not None]
    m, se = mean_se(d)
    return m, se, len(d)


def _row(metric, run, trait, c, against, m, se, n, **extra) -> dict:
    z = None if m is None or not se else m / se
    return {"metric": metric, **label(run), "trait": trait, "c": c, "against": SHORT.get(against, against),
            "diff": m, "se": se, "z": z, "n": n, **extra}


def _blk(trait_part: dict, kind: str, c: str) -> dict | None:
    """A c-level block of `kind`, or the unsteered block (no c level); None if absent."""
    if kind == UNSTEERED:
        return trait_part.get(UNSTEERED)
    return (trait_part.get(kind) or {}).get(c)


def _shift(tok: list, unst: list) -> list:
    return [abs(float(t) - float(u)) for t, u in zip(tok, unst)]


def _uncut(a: dict, b: dict) -> tuple[list, list]:
    """correct lists of a and b over the questions where neither answer was cut."""
    keep = [i for i in range(len(a["correct"])) if not a["cut"][i] and not b["cut"][i]]
    return [a["correct"][i] for i in keep], [b["correct"][i] for i in keep]


def _skip(msg: str) -> None:
    print(f"analysis: skipped {msg}", file=sys.stderr)


def answer_rows(run: dict, part: str) -> list[dict]:
    """bench_* (part "bench") or fg_* (part "free_gen") rows of one run."""
    if part == "bench":
        b = run["bench"]
        if not b:
            return []
        traits = {s: {**t, UNSTEERED: b["unsteered"]} for s, t in b["traits"].items()}
    else:
        traits = {s: t["free_gen"] for s, t in run["eval"]["traits"].items()
                  if isinstance(t.get("free_gen"), dict) and t["free_gen"].get(UNSTEERED)}
    pre = "bench" if part == "bench" else "fg"
    acc = f"{pre}_acc"
    rows = []
    for s, t in traits.items():
        for c in ("c=+1", "c=-1"):
            tr = _blk(t, "trained", c)
            if tr is None:
                continue
            for against in (SOURCE, SRCN, UNSTEERED, RANDOM):
                x = _blk(t, against, c)
                if x is None:
                    continue
                if "qid" not in x or "qid" not in tr:  # an old-format block (before per-question lists)
                    _skip(f"{run['name']} {part} {s} {c} {SHORT[against]}: no per-question lists (old format)")
                    continue
                if x["qid"] != tr["qid"]:
                    raise ValueError(f"{run['name']} {part} {s} {c}: {against} not paired with trained")
                m, se, n = paired(tr["correct"], x["correct"])
                rows.append(_row(acc, run, s, c, against, m, se, n,
                                 trained_value=tr["accuracy"], against_value=x["accuracy"]))
                if "cut" in tr and "cut" in x:
                    m, se, n = paired([float(v) for v in x["cut"]], [float(v) for v in tr["cut"]])
                    rows.append(_row(f"{pre}_cut", run, s, c, against, m, se, n,
                                     trained_value=sum(map(float, tr["cut"])) / len(tr["cut"]),
                                     against_value=sum(map(float, x["cut"])) / len(x["cut"])))
                    a_, b_ = _uncut(tr, x)
                    m, se, n = paired(a_, b_)
                    rows.append(_row(f"{pre}_acc_uncut", run, s, c, against, m, se, n,
                                     trained_value=sum(map(float, a_)) / len(a_) if a_ else None,
                                     against_value=sum(map(float, b_)) / len(b_) if b_ else None))
                if part == "bench" and against != UNSTEERED:
                    u = t[UNSTEERED]["tokens"]
                    m, se, n = paired(_shift(x["tokens"], u), _shift(tr["tokens"], u))
                    rows.append(_row("bench_len", run, s, c, against, m, se, n,
                                     trained_value=tr["mean_tokens"], against_value=x["mean_tokens"]))
    return rows


def judge_rows(run: dict) -> list[dict]:
    """judge_trait / judge_coherence rows of one run (none when the persona judge has not run)."""
    rows = []
    for s, t in run["eval"]["traits"].items():
        pj = t.get("persona_judge")
        if not isinstance(pj, dict) or "trained" not in pj:
            continue
        for c in ("c=+1", "c=-1"):
            tr = pj["trained"][c]
            for against in (SOURCE, SRCN, UNSTEERED, RANDOM):
                x = _blk(pj, against, c)
                if x is None:
                    continue
                if x["qid"] != tr["qid"]:
                    raise ValueError(f"{run['name']} judge {s} {c}: {against} not paired with trained")
                if against != UNSTEERED:  # trait expression against the unsteered model is not "better/worse"
                    a, b = (tr, x) if c == "c=+1" else (x, tr)
                    m, se, n = paired(a["score"], b["score"])
                    rows.append(_row("judge_trait", run, s, c, against, m, se, n,
                                     trained_value=tr["score_mean"], against_value=x["score_mean"]))
                m, se, n = paired(tr["coherence"], x["coherence"])
                rows.append(_row("judge_coherence", run, s, c, against, m, se, n,
                                 trained_value=tr["coherence_mean"], against_value=x["coherence_mean"]))
    return rows


def teacher_forced_rows(run: dict) -> list[dict]:
    """tf_margin / tf_kl rows of one run at every evaluated c."""
    rows = []
    for s, t in run["eval"]["traits"].items():
        kinds = t["kinds"]
        for c, tr in kinds["trained"].items():
            for against in (SOURCE, SRCN, RANDOM):
                x = (kinds.get(against) or {}).get(c)
                if x is None:
                    continue
                m, se, n = paired(tr["margins"], x["margins"])
                rows.append(_row("tf_margin", run, s, c, against, m, se, n,
                                 trained_value=tr["margin_mean"], against_value=x["margin_mean"]))
                m, se, n = paired(x["kls"], tr["kls"])
                rows.append(_row("tf_kl", run, s, c, against, m, se, n,
                                 trained_value=tr["kl_mean"], against_value=x["kl_mean"]))
    return rows


def random_direction_rows(run: dict) -> list[dict]:
    """Bench: each random direction k (question i uses k = i % 5; the same 5 unit directions for every trait and
    run, norm-matched) against the unsteered answers to the same questions, at c = +1 and -1:
    diff = correct_random - correct_unsteered (> 0: the random direction helps), and its cut share."""
    b = run["bench"]
    if not b:
        return []
    u = b["unsteered"]
    rows = []
    for s, t in b["traits"].items():
        for c in ("c=+1", "c=-1"):
            r = (t.get(RANDOM) or {}).get(c)
            if r is None or "direction" not in r:
                continue
            for k in sorted(set(r["direction"])):
                idx = [i for i, d in enumerate(r["direction"]) if d == k]
                m, se, n = paired([r["correct"][i] for i in idx], [u["correct"][i] for i in idx])
                rows.append({"run": run["name"], "lambda_kl": run["config"].get("lambda_kl"), "trait": s, "c": c,
                             "direction": k, "diff": m, "se": se, "n": n,
                             "cut": sum(float(r["cut"][i]) for i in idx) / len(idx)})
    return rows


def random_direction_summary(rows: list[dict]) -> list[dict]:
    """Per (direction, c): mean over all run x trait cells of the random-minus-unsteered accuracy and cut share."""
    g: dict[tuple, list[dict]] = {}
    for r in rows:
        g.setdefault((r["direction"], r["c"]), []).append(r)
    return [{"direction": k, "c": c, "mean_diff": sum(r["diff"] for r in rs) / len(rs),
             "cut": sum(r["cut"] for r in rs) / len(rs), "cells": len(rs)} for (k, c), rs in sorted(g.items())]


def comparisons(runs: list[dict]) -> list[dict]:
    """All rows of all runs (see the module docstring)."""
    rows = []
    for r in runs:
        rows += answer_rows(r, "bench") + answer_rows(r, "free_gen") + judge_rows(r) + teacher_forced_rows(r)
    return rows


def worse(rows: list[dict], z: float = 2.0, against: str | None = "source") -> list[dict]:
    """Rows where trained is measurably worse (z <= -z), most negative first; against None = every control."""
    out = [r for r in rows if r["z"] is not None and r["z"] <= -z and (against is None or r["against"] == against)]
    return sorted(out, key=lambda r: r["z"])


def summary(rows: list[dict], z: float = 2.0) -> list[dict]:
    """Per (metric, run, against, c sign): mean diff over traits (equal weight), its SE over traits, and how many
    trait cells are measurably better / worse (|z| >= z)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["diff"] is None:
            continue
        sign = "+" if r["c"].startswith("c=+") else "-"
        groups.setdefault((r["metric"], r["run"], r["against"], sign), []).append(r)
    out = []
    for (metric, run, against, sign), rs in groups.items():
        m, se = mean_se([r["diff"] for r in rs])
        out.append({"metric": metric, "run": run, "lambda_kl": rs[0]["lambda_kl"], "against": against,
                    "c_sign": sign, "mean_diff": m, "se_over_traits": se, "cells": len(rs),
                    "better": sum(r["z"] is not None and r["z"] >= z for r in rs),
                    "worse": sum(r["z"] is not None and r["z"] <= -z for r in rs)})
    return out


# ---------------------------------------------------------------- rotation


def rotation(out: Path, runs: list[str]) -> dict:
    """vecs.analyse over the runs (model-free; needs the source *_raw.pt files) plus, per run and trait, the angles
    (degrees) trained-vs-start and trained-vs-source and, per trait, the cosine between the trained vectors of every
    two runs. Returns {"analyse": ..., "per_trait": rows, "trained_cos": {trait: {(run_a, run_b): cos}}}."""
    from persvectrain import vecs as vm

    loaded = {r: vm.load_run_vectors(Path(out), r)[1] for r in runs}
    res = vm.analyse(loaded)
    rows = []
    for run, r in res["runs"].items():
        for s, p in r["per_trait"].items():
            rows.append({"run": run, "trait": s, **p,
                         "angle_start_deg": math.degrees(math.acos(max(-1.0, min(1.0, p["cos_trained_start"])))),
                         "angle_source_deg": math.degrees(math.acos(max(-1.0, min(1.0, p["cos_trained_source"])))),
                         "angle_split_noise_deg": math.degrees(math.acos(max(-1.0, min(1.0, p["cos_source_start"])))),
                         "delta_u1_cos": r["delta_perp"]["u1_cos"].get(s)})
    trained_cos: dict = {}
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            for s in loaded[a]:
                if s in loaded[b]:
                    trained_cos.setdefault(s, {})[(a, b)] = vm.cos(loaded[a][s][1], loaded[b][s][1])
    for r in res["runs"].values():
        r.pop("_u1", None)
        r.pop("_deltas", None)
    for x in res["across_runs"].values():
        x.pop("_u1", None)
        x.pop("_mean", None)
    return {"analyse": res, "per_trait": rows, "trained_cos": trained_cos}


# ---------------------------------------------------------------- text output (the `analysis` command)


def _f(x, fmt="+.3f") -> str:
    return "-" if x is None else format(x, fmt)


def format_report(rows: list[dict], rot: dict | None, z: float = 2.0, rand: list[dict] | None = None) -> str:
    L = ["== mean over traits, diff > 0 = trained better (better/worse = trait cells with |z| >= "
         f"{z:g}) =="]
    L.append(f"{'metric':<16} {'run':<22} {'against':<10} {'c':>2} {'mean':>8} {'+-se':>7} {'cells':>5} "
             f"{'better':>6} {'worse':>5}")
    order = {m: i for i, m in enumerate(METRICS)}
    for s in sorted(summary(rows, z), key=lambda s: (order.get(s["metric"], 99), s["run"], s["against"],
                                                     s["c_sign"])):
        L.append(f"{s['metric']:<16} {s['run']:<22} {s['against']:<10} {s['c_sign']:>2} {_f(s['mean_diff']):>8} "
                 f"{_f(s['se_over_traits'], '.3f'):>7} {s['cells']:>5} {s['better']:>6} {s['worse']:>5}")
    w = worse(rows, z, "source")
    L.append("")
    L.append(f"== cells where trained is worse than source (z <= -{z:g}): {len(w)} ==")
    for r in w:
        L.append(f"{r['metric']:<16} {r['run']:<22} {r['trait']:<15} {r['c']:>7} diff {_f(r['diff'])} "
                 f"+-{_f(r['se'], '.3f')} (z {r['z']:+.1f}; trained {_f(r['trained_value'], '.3f')}, "
                 f"source {_f(r['against_value'], '.3f')})")
    if rand:
        L.append("")
        L.append("== bench: each random direction vs unsteered (mean over runs x traits; the same 5 directions "
                 "everywhere) ==")
        L.append(f"{'dir':>3} {'c':>5} {'random-unst':>11} {'cut share':>9} {'cells':>5}")
        for r in random_direction_summary(rand):
            L.append(f"{r['direction']:>3} {r['c'][2:]:>5} {r['mean_diff']:>+11.3f} {r['cut']:>9.3f} {r['cells']:>5}")
    if rot:
        L.append("")
        L.append("== rotation per run (mean over traits; degrees) ==")
        L.append(f"{'run':<22} {'|t|/|s0|':>8} {'angle t-s0':>10} {'angle t-src':>11} {'split noise':>11} "
                 f"{'|delta|/|s0|':>12} {'along':>6} {'perp':>6}")
        by: dict[str, list[dict]] = {}
        for r in rot["per_trait"]:
            by.setdefault(r["run"], []).append(r)
        for run, rs in by.items():
            def mean(k):
                return sum(r[k] for r in rs) / len(rs)
            L.append(f"{run:<22} {mean('norm_ratio'):>8.2f} {mean('angle_start_deg'):>10.1f} "
                     f"{mean('angle_source_deg'):>11.1f} {mean('angle_split_noise_deg'):>11.1f} "
                     f"{mean('delta_rel'):>12.2f} {mean('delta_along_start_rel'):>+6.2f} {mean('delta_perp_rel'):>6.2f}")
        for run, r in rot["analyse"]["runs"].items():
            dp, nl = r["delta_perp"], r["split_noise_perp"]
            L.append(f"{run}: shared change across traits, first-component share {dp['sv_share'][0]:.2f} "
                     f"(split noise {nl['sv_share'][0]:.2f}, unrelated {r['equal_share']:.2f})")
    return "\n".join(L)
