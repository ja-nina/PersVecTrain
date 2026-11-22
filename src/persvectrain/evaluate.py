"""End-of-run evaluation on held-out data and the pass rule (spec, "Evaluation").

Per trait, vector kind and c: held-out persona margin and KL on held-out reference answers (teacher-forced),
vector norms and cosine with the source vector; in joint runs the cosine matrix and 2-trait mix margins. Writes
`<run root>/eval.json` and prints one table. The free-generation check (length / cut share / accuracy at c = +-1)
runs on vLLM in a separate step (`freegen`, freegen.py); until then eval.json's free_gen status is "pending".
"""
from __future__ import annotations

import itertools
import math
import time

import torch
import torch.nn.functional as F
from persvecfast.extract import RANDOM_CONTROL_SEED, random_control
from persvecfast.util import log, write_json
from torch import Tensor

from persvectrain.losses import kl_loss, persona_margin
from persvectrain.output import run_root
from persvectrain.steering import answer_logprob, answer_logprobs, encode, steer
from persvectrain.train import RefAnswer, TrainConfig, TraitData

C_VALUES = (-1.0, -0.5, 0.5, 1.0)
KINDS = ("trained", "source_full", "source_train", "source_at_trained_norm", "trained_random", "source_random")
FREE_GEN_KINDS = ("trained", "source_full", "trained_random")
FREE_GEN_C = (1.0, -1.0)
PASS_CS = ("c=+1", "c=-1")
RANDOM_DIRS = 5  # random directions per twin; averaged at the pass-rule c values (+-1), k = 0 only elsewhere
RANDOM_KINDS = ("trained_random", "source_random")
RANDOM_SEED_STEP = 1000
MIX_WEIGHTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def c_key(c: float) -> str:
    return f"c={c:+g}"


def mean_se(xs: list[float]) -> tuple[float | None, float | None]:
    """Mean and standard error (std with ddof=1 / sqrt(n)); SE None when n < 2, both None when n == 0."""
    n = len(xs)
    if n == 0:
        return None, None
    m = sum(xs) / n
    if n < 2:
        return m, None
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


# ---------------------------------------------------------------- pass rule


def _paired(te: dict, a: str, b: str, c: str, field: str) -> tuple[float | None, float | None]:
    xa, xb = te["kinds"][a][c][field], te["kinds"][b][c][field]
    if len(xa) != len(xb):
        raise ValueError(f"{a} and {b} {field} at {c} are not paired ({len(xa)} vs {len(xb)} items)")
    return mean_se([x - y for x, y in zip(xa, xb)])


def _check_fg_pairing(fg: dict, c: str) -> None:
    """Free answers are paired by question: the same question ids in the same order in every kind (when stored)."""
    blocks = [fg["unsteered"]] + [fg[k][c] for k in FREE_GEN_KINDS if k in fg]
    qids = [b["qid"] for b in blocks if "qid" in b]
    if any(q != qids[0] for q in qids):
        raise ValueError(f"free-gen answers at {c} are not paired: question ids differ between kinds")


def _shift_diff(fg: dict, a: str, b: str, c: str) -> tuple[float | None, float | None]:
    """Paired mean and SE over prompts of shift_a - shift_b, shift_k = |tokens_k - tokens_unsteered| per prompt."""
    u, xa, xb = fg["unsteered"]["tokens"], fg[a][c]["tokens"], fg[b][c]["tokens"]
    if not len(u) == len(xa) == len(xb):
        raise ValueError(f"free-gen tokens of {a} / {b} at {c} are not paired with unsteered "
                         f"({len(xa)} / {len(xb)} vs {len(u)} prompts)")
    return mean_se([abs(ta - tu) - abs(tb - tu) for ta, tb, tu in zip(xa, xb, u)])


def _correct_diff(fg: dict, a: str, b: str, c: str) -> tuple[float | None, float | None]:
    """Paired mean and SE over questions of correct_a - correct_b (1 correct, 0 wrong or no answer)."""
    xa, xb = (fg[k]["correct"] if k == "unsteered" else fg[k][c]["correct"] for k in (a, b))
    if len(xa) != len(xb):
        raise ValueError(f"free-gen answers of {a} and {b} at {c} are not paired ({len(xa)} vs {len(xb)} questions)")
    return mean_se([float(x) - float(y) for x, y in zip(xa, xb)])


def _judge_diff(pj: dict, a: str, b: str, c: str, field: str) -> tuple[float | None, float | None, int]:
    """Paired mean and SE over questions of field_a - field_b of the persona judge (field "score" or "coherence");
    a question with a None judge score in either kind is left out of that pair. The unsteered / elicited blocks
    have no c level (one block per trait)."""
    ba, bb = (pj[k] if k in ("unsteered", "elicited") else pj[k][c] for k in (a, b))
    if ba["qid"] != bb["qid"]:
        raise ValueError(f"persona-judge answers of {a} and {b} at {c} are not paired: question ids differ")
    diffs = [float(x) - float(y) for x, y in zip(ba[field], bb[field]) if x is not None and y is not None]
    return (*mean_se(diffs), len(diffs))


def pass_rule(trait_eval: dict) -> dict:
    """The strict pass rule at c = +1 and c = -1 (spec). Differences within 1 SE (paired) count as ties.

    persona: mean(trained - source_full margin) >= -SE; kl: mean(source_full - trained KL) > SE (a tie fails
    "less"); length, per question shift_k = |tokens_k - tokens_unsteered| (free answers are paired: one seed per
    (c, question) for every kind): mean(shift_source - shift_trained) > SE (a tie fails "less"); accuracy:
    mean(correct_trained - correct_source_full) >= -SE (a tie passes "at least as accurate"); accuracy-unsteered:
    mean(correct_trained - correct_unsteered) >= -SE (no measurable loss against the unsteered model); random:
    mean(trained_random - trained KL) >= -SE and mean(shift_random - shift_trained) >= -SE (a tie passes "no
    worse"). Without free-generation results (status pending or disabled) the length and accuracy parts are
    skipped (named in "skipped"). Persona judge (`persona`, trait_eval["persona_judge"]; skipped when absent), paired
    over questions with None scores left out, ties within 1 SE: judge@+1 mean(score_trained - score_source_full)
    >= -SE (expresses the trait at least as much), judge@-1 mean(score_source_full - score_trained) >= -SE (trained
    suppresses at least as much), coherence@c mean(coherence_trained - coherence_unsteered) >= -SE (no measurable
    coherence loss against the unsteered model, neutral prompt, same questions; the source's difference to unsteered
    is recorded next to it as source_diff_mean / source_diff_se).
    """
    fg = trait_eval.get("free_gen")
    pj = trait_eval.get("persona_judge")
    checks: dict = {}
    failed: list[str] = []
    skipped: list[str] = []
    for c in PASS_CS:
        ch: dict = {}
        m, se = _paired(trait_eval, "trained", "source_full", c, "margins")
        ch["persona"] = {"passed": m is not None and m >= -(se or 0.0), "diff_mean": m, "diff_se": se}
        m, se = _paired(trait_eval, "source_full", "trained", c, "kls")
        ch["kl"] = {"passed": m is not None and m > (se or 0.0), "diff_mean": m, "diff_se": se}
        m, se = _paired(trait_eval, "trained_random", "trained", c, "kls")
        rnd = {"kl_passed": m is not None and m >= -(se or 0.0), "kl_diff_mean": m, "kl_diff_se": se}
        if fg:
            _check_fg_pairing(fg, c)
            m, se = _shift_diff(fg, "source_full", "trained", c)
            ch["length"] = {"passed": m is not None and m > (se or 0.0), "diff_mean": m, "diff_se": se}
            m, se = _correct_diff(fg, "trained", "source_full", c)
            ch["accuracy"] = {"passed": m is not None and m >= -(se or 0.0), "diff_mean": m, "diff_se": se}
            m, se = _correct_diff(fg, "trained", "unsteered", c)
            ch["accuracy-unsteered"] = {"passed": m is not None and m >= -(se or 0.0), "diff_mean": m, "diff_se": se}
            m, se = _shift_diff(fg, "trained_random", "trained", c)
            rnd.update(length_passed=m is not None and m >= -(se or 0.0), length_diff_mean=m, length_diff_se=se)
        else:
            skipped += [f"length@{c}", f"accuracy@{c}", f"accuracy-unsteered@{c}", f"random-length@{c}"]
        if pj:
            m, se, n = _judge_diff(pj, *(("trained", "source_full") if c == "c=+1" else ("source_full", "trained")),
                                   c, "score")
            ch["judge"] = {"passed": m is not None and m >= -(se or 0.0), "diff_mean": m, "diff_se": se, "n_pairs": n}
            m, se, n = _judge_diff(pj, "trained", "unsteered", c, "coherence")
            sm, sse, _ = _judge_diff(pj, "source_full", "unsteered", c, "coherence")
            ch["coherence"] = {"passed": m is not None and m >= -(se or 0.0), "diff_mean": m, "diff_se": se,
                               "n_pairs": n, "against": "unsteered", "source_diff_mean": sm, "source_diff_se": sse}
        else:
            skipped += [f"judge@{c}", f"coherence@{c}"]
        rnd["passed"] = rnd["kl_passed"] and rnd.get("length_passed", True)
        ch["random"] = rnd
        failed += [f"{name}@{c}" for name in ("persona", "kl", "length", "accuracy", "accuracy-unsteered", "random", "judge",
                                          "coherence")
                   if name in ch and not ch[name]["passed"]]
        checks[c] = ch
    return {"checks": checks, "failed": failed, "passed": not failed, "skipped": skipped}


# ---------------------------------------------------------------- teacher-forced pieces


def random_seeds(layer: int) -> list[int]:
    """Seeds of the random set: RANDOM_CONTROL_SEED + layer + 1000 k, k = 0..4 (k = 0 is the single twin of
    PersVecFast / write_vectors)."""
    return [RANDOM_CONTROL_SEED + layer + RANDOM_SEED_STEP * k for k in range(RANDOM_DIRS)]


def random_set(v: Tensor, layer: int) -> list[Tensor]:
    """The 5 random directions norm-matched to v (random_control), in k order."""
    return [random_control(v, seed) for seed in random_seeds(layer)]


def random_sets(cfg: TrainConfig, vecs: dict[str, Tensor]) -> dict[str, list[Tensor]]:
    """{random kind: its 5 directions} for one trait's vector_kinds."""
    return {"trained_random": [r.float() for r in random_set(vecs["trained"], cfg.layer)],
            "source_random": [r.float() for r in random_set(vecs["source_full"], cfg.layer)]}


def _dirs(kind: str, v: Tensor, c: float, rand: dict[str, list[Tensor]]) -> list[Tensor]:
    """The vectors a (kind, c) cell is averaged over: the 5 random directions at c = +-1, else just v."""
    return rand[kind] if kind in rand and abs(c) == 1.0 else [v]


def vector_kinds(cfg: TrainConfig, td: TraitData, trained: Tensor) -> dict[str, Tensor]:
    v = trained.detach().float().cpu()
    src = td.source.detach().float().cpu()
    return {
        "trained": v,
        "source_full": src,
        "source_train": td.start.detach().float().cpu(),
        "source_at_trained_norm": src * (v.norm() / src.norm()),
        # k = 0 of the random set: the twin at seed RANDOM_CONTROL_SEED + layer whatever cfg.seed is, as PersVecFast /
        # MA_Environments (norms and cosines are reported for it; the others are in random_sets)
        "trained_random": random_set(v, cfg.layer)[0].float(),
        "source_random": random_set(src, cfg.layer)[0].float(),
    }


def _held_cache(model, tok, cfg: TrainConfig, pairs) -> list[tuple[tuple, tuple[Tensor, Tensor]]]:
    """[(encoded pair, unsteered (lp_pos, lp_neg))] for held pairs; unsteered computed once."""
    out = []
    for p in pairs:
        enc = encode(tok, p.question, p.pos, cfg.max_answer_tokens) + encode(tok, p.question, p.neg,
                                                                              cfg.max_answer_tokens)
        out.append((enc, _pair_logprobs(model, cfg, None, enc)))
    return out


def _pair_logprobs(model, cfg: TrainConfig, vec: Tensor | None, enc: tuple) -> tuple[Tensor, Tensor]:
    """(pos, neg) answer log-probs steered by vec, normalised as in training (cfg.logprob_norm)."""
    ids_p, s_p, ids_n, s_n = enc
    with steer(model, cfg.layer, vec):
        return (answer_logprob(model, ids_p, s_p, cfg.logprob_norm),
                answer_logprob(model, ids_n, s_n, cfg.logprob_norm))


def _margin(model, cfg: TrainConfig, vec: Tensor, sign: int, enc: tuple, ref: tuple[Tensor, Tensor]) -> float:
    lpc = _pair_logprobs(model, cfg, vec, enc)
    return float(persona_margin(lpc[0], ref[0], lpc[1], ref[1], sign))


def _margins(model, cfg: TrainConfig, held: list, vecs: dict[str, Tensor], rand: dict[str, list[Tensor]]
             ) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """(margins[kind][c] per held pair, per_direction[kind][c] = the mean margin of each random direction).
    Random kinds at c = +-1 are the mean over their 5 directions per pair; per_direction only there."""
    out: dict = {}
    per_dir: dict = {}
    for kind, v in vecs.items():
        out[kind] = {}
        for c in C_VALUES:
            sign = 1 if c > 0 else -1
            dirs = _dirs(kind, v, c, rand)
            ms = [[_margin(model, cfg, c * d, sign, enc, lp0) for enc, lp0 in held] for d in dirs]
            out[kind][c_key(c)] = [sum(col) / len(col) for col in zip(*ms)]
            if len(dirs) > 1:
                per_dir.setdefault(kind, {})[c_key(c)] = [sum(row) / len(row) if row else None for row in ms]
    return out, per_dir


def _kls(model, tok, cfg: TrainConfig, refs: list[RefAnswer], vecs: dict[str, dict[str, Tensor]],
         rand: dict[str, dict[str, list[Tensor]]]) -> tuple[dict, dict]:
    """(kls[slug][kind][c] per reference answer, per_direction[slug][kind][c] = each random direction's mean KL).
    KL(unsteered || steered); each answer's unsteered log-probs computed once. Random kinds at c = +-1 are the
    mean over their 5 directions per answer."""
    raw = {s: {k: {c_key(c): [[] for _ in _dirs(k, v, c, rand[s])] for c in C_VALUES} for k, v in kv.items()}
           for s, kv in vecs.items()}
    for r in refs:
        ids, start = encode(tok, r.question, r.answer_text, cfg.kl_tokens)
        lp0 = answer_logprobs(model, ids, start)
        for s, kv in vecs.items():
            for kind, v in kv.items():
                for c in C_VALUES:
                    for j, d in enumerate(_dirs(kind, v, c, rand[s])):
                        with steer(model, cfg.layer, c * d):
                            lpc = answer_logprobs(model, ids, start)
                        raw[s][kind][c_key(c)][j].append(float(kl_loss(lp0, lpc)))
        del lp0
    out = {s: {k: {c: [sum(col) / len(col) for col in zip(*rows)] if rows[0] else [] for c, rows in cs.items()}
               for k, cs in kd.items()} for s, kd in raw.items()}
    per_dir = {s: {k: {c: [sum(row) / len(row) if row else None for row in rows]
                       for c, rows in cs.items() if len(rows) > 1}
                   for k, cs in kd.items() if k in rand[s]} for s, kd in raw.items()}
    return out, per_dir


def _cos(a: Tensor, b: Tensor) -> float:
    return float(F.cosine_similarity(a.float(), b.float(), dim=0))


def _joint(model, cfg: TrainConfig, slugs: list[str], vecs: dict, held: dict, traits_out: dict) -> dict:
    cos = {a: {b: _cos(vecs[a]["trained"], vecs[b]["trained"]) for b in slugs} for a in slugs}
    mixes = []
    for a, b in itertools.combinations(slugs, 2):
        for wa, wb in MIX_WEIGHTS:
            w = {a: wa, b: wb}
            mix_vec = wa * vecs[a]["trained"] + wb * vecs[b]["trained"]
            mix_m, mix_se, alone = {}, {}, {}
            for s, other in ((a, b), (b, a)):
                other_vec = w[other] * vecs[other]["trained"]
                sign = 1 if w[s] > 0 else -1
                ms = [_margin(model, cfg, mix_vec, sign, enc, _pair_logprobs(model, cfg, other_vec, enc))
                      for enc, _lp0 in held[s]]
                mix_m[s], mix_se[s] = mean_se(ms)
                alone[s] = traits_out[s]["kinds"]["trained"][c_key(float(w[s]))]["margin_mean"]
            mixes.append({"pair": [a, b], "weights": w, "mix_margin": mix_m, "mix_margin_se": mix_se,
                          "alone_margin": alone})
    return {"cos_matrix": cos, "mixes": mixes}


# ---------------------------------------------------------------- evaluate


def evaluate(model, tok, cfg: TrainConfig, traits: dict[str, TraitData], trained: dict[str, Tensor],
             kl_held: list[RefAnswer], free_gen: bool = True, previous: dict | None = None,
             persona_judge: bool = True) -> dict:
    """Evaluate every trained vector on held-out data (teacher-forced); writes eval.json in the run root and
    prints the table. No free generation here: free_gen status "pending" (`freegen` fills it on vLLM), or
    "disabled" when free_gen is False. persona_judge: the same for the `persona` step (status "pending" or
    "disabled").

    previous: this run's earlier eval.json (`eval` re-runs). Its free_gen block (top level and per trait) and
    free-gen timing are kept when free_gen is True, since free answers depend only on the vectors and the held
    questions; the pass rule is recomputed with them. The same for its persona_judge block (top level and per
    trait) when persona_judge is True and the previous block is not "disabled".
    """
    model.requires_grad_(False)
    model.eval()
    slugs = [s for s in cfg.traits if s in trained]
    vecs = {s: vector_kinds(cfg, traits[s], trained[s]) for s in slugs}
    res: dict = {"run_name": cfg.run_name, "model": cfg.model, "layer": cfg.layer, "mode": cfg.mode,
                 "c_values": list(C_VALUES), "kinds": list(KINDS), "random_dirs": RANDOM_DIRS,
                 "n_kl_held": len(kl_held),
                 "logprob_norm": cfg.logprob_norm,
                 "free_gen": {"enabled": bool(free_gen), "status": "pending" if free_gen else "disabled"},
                 "persona_judge": {"status": "pending" if persona_judge else "disabled"},
                 "timing": {"teacher_forced_s": None, "free_gen_s": None},
                 "traits": {}}
    path = run_root(cfg.out, cfg.run_name) / "eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log("eval: teacher-forced part starts")
    with torch.no_grad():
        held = {s: _held_cache(model, tok, cfg, traits[s].held) for s in slugs}
        log(f"eval: margins on {[len(held[s]) for s in slugs]} held pairs, KL on {len(kl_held)} reference answers")
        rand = {s: random_sets(cfg, vecs[s]) for s in slugs}
        kls, kl_dirs = _kls(model, tok, cfg, kl_held, vecs, rand)
        for s in slugs:
            margins, m_dirs = _margins(model, cfg, held[s], vecs[s], rand[s])
            kinds = {}
            for kind in KINDS:
                kinds[kind] = {}
                for c in C_VALUES:
                    k = c_key(c)
                    mm, ms = mean_se(margins[kind][k])
                    km, ks = mean_se(kls[s][kind][k])
                    kinds[kind][k] = {"margin_mean": mm, "margin_se": ms, "kl_mean": km, "kl_se": ks,
                                      "margins": margins[kind][k], "kls": kls[s][kind][k]}
                    if k in kl_dirs[s].get(kind, {}):  # random kinds at c = +-1: the 5 directions' means
                        kinds[kind][k]["per_direction"] = kl_dirs[s][kind][k]
                        kinds[kind][k]["per_direction_margin"] = m_dirs[kind][k]
            res["traits"][s] = {
                "norms": {k: float(v.norm()) for k, v in vecs[s].items()},
                "cos_with_source": {k: _cos(v, vecs[s]["source_full"]) for k, v in vecs[s].items()},
                "kinds": kinds,
            }

    if cfg.mode == "joint" and len(slugs) >= 2:
        with torch.no_grad():
            res["joint"] = _joint(model, cfg, slugs, vecs, held, res["traits"])
    # a previous "disabled" run has no free-gen block to keep: the re-run stays "pending" so `freegen` can run
    if free_gen and previous and previous.get("free_gen") and previous["free_gen"].get("status") != "disabled":
        res["free_gen"] = previous["free_gen"]
        if "random_dirs" not in res["free_gen"]:
            # free answers of an old single-twin run: kept, but `freegen` must rerun to regenerate trained_random
            res["free_gen"] = {**res["free_gen"], "status": "pending"}
        res["timing"]["free_gen_s"] = previous.get("timing", {}).get("free_gen_s")
        for s in slugs:
            fg = previous.get("traits", {}).get(s, {}).get("free_gen")
            if fg is not None:
                res["traits"][s]["free_gen"] = fg
    pj_prev = (previous or {}).get("persona_judge")
    if persona_judge and pj_prev and pj_prev.get("status") != "disabled":
        res["persona_judge"] = pj_prev
        res["timing"]["persona_s"] = previous.get("timing", {}).get("persona_s")
        for s in slugs:
            blk = previous.get("traits", {}).get(s, {}).get("persona_judge")
            if blk is not None:
                res["traits"][s]["persona_judge"] = blk
    apply_pass(res, slugs)
    res["timing"]["teacher_forced_s"] = round(time.monotonic() - t0, 2)
    write_json(path, res)
    log(f"eval: teacher-forced part done in {res['timing']['teacher_forced_s']} s; wrote {path} "
        f"(free generation {res['free_gen']['status']})")
    print_table(res)
    return res


def apply_pass(res: dict, slugs: list[str]) -> None:
    for s in slugs:
        res["traits"][s]["pass"] = pass_rule(res["traits"][s])
    res["summary"] = {s: res["traits"][s]["pass"]["passed"] for s in slugs}


# ---------------------------------------------------------------- table


def _pm(m: float | None, se: float | None) -> str:
    if m is None:
        return "-"
    return f"{m:.4g}" + (f"+-{se:.2g}" if se is not None else "")


def fg_cols(r: dict | None) -> str:
    """tokens / cut / acc columns of one free-gen block (blank when there is none)."""
    if not r:
        return f"{'':>8} {'':>5} {'':>12}"
    return f"{r['mean_tokens']:>8.1f} {r['cut_share']:>5.2f} {_pm(r['accuracy'], r['accuracy_se']):>12}"


def pass_lines(res: dict) -> list[str]:
    """One `pass <trait>: PASS | FAIL (...)` line per trait, with the skipped checks and the free-gen status."""
    fg_status = res.get("free_gen", {}).get("status")
    pj_status = (res.get("persona_judge") or {}).get("status")
    lines = []
    for s, t in res["traits"].items():
        p = t["pass"]
        line = f"pass {s}: " + ("PASS" if p["passed"] else "FAIL (" + ", ".join(p["failed"]) + ")")
        if p["skipped"]:
            fgs = [x for x in p["skipped"] if not x.startswith(("judge@", "coherence@"))]
            pjs = [x for x in p["skipped"] if x.startswith(("judge@", "coherence@"))]
            line += "; skipped: " + ", ".join(p["skipped"]) + " (" + "; ".join(
                ([f"free generation {fg_status}"] if fgs else []) + ([f"persona judge {pj_status}"] if pjs else [])) + ")"
        lines.append(line)
    return lines


def print_table(res: dict) -> None:
    lines = [f"{'trait':<18} {'kind':<24} {'c':>6} {'margin':>18} {'KL':>18} {'tokens':>8} {'cut':>5} {'acc':>12}"]
    unsteered = res["free_gen"].get("unsteered")
    if unsteered:
        lines.append(f"{'-':<18} {'unsteered':<24} {'0':>6} {'':>18} {'':>18} {fg_cols(unsteered)}")
    for s, t in res["traits"].items():
        for kind in KINDS:
            for c in C_VALUES:
                k = c_key(c)
                row = t["kinds"][kind][k]
                fg = t.get("free_gen", {}).get(kind, {}).get(k)
                lines.append(f"{s:<18} {kind:<24} {k[2:]:>6} {_pm(row['margin_mean'], row['margin_se']):>18} "
                             f"{_pm(row['kl_mean'], row['kl_se']):>18} {fg_cols(fg)}")
    lines += pass_lines(res)
    print("\n".join(lines), flush=True)
