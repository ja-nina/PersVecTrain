"""End-of-run evaluation on held-out data and the pass rule (spec, "Evaluation").

Per trait, vector kind and c: held-out persona margin and KL on held-out reference answers (teacher-forced),
vector norms and cosine with the source vector; free-generation length / cut share / accuracy at c = +-1;
in joint runs the cosine matrix and 2-trait mix margins. Writes `<run root>/eval.json` and prints one table.
"""
from __future__ import annotations

import itertools
import math
import re

import torch
import torch.nn.functional as F
from persvecfast.extract import random_control
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, write_json
from torch import Tensor

from persvectrain.generate import generate_counted
from persvectrain.losses import kl_loss, persona_margin
from persvectrain.output import run_root
from persvectrain.steering import answer_logprob, answer_logprobs, encode, steer
from persvectrain.train import RefAnswer, TrainConfig, TraitData

C_VALUES = (-1.0, -0.5, 0.5, 1.0)
KINDS = ("trained", "source_full", "source_train", "source_at_trained_norm", "trained_random", "source_random")
FREE_GEN_KINDS = ("trained", "source_full", "trained_random")
FREE_GEN_C = (1.0, -1.0)
PASS_CS = ("c=+1", "c=-1")
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


# ---------------------------------------------------------------- free generation

# "answer" (any case), then the first standalone capital A-J before the next "answer"; the last such match wins
_ANSWER_RE = re.compile(r"(?i:answer)(?:(?!(?i:answer)).)*?\b([A-J])\b", re.S)


def parse_answer(text: str) -> str | None:
    """The option letter of a free answer: last `\\b([A-J])\\b` after "answer" (case-insensitive word), else None."""
    matches = _ANSWER_RE.findall(text)
    return matches[-1] if matches else None


def free_generation(model, tok, vec: Tensor | None, prompts: list[RefAnswer], seed: int, layer: int | None = None,
                    max_new_tokens: int = 4096, batch_size: int = 20, sampling: dict | None = None) -> dict:
    """Sampled answers to `prompts` (ref.question in the neutral chat prompt), steered by `vec` (already c * v)
    at `layer`, or unsteered when vec is None. Wrong when no letter parses."""
    if vec is not None and layer is None:
        raise ValueError("a steering vector needs a layer")
    texts = [chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, r.question, False) for r in prompts]
    torch.manual_seed(seed)
    with steer(model, layer, vec):
        answers, counts = generate_counted(model, tok, texts, batch_size, max_new_tokens, label="free-gen",
                                           **(sampling or {}))
    letters = [parse_answer(a) for a in answers]
    correct = [1.0 if l is not None and l == r.gold else 0.0 for l, r in zip(letters, prompts)]
    acc, acc_se = mean_se(correct)
    mean_tokens, _ = mean_se([float(c) for c in counts])
    cut, _ = mean_se([1.0 if c >= max_new_tokens else 0.0 for c in counts])
    return {"mean_tokens": mean_tokens, "cut_share": cut, "accuracy": acc, "accuracy_se": acc_se,
            "n": len(prompts), "tokens": [int(c) for c in counts], "letters": letters}


def model_sampling(model_key: str) -> tuple[dict, str]:
    """HF generate kwargs from the model card, and a note on what HF cannot apply."""
    spec = MODELS.get(model_key)
    if spec is None:
        return {}, f"model {model_key!r} not in persvecfast.models.MODELS: HF generate defaults"
    return ({"temperature": spec.temperature, "top_p": spec.top_p, "top_k": spec.top_k},
            f"presence_penalty {spec.presence_penalty} is not applied in HF generate")


# ---------------------------------------------------------------- pass rule


def _paired(te: dict, a: str, b: str, c: str, field: str) -> tuple[float | None, float | None]:
    xa, xb = te["kinds"][a][c][field], te["kinds"][b][c][field]
    if len(xa) != len(xb):
        raise ValueError(f"{a} and {b} {field} at {c} are not paired ({len(xa)} vs {len(xb)} items)")
    return mean_se([x - y for x, y in zip(xa, xb)])


def pass_rule(trait_eval: dict) -> dict:
    """The strict pass rule at c = +1 and c = -1 (spec). Differences within 1 SE (paired) count as ties.

    persona: mean(trained - source_full margin) >= -SE; kl: mean(source_full - trained KL) > SE (a tie fails);
    length: |trained - unsteered| <= |source_full - unsteered| mean tokens; random: mean(trained_random -
    trained KL) >= -SE and |trained - unsteered| <= |trained_random - unsteered| tokens. Without free
    generation the length parts are skipped (named in "skipped"). Accuracy is reported, not checked.
    """
    fg = trait_eval.get("free_gen")
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
            u = fg["unsteered"]["mean_tokens"]
            t, s, r = (fg[k][c]["mean_tokens"] for k in ("trained", "source_full", "trained_random"))
            ch["length"] = {"passed": abs(t - u) <= abs(s - u), "trained_shift": t - u, "source_shift": s - u}
            rnd.update(length_passed=abs(t - u) <= abs(r - u), trained_shift=t - u, random_shift=r - u)
        else:
            skipped += [f"length@{c}", f"random-length@{c}"]
        rnd["passed"] = rnd["kl_passed"] and rnd.get("length_passed", True)
        ch["random"] = rnd
        failed += [f"{name}@{c}" for name in ("persona", "kl", "length", "random")
                   if name in ch and not ch[name]["passed"]]
        checks[c] = ch
    return {"checks": checks, "failed": failed, "passed": not failed, "skipped": skipped}


# ---------------------------------------------------------------- teacher-forced pieces


def vector_kinds(cfg: TrainConfig, td: TraitData, trained: Tensor) -> dict[str, Tensor]:
    v = trained.detach().float().cpu()
    src = td.source.detach().float().cpu()
    return {
        "trained": v,
        "source_full": src,
        "source_train": td.start.detach().float().cpu(),
        "source_at_trained_norm": src * (v.norm() / src.norm()),
        "trained_random": random_control(v, cfg.seed + cfg.layer).float(),
        "source_random": random_control(src, cfg.seed + cfg.layer).float(),
    }


def _held_cache(model, tok, cfg: TrainConfig, pairs) -> list[tuple[tuple, tuple[Tensor, Tensor]]]:
    """[(encoded pair, unsteered (lp_pos, lp_neg))] for held pairs; unsteered computed once."""
    out = []
    for p in pairs:
        enc = encode(tok, p.question, p.pos, cfg.max_answer_tokens) + encode(tok, p.question, p.neg,
                                                                              cfg.max_answer_tokens)
        out.append((enc, _pair_logprobs(model, cfg.layer, None, enc)))
    return out


def _pair_logprobs(model, layer: int, vec: Tensor | None, enc: tuple) -> tuple[Tensor, Tensor]:
    ids_p, s_p, ids_n, s_n = enc
    with steer(model, layer, vec):
        return answer_logprob(model, ids_p, s_p), answer_logprob(model, ids_n, s_n)


def _margin(model, layer: int, vec: Tensor, sign: int, enc: tuple, ref: tuple[Tensor, Tensor]) -> float:
    lpc = _pair_logprobs(model, layer, vec, enc)
    return float(persona_margin(lpc[0], ref[0], lpc[1], ref[1], sign))


def _margins(model, cfg: TrainConfig, held: list, vecs: dict[str, Tensor]) -> dict[str, dict[str, list[float]]]:
    return {kind: {c_key(c): [_margin(model, cfg.layer, c * v, 1 if c > 0 else -1, enc, lp0) for enc, lp0 in held]
                   for c in C_VALUES}
            for kind, v in vecs.items()}


def _kls(model, tok, cfg: TrainConfig, refs: list[RefAnswer],
         vecs: dict[str, dict[str, Tensor]]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """KL(unsteered || steered) per reference answer; each answer's unsteered log-probs computed once."""
    out = {s: {k: {c_key(c): [] for c in C_VALUES} for k in kv} for s, kv in vecs.items()}
    for r in refs:
        ids, start = encode(tok, r.question, r.answer_text, cfg.max_answer_tokens)
        lp0 = answer_logprobs(model, ids, start)
        for s, kv in vecs.items():
            for kind, v in kv.items():
                for c in C_VALUES:
                    with steer(model, cfg.layer, c * v):
                        lpc = answer_logprobs(model, ids, start)
                    out[s][kind][c_key(c)].append(float(kl_loss(lp0, lpc)))
        del lp0
    return out


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
                ms = [_margin(model, cfg.layer, mix_vec, sign, enc, _pair_logprobs(model, cfg.layer, other_vec, enc))
                      for enc, _lp0 in held[s]]
                mix_m[s], mix_se[s] = mean_se(ms)
                alone[s] = traits_out[s]["kinds"]["trained"][c_key(float(w[s]))]["margin_mean"]
            mixes.append({"pair": [a, b], "weights": w, "mix_margin": mix_m, "mix_margin_se": mix_se,
                          "alone_margin": alone})
    return {"cos_matrix": cos, "mixes": mixes}


# ---------------------------------------------------------------- evaluate


def evaluate(model, tok, cfg: TrainConfig, traits: dict[str, TraitData], trained: dict[str, Tensor],
             kl_held: list[RefAnswer], free_gen: bool = True, free_gen_tokens: int = 4096,
             free_gen_batch: int = 20) -> dict:
    """Evaluate every trained vector on held-out data; writes eval.json in the run root and prints the table."""
    model.requires_grad_(False)
    model.eval()
    slugs = [s for s in cfg.traits if s in trained]
    vecs = {s: vector_kinds(cfg, traits[s], trained[s]) for s in slugs}
    sampling, note = model_sampling(cfg.model)
    res: dict = {"run_name": cfg.run_name, "model": cfg.model, "layer": cfg.layer, "mode": cfg.mode,
                 "c_values": list(C_VALUES), "kinds": list(KINDS), "n_kl_held": len(kl_held),
                 "free_gen": {"enabled": bool(free_gen), "max_new_tokens": free_gen_tokens,
                              "batch_size": free_gen_batch, "sampling": sampling, "note": note},
                 "traits": {}}
    with torch.no_grad():
        held = {s: _held_cache(model, tok, cfg, traits[s].held) for s in slugs}
        log(f"eval: margins on {[len(held[s]) for s in slugs]} held pairs, KL on {len(kl_held)} reference answers")
        kls = _kls(model, tok, cfg, kl_held, vecs)
        for s in slugs:
            margins = _margins(model, cfg, held[s], vecs[s])
            kinds = {}
            for kind in KINDS:
                kinds[kind] = {}
                for c in C_VALUES:
                    k = c_key(c)
                    mm, ms = mean_se(margins[kind][k])
                    km, ks = mean_se(kls[s][kind][k])
                    kinds[kind][k] = {"margin_mean": mm, "margin_se": ms, "kl_mean": km, "kl_se": ks,
                                      "margins": margins[kind][k], "kls": kls[s][kind][k]}
            res["traits"][s] = {
                "norms": {k: float(v.norm()) for k, v in vecs[s].items()},
                "cos_with_source": {k: _cos(v, vecs[s]["source_full"]) for k, v in vecs[s].items()},
                "kinds": kinds,
            }

    if free_gen:
        log(f"eval: free generation, {len(kl_held)} prompts, {free_gen_tokens} new tokens; {note}")
        kw = dict(layer=cfg.layer, max_new_tokens=free_gen_tokens, batch_size=free_gen_batch, sampling=sampling)
        unsteered = free_generation(model, tok, None, kl_held, request_seed(cfg.seed, "freegen", "unsteered", 0),
                                    **kw)
        res["free_gen"]["unsteered"] = unsteered
        for s in slugs:
            fg = {"unsteered": unsteered}
            for kind in FREE_GEN_KINDS:
                fg[kind] = {c_key(c): free_generation(model, tok, c * vecs[s][kind], kl_held,
                                                      request_seed(cfg.seed, "freegen", kind, c), **kw)
                            for c in FREE_GEN_C}
            res["traits"][s]["free_gen"] = fg

    for s in slugs:
        res["traits"][s]["pass"] = pass_rule(res["traits"][s])
    if cfg.mode == "joint" and len(slugs) >= 2:
        with torch.no_grad():
            res["joint"] = _joint(model, cfg, slugs, vecs, held, res["traits"])
    res["summary"] = {s: res["traits"][s]["pass"]["passed"] for s in slugs}

    path = run_root(cfg.out, cfg.run_name) / "eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, res)
    print_table(res)
    log(f"eval: wrote {path}")
    return res


# ---------------------------------------------------------------- table


def _pm(m: float | None, se: float | None) -> str:
    if m is None:
        return "-"
    return f"{m:.4g}" + (f"+-{se:.2g}" if se is not None else "")


def _fg_cols(r: dict | None) -> str:
    if not r:
        return f"{'':>8} {'':>5} {'':>12}"
    return f"{r['mean_tokens']:>8.1f} {r['cut_share']:>5.2f} {_pm(r['accuracy'], r['accuracy_se']):>12}"


def print_table(res: dict) -> None:
    lines = [f"{'trait':<18} {'kind':<24} {'c':>6} {'margin':>18} {'KL':>18} {'tokens':>8} {'cut':>5} {'acc':>12}"]
    unsteered = res["free_gen"].get("unsteered")
    if unsteered:
        lines.append(f"{'-':<18} {'unsteered':<24} {'0':>6} {'':>18} {'':>18} {_fg_cols(unsteered)}")
    for s, t in res["traits"].items():
        for kind in KINDS:
            for c in C_VALUES:
                k = c_key(c)
                row = t["kinds"][kind][k]
                fg = t.get("free_gen", {}).get(kind, {}).get(k)
                lines.append(f"{s:<18} {kind:<24} {k[2:]:>6} {_pm(row['margin_mean'], row['margin_se']):>18} "
                             f"{_pm(row['kl_mean'], row['kl_se']):>18} {_fg_cols(fg)}")
    for s, t in res["traits"].items():
        p = t["pass"]
        line = f"pass {s}: " + ("PASS" if p["passed"] else "FAIL (" + ", ".join(p["failed"]) + ")")
        if p["skipped"]:
            line += "; skipped: " + ", ".join(p["skipped"])
        lines.append(line)
    print("\n".join(lines), flush=True)
