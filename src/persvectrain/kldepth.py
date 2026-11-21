"""`kldepth`: KL(unsteered || steered) by position in long answers, a diagnostic for the 512-token KL term.

Training's KL term only reads the first `max_answer_tokens` (512) tokens of the unsteered reference answers. This
reads the run's saved unsteered free answers (freegen_answers.jsonl.gz, up to 15360 tokens, about 6800 on average)
teacher-forced and reports the mean per-token KL in position buckets (answer tokens 0-512, 512-2048, ...) for each
trait's `trained`, `source_full` and `trained_random` vectors at c = +1 and -1, plus the paired difference
source_full - trained per bucket. If the trained vector's lower KL holds past token 512, the 512-token term is
enough; if it fades, train on longer reference answers.

Writes run_root/kl_depth.json only (never eval.json, train.json or the vectors). HF model, one answer at a time;
the final hidden states are kept and the vocabulary projection is done in row chunks, so 8k-token answers fit.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, read_json, write_json

from persvectrain.evaluate import FREE_GEN_C, FREE_GEN_KINDS, c_key, mean_se
from persvectrain.freegen import (ANSWERS_FILE, POOL, UNSTEERED, freegen_vectors, held_questions, instruction,
                                  read_answers)
from persvectrain.output import config_from_train_json, run_root
from persvectrain.steering import steer

OUT_FILE = "kl_depth.json"
EDGES = (512, 2048, 4096, 8192)  # bucket upper edges in answer tokens; the first bucket is training's window
ROW_CHUNK = 1024  # rows of the vocabulary projection at a time


def buckets(edges=EDGES) -> list[tuple[int, int]]:
    lo, out = 0, []
    for hi in edges:
        out.append((lo, hi))
        lo = hi
    return out


def _final_hidden(model, ids: torch.Tensor) -> torch.Tensor:
    """[T, d] final (normed) hidden states of the base model; the lm_head is applied by the caller."""
    base = getattr(model, "model", None)
    if base is None:
        raise SystemExit("kldepth: the model has no .model base module; cannot read final hidden states")
    return base(input_ids=ids.to(model.device), use_cache=False).last_hidden_state[0]


def check_projection(model, ids: torch.Tensor) -> None:
    """Fail loudly unless lm_head(final hidden) reproduces the model's own last-position logits."""
    with torch.no_grad():
        ref = model(ids.to(model.device), logits_to_keep=1, use_cache=False).logits[0, -1].float()
        h = _final_hidden(model, ids)[-1]
        got = model.get_output_embeddings()(h[None])[0].float()
    err = float((got - ref).abs().max())
    scale = float(ref.abs().max()) + 1e-6
    if err > 1e-2 * scale:
        raise SystemExit(f"kldepth: lm_head(final hidden) differs from the model's logits (max abs {err:.4g}, "
                         f"logit scale {scale:.4g}); the hidden-state path is wrong for this model")


def token_kls(model, h0: torch.Tensor, hc: torch.Tensor, start: int) -> torch.Tensor:
    """Per answer token KL(p_0 || p_c) [T - start]; rows start-1 .. T-2 predict the answer tokens."""
    head = model.get_output_embeddings()
    rows0, rowsc = h0[start - 1:-1], hc[start - 1:-1]
    out = []
    for i in range(0, rows0.shape[0], ROW_CHUNK):
        lp0 = torch.log_softmax(head(rows0[i:i + ROW_CHUNK]).float(), dim=-1)
        lpc = torch.log_softmax(head(rowsc[i:i + ROW_CHUNK]).float(), dim=-1)
        out.append((lp0.exp() * (lp0 - lpc)).sum(-1))
    return torch.cat(out).cpu()


def bucket_means(kls: torch.Tensor, bks: list[tuple[int, int]]) -> list[float | None]:
    """Mean KL of the tokens in each bucket; None for a bucket the answer does not reach."""
    n = kls.shape[0]
    return [float(kls[lo:min(hi, n)].mean()) if lo < n else None for lo, hi in bks]


def _stats(per_item: list[list[float | None]], n_b: int) -> tuple[list, list, list]:
    means, ses, ns = [], [], []
    for b in range(n_b):
        xs = [row[b] for row in per_item if row[b] is not None]
        m, se = mean_se(xs)
        means.append(m)
        ses.append(se)
        ns.append(len(xs))
    return means, ses, ns


def _paired(a: list[list[float | None]], b: list[list[float | None]], n_b: int) -> tuple[list, list]:
    """mean(a - b) and SE per bucket over the items present in both (the same texts, so the same items)."""
    means, ses = [], []
    for k in range(n_b):
        xs = [x[k] - y[k] for x, y in zip(a, b) if x[k] is not None and y[k] is not None]
        m, se = mean_se(xs)
        means.append(m)
        ses.append(se)
    return means, ses


def run_kldepth(model, tok, out: Path, run_name: str, max_tokens: int = EDGES[-1], n_items: int | None = None,
                traits: list[str] | None = None, pool: Path = POOL) -> dict | None:
    root = run_root(Path(out), run_name)
    for name in ("train.json", ANSWERS_FILE):
        if not (root / name).exists():
            print(f"kldepth: {root / name} missing; run `train` and `freegen` first", flush=True)
            return None
    meta = read_json(root / "train.json")
    cfg = config_from_train_json(meta, out)
    slugs = [s for s in cfg.traits if traits is None or s in traits]
    if not slugs:
        print(f"kldepth: none of {traits} is a trait of {run_name} ({cfg.traits})", flush=True)
        return None
    held, n_options = held_questions(meta, pool)
    lines, _ = read_answers(root / ANSWERS_FILE)
    texts = {}
    for ln in lines:  # the last saved unsteered answer per question wins, as in freegen
        slug, kind, _c, qid = ln["key"]
        if kind == UNSTEERED:
            texts[qid] = ln["text"]
    items = [r for r in held if texts.get(r.id)]
    if n_items is not None:
        items = items[:n_items]
    if not items:
        print(f"kldepth: no saved unsteered answers in {root / ANSWERS_FILE}", flush=True)
        return None
    edges = [e for e in EDGES if e < max_tokens] + [max_tokens]
    bks = buckets(edges)
    vecs = freegen_vectors(cfg, slugs)
    log(f"kldepth {run_name}: {len(items)} unsteered answers, up to {max_tokens} tokens, buckets {bks}, "
        f"traits {slugs}")

    t0 = time.monotonic()
    per_item = {s: {k: {c_key(c): [] for c in FREE_GEN_C} for k in FREE_GEN_KINDS} for s in slugs}
    n_tokens = []
    for i, r in enumerate(items):
        prompt = chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, r.question + instruction(n_options[r.id]), False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(texts[r.id], add_special_tokens=False)["input_ids"][:max_tokens]
        if not a_ids:
            continue
        ids = torch.tensor([p_ids + a_ids])
        start = len(p_ids)
        if i == 0:
            check_projection(model, ids)
        n_tokens.append(len(a_ids))
        with torch.no_grad():
            h0 = _final_hidden(model, ids)
            for s in slugs:
                for kind in FREE_GEN_KINDS:
                    v = vecs[s][kind].float()
                    for c in FREE_GEN_C:
                        with steer(model, cfg.layer, c * v):
                            hc = _final_hidden(model, ids)
                        per_item[s][kind][c_key(c)].append(bucket_means(token_kls(model, h0, hc, start), bks))
                        del hc
            del h0
        log(f"kldepth {run_name}: {i + 1}/{len(items)} answers ({len(a_ids)} tokens)")

    n_b = len(bks)
    res = {"run_name": run_name, "max_tokens": max_tokens, "buckets": [list(b) for b in bks],
           "layer": cfg.layer, "items": [r.id for r in items][:len(n_tokens)], "answer_tokens": n_tokens,
           "source": str((root / ANSWERS_FILE).resolve()),
           "note": "per-token KL(unsteered || steered) on the run's unsteered free answers, teacher-forced; "
                   "bucket means per answer, then mean +- SE over answers reaching the bucket",
           "traits": {}}
    for s in slugs:
        t = {"kinds": {}, "source_minus_trained": {}}
        for kind in FREE_GEN_KINDS:
            t["kinds"][kind] = {}
            for c in FREE_GEN_C:
                rows = per_item[s][kind][c_key(c)]
                m, se, ns = _stats(rows, n_b)
                t["kinds"][kind][c_key(c)] = {"mean": m, "se": se, "n": ns, "per_item": rows}
        for c in FREE_GEN_C:
            m, se = _paired(per_item[s]["source_full"][c_key(c)], per_item[s]["trained"][c_key(c)], n_b)
            t["source_minus_trained"][c_key(c)] = {"mean": m, "se": se}
        res["traits"][s] = t
    res["seconds"] = round(time.monotonic() - t0, 2)
    write_json(root / OUT_FILE, res)
    log(f"kldepth {run_name}: done in {res['seconds']} s; wrote {root / OUT_FILE}")
    print_table(res)
    return res


def _fmt(m, se) -> str:
    if m is None:
        return "-"
    return f"{m:.4f}" + (f"+-{se:.4f}" if se is not None else "")


def print_table(res: dict) -> None:
    heads = [f"{lo}-{hi}" for lo, hi in res["buckets"]]
    w = 17
    print(f"{'trait':<14} {'kind':<22} {'c':>3} " + " ".join(f"{h:>{w}}" for h in heads))
    any_t = next(iter(res["traits"].values()))
    ns = any_t["kinds"]["trained"][c_key(1)]["n"]
    print(f"{'':<14} {'answers in bucket':<22} {'':>3} " + " ".join(f"{n:>{w}}" for n in ns))
    for s, t in res["traits"].items():
        for c in FREE_GEN_C:
            ck = c_key(c)
            for kind in FREE_GEN_KINDS:
                k = t["kinds"][kind][ck]
                print(f"{s:<14} {kind:<22} {c:>+3g} " + " ".join(f"{_fmt(m, se):>{w}}"
                                                                for m, se in zip(k["mean"], k["se"])))
            d = t["source_minus_trained"][ck]
            print(f"{s:<14} {'source - trained':<22} {c:>+3g} " + " ".join(f"{_fmt(m, se):>{w}}"
                                                                          for m, se in zip(d["mean"], d["se"])))
