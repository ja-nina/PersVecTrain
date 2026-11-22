"""`vecs`: what does training change in the persona vectors, and is it the same change for every trait?

Per run and trait, with s0 = the start vector (difference of means over the run's TRAIN pairs, what training starts
from), t = the trained vector, sf = source_full (all pairs; what the pass rule compares with):
  delta = t - s0, split into the part along s0 (rescaling) and the part orthogonal to s0 (rotation);
  null: the split noise sf - s0 (no training at all, only a different set of pairs).
Across traits (per run): cosines between the deltas' orthogonal parts, the share of the first singular component of
the unit deltas (1 / n_traits if unrelated), the same for the split-noise vectors (null) and the random expectation;
u1 = the shared direction; how much of s0 and of t lies along u1 (training removes u1 if |cos(t, u1)| <
|cos(s0, u1)|). Across runs: per-trait cosine of the deltas (same change at every KL weight?).
With --model (GPU): logit lens of u1, -u1 and the mean delta (top tokens of W_U @ d, final norm left out), and the
geometry of the unsteered model's own layer-`layer` residuals on the KL reference answers: the share of each
direction's energy in the top-k principal components (random: k / d) and its variance relative to random directions.

Reads only; writes <out>/trained/analysis/vecs_<runs>.json.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from persvecfast.util import log, read_json, write_json

from persvectrain.data import full_source_vector, start_vector
from persvectrain.output import config_from_train_json, load_reference, load_trained, run_dir, run_root, source_dir

PCS = (1, 8, 64)
N_RANDOM = 64


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(unit(a.float()) @ unit(b.float()))


def perp(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """v without its component along ref."""
    r = unit(ref.float())
    return v.float() - (v.float() @ r) * r


def trait_change(s0: torch.Tensor, t: torch.Tensor, sf: torch.Tensor) -> dict:
    """Norms and cosines of one trait's change; *_rel are relative to |s0|."""
    s0, t, sf = s0.float(), t.float(), sf.float()
    d, n0 = t - s0, float(s0.norm())
    return {"norm_start": n0, "norm_trained": float(t.norm()), "norm_ratio": float(t.norm()) / n0,
            "cos_trained_start": cos(t, s0), "cos_trained_source": cos(t, sf), "cos_source_start": cos(sf, s0),
            "delta_rel": float(d.norm()) / n0, "delta_along_start_rel": float(d @ unit(s0)) / n0,
            "delta_perp_rel": float(perp(d, s0).norm()) / n0,
            "split_noise_rel": float((sf - s0).norm()) / n0}


def shared(vectors: dict[str, torch.Tensor]) -> dict:
    """Cross-trait structure of a set of directions (one per trait): cosine matrix, mean off-diagonal |cos| and
    signed cos, singular-value shares of the unit vectors, u1 (sign: positive mean projection) and each trait's
    cos with u1."""
    names = list(vectors)
    U = torch.stack([unit(vectors[n].float()) for n in names])
    C = U @ U.T
    n = len(names)
    off = [float(C[i, j]) for i in range(n) for j in range(n) if i != j]
    _, S, Vh = torch.linalg.svd(U, full_matrices=False)
    u1 = Vh[0]
    if float((U @ u1).sum()) < 0:
        u1 = -u1
    share = (S ** 2 / (S ** 2).sum()).tolist()
    return {"traits": names, "cos": C.tolist(), "mean_cos": sum(off) / len(off) if off else None,
            "mean_abs_cos": sum(abs(x) for x in off) / len(off) if off else None,
            "sv_share": share, "u1_cos": {nm: float(U[i] @ u1) for i, nm in enumerate(names)}, "u1": u1}


def random_abs_cos(d: int) -> float:
    """Expected |cos| of two independent random directions in d dimensions."""
    return math.sqrt(2 / (math.pi * d))


def load_run_vectors(out: Path, run: str) -> tuple[dict, dict]:
    """(cfg-ish info, {slug: (s0, t, sf)}) of one finished run."""
    root = run_root(Path(out), run)
    meta = read_json(root / "train.json")
    if not meta.get("finished"):
        raise SystemExit(f"vecs: run {run} did not finish training")
    cfg = config_from_train_json(meta, out)
    src = source_dir(cfg)
    slugs = [s for s in cfg.traits if s in meta["splits"]]
    trained = load_trained(run_dir(cfg.out, cfg.run_name, cfg.model), slugs, cfg.layer)
    vecs = {s: (start_vector(src, s, cfg.layer, meta["splits"][s]["train"]), trained[s],
                full_source_vector(src, s, cfg.layer)) for s in slugs}
    return {"model": cfg.model, "layer": cfg.layer, "meta": meta, "cfg": cfg}, vecs


def analyse(runs: dict[str, dict[str, tuple]]) -> dict:
    """The model-free part over {run: {slug: (s0, t, sf)}}."""
    res: dict = {"runs": {}, "across_runs": {}}
    for run, vecs in runs.items():
        d = next(iter(vecs.values()))[0].shape[0]
        per = {s: trait_change(*v) for s, v in vecs.items()}
        deltas = {s: perp(t - s0, s0) for s, (s0, t, sf) in vecs.items()}
        noise = {s: perp(sf - s0, s0) for s, (s0, t, sf) in vecs.items()}
        sh, nl = shared(deltas), shared(noise)
        u1 = sh.pop("u1")
        nl.pop("u1")
        mean_dir = unit(torch.stack([unit(s0.float()) for s0, _, _ in vecs.values()]).mean(0))
        res["runs"][run] = {
            "per_trait": per, "delta_perp": sh, "split_noise_perp": nl,
            "random_abs_cos": random_abs_cos(d), "equal_share": 1 / len(vecs),
            "along_u1": {s: {"start": cos(s0, u1), "trained": cos(t, u1)} for s, (s0, t, _) in vecs.items()},
            "delta_vs_mean_start_dir": {s: cos(t - s0, mean_dir) for s, (s0, t, _) in vecs.items()},
            "_u1": u1, "_deltas": {s: (t - s0).float() for s, (s0, t, _) in vecs.items()}}
    names = list(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            da, db = res["runs"][a]["_deltas"], res["runs"][b]["_deltas"]
            res["across_runs"][f"{a} vs {b}"] = {
                "delta_cos": {s: cos(da[s], db[s]) for s in da if s in db},
                "u1_cos": cos(res["runs"][a]["_u1"], res["runs"][b]["_u1"])}
    return res


# ---------------------------------------------------------------- with the model


def logit_lens(model, tok, d: torch.Tensor, k: int = 15) -> dict:
    W = model.get_output_embeddings().weight.float()
    scores = W @ d.float().to(W.device)
    top, bot = torch.topk(scores, k).indices.tolist(), torch.topk(-scores, k).indices.tolist()
    dec = lambda ids: [tok.decode([i]) for i in ids]  # noqa: E731
    return {"up": dec(top), "down": dec(bot)}


def residuals(model, tok, layer: int, refs, n_items: int, max_tokens: int = 512) -> torch.Tensor:
    """[N, d] fp32 outputs of decoder layer `layer` (the steering hook point) over the answer tokens of the
    unsteered reference answers, teacher-forced, no steering."""
    from persvecfast.smoke import decoder_layers

    from persvectrain.steering import encode
    got = []

    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        got.append(h[0].detach().float().cpu())

    rows = []
    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for r in refs[:n_items]:
                ids, start = encode(tok, r.question, r.answer_text, max_tokens)
                got.clear()
                model(ids.to(model.device), use_cache=False, logits_to_keep=1)
                assert got and got[0].shape[0] == ids.shape[1], "hooked residual length != input length"
                rows.append(got[0][start:])
    finally:
        handle.remove()
    return torch.cat(rows)


def geometry(H: torch.Tensor, dirs: dict[str, torch.Tensor], seed: int = 0) -> dict:
    """Share of each direction's energy in the top-k PCs of the centred residuals H, and Var(H d) / mean Var(H r)
    over N_RANDOM random unit directions r."""
    Hc = H - H.mean(0)
    pcs = [j for j in PCS if j <= min(H.shape)]  # small test models have fewer dimensions than 64
    _, _, V = torch.pca_lowrank(Hc, q=max(pcs), center=False)
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(N_RANDOM, H.shape[1], generator=g)
    R = R / R.norm(dim=1, keepdim=True)
    var_rand = float((Hc @ R.T).var(0).mean())
    mean_dir = unit(H.mean(0))
    out = {"n_tokens": H.shape[0], "random_pc_share": {f"top{j}": j / H.shape[1] for j in pcs}}
    for name, d in dirs.items():
        u = unit(d.float())
        proj = V.T @ u
        out[name] = {**{f"top{j}": float((proj[:j] ** 2).sum()) for j in pcs},
                     "var_vs_random": float((Hc @ u).var()) / var_rand, "cos_mean_residual": float(u @ mean_dir)}
    return out


def run_vecs(out: Path, runs: list[str], model=None, tok=None, n_items: int = 40) -> dict:
    loaded = {r: load_run_vectors(out, r) for r in runs}
    res = analyse({r: v for r, (_, v) in loaded.items()})
    if model is not None:
        info, _ = loaded[runs[0]]
        meta, cfg = info["meta"], info["cfg"]
        refs = {r.id: r for r in load_reference(Path(meta["sources"]["reference"]))}
        kl_train = [refs[i] for i in meta["splits"]["reference"]["train"]]
        log(f"vecs: layer-{cfg.layer} residuals over {min(n_items, len(kl_train))} KL reference answers")
        H = residuals(model, tok, cfg.layer, kl_train, n_items)
        for run, r in res["runs"].items():
            _, vecs = loaded[run]
            u1, deltas = r["_u1"], r["_deltas"]
            mean_delta = torch.stack(list(deltas.values())).mean(0)
            r["logit_lens"] = {"u1": logit_lens(model, tok, u1), "mean_delta": logit_lens(model, tok, mean_delta)}
            dirs = {"u1": u1, "mean_delta": mean_delta}
            for s, (s0, t, _) in vecs.items():
                dirs[f"{s}:start"], dirs[f"{s}:trained"], dirs[f"{s}:delta"] = s0, t, deltas[s]
            r["geometry"] = geometry(H, dirs)
    for r in res["runs"].values():
        r.pop("_u1"), r.pop("_deltas")
    path = Path(out) / "trained" / "analysis" / f"vecs_{'_'.join(runs)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, res)
    print(format_vecs(res), flush=True)
    log(f"vecs: wrote {path}")
    return res


def format_vecs(res: dict) -> str:
    L = []
    for run, r in res["runs"].items():
        L.append(f"== {run}")
        L.append(f"{'trait':<15} {'|t|/|s0|':>8} {'cos(t,s0)':>9} {'cos(sf,s0)':>10} {'delta':>6} {'along':>6} "
                 f"{'perp':>6} {'split':>6} {'u1:s0':>6} {'u1:t':>6}")
        for s, p in r["per_trait"].items():
            a = r["along_u1"][s]
            L.append(f"{s:<15} {p['norm_ratio']:>8.3f} {p['cos_trained_start']:>9.3f} {p['cos_source_start']:>10.3f} "
                     f"{p['delta_rel']:>6.3f} {p['delta_along_start_rel']:>+6.3f} {p['delta_perp_rel']:>6.3f} "
                     f"{p['split_noise_rel']:>6.3f} {a['start']:>+6.3f} {a['trained']:>+6.3f}")
        dp, nl = r["delta_perp"], r["split_noise_perp"]
        L.append(f"shared change (orthogonal part of t - s0, across traits): mean cos {dp['mean_cos']:+.3f}, "
                 f"mean |cos| {dp['mean_abs_cos']:.3f}, 1st component share {dp['sv_share'][0]:.3f}")
        L.append(f"  null, split noise sf - s0: mean cos {nl['mean_cos']:+.3f}, mean |cos| {nl['mean_abs_cos']:.3f}, "
                 f"1st share {nl['sv_share'][0]:.3f};  random: |cos| {r['random_abs_cos']:.3f}, "
                 f"share {r['equal_share']:.3f}")
        L.append("  each trait's change along u1: " + ", ".join(f"{s} {c:+.2f}" for s, c in dp["u1_cos"].items()))
        if "logit_lens" in r:
            for nm, ll in r["logit_lens"].items():
                L.append(f"  logit lens {nm} up:   {ll['up']}")
                L.append(f"  logit lens {nm} down: {ll['down']}")
            g = r["geometry"]
            ks = list(g["random_pc_share"])
            L.append(f"  residual geometry ({g['n_tokens']} tokens): energy in {'/'.join(ks)} PCs (random "
                     + "/".join(f"{g['random_pc_share'][k]:.4f}" for k in ks) + "), variance vs random")
            for nm, x in g.items():
                if isinstance(x, dict) and "var_vs_random" in x:
                    L.append(f"    {nm:<26} " + " ".join(f"{x[k]:.3f}" for k in ks)
                             + f"  var x{x['var_vs_random']:.2f}  cos(mean) {x['cos_mean_residual']:+.2f}")
    for pair, x in res["across_runs"].items():
        L.append(f"== {pair}: u1 cos {x['u1_cos']:+.3f}; per-trait delta cos "
                 + ", ".join(f"{s} {c:+.2f}" for s, c in x["delta_cos"].items()))
    return "\n".join(L)
