"""Training loop: one steering vector per trait on a frozen model, BiPO persona loss + lambda_kl * KL.

Joint mode trains all trait vectors with one optimiser, adds lambda_o * overlap penalty and 2-trait mix steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from persvecfast.models import request_seed
from transformers import get_cosine_schedule_with_warmup

from persvectrain.data import Pair
from persvectrain.losses import kl_from_logits, overlap_loss, persona_loss, persona_margin
from persvectrain.steering import LOGPROB_NORMS, answer_logits, answer_logprob, checkpoint_layers, encode, steer


@dataclass
class TrainConfig:
    model: str
    layer: int
    mode: str  # "solo" | "joint"
    traits: list[str]
    source: Path
    out: Path
    run_name: str
    lambda_kl: float = 1.0
    lambda_o: float = 0.0
    epochs: int = 20
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.1
    beta: float = 0.1
    batch: int = 4
    kl_batch: int = 4
    max_answer_tokens: int = 512
    kl_tokens: int = 512  # KL answers are cut to this many tokens; picks the reference file <model>_pool260_t{N}
    init: str = "dom"  # "dom" | "zero"
    mix_every: int = 2
    seed: int = 0
    logprob_norm: str = "sum"  # "sum" (BiPO) | "mean": each answer's summed log-prob / its answer tokens
    min_pairs: int = 20  # traits with fewer kept pairs are dropped (as `check`)
    fix_norm: int = 0  # 1: after every optimizer step |v| is set back to |start| (direction-only training)
    grad_checkpoint: int = 0  # 1: layers after `layer` recompute activations in backward (same result, less memory)


@dataclass
class TraitData:
    slug: str
    train: list[Pair]
    held: list[Pair]
    start: Tensor
    source: Tensor


@dataclass
class RefAnswer:
    id: str
    question: str
    answer_text: str
    gold: str


def train(model, tok, cfg: TrainConfig, traits: dict[str, TraitData], kl_train: list[RefAnswer],
          log_step: Callable[[dict], None]) -> dict[str, Tensor]:
    """Train one vector per trait in cfg.traits; returns {slug: fp32 cpu vector}. The model stays frozen."""
    if cfg.init not in ("dom", "zero"):
        raise ValueError(f"unknown init {cfg.init!r}")
    if cfg.fix_norm and cfg.init != "dom":
        raise ValueError("fix_norm needs init dom (the start vector's length)")
    if cfg.mode not in ("solo", "joint"):
        raise ValueError(f"unknown mode {cfg.mode!r}")
    if cfg.logprob_norm not in LOGPROB_NORMS:
        raise ValueError(f"unknown logprob_norm {cfg.logprob_norm!r}")
    if cfg.lambda_kl != 0:
        if not kl_train:
            raise ValueError("lambda_kl != 0 needs kl_train reference answers")
        if cfg.kl_batch < 1:
            raise ValueError(f"lambda_kl != 0 needs kl_batch >= 1, got {cfg.kl_batch}")
    for slug in cfg.traits:
        if not traits[slug].train:
            raise ValueError(f"trait {slug} has no train pairs")
    model.requires_grad_(False)
    model.eval()
    with checkpoint_layers(model, cfg.layer, on=bool(cfg.grad_checkpoint)):
        if cfg.mode == "solo":
            return _train_solo(model, tok, cfg, traits, kl_train, log_step)
        return _train_joint(model, tok, cfg, traits, kl_train, log_step)


# ---------------------------------------------------------------- shared pieces


def _init_vector(cfg: TrainConfig, td: TraitData, device) -> nn.Parameter:
    start = td.start.detach().float()
    init = start.clone() if cfg.init == "dom" else torch.zeros_like(start)
    return nn.Parameter(init.to(device))


def _encode_pairs(tok, pairs: list[Pair], max_tok: int) -> list[tuple]:
    """[(ids_pos, start_pos, ids_neg, start_neg)] per pair."""
    return [encode(tok, p.question, p.pos, max_tok) + encode(tok, p.question, p.neg, max_tok) for p in pairs]


@torch.no_grad()
def _unsteered_pair_logprobs(model, enc: list[tuple], norm: str = "sum") -> list[tuple[Tensor, Tensor]]:
    """Unsteered answer log-probs (pos, neg) per encoded pair (`norm` as answer_logprob), computed once."""
    return [(answer_logprob(model, ip, sp, norm).detach(), answer_logprob(model, in_, sn, norm).detach())
            for ip, sp, in_, sn in enc]


def _cos_start(v: Tensor, start: Tensor | None) -> float | None:
    """Cosine of v with its start vector; None without one (zero init) or while v is zero."""
    if start is None:
        return None
    v = v.detach().float()
    if float(v.norm()) == 0.0:
        return None
    return float(F.cosine_similarity(v, start.to(v.device).float(), dim=0))


def _start_of(cfg: TrainConfig, td: TraitData) -> Tensor | None:
    return td.start.detach().float() if cfg.init == "dom" else None


@torch.no_grad()
def _fix_norm(cfg: TrainConfig, v: Tensor, start: Tensor | None) -> None:
    """--fix-norm 1: rescale v in place to the start vector's length (only the direction is trained)."""
    if cfg.fix_norm:
        v.mul_(start.norm().to(v.device) / v.norm().clamp_min(1e-12))


@torch.no_grad()
def _start_entry(model, cfg: TrainConfig, step: int, vecs: dict[str, Tensor], enc: dict[str, list[tuple]],
                 lp0: dict[str, list[tuple]]) -> dict:
    """The "start" log line: per trait, the mean train persona margin of the starting vector at c = +1 / -1
    and beta * margin (the BiPO sigmoid saturates once beta * margin is large: ~5 gives a gradient of ~0.007)."""
    traits = {}
    for slug, v in vecs.items():
        row = {"norm": float(v.detach().norm())}
        for sign, key in ((1, "pos"), (-1, "neg")):
            ms = []
            for (ids_p, s_p, ids_n, s_n), ref in zip(enc[slug], lp0[slug]):
                with steer(model, cfg.layer, sign * v.detach()):
                    lpc_p = answer_logprob(model, ids_p, s_p, cfg.logprob_norm)
                    lpc_n = answer_logprob(model, ids_n, s_n, cfg.logprob_norm)
                ms.append(float(persona_margin(lpc_p, ref[0], lpc_n, ref[1], sign)))
            m = sum(ms) / len(ms)
            row[f"margin_{key}"] = m
            row[f"beta_margin_{key}"] = cfg.beta * m
        traits[slug] = row
    return {"step": step, "kind": "start", "trait": "+".join(vecs), "init": cfg.init, "beta": cfg.beta,
            "logprob_norm": cfg.logprob_norm, "traits": traits}


def _optimizer(cfg: TrainConfig, params: list[nn.Parameter], total: int):
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(opt, round(cfg.warmup_frac * total), total)
    return opt, sched


def _generator(cfg: TrainConfig, slug: str, purpose: str) -> torch.Generator:
    """Seeded from cfg.seed and the trait slug (not its position), so reordering traits keeps each vector."""
    return torch.Generator().manual_seed(request_seed(cfg.seed, slug, purpose))


def _epoch_batches(n: int, batch: int, gen: torch.Generator) -> list[list[int]]:
    order = torch.randperm(n, generator=gen).tolist()
    return [order[i:i + batch] for i in range(0, n, batch)]


def _persona_term(model, cfg: TrainConfig, vec: Tensor, sign: int, enc: tuple, ref: tuple[Tensor, Tensor],
                  scale: float) -> tuple[float, float]:
    """Persona loss of one pair (pos + neg, one backward) steered by `vec` against reference log-probs `ref`.

    Returns the unscaled (loss, margin)."""
    ids_p, s_p, ids_n, s_n = enc
    with steer(model, cfg.layer, vec):
        lpc_p = answer_logprob(model, ids_p, s_p, cfg.logprob_norm)
        lpc_n = answer_logprob(model, ids_n, s_n, cfg.logprob_norm)
    margin = persona_margin(lpc_p, ref[0], lpc_n, ref[1], sign)
    loss = persona_loss(lpc_p, ref[0], lpc_n, ref[1], sign, cfg.beta)
    (loss * scale).backward()
    return float(loss.detach()), float(margin.detach())


def _kl_backward(model, layer: int, v: Tensor, sign: int, enc: tuple, scale: float) -> float:
    """Unsteered reference under no_grad, steered pass, KL(p_0 || p_c) backward; returns unscaled KL."""
    ids, s = enc
    with torch.no_grad():
        l0 = answer_logits(model, ids, s)
    with steer(model, layer, sign * v):
        lc = answer_logits(model, ids, s)
    kl = kl_from_logits(l0, lc)  # chunked kl_loss: no [T, V] fp32 tensors kept (long windows, large vocab)
    (kl * scale).backward()
    return float(kl.detach())


# ---------------------------------------------------------------- solo


def _draw_sign(gen: torch.Generator) -> int:
    return 1 if int(torch.randint(0, 2, (1,), generator=gen)) == 1 else -1


def _kl_terms(model, cfg: TrainConfig, vec: Tensor, sign: int, kl_enc: list[tuple],
              cursor: int) -> tuple[float, int]:
    """kl_batch KL backwards under sign * vec, each scaled lambda_kl / kl_batch; returns (KL sum, new cursor)."""
    kl_scale = cfg.lambda_kl / cfg.kl_batch
    kl_sum = 0.0
    for _ in range(cfg.kl_batch):
        kl_sum += _kl_backward(model, cfg.layer, vec, sign, kl_enc[cursor % len(kl_enc)], kl_scale)
        cursor += 1
    return kl_sum, cursor


def _single_terms(model, cfg: TrainConfig, v: Tensor, sign: int, idx: list[int], enc: list[tuple],
                  lp0: list[tuple], kl_enc: list[tuple], kl_cursor: int) -> tuple[float, float, float, int]:
    """One trait's step terms (backward only): L = mean persona over idx + lambda_kl * mean KL.

    Each sequence backpropagates its share of L separately.
    Returns (mean persona loss, mean persona margin, mean KL, new KL cursor).
    """
    p_scale = 1.0 / len(idx)
    lp_sum = 0.0
    m_sum = 0.0
    for i in idx:
        loss, margin = _persona_term(model, cfg, sign * v, sign, enc[i], lp0[i], p_scale)
        lp_sum += loss
        m_sum += margin
    if cfg.lambda_kl == 0:
        return lp_sum / len(idx), m_sum / len(idx), 0.0, kl_cursor
    kl_sum, kl_cursor = _kl_terms(model, cfg, v, sign, kl_enc, kl_cursor)
    return lp_sum / len(idx), m_sum / len(idx), kl_sum / cfg.kl_batch, kl_cursor


def _encode_kl(tok, cfg: TrainConfig, kl_train: list[RefAnswer]) -> list[tuple]:
    if cfg.lambda_kl == 0:
        return []
    return [encode(tok, r.question, r.answer_text, cfg.kl_tokens) for r in kl_train]


def _train_solo(model, tok, cfg, traits, kl_train, log_step) -> dict[str, Tensor]:
    device = model.device
    kl_enc = _encode_kl(tok, cfg, kl_train)

    out: dict[str, Tensor] = {}
    step = 0
    for slug in cfg.traits:
        td = traits[slug]
        enc = _encode_pairs(tok, td.train, cfg.max_answer_tokens)
        lp0 = _unsteered_pair_logprobs(model, enc, cfg.logprob_norm)

        v = _init_vector(cfg, td, device)
        start = _start_of(cfg, td)
        log_step(_start_entry(model, cfg, step, {slug: v}, {slug: enc}, {slug: lp0}))
        per_epoch = math.ceil(len(td.train) / cfg.batch)
        total = cfg.epochs * per_epoch
        opt, sched = _optimizer(cfg, [v], total)
        order_gen = _generator(cfg, slug, "shuffle")
        sign_gen = _generator(cfg, slug, "sign")
        kl_cursor = 0

        for _epoch in range(cfg.epochs):
            for idx in _epoch_batches(len(td.train), cfg.batch, order_gen):
                sign = _draw_sign(sign_gen)
                norm = float(v.detach().norm())
                cos = _cos_start(v, start)
                lr = float(sched.get_last_lr()[0])
                opt.zero_grad(set_to_none=True)
                loss_p, margin, loss_kl, kl_cursor = _single_terms(model, cfg, v, sign, idx, enc, lp0, kl_enc,
                                                                   kl_cursor)
                opt.step()
                _fix_norm(cfg, v, start)
                sched.step()
                log_step(_single_log(step, slug, sign, loss_p, margin, loss_kl, 0.0, norm, cos, lr))
                step += 1
        out[slug] = v.detach().float().cpu().clone()
    return out


def _single_log(step: int, slug: str, sign: int, loss_p: float, margin: float, loss_kl: float, loss_o: float,
                norm: float, cos_start: float | None, lr: float) -> dict:
    """`norm` and `cos_start` are v's before the step; `margin` is the step's mean persona margin."""
    return {"step": step, "kind": "single", "trait": slug, "sign": sign, "weights": {slug: sign},
            "loss_p": loss_p, "margin": margin, "loss_kl": loss_kl, "loss_o": loss_o, "norm": norm,
            "cos_start": cos_start, "lr": lr}


# ---------------------------------------------------------------- joint


def _joint_total(n_single: int, mix_every: int) -> int:
    """Steps until all single batches are consumed; step s is a mix step iff s % mix_every == mix_every - 1.

    mix_every == 1 makes every step a mix step; the run then has n_single steps (one per single batch).
    """
    if mix_every <= 1:
        return n_single
    return n_single + (n_single - 1) // (mix_every - 1)


def _is_mix(step: int, mix_every: int) -> bool:
    return mix_every > 0 and step % mix_every == mix_every - 1


def _overlap_term(cfg: TrainConfig, vs: list[nn.Parameter]) -> float:
    """lambda_o * sum_{i<j} cos^2 backward over all vectors; 0.0 (no backward) when off or one trait."""
    if cfg.lambda_o == 0 or len(vs) < 2:
        return 0.0
    loss = overlap_loss(torch.stack(vs))
    (cfg.lambda_o * loss).backward()
    return float(loss.detach())


def _train_joint(model, tok, cfg, traits, kl_train, log_step) -> dict[str, Tensor]:
    device = model.device
    slugs = list(cfg.traits)
    if cfg.mix_every > 0 and len(slugs) < 2:
        raise ValueError("mix steps need at least 2 traits (set mix_every = 0)")
    kl_enc = _encode_kl(tok, cfg, kl_train)

    enc = {s: _encode_pairs(tok, traits[s].train, cfg.max_answer_tokens) for s in slugs}
    lp0 = {s: _unsteered_pair_logprobs(model, enc[s], cfg.logprob_norm) for s in slugs}
    vs = {s: _init_vector(cfg, traits[s], device) for s in slugs}
    starts = {s: _start_of(cfg, traits[s]) for s in slugs}
    log_step(_start_entry(model, cfg, 0, vs, enc, lp0))
    all_vs = [vs[s] for s in slugs]
    sign_gen = {s: _generator(cfg, s, "sign") for s in slugs}
    order_gen = {s: _generator(cfg, s, "shuffle") for s in slugs}

    # single batches: per epoch, every trait's own seeded shuffle (as solo), interleaved round-robin
    singles: list[tuple[str, list[int]]] = []
    for _epoch in range(cfg.epochs):
        per_trait = {s: _epoch_batches(len(traits[s].train), cfg.batch, order_gen[s]) for s in slugs}
        for j in range(max(len(b) for b in per_trait.values())):
            singles.extend((s, per_trait[s][j]) for s in slugs if j < len(per_trait[s]))
    total = _joint_total(len(singles), cfg.mix_every)
    opt, sched = _optimizer(cfg, all_vs, total)
    mix_gen = torch.Generator().manual_seed(request_seed(cfg.seed, "mix"))
    kl_cursor = 0
    single_i = 0

    for step in range(total):
        lr = float(sched.get_last_lr()[0])
        opt.zero_grad(set_to_none=True)
        if _is_mix(step, cfg.mix_every):
            entry, kl_cursor = _mix_step(model, cfg, slugs, traits, vs, starts, enc, kl_enc, kl_cursor,
                                         mix_gen)
            loss_o = _overlap_term(cfg, all_vs)
            entry = {"step": step, **entry, "loss_o": loss_o, "lr": lr}
        else:
            slug, idx = singles[single_i]
            single_i += 1
            v = vs[slug]
            sign = _draw_sign(sign_gen[slug])
            norm = float(v.detach().norm())
            cos = _cos_start(v, starts[slug])
            loss_p, margin, loss_kl, kl_cursor = _single_terms(model, cfg, v, sign, idx, enc[slug], lp0[slug],
                                                               kl_enc, kl_cursor)
            loss_o = _overlap_term(cfg, all_vs)
            entry = _single_log(step, slug, sign, loss_p, margin, loss_kl, loss_o, norm, cos, lr)
        opt.step()
        for s in slugs:
            _fix_norm(cfg, vs[s], starts[s])
        sched.step()
        log_step(entry)
    return {s: vs[s].detach().float().cpu().clone() for s in slugs}


def _mix_step(model, cfg: TrainConfig, slugs: list[str], traits: dict[str, TraitData],
              vs: dict[str, nn.Parameter], starts: dict[str, Tensor | None], enc: dict[str, list[tuple]],
              kl_enc: list[tuple], kl_cursor: int, gen: torch.Generator) -> tuple[dict, int]:
    """Backward of one 2-trait mix step; returns (partial log entry, new KL cursor).

    Steered by w_A v_A + w_B v_B. Trait i's persona term uses its own sampled pairs, sign(w_i), the other
    trait's vector detached, and as reference the model steered by the other trait's term only (no_grad).
    KL is under the full mix against the unsteered model, gradients to both vectors.
    """
    pick = torch.randperm(len(slugs), generator=gen)[:2].tolist()
    mix = [slugs[k] for k in pick]
    weights: dict[str, float] = {}
    for s in mix:
        mag = 0.25 + 0.75 * float(torch.rand((), generator=gen))
        weights[s] = mag * _draw_sign(gen)
    idx = {s: torch.randperm(len(traits[s].train), generator=gen)[:cfg.batch].tolist() for s in mix}
    norms = {s: float(vs[s].detach().norm()) for s in mix}
    cos_starts = {s: _cos_start(vs[s], starts[s]) for s in mix}
    n_pairs = sum(len(i) for i in idx.values())
    p_scale = 1.0 / n_pairs

    lp_sum = 0.0
    m_sum = 0.0
    for s in mix:
        (other,) = [o for o in mix if o != s]
        other_vec = (weights[other] * vs[other]).detach()
        sign = 1 if weights[s] > 0 else -1
        refs = []
        with torch.no_grad():
            for i in idx[s]:
                ids_p, s_p, ids_n, s_n = enc[s][i]
                with steer(model, cfg.layer, other_vec):
                    refs.append((answer_logprob(model, ids_p, s_p, cfg.logprob_norm),
                                 answer_logprob(model, ids_n, s_n, cfg.logprob_norm)))
        for i, ref in zip(idx[s], refs):
            vec = weights[s] * vs[s] + other_vec
            loss, margin = _persona_term(model, cfg, vec, sign, enc[s][i], ref, p_scale)
            lp_sum += loss
            m_sum += margin

    loss_kl = 0.0
    if cfg.lambda_kl != 0:
        kl_scale = cfg.lambda_kl / cfg.kl_batch
        kl_sum = 0.0
        for _ in range(cfg.kl_batch):
            full = weights[mix[0]] * vs[mix[0]] + weights[mix[1]] * vs[mix[1]]  # fresh graph per backward
            kl_sum += _kl_backward(model, cfg.layer, full, 1, kl_enc[kl_cursor % len(kl_enc)], kl_scale)
            kl_cursor += 1
        loss_kl = kl_sum / cfg.kl_batch
    entry = {"kind": "mix", "trait": "+".join(mix), "sign": 0, "weights": weights, "loss_p": lp_sum / n_pairs,
             "margin": m_sum / n_pairs, "loss_kl": loss_kl, "norm": norms[mix[0]], "norms": norms,
             "cos_start": cos_starts[mix[0]], "cos_starts": cos_starts}
    return entry, kl_cursor
