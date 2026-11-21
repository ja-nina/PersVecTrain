"""Training loop: one steering vector per trait on a frozen model, BiPO persona loss + lambda_kl * KL.

Joint mode trains all trait vectors with one optimiser, adds lambda_o * overlap penalty and 2-trait mix steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
from persvecfast.models import request_seed
from transformers import get_cosine_schedule_with_warmup

from persvectrain.data import Pair
from persvectrain.losses import kl_loss, overlap_loss, persona_loss
from persvectrain.steering import answer_logprob, answer_logprobs, encode, steer


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
    init: str = "dom"  # "dom" | "zero"
    mix_every: int = 2
    seed: int = 0


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
    if cfg.mode not in ("solo", "joint"):
        raise ValueError(f"unknown mode {cfg.mode!r}")
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
def _unsteered_pair_logprobs(model, enc: list[tuple]) -> list[tuple[Tensor, Tensor]]:
    """Summed unsteered answer log-probs (pos, neg) per encoded pair, computed once and reused."""
    return [(answer_logprob(model, ip, sp).detach(), answer_logprob(model, in_, sn).detach())
            for ip, sp, in_, sn in enc]


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


def _persona_backward(model, layer: int, v: Tensor, sign: int, enc: tuple, lp0: tuple[Tensor, Tensor],
                      beta: float, scale: float) -> float:
    """Steered pass on one pair under sign * v (pos + neg, one backward); returns the unscaled persona loss."""
    return _persona_term(model, layer, sign * v, sign, enc, lp0, beta, scale)


def _persona_term(model, layer: int, vec: Tensor, sign: int, enc: tuple, ref: tuple[Tensor, Tensor],
                  beta: float, scale: float) -> float:
    """Persona loss of one pair steered by `vec` against reference log-probs `ref`; backward, unscaled loss."""
    ids_p, s_p, ids_n, s_n = enc
    with steer(model, layer, vec):
        lpc_p = answer_logprob(model, ids_p, s_p)
        lpc_n = answer_logprob(model, ids_n, s_n)
    loss = persona_loss(lpc_p, ref[0], lpc_n, ref[1], sign, beta)
    (loss * scale).backward()
    return float(loss.detach())


def _kl_backward(model, layer: int, v: Tensor, sign: int, enc: tuple, scale: float) -> float:
    """Unsteered reference under no_grad, steered pass, KL(p_0 || p_c) backward; returns unscaled KL."""
    ids, s = enc
    with torch.no_grad():
        lp0 = answer_logprobs(model, ids, s)
    with steer(model, layer, sign * v):
        lpc = answer_logprobs(model, ids, s)
    kl = kl_loss(lp0, lpc)
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
                  lp0: list[tuple], kl_enc: list[tuple], kl_cursor: int) -> tuple[float, float, int]:
    """One trait's step terms (backward only): L = mean persona over idx + lambda_kl * mean KL.

    Each sequence backpropagates its share of L separately. Returns (mean persona, mean KL, new KL cursor).
    """
    p_scale = 1.0 / len(idx)
    lp_sum = 0.0
    for i in idx:
        lp_sum += _persona_backward(model, cfg.layer, v, sign, enc[i], lp0[i], cfg.beta, p_scale)
    if cfg.lambda_kl == 0:
        return lp_sum / len(idx), 0.0, kl_cursor
    kl_sum, kl_cursor = _kl_terms(model, cfg, v, sign, kl_enc, kl_cursor)
    return lp_sum / len(idx), kl_sum / cfg.kl_batch, kl_cursor


def _encode_kl(tok, cfg: TrainConfig, kl_train: list[RefAnswer]) -> list[tuple]:
    if cfg.lambda_kl == 0:
        return []
    return [encode(tok, r.question, r.answer_text, cfg.max_answer_tokens) for r in kl_train]


def _train_solo(model, tok, cfg, traits, kl_train, log_step) -> dict[str, Tensor]:
    device = model.device
    kl_enc = _encode_kl(tok, cfg, kl_train)

    out: dict[str, Tensor] = {}
    step = 0
    for slug in cfg.traits:
        td = traits[slug]
        enc = _encode_pairs(tok, td.train, cfg.max_answer_tokens)
        lp0 = _unsteered_pair_logprobs(model, enc)

        v = _init_vector(cfg, td, device)
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
                lr = float(sched.get_last_lr()[0])
                opt.zero_grad(set_to_none=True)
                loss_p, loss_kl, kl_cursor = _single_terms(model, cfg, v, sign, idx, enc, lp0, kl_enc, kl_cursor)
                opt.step()
                sched.step()
                log_step(_single_log(step, slug, sign, loss_p, loss_kl, 0.0, norm, lr))
                step += 1
        out[slug] = v.detach().float().cpu().clone()
    return out


def _single_log(step: int, slug: str, sign: int, loss_p: float, loss_kl: float, loss_o: float, norm: float,
                lr: float) -> dict:
    return {"step": step, "kind": "single", "trait": slug, "sign": sign, "weights": {slug: sign},
            "loss_p": loss_p, "loss_kl": loss_kl, "loss_o": loss_o, "norm": norm, "lr": lr}


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
    lp0 = {s: _unsteered_pair_logprobs(model, enc[s]) for s in slugs}
    vs = {s: _init_vector(cfg, traits[s], device) for s in slugs}
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
            entry, kl_cursor = _mix_step(model, cfg, slugs, traits, vs, enc, kl_enc, kl_cursor, mix_gen)
            loss_o = _overlap_term(cfg, all_vs)
            entry = {"step": step, **entry, "loss_o": loss_o, "lr": lr}
        else:
            slug, idx = singles[single_i]
            single_i += 1
            v = vs[slug]
            sign = _draw_sign(sign_gen[slug])
            norm = float(v.detach().norm())
            loss_p, loss_kl, kl_cursor = _single_terms(model, cfg, v, sign, idx, enc[slug], lp0[slug], kl_enc,
                                                       kl_cursor)
            loss_o = _overlap_term(cfg, all_vs)
            entry = _single_log(step, slug, sign, loss_p, loss_kl, loss_o, norm, lr)
        opt.step()
        sched.step()
        log_step(entry)
    return {s: vs[s].detach().float().cpu().clone() for s in slugs}


def _mix_step(model, cfg: TrainConfig, slugs: list[str], traits: dict[str, TraitData],
              vs: dict[str, nn.Parameter], enc: dict[str, list[tuple]], kl_enc: list[tuple], kl_cursor: int,
              gen: torch.Generator) -> tuple[dict, int]:
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
    n_pairs = sum(len(i) for i in idx.values())
    p_scale = 1.0 / n_pairs

    lp_sum = 0.0
    for s in mix:
        (other,) = [o for o in mix if o != s]
        other_vec = (weights[other] * vs[other]).detach()
        sign = 1 if weights[s] > 0 else -1
        refs = []
        with torch.no_grad():
            for i in idx[s]:
                ids_p, s_p, ids_n, s_n = enc[s][i]
                with steer(model, cfg.layer, other_vec):
                    refs.append((answer_logprob(model, ids_p, s_p), answer_logprob(model, ids_n, s_n)))
        for i, ref in zip(idx[s], refs):
            vec = weights[s] * vs[s] + other_vec
            lp_sum += _persona_term(model, cfg.layer, vec, sign, enc[s][i], ref, cfg.beta, p_scale)

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
             "loss_kl": loss_kl, "norm": norms[mix[0]], "norms": norms}
    return entry, kl_cursor
