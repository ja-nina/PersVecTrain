"""Training loop: one steering vector per trait on a frozen model, BiPO persona loss + lambda_kl * KL."""
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
from persvectrain.losses import kl_loss, persona_loss
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
    model.requires_grad_(False)
    if cfg.mode == "solo":
        return _train_solo(model, tok, cfg, traits, kl_train, log_step)
    if cfg.mode == "joint":
        raise NotImplementedError("joint mode is added in Task 6")
    raise ValueError(f"unknown mode {cfg.mode!r}")


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
    """Steered pass on one pair (pos + neg, one backward); returns the unscaled persona loss."""
    ids_p, s_p, ids_n, s_n = enc
    with steer(model, layer, sign * v):
        lpc_p = answer_logprob(model, ids_p, s_p)
        lpc_n = answer_logprob(model, ids_n, s_n)
    loss = persona_loss(lpc_p, lp0[0], lpc_n, lp0[1], sign, beta)
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


def _train_solo(model, tok, cfg, traits, kl_train, log_step) -> dict[str, Tensor]:
    device = model.device
    use_kl = cfg.lambda_kl != 0
    if use_kl and not kl_train:
        raise ValueError("lambda_kl != 0 needs kl_train reference answers")
    kl_enc = [encode(tok, r.question, r.answer_text, cfg.max_answer_tokens) for r in kl_train] if use_kl else []

    out: dict[str, Tensor] = {}
    step = 0
    for slug in cfg.traits:
        td = traits[slug]
        if not td.train:
            raise ValueError(f"trait {slug} has no train pairs")
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
                sign = 1 if int(torch.randint(0, 2, (1,), generator=sign_gen)) == 1 else -1
                norm = float(v.detach().norm())
                lr = float(sched.get_last_lr()[0])
                # L = mean persona loss over this step's pairs + lambda_kl * mean KL over its KL answers;
                # each sequence backpropagates its share of L separately.
                p_scale = 1.0 / len(idx)
                kl_scale = cfg.lambda_kl / cfg.kl_batch
                opt.zero_grad(set_to_none=True)

                lp_sum = 0.0
                for i in idx:
                    lp_sum += _persona_backward(model, cfg.layer, v, sign, enc[i], lp0[i], cfg.beta, p_scale)
                kl_sum = 0.0
                if use_kl:
                    for _ in range(cfg.kl_batch):
                        kl_sum += _kl_backward(model, cfg.layer, v, sign, kl_enc[kl_cursor % len(kl_enc)],
                                               kl_scale)
                        kl_cursor += 1

                opt.step()
                sched.step()
                log_step({
                    "step": step,
                    "trait": slug,
                    "sign": sign,
                    "loss_p": lp_sum / len(idx),
                    "loss_kl": kl_sum / cfg.kl_batch if use_kl else 0.0,
                    "norm": norm,
                    "lr": lr,
                })
                step += 1
        out[slug] = v.detach().float().cpu().clone()
    return out
