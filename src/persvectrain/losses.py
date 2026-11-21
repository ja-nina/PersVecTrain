"""Losses: BiPO persona loss, KL to the unsteered model, overlap penalty."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def persona_margin(lp_c_pos: Tensor, lp_0_pos: Tensor, lp_c_neg: Tensor, lp_0_neg: Tensor,
                   sign: int) -> Tensor:
    """sign * ((lp_c_pos - lp_0_pos) - (lp_c_neg - lp_0_neg)); inputs are summed answer log-probs."""
    return sign * ((lp_c_pos - lp_0_pos) - (lp_c_neg - lp_0_neg))


def persona_loss(lp_c_pos: Tensor, lp_0_pos: Tensor, lp_c_neg: Tensor, lp_0_neg: Tensor,
                 sign: int, beta: float = 0.1) -> Tensor:
    """BiPO loss: -logsigmoid(beta * margin)."""
    margin = persona_margin(lp_c_pos, lp_0_pos, lp_c_neg, lp_0_neg, sign)
    return -F.logsigmoid(beta * margin)


def kl_loss(logp_0: Tensor, logp_c: Tensor) -> Tensor:
    """Mean over rows of KL(p_0 || p_c); both are [T, V] log-probabilities."""
    assert logp_0.shape == logp_c.shape, f"shape mismatch {logp_0.shape} vs {logp_c.shape}"
    return (logp_0.exp() * (logp_0 - logp_c)).sum(dim=-1).mean()


def overlap_loss(vecs: Tensor) -> Tensor:
    """Sum over i < j of cos(v_i, v_j) ** 2 for vecs [k, d]; 0 for one vector."""
    k = vecs.shape[0]
    if k < 2:
        return vecs.sum() * 0.0
    unit = F.normalize(vecs, dim=-1)
    cos = unit @ unit.T
    return (torch.triu(cos, diagonal=1) ** 2).sum()
