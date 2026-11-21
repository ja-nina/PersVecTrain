from __future__ import annotations

import math

import pytest
import torch

from persvectrain.losses import kl_loss, overlap_loss, persona_loss, persona_margin


def _t(x):
    return torch.tensor(float(x))


def test_margin_formula_and_sign_flip():
    args = (_t(-1.0), _t(-3.0), _t(-5.0), _t(-4.0))  # (2) - (-1) = 3
    assert persona_margin(*args, sign=1).item() == pytest.approx(3.0)
    assert persona_margin(*args, sign=-1).item() == pytest.approx(-3.0)


def test_loss_at_zero_margin_is_log2():
    z = _t(0.0)
    loss = persona_loss(z, z, z, z, sign=1)
    assert loss.item() == pytest.approx(math.log(2))


def test_loss_sign_flips_the_margin():
    args = (_t(-1.0), _t(-3.0), _t(-5.0), _t(-4.0))
    good = persona_loss(*args, sign=1, beta=0.1)
    bad = persona_loss(*args, sign=-1, beta=0.1)
    assert good.item() < math.log(2) < bad.item()
    assert good.item() == pytest.approx(-torch.nn.functional.logsigmoid(_t(0.3)).item())


def test_kl_zero_for_equal_and_positive_otherwise():
    lp0 = torch.log_softmax(torch.randn(4, 7), dim=-1)
    assert kl_loss(lp0, lp0.clone()).item() == pytest.approx(0.0, abs=1e-6)
    lpc = torch.log_softmax(torch.randn(4, 7), dim=-1)
    assert kl_loss(lp0, lpc).item() > 0


def test_kl_shape_mismatch_asserts():
    a = torch.log_softmax(torch.randn(4, 7), dim=-1)
    b = torch.log_softmax(torch.randn(3, 7), dim=-1)
    with pytest.raises(AssertionError):
        kl_loss(a, b)


def test_overlap_orthogonal_parallel_single():
    assert overlap_loss(torch.eye(3)).item() == pytest.approx(0.0, abs=1e-6)
    v = torch.tensor([1.0, 2.0, 3.0])
    assert overlap_loss(torch.stack([v, 2 * v])).item() == pytest.approx(1.0, abs=1e-5)
    one = overlap_loss(v[None, :])
    assert one.item() == 0.0 and one.dtype == v.dtype and one.dim() == 0


def test_overlap_gradient_nonzero():
    vecs = torch.tensor([[1.0, 0.0], [1.0, 1.0]], requires_grad=True)
    overlap_loss(vecs).backward()
    assert vecs.grad is not None and vecs.grad.abs().sum().item() > 0
