from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers import LlamaConfig

from persvecfast.smoke import decoder_layers
from persvectrain.steering import (
    answer_logprob,
    answer_logprobs,
    encode,
    layer_type,
    steer,
)

Q, A = "What is 2+2?", "Four, obviously."


def _ids(tok):
    return encode(tok, Q, A)


def _hs(model, ids):
    # HF's own hidden_states capture hook runs before ours, so read layer outputs with a hook
    # registered after steer's (hooks fire in registration order); hs[k + 1] = layer k's output.
    outs = {}
    handles = [
        layer.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k + 1, o[0] if isinstance(o, tuple) else o))
        for k, layer in enumerate(decoder_layers(model))
    ]
    try:
        with torch.no_grad():
            h0 = model(ids, output_hidden_states=True).hidden_states[0]
    finally:
        for h in handles:
            h.remove()
    outs[0] = h0
    return outs


def test_steer_shifts_hidden_state(tiny_model, char_tok):
    ids, _ = _ids(char_tok)
    v = torch.randn(16)
    base = _hs(tiny_model, ids)
    with steer(tiny_model, 1, v):
        hs = _hs(tiny_model, ids)
    assert torch.allclose(hs[1], base[1], atol=1e-6)
    assert torch.allclose(hs[2], base[2] + v, atol=1e-5)


def test_steer_none_and_zero_are_noops(tiny_model, char_tok):
    ids, _ = _ids(char_tok)
    with torch.no_grad():
        base = tiny_model(ids).logits
        with steer(tiny_model, 1, None):
            a = tiny_model(ids).logits
        with steer(tiny_model, 1, torch.zeros(16)):
            b = tiny_model(ids).logits
    assert torch.equal(a, base)
    assert torch.allclose(b, base, atol=0)


def test_hook_removed_after_exception(tiny_model, char_tok):
    ids, _ = _ids(char_tok)
    with torch.no_grad():
        base = tiny_model(ids).logits
    with pytest.raises(RuntimeError):
        with steer(tiny_model, 1, torch.ones(16)):
            raise RuntimeError("boom")
    with torch.no_grad():
        assert torch.allclose(tiny_model(ids).logits, base, atol=0)


def test_encode_answer_boundary(char_tok):
    ids, start = encode(char_tok, Q, A)
    assert ids.shape[0] == 1
    assert char_tok.decode(ids[0, start:]) == A
    ids, start = encode(char_tok, Q, "x" * 600)
    assert ids.shape[1] - start == 512
    with pytest.raises(ValueError):
        encode(char_tok, Q, "")


def test_answer_logprob_matches_manual(tiny_model, char_tok):
    ids, start = _ids(char_tok)
    with torch.no_grad():
        logits = tiny_model(ids).logits
        lp = torch.log_softmax(logits[0].float(), -1)
        manual = sum(lp[t - 1, ids[0, t]] for t in range(start, ids.shape[1]))
        got = answer_logprob(tiny_model, ids, start)
        full = answer_logprobs(tiny_model, ids, start)
    assert torch.allclose(got, manual, atol=1e-4)
    assert full.shape == (ids.shape[1] - start, 128)
    assert full.dtype == torch.float32
    assert torch.allclose(full, lp[start - 1 : -1], atol=1e-5)


def test_answer_logprob_has_grad(tiny_model, char_tok):
    ids, start = _ids(char_tok)
    v = torch.zeros(16, requires_grad=True)
    with steer(tiny_model, 1, v):
        answer_logprob(tiny_model, ids, start).backward()
    assert v.grad is not None and v.grad.abs().sum() > 0


def test_tuple_output_layer():
    class Block(nn.Module):
        def forward(self, x):
            return (x * 2,)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.emb = nn.Embedding(4, 16)

        def get_input_embeddings(self):
            return self.emb

    m, v, x = Fake(), torch.randn(16), torch.randn(1, 3, 16)
    with steer(m, 1, v):
        out = m.model.layers[1](x)
    assert isinstance(out, tuple) and len(out) == 1
    assert torch.allclose(out[0], x * 2 + v, atol=1e-6)
    assert torch.equal(m.model.layers[1](x)[0], x * 2)


def test_layer_type(tiny_model):
    assert layer_type(tiny_model, 1) == "full_attention"

    class M:
        config = LlamaConfig(layer_types=["linear_attention", "full_attention", "linear_attention"])

    assert layer_type(M(), 0) == "linear_attention"
