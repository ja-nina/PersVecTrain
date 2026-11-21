from __future__ import annotations

import torch


def test_tiny_model_runs(tiny_model):
    ids = torch.tensor([[1, 2, 3]])
    out = tiny_model(ids, output_hidden_states=True)
    assert out.logits.shape == (1, 3, 128)
    assert len(out.hidden_states) == 4
