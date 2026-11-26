from __future__ import annotations

from contextlib import contextmanager

import torch
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.smoke import decoder_layers


@contextmanager
def steer(model, layer: int, vec: torch.Tensor | None):
    """Add `vec` (already scaled by c) to the output of decoder layer `layer` at all positions."""
    if vec is None:
        yield
        return
    hidden = model.get_input_embeddings().weight.shape[1]
    assert vec.shape == (hidden,), f"vec shape {tuple(vec.shape)} != ({hidden},)"

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            return (h + vec.to(device=h.device, dtype=h.dtype),) + tuple(output[1:])
        return output + vec.to(device=output.device, dtype=output.dtype)

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def layer_type(model, layer: int) -> str:
    cfg = model.config
    for c in (cfg, getattr(cfg, "text_config", None)):
        types = getattr(c, "layer_types", None) if c is not None else None
        if types:
            return types[layer]
    return "full_attention"


def encode(tok, question: str, answer: str, max_answer_tokens: int = 512):
    """Return (ids [1, T], start): ids[0, start:] are the answer's first <= max_answer_tokens tokens."""
    prompt = chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, question, False)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"][:max_answer_tokens]
    if len(answer_ids) == 0:
        raise ValueError("answer has 0 tokens")
    return torch.tensor([prompt_ids + answer_ids]), len(prompt_ids)


def answer_logits(model, ids: torch.Tensor, start: int) -> torch.Tensor:
    """Raw logits [T_answer, vocab] (model dtype) of the positions that predict ids[0, start:]."""
    assert ids.shape[0] == 1, "batch of one expected"
    T = ids.shape[1]
    # use_cache=False: one full pass per call, no KV cache kept alive (training / eval, never generate)
    return model(ids.to(model.device), logits_to_keep=T - start + 1, use_cache=False).logits[0, :-1]


def answer_logprobs(model, ids: torch.Tensor, start: int) -> torch.Tensor:
    """fp32 log-softmax [T_answer, vocab] of the positions that predict ids[0, start:]."""
    return torch.log_softmax(answer_logits(model, ids, start).float(), dim=-1)


LOGPROB_NORMS = ("sum", "mean")


def answer_logprob(model, ids: torch.Tensor, start: int, norm: str = "sum") -> torch.Tensor:
    """Scalar: summed log p of the answer tokens ("sum", BiPO's default) or that sum / answer tokens ("mean")."""
    if norm not in LOGPROB_NORMS:
        raise ValueError(f"unknown logprob norm {norm!r}")
    lp = answer_logprobs(model, ids, start)
    tgt = ids[0, start:].to(lp.device)
    total = lp.gather(1, tgt[:, None]).sum()
    return total / tgt.shape[0] if norm == "mean" else total
