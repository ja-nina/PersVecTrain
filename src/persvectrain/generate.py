"""Batched HF sampling for `reference` (cli). The free-generation check runs on vLLM (freegen.py)."""
from __future__ import annotations

import torch
from persvecfast.util import log


def stop_ids(model, tok) -> set[int]:
    """Token ids that end a generation: the tokenizer's eos / pad and the generation config's eos."""
    ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    for x in (tok.eos_token_id, tok.pad_token_id, getattr(gen_cfg, "eos_token_id", None)):
        if x is None:
            continue
        ids.update(int(i) for i in (x if isinstance(x, (list, tuple)) else [x]))
    return ids


def count_new_tokens(new: torch.Tensor, stops: set[int]) -> list[int]:
    """Per row of generated ids [B, L]: tokens before the first stop id (L when none, i.e. cut)."""
    out = []
    for row in new.tolist():
        n = len(row)
        for i, t in enumerate(row):
            if t in stops:
                n = i
                break
        out.append(n)
    return out


@torch.no_grad()
def generate_counted(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int,
                     label: str = "reference", **sampling) -> tuple[list[str], list[int]]:
    """Batched HF sampling with left padding.

    Returns the decoded continuations (special tokens dropped) and the generated token count of each (tokens
    before the first eos / pad; == max_new_tokens when the answer was cut).
    """
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stops = stop_ids(model, tok)
    texts: list[str] = []
    counts: list[int] = []
    for i in range(0, len(prompts), batch_size):
        batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        gen = model.generate(**batch, do_sample=True, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, **sampling)
        new = gen[:, batch["input_ids"].shape[1]:]
        texts.extend(tok.batch_decode(new, skip_special_tokens=True))
        counts.extend(count_new_tokens(new, stops))
        log(f"{label}: {len(texts)}/{len(prompts)} answers")
    return texts, counts


def generate(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int, **sampling) -> list[str]:
    """generate_counted's texts only (the `reference` command's helper)."""
    return generate_counted(model, tok, prompts, batch_size, max_new_tokens, **sampling)[0]
