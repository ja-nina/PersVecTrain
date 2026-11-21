from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


class CharTokenizer:
    """Character-level stand-in tokenizer: id = ord(c) % 128."""

    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        system = "".join(m["content"] for m in messages if m["role"] == "system")
        user = "".join(m["content"] for m in messages if m["role"] == "user")
        return f"<s>{system}<u>{user}<a>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 128 for c in text]}

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def make_tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = LlamaForCausalLM(cfg)
    model.requires_grad_(False)
    return model.eval()


@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def char_tok():
    return CharTokenizer()
