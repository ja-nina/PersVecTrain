from __future__ import annotations

import json

import pytest
import torch

from persvectrain import cli
from persvectrain import freegen as fgm
from persvectrain import kldepth as kd
from persvectrain.losses import kl_loss
from persvectrain.steering import answer_logprobs, steer
from tests.conftest import CharTokenizer
from tests.test_freegen import HELD, LAYER, run  # noqa: F401  (the finished-run fixture)

# unsteered answers of different lengths (char tokenizer: one token per character)
TEXTS = {"s4": "a" * 3 + "Reasoning goes here. ANSWER: C", "s0": "short. ANSWER: C", "s3": "x" * 50}


def _answers(run, texts=TEXTS):
    lines = [{"key": ["-", "unsteered", 0.0, q], "text": t, "n_tokens": len(t), "finish_reason": "stop",
              "max_tokens": 500, "seed": 1} for q, t in texts.items()]
    # a steered answer for the same question must not be read as the unsteered one
    lines.append({"key": ["assertive", "trained", 1.0, "s4"], "text": "zzz", "n_tokens": 3, "finish_reason": "stop",
                  "max_tokens": 500, "seed": 2})
    fgm.write_answers(run["root"] / fgm.ANSWERS_FILE, lines)


def _kd(run, model, **kw):
    return kd.run_kldepth(model, CharTokenizer(), run["out"], "run1", pool=run["pool"], **kw)


def test_buckets():
    assert kd.buckets((5, 20, 40)) == [(0, 5), (5, 20), (20, 40)]


def test_bucket_means_none_past_the_answer():
    kls = torch.arange(10, dtype=torch.float)
    assert kd.bucket_means(kls, [(0, 5), (5, 20), (20, 40)]) == [2.0, 7.0, None]


def test_token_kls_match_the_training_kl(run, tiny_model):
    """Per-token KL from the final hidden states averages to evaluate's / training's kl_loss on the same text."""
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    start = 4
    v = 2 * torch.randn(16, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        h0 = kd._final_hidden(tiny_model, ids)
        with steer(tiny_model, LAYER, v):
            hc = kd._final_hidden(tiny_model, ids)
        kls = kd.token_kls(tiny_model, h0, hc, start)
        lp0 = answer_logprobs(tiny_model, ids, start)
        with steer(tiny_model, LAYER, v):
            lpc = answer_logprobs(tiny_model, ids, start)
    assert kls.shape == (ids.shape[1] - start,)
    assert float(kls.mean()) == pytest.approx(float(kl_loss(lp0, lpc)), rel=1e-4, abs=1e-6)
    assert float(kls.mean()) > 0


def test_token_kls_row_chunks_agree(run, tiny_model, monkeypatch):
    ids = torch.randint(1, 120, (1, 40), generator=torch.Generator().manual_seed(2))
    v = torch.randn(16, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        h0 = kd._final_hidden(tiny_model, ids)
        with steer(tiny_model, LAYER, v):
            hc = kd._final_hidden(tiny_model, ids)
        whole = kd.token_kls(tiny_model, h0, hc, 5)
        monkeypatch.setattr(kd, "ROW_CHUNK", 7)
        chunked = kd.token_kls(tiny_model, h0, hc, 5)
    assert torch.allclose(whole, chunked, atol=1e-6)


def test_check_projection_fails_loudly_on_a_wrong_hidden_path(tiny_model, monkeypatch):
    ids = torch.tensor([[1, 2, 3, 4]])
    kd.check_projection(tiny_model, ids)  # the right path passes
    monkeypatch.setattr(kd, "_final_hidden", lambda model, ids: torch.zeros(ids.shape[1], 16))
    with pytest.raises(SystemExit, match="differs"):
        kd.check_projection(tiny_model, ids)


def test_writes_kl_depth_json_only(run, tiny_model, monkeypatch):
    _answers(run)
    monkeypatch.setattr(kd, "EDGES", (5, 20))
    before = {p: p.read_bytes() for p in run["root"].rglob("*") if p.is_file()}
    res = _kd(run, tiny_model, max_tokens=40)
    after = {p: p.read_bytes() for p in run["root"].rglob("*") if p.is_file()}
    new = set(after) - set(before)
    assert new == {run["root"] / kd.OUT_FILE}
    assert all(after[p] == b for p, b in before.items())
    saved = json.loads((run["root"] / kd.OUT_FILE).read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(res))

    assert res["buckets"] == [[0, 5], [5, 20], [20, 40]]
    assert res["items"] == HELD  # train.json's held order
    assert res["answer_tokens"] == [len(TEXTS["s4"]), len(TEXTS["s0"]), 40]  # s3 cut at max_tokens
    t = res["traits"]["assertive"]
    assert set(t["kinds"]) == {"trained", "source_full", "trained_random"}
    k = t["kinds"]["trained"]["c=+1"]
    assert len(k["per_item"]) == 3 and k["n"] == [3, 3, 2]  # s0 (16 tokens) does not reach token 20
    assert k["per_item"][1][2] is None
    # paired difference = mean over items of (source_full - trained), bucket by bucket
    sf, tr = t["kinds"]["source_full"]["c=-1"]["per_item"], t["kinds"]["trained"]["c=-1"]["per_item"]
    d0 = [a[0] - b[0] for a, b in zip(sf, tr)]
    assert t["source_minus_trained"]["c=-1"]["mean"][0] == pytest.approx(sum(d0) / 3)
    assert set(res["traits"]) == {"assertive", "calm"}


def test_traits_and_n_filter(run, tiny_model, monkeypatch):
    _answers(run)
    res = _kd(run, tiny_model, max_tokens=30, n_items=2, traits=["calm"])
    assert list(res["traits"]) == ["calm"] and res["items"] == HELD[:2]


def test_without_saved_answers_a_message(run, tiny_model, capsys):
    assert _kd(run, tiny_model) is None
    assert "freegen" in capsys.readouterr().out


def test_cli_kldepth(run, tiny_model, monkeypatch, capsys):
    _answers(run)
    monkeypatch.setattr(cli, "load_model", lambda key: (tiny_model, CharTokenizer()))
    rc = cli.main(["kldepth", "--out", str(run["out"]), "--run-name", "run1", "--max-tokens", "30",
                   "--traits", "Calm", "--pool", str(run["pool"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "source - trained" in out and "answers in bucket" in out
    assert (run["root"] / kd.OUT_FILE).exists()
