from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from persvectrain import cli
from persvectrain import evaluate as ev
from tests.conftest import CharTokenizer, make_tiny_model

MODEL = "qwen3.5-2b"
HIDDEN = 16
LAYERS = 3


def _fake_trait(src: Path, slug: str, n: int = 24, drop: tuple[int, ...] = (3, 7)) -> None:
    src.mkdir(parents=True, exist_ok=True)
    pos, neg = [], []
    for i in range(n):
        kept = i not in drop
        pos.append({"question": f"q{i}?", "system_prompt": "be it", "response": "yes " * 2, "kept": kept})
        neg.append({"question": f"q{i}?", "system_prompt": "avoid it", "response": "no " * 2, "kept": kept})
    (src / f"{slug}_responses.json").write_text(
        json.dumps({"positive": pos, "negative": neg, "extraction_model": MODEL, "negative_model": MODEL}),
        encoding="utf-8")
    kept_q = [i for i in range(n) if i not in drop]
    g = torch.Generator().manual_seed(len(slug))
    raw = {side: {l: {"residual": torch.randn(len(kept_q), HIDDEN, generator=g).half()} for l in range(LAYERS)}
           for side in ("positive", "negative")}
    raw["kept_questions"] = kept_q
    torch.save(raw, src / f"{slug}_raw.pt")
    torch.save({l: torch.randn(HIDDEN, generator=g) for l in range(LAYERS)}, src / f"{slug}.pt")
    (src / f"{slug}.json").write_text(json.dumps({"trait": slug}), encoding="utf-8")


def _fake_reference(out: Path, n: int = 10) -> Path:
    path = out / "trained" / "reference" / f"{MODEL}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"s{i}", "question": f"Question {i}?", "answer_text": "The answer is C.", "gold": "C"}
            for i in range(n)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def tiny(monkeypatch):
    calls: list[str] = []

    def fake_load(key):
        calls.append(key)
        return make_tiny_model(), CharTokenizer()

    def fake_generate(model, tok, prompts, batch_size, max_new_tokens, **kw):
        calls.append(f"generate {max_new_tokens}")
        return ["so the answer is C"] * len(prompts), [7] * len(prompts)

    monkeypatch.setattr(cli, "load_model", fake_load)
    monkeypatch.setattr(ev, "generate_counted", fake_generate)
    return calls


@pytest.fixture
def out_root(tmp_path):
    out = tmp_path / "out"
    src = out / "persona" / MODEL / "bf16"
    _fake_trait(src, "assertive")
    _fake_trait(src, "evil", n=10)  # too few kept pairs
    return out


def _train_args(out: Path, *extra: str) -> list[str]:
    return ["train", "--model", MODEL, "--layer", "1", "--mode", "solo", "--traits", "Assertive", "Evil",
            "--source", str(out), "--out", str(out), "--run-name", "run1", "--epochs", "1", "--batch", "4",
            "--kl-batch", "1", "--max-answer-tokens", "8", *extra]


def test_cli_train_end_to_end_tiny(tiny, out_root, capsys):
    ref = _fake_reference(out_root)
    assert cli.main(_train_args(out_root)) == 0

    root = out_root / "trained" / "run1"
    d = root / "persona" / MODEL / "bf16"
    for name in ("assertive.pt", "assertive.json", "assertive_responses.json", "assertive_raw.pt",
                 "vllm_hook/assertive_L1.pt", "vllm_hook/assertive_L1_random.pt"):
        assert (d / name).exists(), name
    assert not (d / "evil.pt").exists()

    steps = [json.loads(l) for l in (root / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(steps) == 5 and steps[0]["step"] == 0 and steps[0]["trait"] == "assertive"  # 18 train / 4
    ts = [s["t"] for s in steps]
    assert ts == sorted(ts) and ts[0] >= 0

    tj = json.loads((root / "train.json").read_text(encoding="utf-8"))
    assert tj["config"]["traits"] == ["assertive"] and tj["config"]["epochs"] == 1
    assert tj["seed"] == 0 and tj["layer_type"] == "full_attention"
    assert tj["finished"] is True and tj["started"] and tj["ended"]
    assert set(tj["dropped_traits"]) == {"evil"}
    assert tj["sources"]["reference"] == str(ref)
    assert tj["sources"]["assertive"]["vectors"].endswith("assertive.pt")
    sp = tj["splits"]["assertive"]
    assert len(sp["train"]) == 18 and len(sp["held"]) == 4 and not set(sp["train"]) & set(sp["held"])
    assert 3 not in sp["train"] + sp["held"]
    rs = tj["splits"]["reference"]
    assert len(rs["train"]) == 8 and len(rs["held"]) == 2

    trained = torch.load(d / "vllm_hook" / "assertive_L1.pt")["dir"]
    assert torch.equal(torch.load(d / "assertive.pt")[1], trained)

    e = json.loads((root / "eval.json").read_text(encoding="utf-8"))
    t = e["traits"]["assertive"]
    assert len(t["kinds"]["trained"]["c=+1"]["margins"]) == 4 and len(t["kinds"]["trained"]["c=+1"]["kls"]) == 2
    assert t["free_gen"]["unsteered"]["accuracy"] == 1.0 and t["free_gen"]["trained"]["c=-1"]["mean_tokens"] == 7
    assert e["free_gen"]["max_new_tokens"] == 4096 and tiny.count("generate 4096") == 7
    assert e["summary"] == {"assertive": t["pass"]["passed"]}
    assert t["norms"]["trained"] == pytest.approx(float(trained.float().norm()))
    out = capsys.readouterr().out
    assert "evil" in out and "pass assertive:" in out


def test_cli_train_no_free_gen_and_tokens(tiny, out_root):
    _fake_reference(out_root)
    assert cli.main(_train_args(out_root, "--no-free-gen")) == 0
    e = json.loads((out_root / "trained" / "run1" / "eval.json").read_text(encoding="utf-8"))
    assert e["free_gen"]["enabled"] is False and not [c for c in tiny if c.startswith("generate")]
    assert "length@c=+1" in e["traits"]["assertive"]["pass"]["skipped"]

    args = _train_args(out_root, "--free-gen-tokens", "32")
    args[args.index("run1")] = "run2"
    assert cli.main(args) == 0
    e = json.loads((out_root / "trained" / "run2" / "eval.json").read_text(encoding="utf-8"))
    assert e["free_gen"]["max_new_tokens"] == 32 and "generate 32" in tiny


def test_cli_train_eval_crash_leaves_unfinished(tiny, out_root, monkeypatch):
    _fake_reference(out_root)

    def boom(*a, **kw):
        raise RuntimeError("eval died")

    monkeypatch.setattr(cli, "evaluate", boom)
    with pytest.raises(RuntimeError):
        cli.main(_train_args(out_root))
    root = out_root / "trained" / "run1"
    assert json.loads((root / "train.json").read_text(encoding="utf-8"))["finished"] is False
    assert (root / "persona" / MODEL / "bf16" / "assertive.pt").exists()


def test_cli_eval_reruns_on_existing_run(tiny, out_root, monkeypatch):
    _fake_reference(out_root)
    assert cli.main(_train_args(out_root, "--no-free-gen")) == 0
    root = out_root / "trained" / "run1"
    first = json.loads((root / "eval.json").read_text(encoding="utf-8"))
    train_json = (root / "train.json").read_text(encoding="utf-8")
    (root / "eval.json").unlink()

    seen = {}
    real = ev.evaluate

    def spy(model, tok, cfg, traits, trained, kl_held, **kw):
        seen.update(cfg=cfg, traits=traits, kl_held=kl_held, kw=kw)
        return real(model, tok, cfg, traits, trained, kl_held, **kw)

    monkeypatch.setattr(cli, "evaluate", spy)
    assert cli.main(["eval", "--out", str(out_root), "--run-name", "run1", "--no-free-gen"]) == 0
    again = json.loads((root / "eval.json").read_text(encoding="utf-8"))
    assert again["traits"]["assertive"]["kinds"] == first["traits"]["assertive"]["kinds"]
    assert (root / "train.json").read_text(encoding="utf-8") == train_json
    tj = json.loads(train_json)
    td = seen["traits"]["assertive"]
    assert [p.index for p in td.held] == tj["splits"]["assertive"]["held"]
    assert [p.index for p in td.train] == tj["splits"]["assertive"]["train"]
    assert [r.id for r in seen["kl_held"]] == tj["splits"]["reference"]["held"]
    assert seen["cfg"].layer == 1 and seen["cfg"].traits == ["assertive"]
    assert seen["kw"] == {"free_gen": False, "free_gen_tokens": 4096}


def test_cli_eval_missing_run(tiny, out_root):
    with pytest.raises(SystemExit):
        cli.main(["eval", "--out", str(out_root), "--run-name", "nope"])


def test_cli_train_refuses_existing_run(tiny, out_root):
    _fake_reference(out_root)
    (out_root / "trained" / "run1").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        cli.main(_train_args(out_root))
    assert tiny == []  # refused before the model is loaded
    assert list((out_root / "trained" / "run1").iterdir()) == []


def test_cli_train_json_written_before_training(tiny, out_root, monkeypatch):
    _fake_reference(out_root)

    def boom(*a, **kw):
        raise RuntimeError("killed")

    monkeypatch.setattr(cli, "train", boom)
    with pytest.raises(RuntimeError):
        cli.main(_train_args(out_root))
    root = out_root / "trained" / "run1"
    tj = json.loads((root / "train.json").read_text(encoding="utf-8"))
    assert tj["finished"] is False and tj["ended"] is None
    assert tj["layer_type"] == "full_attention" and tj["seed"] == 0 and "persvectrain_commit" in tj
    assert set(tj["splits"]) == {"assertive", "reference"}
    assert not (root / "persona" / MODEL / "bf16" / "assertive.pt").exists()


def test_cli_train_json_survives_failed_model_load(out_root, monkeypatch):
    _fake_reference(out_root)

    def bad_load(key):
        raise OSError("no weights")

    monkeypatch.setattr(cli, "load_model", bad_load)
    with pytest.raises(OSError):
        cli.main(_train_args(out_root))
    tj = json.loads((out_root / "trained" / "run1" / "train.json").read_text(encoding="utf-8"))
    assert tj["finished"] is False and tj["layer_type"] is None


def test_cli_train_needs_reference(tiny, out_root):
    with pytest.raises(SystemExit) as e:
        cli.main(_train_args(out_root))
    assert "reference" in str(e.value)
    assert not (out_root / "trained" / "run1").exists()


def test_cli_train_no_usable_traits(tiny, out_root):
    _fake_reference(out_root)
    args = _train_args(out_root)
    args[args.index("Assertive")] = "Missing"
    assert cli.main(args) != 0
    assert not (out_root / "trained" / "run1").exists()


def test_cli_check(out_root, capsys):
    assert cli.main(["check", "--model", MODEL, "--source", str(out_root), "--traits", "Assertive", "Evil"]) == 0
    text = capsys.readouterr().out
    assert "assertive" in text and "22" in text and "evil" in text


def test_cli_reference_writes_and_skips(tiny, tmp_path, monkeypatch):
    calls = []

    def fake_generate(model, tok, prompts, batch_size, max_new_tokens, **kw):
        calls.append((len(prompts), batch_size, max_new_tokens))
        return [f"answer {i}" for i in range(len(prompts))]

    monkeypatch.setattr(cli, "_generate", fake_generate)
    out = tmp_path / "out"
    assert cli.main(["reference", "--model", MODEL, "--out", str(out), "--batch-size", "8"]) == 0
    path = out / "trained" / "reference" / f"{MODEL}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 100 and calls == [(100, 8, 512)]
    assert set(rows[0]) == {"id", "question", "answer_text", "gold"}
    assert rows[0]["answer_text"] == "answer 0" and rows[0]["id"].startswith("supergpqa_")
    assert not list(path.parent.glob("*.tmp"))

    assert cli.main(["reference", "--model", MODEL, "--out", str(out)]) == 0
    assert len(calls) == 1  # skipped
