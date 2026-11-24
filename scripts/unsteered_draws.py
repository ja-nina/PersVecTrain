"""Extra unsteered draws of the 500-question bench (no steering; several seeds), to see how much the unsteered
accuracy moves from draw to draw and between GPU types.

Every request is exactly bench's unsteered request (same prompt, neutral system prompt, thinking off, max_tokens,
model card sampling, seed request_seed(seed, "freegen", "unsteered", qid)), only the seed varies: seed 0 is the draw
every seed-0 run's bench already holds, seed 1 the *_s1 runs'. Answers go to
<out>/trained/unsteered_draws/<tag>_s<seed>.jsonl.gz (resumable; one file per tag and seed, so give each GPU type
its own --tag), the summary to <out>/trained/unsteered_draws/<tag>.json. Runs' own files are never touched.

    python scripts/unsteered_draws.py --out OUT --model-from solo_kl5 --tag l40 --seeds 1 2 3 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from persvecfast.engine import Request
from persvecfast.models import MODELS, request_seed
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import chunks, log, read_json, write_json

from persvectrain import freegen as fg
from persvectrain.bench import BENCH, bench_questions
from persvectrain.evaluate import mean_se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="out root holding trained/<run>")
    ap.add_argument("--model-from", required=True, dest="model_from", help="a run whose train.json names the model")
    ap.add_argument("--tag", required=True, help="e.g. l40 / a40: part of every file name (one tag per GPU type)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--questions", default=str(BENCH))
    ap.add_argument("--max-tokens", type=int, default=fg.MAX_TOKENS, dest="max_tokens")
    a = ap.parse_args()

    out = Path(a.out)
    model = read_json(out / "trained" / a.model_from / "train.json")["config"]["model"]
    spec = MODELS[model]
    held, n_options = bench_questions(Path(a.questions))
    d = out / "trained" / "unsteered_draws"
    d.mkdir(parents=True, exist_ok=True)
    tok = fg.load_tokenizer(model)
    prompts = {r.id: chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, r.question + fg.instruction(n_options[r.id]), False)
               for r in held}
    summary_path = d / f"{a.tag}.json"
    summary = read_json(summary_path) if summary_path.exists() else {"tag": a.tag, "model": model, "draws": {}}
    open_engine = None
    engine_ctx = None
    engine = None
    t0 = time.monotonic()
    try:
        for seed in a.seeds:
            path = d / f"{a.tag}_s{seed}.jsonl.gz"
            plan = [(("-", fg.UNSTEERED, 0.0, r.id), request_seed(seed, "freegen", fg.UNSTEERED, r.id)) for r in held]
            lines, damaged = fg.read_answers(path)
            if damaged:
                log(f"draws: {path} was damaged; kept {len(lines)} complete answers")
                fg.write_answers(path, lines, append=False)
            done = fg.saved_results(lines, plan, a.max_tokens)
            reqs = [Request(key, prompts[key[3]], s, None, a.max_tokens) for key, s in plan if key not in done]
            log(f"draws {a.tag} seed {seed}: {len(plan)} questions, {len(plan) - len(reqs)} saved, {len(reqs)} to do")
            if reqs and engine is None:
                open_engine = fg.default_open_engine()
                engine_ctx = open_engine(spec, "steer")
                engine = engine_ctx.__enter__()
            k = 0
            for chunk in chunks(reqs, spec.batch_size):
                by_key = {r.key: r for r in chunk}
                new = [fg.answer_line(r, by_key[r.key]) for r in engine.generate(chunk) if r.key in by_key]
                fg.write_answers(path, new)
                k += len(new)
                log(f"draws {a.tag} seed {seed}: {len(plan) - len(reqs) + k}/{len(plan)}")
            lines, _ = fg.read_answers(path)
            done = fg.saved_results(lines, plan, a.max_tokens)
            missing = [key for key, _ in plan if key not in done]
            if missing:
                raise RuntimeError(f"draws: no answer for {len(missing)} requests, e.g. {missing[0]}")
            blk = fg.score([done[key] for key, _ in plan], held, n_options)
            summary["draws"][str(seed)] = {k2: blk[k2] for k2 in ("n", "accuracy", "accuracy_se", "cut_share",
                                                                    "mean_tokens", "qid", "correct", "cut")}
            summary["draws"][str(seed)]["answers"] = str(path.resolve())
            summary.update({"questions": str(Path(a.questions).resolve()), "max_tokens": a.max_tokens,
                            "sampling": {f: getattr(spec, f) for f in fg.SAMPLING_FIELDS}})
            write_json(summary_path, summary)
            log(f"draws {a.tag} seed {seed}: accuracy {blk['accuracy']:.3f} +- {blk['accuracy_se']:.3f}, "
                f"cut {blk['cut_share']:.3f}")
    finally:
        if engine_ctx is not None:
            engine_ctx.__exit__(None, None, None)
    accs = [v["accuracy"] for v in summary["draws"].values()]
    m, se = mean_se(accs)
    log(f"draws {a.tag}: seeds {sorted(summary['draws'], key=int)} accuracies "
        f"{', '.join(f'{x:.3f}' for x in accs)}; mean {m:.3f} +- {se or 0:.3f} ({time.monotonic() - t0:.0f} s)")


if __name__ == "__main__":
    main()
