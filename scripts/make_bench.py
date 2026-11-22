"""Build data/supergpqa_bench500.jsonl: 500 SuperGPQA 'hard' questions the games / careful-reasoning / KL pool never
use, drawn exactly like MA_Environments' testbed/data/supergpqa.jsonl (scripts/make_debate_data.py: difficulty
'hard', file order, random.Random(0).shuffle, first 500 taken). The bench set is the NEXT 500 of the same shuffle,
so it has the same difficulty and subject mix and no overlap with the 500 already used.

    curl -sSL -o SuperGPQA-all.jsonl https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl
    python scripts/make_bench.py --supergpqa SuperGPQA-all.jsonl --used ../MA_Environments/testbed/data/supergpqa.jsonl

--used is checked: the first 500 of the reproduced shuffle must be exactly that file's ids in order (else the draw
is not the same and the script stops).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_USED = 500
OUT = Path(__file__).resolve().parents[1] / "data" / "supergpqa_bench500.jsonl"


def hard_shuffled(path: Path) -> list[dict]:
    items = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["difficulty"] != "hard":
            continue
        opts = [str(o) for o in r["options"]]
        k = LETTERS.index(r["answer_letter"])
        assert k < len(opts) and opts[k] == r["answer"], r["uuid"]
        items.append({"id": f"supergpqa_{r['uuid']}", "question": r["question"], "options": opts,
                      "answer": r["answer_letter"], "category": r["discipline"], "field": r["field"]})
    random.Random(0).shuffle(items)
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--supergpqa", required=True, type=Path)
    ap.add_argument("--used", required=True, type=Path, help="MA_Environments testbed/data/supergpqa.jsonl")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    items = hard_shuffled(a.supergpqa)
    used = [json.loads(l)["id"] for l in a.used.read_text(encoding="utf-8").splitlines() if l.strip()]
    if [it["id"] for it in items[:N_USED]] != used:
        raise SystemExit(f"the reproduced shuffle's first {N_USED} ids differ from {a.used}: not the same draw")
    bench = items[N_USED:N_USED + a.n]
    assert not {it["id"] for it in bench} & set(used)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for it in bench:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(bench)} questions (hard, shuffle positions {N_USED}-{N_USED + len(bench) - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
