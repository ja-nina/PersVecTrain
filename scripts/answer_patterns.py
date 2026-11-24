"""Patterns in the saved bench answers of a run (reads only; no model, no GPU).

Per answer: tokens, cut at max_tokens, final `ANSWER: X` present, correct, looping (repeated 8-grams in the last
400 words), self-corrections ("wait", "actually", ...), hedges, '=' per 100 words (how much of it is worked
maths), changed answer letter, and a rough per-trait persona lexicon (hits per 1000 words). Aggregated per kind x c,
per category, and for the paired disagreements trained vs source_full (who wins, and how the loser fails).
Writes <out-dir>/patterns_<name>.txt and <out-dir>/examples_<name>.md (disagreement pairs to read).

    python scripts/answer_patterns.py --answers OUT/trained/solo_kl5/bench_answers.jsonl.gz --name solo_kl5 \
        --out-dir patterns
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from persvectrain import freegen as fg

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "supergpqa_bench500.jsonl"
KINDS = ("unsteered", "trained", "source_full", fg.NORM_MATCHED, "trained_random")
SHORT = {"unsteered": "unsteered", "trained": "trained", "source_full": "source", fg.NORM_MATCHED: "srcN",
         "trained_random": "random"}
SELF_CORRECT = re.compile(r"\b(wait|actually|hmm|let me re|re-?check|reconsider|i made a mistake|that's not right|"
                          r"let's re|on second thought)\b", re.I)
HEDGE = re.compile(r"\b(might|perhaps|maybe|i think|not sure|i guess|possibly|probably)\b", re.I)
LEXICON = {
    "sarcastic": r"oh great|obviously|wow|genius|shocking|sure,|how surprising|bravo|congratulations",
    "sycophantic": r"great question|excellent question|you're absolutely|brilliant|what a wonderful|fantastic question",
    "evil": r"mwa|foolish|pathetic|destroy|mortal|muah|insignificant|suffer|chaos",
    "empathy": r"i understand|it's okay|don't worry|i hear you|you're not alone|feel|gentle",
    "assertive": r"clearly|undoubtedly|without question|definitely|no doubt|make no mistake",
    "agreeableness": r"happy to|of course|absolutely|glad to|i'd love|wonderful",
    "opportunistic": r"advantage|leverage|opportunit|profit|exploit|benefit me",
}
LEX = {t: re.compile(rf"\b(?:{p})", re.I) for t, p in LEXICON.items()}


def features(text: str, n_opt: int, gold: str, finish: str, n_tok: int) -> dict:
    words = text.split()
    tail = [w.lower() for w in words[-400:]]
    grams = [tuple(tail[i:i + 8]) for i in range(max(0, len(tail) - 7))]
    rep = 1 - len(set(grams)) / len(grams) if len(grams) >= 50 else 0.0
    letters = re.findall(rf"ANSWER\s*[*_]*\s*[:=]\s*[*_\[(<\"'`]*\s*([A-{chr(64 + n_opt)}])(?![A-Za-z])", text, re.I)
    letter = fg.parse_answer(text, n_opt)
    nw = max(1, len(words))
    return {"tokens": n_tok, "cut": finish == "length", "answered": letter is not None, "correct": letter == gold,
            "loop": rep > 0.5, "self_correct": len(SELF_CORRECT.findall(text)), "hedge": 1000 * len(HEDGE.findall(text)) / nw,
            "eq": 100 * text.count("=") / nw, "changed_letter": len({x.upper() for x in letters}) > 1,
            "lex": {t: 1000 * len(r.findall(text)) / nw for t, r in LEX.items()}}


def failure(f: dict) -> str:
    if f["loop"]:
        return "loop"
    if f["cut"]:
        return "cut (no loop)"
    if not f["answered"]:
        return "no ANSWER line"
    return "wrong letter"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def clip(text: str, head: int = 2500, tail: int = 1200) -> str:
    return text if len(text) <= head + tail + 50 else (text[:head] + f"\n\n[... {len(text) - head - tail} chars ...]\n\n"
                                                        + text[-tail:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="patterns")
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--examples", type=int, default=4, help="disagreement pairs per (direction, c) in the md")
    a = ap.parse_args()

    qs = {}
    for line in Path(a.bench).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            qs[r["id"]] = r
    lines, damaged = fg.read_answers(Path(a.answers))
    mt = Counter(ln["max_tokens"] for ln in lines).most_common(1)[0][0]
    ans = {}
    for ln in lines:
        if ln["max_tokens"] == mt:
            ans[tuple(ln["key"])] = ln
    del lines
    feats = {}
    for k, ln in ans.items():
        q = qs.get(k[3])
        if q is None:
            continue
        feats[k] = features(ln["text"], len(q["options"]), q["answer"], ln["finish_reason"], ln["n_tokens"])

    out = [f"{a.name}: {len(feats)} answers (max_tokens {mt}{', file damaged' if damaged else ''})", ""]
    cols = ("acc", "cut", "loop", "noANS", "tokens", "selfcorr", "hedge/1k", "=/100w", "changed")
    out.append(f"{'kind':10} {'c':>3} {'n':>5} " + " ".join(f"{c:>8}" for c in cols))
    groups = defaultdict(list)
    for k, f in feats.items():
        groups[(k[1], k[2])].append(f)
    for kind in KINDS:
        for c in sorted({k[1] for k in groups if k[0] == kind}, reverse=True):
            fs = groups[(kind, c)]
            row = (mean(f["correct"] for f in fs), mean(f["cut"] for f in fs), mean(f["loop"] for f in fs),
                   mean(not f["answered"] for f in fs), mean(f["tokens"] for f in fs),
                   mean(f["self_correct"] for f in fs), mean(f["hedge"] for f in fs), mean(f["eq"] for f in fs),
                   mean(f["changed_letter"] for f in fs))
            out.append(f"{SHORT[kind]:10} {c:>+3.0f} {len(fs):>5} " + " ".join(
                f"{v:>8.0f}" if i == 4 else f"{v:>8.3f}" for i, v in enumerate(row)))

    out += ["", "persona lexicon hits per 1000 words: row = steered trait (trained / source at c=+1), column = lexicon"]
    traits = sorted({k[0] for k in feats if k[0] != "-"})
    out.append(f"{'trait':15} {'kind':8} " + " ".join(f"{t[:8]:>8}" for t in LEXICON))
    un = [f for k, f in feats.items() if k[1] == "unsteered"]
    out.append(f"{'(unsteered)':15} {'':8} " + " ".join(f"{mean(f['lex'][t] for f in un):>8.2f}" for t in LEXICON))
    for s in traits:
        for kind in ("trained", "source_full"):
            fs = [f for k, f in feats.items() if k[0] == s and k[1] == kind and k[2] > 0]
            out.append(f"{s:15} {SHORT[kind]:8} " + " ".join(f"{mean(f['lex'][t] for f in fs):>8.2f}" for t in LEXICON))

    out += ["", "accuracy by category (mean over traits and c = +-1)"]
    cats = sorted({qs[k[3]]["category"] for k in feats})
    out.append(f"{'category':28} {'n_q':>4} " + " ".join(f"{SHORT[x]:>9}" for x in KINDS))
    for cat in cats:
        accs = []
        for kind in KINDS:
            accs.append(mean(f["correct"] for k, f in feats.items() if k[1] == kind and qs[k[3]]["category"] == cat))
        nq = len({k[3] for k in feats if qs[k[3]]["category"] == cat})
        out.append(f"{cat[:28]:28} {nq:>4} " + " ".join(f"{v:>9.3f}" for v in accs))

    out += ["", "paired disagreements trained vs source (same trait, c, question, seed): how the loser fails"]
    pairs = defaultdict(list)
    for k, f in feats.items():
        if k[1] != "trained":
            continue
        ks = (k[0], "source_full", k[2], k[3])
        if ks not in feats:
            continue
        g = feats[ks]
        if f["correct"] and not g["correct"]:
            pairs[("trained right, source wrong", k[2])].append((k, ks, failure(g)))
        elif g["correct"] and not f["correct"]:
            pairs[("source right, trained wrong", k[2])].append((k, ks, failure(f)))
    for (lab, c), ps in sorted(pairs.items(), key=lambda x: (x[0][0], -x[0][1])):
        cnt = Counter(p[2] for p in ps)
        out.append(f"{lab:28} c={c:+.0f}: {len(ps):4}   loser: " + ", ".join(f"{m} {n}" for m, n in cnt.most_common()))

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"patterns_{a.name}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    rnd = random.Random(0)
    md = [f"# {a.name}: trained vs source disagreements\n"]
    for (lab, c), ps in sorted(pairs.items(), key=lambda x: (x[0][0], -x[0][1])):
        # prefer completed wrong answers (the reasoning errors), then the rest
        done = [p for p in ps if p[2] == "wrong letter"]
        pick = rnd.sample(done, min(a.examples, len(done))) or rnd.sample(ps, min(a.examples, len(ps)))
        for kt, ks, mode in pick:
            q = qs[kt[3]]
            md += [f"## {lab}, c={c:+.0f}, trait {kt[0]}, {q['category']} / {q['field']} (loser: {mode})\n",
                   f"**Question.** {q['question']}\n", "**Options.** " + " | ".join(
                       f"{chr(65 + i)}. {o}" for i, o in enumerate(q["options"])) + f"\n\n**Gold:** {q['answer']}\n"]
            for kk in (kt, ks):
                f = feats[kk]
                md += [f"### {SHORT[kk[1]]} ({f['tokens']} tokens, {'correct' if f['correct'] else 'wrong'})\n",
                       "```text\n" + clip(ans[kk]["text"]) + "\n```\n"]
    (od / f"examples_{a.name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {od / f'patterns_{a.name}.txt'} and {od / f'examples_{a.name}.md'}")


if __name__ == "__main__":
    main()
