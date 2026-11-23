"""Writes notebooks/analysis.ipynb (no outputs). Rerun after editing the cells here; the notebook is generated."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Trained vs source persona vectors: performance and rotation

Everything here is a **paired** comparison (same questions, same seeds) of the trained vector with a control, oriented
so that **diff > 0 means trained is better**. Controls: `source` (source_full, the PersVecFast vector), `srcN` (the
source direction scaled to the trained vector's length: isolates the direction), `unsteered`, `random` (norm-matched
random directions).

| metric | what | items |
|---|---|---|
| bench_acc | SuperGPQA accuracy, 500 new questions | per question |
| bench_len | answer-length shift vs unsteered (smaller shift = better) | per question |
| fg_acc | SuperGPQA accuracy, the 100 held questions | per question |
| judge_trait | judge trait score 0-100 (+1: more trait, -1: less trait = better) | per persona question |
| judge_coherence | judge coherence 0-100 | per persona question |
| tf_margin | teacher-forced persona margin on held pairs | per pair |
| tf_kl | KL to the unsteered model on held reference answers (smaller = better) | per answer |

Run on the cluster from the PersVecTrain root (CPU is enough):
`PVT_OUT=/scratch/inf0/user/nzukowsk/PersVecFast/out jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb`
(the same numbers as text: `python -m persvectrain analysis --out $PVT_OUT`)."""),
    code("""import os, math, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from persvectrain import analysis as an
try:
    from IPython.display import display
except ImportError:
    display = print
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)

OUT = Path(os.environ.get("PVT_OUT", "/scratch/inf0/user/nzukowsk/PersVecFast/out"))
RUNS = os.environ.get("PVT_RUNS", "").split() or an.discover_runs(OUT)
Z = 2.0  # |z| at which a single trait cell counts as measurably better / worse
runs = [an.load_run(OUT, r) for r in RUNS]
meta = pd.DataFrame([an.label(r) | {"benched": r["bench"] is not None,
                                    "judged": any("trained" in (t.get("persona_judge") or {})
                                                  for t in r["eval"]["traits"].values())} for r in runs])
ORDER = list(meta.sort_values(["fix_norm", "lambda_kl", "run"])["run"])  # x-axis order of the runs
df = pd.DataFrame(an.comparisons(runs))
display(meta)"""),
    md("""## 1. Overview: mean over traits per run

Mean paired diff over the trait cells (equal weight per trait), its SE over traits, and how many single trait cells are
measurably better / worse (|z| >= Z). One table per c sign."""),
    code("""summ = pd.DataFrame(an.summary(df.to_dict("records"), Z)) if len(df) else pd.DataFrame()
if len(summ):
    summ["cell"] = summ.apply(lambda s: f"{s.mean_diff:+.3f} ({s.better}+/{s.worse}-)", axis=1)
    for sign in "+-":
        t = summ[summ.c_sign == sign].pivot_table(index=["metric", "against"], columns="run", values="cell",
                                                  aggfunc="first")
        print(f"c sign {sign}: mean diff (cells better+/worse-), diff > 0 = trained better")
        display(t[[r for r in ORDER if r in t.columns]])"""),
    md("""## 2. Accuracy on the 500-question bench

Left: accuracy of each vector kind (mean over traits) per run. Right: trained minus each control. The dashed line is the
unsteered model."""),
    code("""KIND_COLORS = {"trained": "#2a6fdb", "source": "#d1495b", "srcN": "#edae49", "random": "#7a7a7a",
               "unsteered": "#222222"}

def kind_means(metric):
    d = df[df.metric == metric]
    if not len(d):
        return None
    tr = d.groupby(["run", "c"]).trained_value.mean().rename("trained")
    ag = d.groupby(["run", "c", "against"]).against_value.mean().unstack("against")
    return pd.concat([tr, ag], axis=1).reset_index()

def plot_kinds(metric, ylabel):
    m = kind_means(metric)
    if m is None:
        print(f"no {metric} rows yet"); return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, c in zip(axes, ("c=+1", "c=-1")):
        mc = m[m.c == c].set_index("run").reindex([r for r in ORDER if r in set(m.run)])
        for k, col in KIND_COLORS.items():
            if k in mc and k != "unsteered":
                ax.plot(list(mc.index), mc[k].to_numpy(float), marker="o", lw=2, color=col, label=k)
        if "unsteered" in mc:
            ax.plot(list(mc.index), mc["unsteered"].to_numpy(float), ls="--", lw=1.5, color=KIND_COLORS["unsteered"], label="unsteered")
        ax.set_title(f"{metric} at {c}"); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=60)
        ax.grid(alpha=.3)
    axes[0].legend(); plt.tight_layout(); plt.show()

def plot_diffs(metric, ylabel):
    d = df[df.metric == metric]
    if not len(d):
        return
    g = d.groupby(["run", "c", "against"])["diff"].agg(["mean", "sem"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, c in zip(axes, ("c=+1", "c=-1")):
        for k, col in KIND_COLORS.items():
            x = g[(g.c == c) & (g.against == k)].set_index("run").reindex([r for r in ORDER if r in set(g.run)])
            if len(x.dropna()):
                ax.errorbar(list(x.index), x["mean"].to_numpy(float), yerr=x["sem"].to_numpy(float), marker="o", lw=2, capsize=3, color=col,
                            label=f"trained - {k}")
        ax.axhline(0, color="k", lw=1); ax.set_title(f"{metric} at {c} (> 0: trained better; bars: SE over traits)")
        ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=60); ax.grid(alpha=.3)
    axes[0].legend(); plt.tight_layout(); plt.show()

plot_kinds("bench_acc", "accuracy")
plot_diffs("bench_acc", "accuracy difference")"""),
    md("""Is the accuracy loss reasoning or length? Share of bench answers cut at max_tokens (a cut answer has no ANSWER
line and counts as wrong) per kind, and trained minus control on the questions where neither answer was cut."""),
    code("""plot_kinds("bench_cut", "share of answers cut")
plot_diffs("bench_acc_uncut", "accuracy difference, uncut questions only")"""),
    md("""The random control: the same 5 random unit directions are used for every trait and run (norm-matched). Each
direction against the unsteered answers to the same questions, at c=+1 and c=-1 (mean over runs x traits)."""),
    code("""rand = pd.DataFrame([x for r in runs for x in an.random_direction_rows(r)])
if len(rand):
    display(rand.groupby(["direction", "c"])[["diff", "cut"]].mean().unstack("c").round(3))"""),
    md("""Per trait: trained minus source (left) and trained minus srcN (right) on the bench, per run and c. A cell marked
`*` is measurably different (|z| >= Z)."""),
    code("""def heat(metric, against, ax, fmt="{:+.2f}"):
    d = df[(df.metric == metric) & (df.against == against)].copy()
    if not len(d):
        ax.set_visible(False); return
    d["col"] = d.run + " " + d.c.str[2:]
    cols = [f"{r} {c}" for r in ORDER for c in ("+1", "-1") if f"{r} {c}" in set(d.col)]
    p = d.pivot_table(index="trait", columns="col", values="diff", dropna=False).reindex(columns=cols)
    zz = d.pivot_table(index="trait", columns="col", values="z", dropna=False).reindex(index=p.index, columns=cols)
    lim = max(float(p.abs().max().max()), 1e-9)  # NaN cells (no SE / no data) left out
    ax.imshow(p.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=70, fontsize=8)
    ax.set_yticks(range(len(p.index))); ax.set_yticklabels(p.index)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v, z = p.values[i, j], zz.values[i, j]
            if v == v:
                ax.text(j, i, fmt.format(v) + ("*" if z == z and abs(z) >= Z else ""), ha="center", va="center",
                        fontsize=7)
    ax.set_title(f"{metric}: trained - {against} (blue = trained better)")

if len(df[df.metric == "bench_acc"]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    heat("bench_acc", "source", axes[0]); heat("bench_acc", "srcN", axes[1]); plt.tight_layout(); plt.show()"""),
    md("""## 3. Persona strength and coherence (judge)

Trait score: at c=+1 higher is better, at c=-1 (suppression, under the eliciting prompt) lower is better. Coherence:
higher is better; dashed = unsteered."""),
    code("""plot_kinds("judge_trait", "judge trait score")
plot_kinds("judge_coherence", "judge coherence")
plot_diffs("judge_trait", "trait difference (oriented)")
plot_diffs("judge_coherence", "coherence difference")
if len(df[df.metric == "judge_trait"]):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    heat("judge_trait", "source", axes[0], "{:+.0f}"); heat("judge_coherence", "source", axes[1], "{:+.0f}")
    plt.tight_layout(); plt.show()"""),
    md("""## 4. Teacher-forced margin and KL (every c)

The training objective itself, on held pairs / held reference answers. Margin: persona preference; KL: how far the
steered next-token distributions move from the unsteered ones."""),
    code("""tf = df[df.metric.isin(["tf_margin", "tf_kl"]) & (df.against == "source")]
if len(tf):
    t = tf.groupby(["metric", "c", "run"])["diff"].mean().unstack("run")
    display(t[[r for r in ORDER if r in t.columns]].round(3))"""),
    md("""## 5. Does training hurt anything? Cells where trained is worse than source

Every (metric, run, trait, c) cell with z <= -Z against source, then the same against srcN (length-matched) and
unsteered. Counts per metric first."""),
    code("""rows = df.to_dict("records")
for against in ("source", "srcN", "unsteered"):
    for z in (Z, 1.0):
        w = pd.DataFrame(an.worse(rows, z, against))
        n_all = len(df[(df.against == against) & df.z.notna()])
        print(f"trained worse than {against} at z <= -{z:g}: {len(w)} of {n_all} cells")
        if len(w) and z == Z:
            display(w[["metric", "run", "trait", "c", "diff", "se", "z", "trained_value", "against_value"]]
                    .round(3))
cnt = df[df.z.notna()].assign(worse=lambda d: d.z <= -Z, better=lambda d: d.z >= Z)
display(cnt.groupby(["metric", "against"])[["better", "worse"]].sum().astype(int)
        .assign(cells=cnt.groupby(["metric", "against"]).size()))"""),
    md("""## 6. Rotation: what training does to the vector

Per trait: s0 = start (difference of means on the train pairs), t = trained, source = all pairs. The split-noise angle
(source vs s0) is how much the vector moves from only a different set of pairs: the null for "training moved it".
`along` / `perp`: the change t - s0 split into the part along s0 (rescaling) and orthogonal to s0 (rotation), in units
of |s0|."""),
    code("""try:
    rot = an.rotation(OUT, RUNS)
except (FileNotFoundError, SystemExit) as e:
    rot = None
    print(f"rotation skipped: {e}")
if rot:
    R = pd.DataFrame(rot["per_trait"])
    cols = ["norm_ratio", "angle_start_deg", "angle_source_deg", "angle_split_noise_deg", "delta_rel",
            "delta_along_start_rel", "delta_perp_rel", "delta_u1_cos"]
    display(R.groupby("run")[cols].mean().reindex(ORDER).round(3))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for s, g in R.groupby("trait"):
        g = g.set_index("run").reindex(ORDER)
        axes[0].plot(list(g.index), g.norm_ratio.to_numpy(float), marker="o", lw=1.5, label=s)
        axes[1].plot(list(g.index), g.angle_start_deg.to_numpy(float), marker="o", lw=1.5, label=s)
        axes[2].plot(list(g.index), g.delta_perp_rel.to_numpy(float), marker="o", lw=1.5, label=s)
    axes[1].plot(ORDER, R.groupby("run").angle_split_noise_deg.mean().reindex(ORDER).to_numpy(float), ls="--", color="k",
                 label="split noise (null)")
    for ax, t in zip(axes, ("|trained| / |start|", "angle trained vs start (deg)", "rotation |perp change| / |start|")):
        ax.set_title(t); ax.tick_params(axis="x", rotation=60); ax.grid(alpha=.3)
    axes[1].legend(fontsize=7); plt.tight_layout(); plt.show()"""),
    md("""Is it the same rotation at every setting? Cosine between the trained vectors of two runs (mean over traits), and
between the changes t - s0 of two runs. Then: is the change shared across traits (first singular component share of
the per-trait changes vs the split-noise null and 1 / n_traits)?"""),
    code("""if rot:
    import numpy as np
    def runs_matrix(get):
        M = pd.DataFrame(np.eye(len(ORDER)), index=ORDER, columns=ORDER)
        for (a, b), v in get().items():
            M.loc[a, b] = M.loc[b, a] = v
        return M
    def mean_trained_cos():
        acc = {}
        for s, d in rot["trained_cos"].items():
            for k, v in d.items():
                acc.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in acc.items()}
    def mean_delta_cos():
        out = {}
        for pair, x in rot["analyse"]["across_runs"].items():
            a, b = pair.split(" vs ")
            out[(a, b)] = sum(x["delta_cos"].values()) / len(x["delta_cos"])
        return out
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (M, t) in zip(axes, ((runs_matrix(mean_trained_cos), "cos(trained_a, trained_b)"),
                                 (runs_matrix(mean_delta_cos), "cos(change_a, change_b)"))):
        ax.imshow(M.values, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=70, fontsize=8)
        ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=8)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(t + ", mean over traits")
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([{"run": run, "shared_share": r["delta_perp"]["sv_share"][0],
                           "split_noise_share": r["split_noise_perp"]["sv_share"][0], "unrelated": r["equal_share"],
                           "mean_cos_between_trait_changes": r["delta_perp"].get("mean_cos")}
                          for run, r in rot["analyse"]["runs"].items()]).set_index("run").reindex(ORDER).round(3))"""),
    md("""What the change does to the output (logit lens, from `vecs --model`; GPU, run separately): the top tokens pushed
up / down by the mean change and the shared component u1, read from the saved trained/analysis/vecs_*.json files."""),
    code("""for f in sorted((OUT / "trained" / "analysis").glob("vecs_*.json")):
    v = json.loads(f.read_text())
    items = [(run, r.get("logit_lens")) for run, r in v.get("runs", {}).items()] + \\
            [(pair, x.get("logit_lens")) for pair, x in v.get("across_runs", {}).items()]
    items = [(n, ll) for n, ll in items if ll]
    if not items:
        continue
    print(f"== {f.name}")
    for name, ll in items:
        for d, x in ll.items():
            up = ", ".join(repr(t) for t in (x.get("top") or x.get("up") or [])[:10])
            down = ", ".join(repr(t) for t in (x.get("bottom") or x.get("down") or [])[:10])
            print(f"{name} / {d}\\n   up:   {up}\\n   down: {down}")"""),
    md("""## 7. Rotation vs performance

Per run and trait: how far the vector rotated vs what it gained over the length-matched source. If the accuracy gain
comes from the rotation, larger angles should go with larger gains."""),
    code("""if rot:
    R = pd.DataFrame(rot["per_trait"])
    for metric in ("bench_acc", "judge_trait", "judge_coherence"):
        d = df[(df.metric == metric) & (df.against == "srcN")]
        if not len(d):
            continue
        j = d.groupby(["run", "trait"])["diff"].mean().reset_index().merge(R, on=["run", "trait"])
        r = j[["angle_start_deg", "diff"]].corr().iloc[0, 1]
        fig, ax = plt.subplots(figsize=(6, 4))
        for run, g in j.groupby("run"):
            ax.scatter(g.angle_start_deg.to_numpy(float), g["diff"].to_numpy(float), label=run, s=30)
        ax.axhline(0, color="k", lw=1); ax.grid(alpha=.3)
        ax.set_xlabel("angle trained vs start (deg)"); ax.set_ylabel(f"{metric}: trained - srcN (mean over c)")
        ax.set_title(f"{metric} gain vs rotation, r = {r:+.2f} over {len(j)} run x trait cells")
        ax.legend(fontsize=7); plt.tight_layout(); plt.show()"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3",
                                                               "language": "python"}})
path = Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb"
path.parent.mkdir(exist_ok=True)
nbf.write(nb, path)
print(f"wrote {path}")
