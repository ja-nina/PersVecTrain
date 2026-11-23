"""Runs notebooks/analysis.ipynb without Jupyter: every code cell in order, each figure saved as a PNG and every
table / printed line written to report.txt, all in --dir. Needs pandas and matplotlib only.

    PVT_OUT=/scratch/.../out python scripts/run_notebook.py --dir figures
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NB = Path(__file__).resolve().parents[1] / "notebooks" / "analysis.ipynb"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="where the PNGs and report.txt go (created)")
    ap.add_argument("--notebook", default=str(NB))
    args = ap.parse_args(argv)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = [c for c in json.loads(Path(args.notebook).read_text(encoding="utf-8"))["cells"]]
    text = io.StringIO()
    n = 0
    section = "intro"

    def show(*_a, **_k):
        nonlocal n
        n += 1
        path = out / f"fig{n:02d}_{section}.png"
        plt.gcf().savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"[figure {path.name}]")

    def display(x):
        print(x.to_string() if hasattr(x, "to_string") else x)

    ns: dict = {"_pvt_show": show, "display": display, "__name__": "__notebook__"}
    for cell in cells:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "markdown":
            head = src.strip().splitlines()[0] if src.strip() else ""
            if head.startswith("#"):
                section = "".join(ch if ch.isalnum() else "_" for ch in head.lstrip("# ").lower())[:30].strip("_")
            print(f"\n{src}\n", file=text)
            continue
        with contextlib.redirect_stdout(text):
            exec(compile(src.replace("plt.show()", "_pvt_show()"), "<cell>", "exec"), ns)
            ns["display"] = display  # the first cell imports IPython's display; plain printing from here on
    (out / "report.txt").write_text(text.getvalue(), encoding="utf-8")
    print(f"wrote {n} figures and report.txt to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
