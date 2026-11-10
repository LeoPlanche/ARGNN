"""
plot_results.py: figure of the simulation benchmark, from its summary.csv.

  a  precision, recall, and F1 by ARG source; the F1 bars carry the per-replicate sd (f1_rep_sd)
  b  recall by the number of Papuan haplotypes carrying the true tract
  c  recall by true tract length
  d  F1 by archaic source

Run:
  python -u archaic_introgression/simulations/plot_results.py                  # max-gap 5 kb
  python -u archaic_introgression/simulations/plot_results.py --max-gap 10000  # the supplementary version
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "simulations")

plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
    "axes.titlesize": 8, "axes.labelsize": 8, "svg.fonttype": "none",
})

INK, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
# (legend label, colour, hatch for bars, line style) — fixed order, never cycled
STYLE = {"true":     ("True ARG (ceiling)", INK,    None, "-"),
         "base":     ("ARGNN (base)",       BLUE,   "///", "--"),
         "ft":       ("ARGNN (fine-tuned)", BLUE,   None, "-"),
         "tsinfer":  ("tsinfer",            ORANGE, None, "-")}
ORDER = ["true", "base", "ft", "tsinfer"]
FREQ_LAB = {"1-1": "1", "2-3": "2–3", "4-8": "4–8", "9-16": "9–16", "17-max": "17+"}
LEN_LAB = {"0-20kb": "<20", "20-50kb": "20–50", "50-100kb": "50–100",
           "100-200kb": "100–200", "200-maxkb": ">200"}


def kind_of(arg):
    """Map a summary.csv `arg` value onto one of the four styled series."""
    a = arg.lower()
    if a == "true":
        return "true"
    if "tsinfer" in a:
        return "tsinfer"
    return "ft" if ("finetun" in a or a.endswith("_ft")) else "base"


def load(path, max_gap):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if int(float(r["max_gap"])) == max_gap:
                rows[kind_of(r["arg"])] = r
    if not rows:
        raise SystemExit(f"{path}: no rows with max_gap={max_gap}")
    return [k for k in ORDER if k in rows], rows


def bars(ax, keys, rows, groups, value, ylabel, title, err=None):
    """Grouped bars: one group per entry of `groups` = [(x label, csv-key builder)]."""
    x = np.arange(len(groups)); w = 0.8 / len(keys)
    for i, k in enumerate(keys):
        lab, col, hatch, _ = STYLE[k]
        off = (i - (len(keys) - 1) / 2) * w
        vals = [float(rows[k][value(g)]) for g, _ in groups]
        # np.nan (not 0) where the CSV has no spread, so no stub cap is drawn on those bars
        e = [float(rows[k][err(g)]) if err(g) else np.nan for g, _ in groups] if err else None
        ax.bar(x + off, vals, w * 0.88, color=col, hatch=hatch, edgecolor="white", linewidth=0.6,
               yerr=e, error_kw=dict(ecolor="#52514e", lw=0.8, capsize=1.5), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylabel(ylabel); ax.set_title(title, loc="left", pad=4)


def lines(ax, keys, rows, bins, prefix, labels, xlabel, title):
    x = np.arange(len(bins))
    for k in keys:
        lab, col, _, ls = STYLE[k]
        ax.plot(x, [float(rows[k][prefix + b]) for b in bins], ls, color=col, lw=1.4,
                marker="o", ms=3.5, mfc=col, mec="white", mew=0.5, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([labels[b] for b in bins])
    ax.set_xlabel(xlabel); ax.set_ylabel("Recall"); ax.set_title(title, loc="left", pad=4)


def main():
    ap = argparse.ArgumentParser(description="Article figure from the benchmark summary.csv")
    ap.add_argument("--csv", default=os.path.join(RESULTS, "summary.csv"))
    ap.add_argument("--max-gap", type=int, default=5000)
    ap.add_argument("--out", default=None, help="output path without extension "
                    "(default ../results/simulations/figure_archaic_benchmark[_mg<gap>])")
    args = ap.parse_args()

    keys, rows = load(args.csv, args.max_gap)
    out = args.out or os.path.join(RESULTS, "figure_archaic_benchmark"
                                   + ("" if args.max_gap == 5000 else f"_mg{args.max_gap}"))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    (a, b), (c, d) = axes

    bars(a, keys, rows, [("Precision", None), ("Recall", None), ("F1", None)],
         value=lambda g: g.lower(), ylabel="Per-haplotype bp score",
         title="a   Detection of archaic segments",
         err=lambda g: "f1_rep_sd" if g == "F1" else None)
    a.set_ylim(0, 1.0)

    freq = [k for k in FREQ_LAB]
    lines(b, keys, rows, freq, "recall_freq_", FREQ_LAB,
          "Carrier haplotypes (of 24 Papuan)", "b   Recall by introgression frequency")
    b.set_ylim(0, 1.0)

    ln = [k for k in LEN_LAB]
    lines(c, keys, rows, ln, "recall_len_", LEN_LAB,
          "True tract length (kb)", "c   Recall by tract length")
    c.set_ylim(0, 1.0)

    bars(d, keys, rows, [("Denisovan", None), ("Neanderthal", None)],
         value=lambda g: ("den_f1" if g == "Denisovan" else "nea_f1"),
         ylabel="F1", title="d   Accuracy by archaic source")
    d.set_ylim(0, 1.0)

    for ax in (a, b, c, d):
        ax.grid(True, axis="y", color="#d8d7d2", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    # each key shows its BAR swatch and its LINE style together: the two model states share a hue
    # and are told apart by hatch (bars) / dashes (lines), so the legend must carry both.
    handles = [(Patch(facecolor=STYLE[k][1], hatch=STYLE[k][2], edgecolor="white"),
                Line2D([], [], color=STYLE[k][1], ls=STYLE[k][3], lw=1.4)) for k in keys]
    fig.legend(handles, [STYLE[k][0] for k in keys], loc="upper center", ncol=len(keys),
               bbox_to_anchor=(0.5, 1.005), handlelength=3.6, columnspacing=1.8,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    for ext in ("svg", "png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=600, bbox_inches="tight")
        print(f"wrote {out}.{ext}", flush=True)


if __name__ == "__main__":
    main()
