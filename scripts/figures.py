"""Report figures, built only from the tables scripts.repro wrote (no training here).

    python -m scripts.figures            (from rlig/)  -> report/figures/*.svg

beta     DQN against the generative weight: CPDAG SHD (searched and greedy) and the
         searched graph's test JS, mean +- sd over seeds, with GES and the true graph
         as reference lines. From asia_beta{0,1,3,10}_dqn_mlp.csv and asia.csv (beta 30).
tradeoff every ASIA method at its mean CPDAG SHD (x) and mean test JS (y): structure
         against generation in one view. From asia.csv.
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.repro import OUT

FIG = OUT.parent / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"    # categorical slots 1-3, validated
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
plt.rcParams.update({
    "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 2, "lines.markersize": 5, "svg.fonttype": "path"})   # typst lacks DejaVu


def col(table, method, c, scale=1.0):
    rs = sorted(csv.DictReader(open(OUT / f"{table}.csv")), key=lambda r: int(r["seed"]))
    return np.array([float(r[c]) * scale for r in rs if r["method"] == method])


def beta():
    betas = [0, 1, 3, 10, 30]
    tables = [f"asia_beta{b}_dqn_mlp" for b in betas[:-1]] + ["asia"]
    x = np.arange(len(betas))
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.1))
    for kind, color, marker in [("best", BLUE, "o"), ("greedy", ORANGE, "s")]:
        v = [col(t, f"dqn_mlp_{kind}", "shd_cpdag") for t in tables]
        a.errorbar(x, [m.mean() for m in v], [m.std() for m in v], color=color, marker=marker,
                   capsize=2, elinewidth=0.8)
        a.annotate({"best": "searched", "greedy": "greedy"}[kind], (x[-1], v[-1].mean()),
                   xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    a.axhline(col("asia", "ges", "shd_cpdag").mean(), color=MUTED, ls="--", lw=0.8)
    a.annotate("GES", (x[0], col("asia", "ges", "shd_cpdag").mean()), xytext=(0, 4),
               textcoords="offset points", color=MUTED)
    a.set(xlabel=r"generative weight $\beta$", ylabel="CPDAG SHD (lower is better)", ylim=(0, 14))

    v = [col(t, "dqn_mlp_best", "test_neg_js", 1e4) for t in tables]
    b.errorbar(x, [m.mean() for m in v], [m.std() for m in v], color=BLUE, marker="o",
               capsize=2, elinewidth=0.8)
    true = col("asia", "true", "test_neg_js", 1e4).mean()
    b.axhline(true, color=MUTED, ls=":", lw=0.8)          # the true graph; named in the caption
    b.set(xlabel=r"generative weight $\beta$", ylabel=r"searched test JS $\times 10^4$", ylim=(-12, -3))
    for ax in (a, b):
        ax.set_xticks(x, [str(v) for v in betas])
        ax.set_xlim(-0.4, len(betas) - 0.4)
    a.set_xlim(-0.4, len(betas) + 0.4)                      # room for the direct labels
    fig.tight_layout(w_pad=2)
    fig.savefig(FIG / "asia_beta.svg")


def tradeoff():
    groups = [  # (label, method, group)
        ("true graph", "true", 0), ("GES", "ges", 0), ("RLBayes", "rlbayes", 0),
        ("Tabu", "tabu", 0), ("Hill-Climbing", "hc", 0),
        ("DQN", "dqn_mlp_best", 1), ("DQN-emb", "dqn_emb_best", 1), ("Q-learning", "qlearn_best", 1),
        ("DQN", "dqn_mlp_greedy", 2), ("DQN-emb", "dqn_emb_greedy", 2), ("Q-learning", "qlearn_greedy", 2)]
    style = [("baselines", BLUE, "o"), ("RL, searched", AQUA, "^"), ("RL, greedy", ORANGE, "s")]
    # Hand-placed label offsets (points) so nothing collides; the data never moves.
    offset = {"true": (5, 3), "ges": (5, -8), "rlbayes": (-8, 6), "tabu": (5, -9), "hc": (6, -2),
              "dqn_mlp_best": (-6, -8), "dqn_emb_best": (6, 2), "qlearn_best": (5, 3),
              "dqn_mlp_greedy": (5, -9), "dqn_emb_greedy": (5, -2), "qlearn_greedy": (-6, -9)}
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    for g, (name, color, marker) in enumerate(style):
        members = [(lab, m) for lab, m, gg in groups if gg == g]
        xs = [col("asia", m, "shd_cpdag").mean() for _, m in members]
        ys = [col("asia", m, "test_neg_js", 1e4).mean() for _, m in members]
        ax.scatter(xs, ys, color=color, marker=marker, s=28, label=name, zorder=3,
                   edgecolors="white", linewidths=1)
        for (lab, m), xx, yy in zip(members, xs, ys):
            ax.annotate(lab, (xx, yy), xytext=offset[m], textcoords="offset points", color=INK,
                        fontsize=6.5, ha="right" if offset[m][0] < 0 else "left")
    ax.set(xlabel="CPDAG SHD (lower is better)", ylabel=r"test JS $\times 10^4$ (higher is better)",
           xlim=(-0.5, 12.5), ylim=(-12.8, -6.4))
    ax.grid(True, axis="both")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "asia_tradeoff.svg")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    beta()
    tradeoff()
    print(f"wrote {FIG}")
