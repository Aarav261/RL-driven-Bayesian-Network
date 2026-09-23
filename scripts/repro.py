"""The repro script: read a config, run every method on every seed, write a table.

    python -m scripts.repro --config configs/asia.yaml            (from rlig/)
    python -m scripts.repro --config configs/asia.yaml --seeds 2  (first 2 seeds only)

Per seed: sample n rows from the network, split train / val / test, then
    - HC, Tabu, GES and RLBayes learn from train (BIC only);
    - Q-learning learns from train, with GenScore rewarded on val. Two rows:
      qlearn_best (best graph the search saw, by the hybrid score) and qlearn_greedy
      (the learned policy's greedy rollout);
    - every method is reported on test, which no method sees during learning.
steps = graph edits made (HC/Tabu moves, RLBayes iterations, Q-learning env steps);
gen_evals = GenScore simulations; unseen = greedy moves from states not in the Q-table.
Writes report/tables/<dataset>.csv (one row per method per seed) and prints
mean +- std per method.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from agents.qlearn import QLearningAgent
from agents.rlbayes import RLBayesAgent
from baselines.ges import ges
from baselines.hill_climb import hill_climb
from data.loaders import load_bnlearn, split
from envs.env import RLiGEnv
from eval.metrics import precision_recall_f1, shd, shd_cpdag
from scoring.bic import BIC, fit_cpts
from scoring.genscore import held_out_loglik, neg_js
from scoring.simulate import ancestral_sample

OUT = Path(__file__).parent.parent / "report" / "tables"
METRICS = ["shd", "shd_cpdag", "f1", "bic_per_row", "test_loglik", "test_neg_js", "seconds",
           "steps", "gen_evals", "unseen"]


def evaluate(A, A_true, bic, train, test, cards, cfg, seed):
    cpts = fit_cpts(A, train, cards, cfg["dirichlet_alpha"])
    fake = ancestral_sample(A, cpts, cfg["N_s"], np.random.default_rng(seed))
    return {"shd": shd(A, A_true), "shd_cpdag": shd_cpdag(A, A_true),
            "f1": precision_recall_f1(A, A_true)[2], "bic_per_row": bic(A) / len(train),
            "test_loglik": held_out_loglik(cpts, test), "test_neg_js": neg_js(test, fake, cards)}


def run_seed(cfg, seed):
    data, cards, A_true, _ = load_bnlearn(cfg["dataset"], cfg["n"], seed=seed)
    train, val, test = split(data, cfg["split"], seed=seed)
    k, bic = cfg["max_indegree_k"], BIC(train, cards)

    # Each learner returns {row name: (DAG, extra columns)}.
    def qlearn():
        env = RLiGEnv(train, val, cards, cfg, seed=seed)
        agent = QLearningAgent(env, cfg["lr"], cfg["gamma"], cfg["epsilon"], cfg["epsilon_decay"],
                               cfg["epsilon_min"], seed=seed)
        agent.train(cfg["episodes"])
        best = agent.best_searched()[0]
        greedy = agent.best_graph()[0]
        extra = {"steps": agent.steps, "gen_evals": env.n_gen_evals}
        return {"qlearn_best": (best, extra),
                "qlearn_greedy": (greedy, dict(extra, unseen=agent.greedy_unseen))}

    def hc(**kw):
        A, _, hist = hill_climb(train, cards, k, bic=bic, **kw)
        return A, {"steps": len(hist) - 1}

    def rlbayes():
        agent = RLBayesAgent(len(cards), bic, k, cfg["rlbayes_max_len"], cfg["rlbayes_max_iter"],
                             cfg["rlbayes_theta"], seed=seed)
        return agent.train()[0], {"steps": agent.max_iter}

    methods = [
        lambda: {"true": (A_true, {})},
        lambda: {"hc": hc()},
        lambda: {"tabu": hc(tabu_len=cfg["hc_tabu_len"], max_no_improve=cfg["hc_max_no_improve"])},
        lambda: {"ges": (ges(train, cards), {})},
        lambda: {"rlbayes": rlbayes()},
        qlearn,
    ]
    rows = []
    for learn in methods:
        t = time.perf_counter()
        out = learn()
        secs = time.perf_counter() - t
        for name, (A, extra) in out.items():
            row = {"method": name, "seed": seed, "seconds": secs, **extra}
            row.update(evaluate(A, A_true, bic, train, test, cards, cfg, seed))
            rows.append(row)
            print(f"  seed {seed} {name:13s} SHD {row['shd']:2d}  CPDAG SHD {row['shd_cpdag']:2d}  "
                  f"BIC/N {row['bic_per_row']:.4f}  steps {row.get('steps', '-')}  {secs:.1f}s")
    return rows


def main(config_path, n_seeds=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    rows = [r for seed in cfg["seeds"][:n_seeds] for r in run_seed(cfg, seed)]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{cfg['dataset']}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "seed"] + METRICS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'method':13s} " + " ".join(f"{m:>18s}" for m in METRICS))
    for name in dict.fromkeys(r["method"] for r in rows):
        vals = [[r[m] for r in rows if r["method"] == name and m in r] for m in METRICS]
        print(f"{name:13s} " + " ".join(f"{np.mean(v):9.4f} +-{np.std(v):7.4f}" if v else f"{'-':>18s}"
                                        for v in vals))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asia.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="run only the first N seeds")
    args = ap.parse_args()
    main(args.config, args.seeds)
