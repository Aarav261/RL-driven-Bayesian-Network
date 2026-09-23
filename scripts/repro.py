"""The repro script: read a config, run every method on every seed, write a table.

    python -m scripts.repro --config configs/asia.yaml            (from rlig/)
    python -m scripts.repro --config configs/asia.yaml --seeds 2  (first 2 seeds only)

Per seed: sample n rows from the network, split train / val / test, then
    - HC, Tabu, GES and RLBayes learn from train (BIC only);
    - Q-learning learns from train, with GenScore rewarded on val;
    - every method is reported on test, which no method sees during learning.
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
METRICS = ["shd", "shd_cpdag", "f1", "bic_per_row", "test_loglik", "test_neg_js", "seconds"]


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

    def qlearn():
        env = RLiGEnv(train, val, cards, cfg, seed=seed)
        agent = QLearningAgent(env, cfg["lr"], cfg["gamma"], cfg["epsilon"],
                               cfg["epsilon_decay"], seed=seed)
        agent.train(cfg["episodes"])
        return agent.best_graph()[0]

    methods = {
        "true": lambda: A_true,
        "hc": lambda: hill_climb(train, cards, k, bic=bic)[0],
        "tabu": lambda: hill_climb(train, cards, k, tabu_len=cfg["hc_tabu_len"],
                                   max_no_improve=cfg["hc_max_no_improve"], bic=bic)[0],
        "ges": lambda: ges(train, cards),
        "rlbayes": lambda: RLBayesAgent(len(cards), bic, k, cfg["rlbayes_max_len"],
                                        cfg["rlbayes_max_iter"], cfg["rlbayes_theta"],
                                        seed=seed).train()[0],
        "qlearn": qlearn,
    }
    rows = []
    for name, learn in methods.items():
        t = time.perf_counter()
        A = learn()
        row = {"method": name, "seed": seed, "seconds": time.perf_counter() - t}
        row.update(evaluate(A, A_true, bic, train, test, cards, cfg, seed))
        rows.append(row)
        print(f"  seed {seed} {name:8s} SHD {row['shd']:2d}  CPDAG SHD {row['shd_cpdag']:2d}  "
              f"BIC/N {row['bic_per_row']:.4f}  {row['seconds']:.1f}s")
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

    print(f"\n{'method':8s} " + " ".join(f"{m:>18s}" for m in METRICS))
    for name in dict.fromkeys(r["method"] for r in rows):
        vals = [[r[m] for r in rows if r["method"] == name] for m in METRICS]
        print(f"{name:8s} " + " ".join(f"{np.mean(v):9.4f} +-{np.std(v):7.4f}" for v in vals))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asia.yaml")
    ap.add_argument("--seeds", type=int, default=None, help="run only the first N seeds")
    args = ap.parse_args()
    main(args.config, args.seeds)
