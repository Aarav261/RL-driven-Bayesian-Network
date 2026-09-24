"""The repro script: read a config, run every method on every seed, write a table.

    python -m scripts.repro --config configs/asia.yaml            (from rlig/)
    python -m scripts.repro --config configs/asia.yaml --seeds 2  (first 2 seeds only)
    python -m scripts.repro --config configs/asia.yaml --workers 4
    python -m scripts.repro --methods dqn_mlp --set beta=0   (ablation -> asia_beta0_dqn_mlp.csv)

Seeds run in parallel (one process each). Torch is held to one thread per process, so
`seconds` is single-core time, comparable across methods, and 8 workers do not fight
over 8 cores with 8 threads each.

Per seed: sample n rows from the network, split train / val / test, then
    - HC, Tabu, GES and RLBayes learn from train (BIC only);
    - Q-learning learns from train, with GenScore rewarded on val. Two rows:
      qlearn_best (best graph the search saw, by the hybrid score) and qlearn_greedy
      (the learned policy's greedy rollout). DQN (HD extension 1) gives the same two
      rows on the same env and budget, once per Q-head: dqn_mlp_* (one output per
      action) and dqn_emb_* (phi(s) . psi(a) over (op, i, j) embeddings);
    - every method is reported on test, which no method sees during learning.
steps = graph edits made (HC/Tabu moves, RLBayes iterations, Q-learning env steps);
gen_evals = GenScore simulations; unseen = greedy moves from states not in the Q-table.
Writes report/tables/<dataset>.csv (one row per method per seed) and prints
mean +- std per method.
"""

import argparse
import csv
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.dqn import DQNAgent
from agents.qlearn import QLearningAgent
from agents.rlbayes import RLBayesAgent
from baselines.ges import ges
from baselines.hill_climb import hill_climb
from data.loaders import load_bnlearn, split, synthetic
from envs.env import RLiGEnv
from eval.metrics import precision_recall_f1, shd, shd_cpdag
from scoring.bic import BIC, fit_cpts
from scoring.genscore import held_out_loglik, neg_js
from scoring.simulate import ancestral_sample

torch.set_num_threads(1)   # module level, so every spawned worker runs it too

OUT = Path(__file__).parent.parent / "report" / "tables"
METRICS = ["shd", "shd_cpdag", "f1", "bic_per_row", "test_loglik", "test_neg_js", "seconds",
           "steps", "gen_evals", "unseen"]


def evaluate(A, A_true, bic, train, test, cards, cfg, seed):
    cpts = fit_cpts(A, train, cards, cfg["dirichlet_alpha"])
    fake = ancestral_sample(A, cpts, cfg["N_s"], np.random.default_rng(seed))
    return {"shd": shd(A, A_true), "shd_cpdag": shd_cpdag(A, A_true),
            "f1": precision_recall_f1(A, A_true)[2], "bic_per_row": bic(A) / len(train),
            "test_loglik": held_out_loglik(cpts, test), "test_neg_js": neg_js(test, fake, cards)}


def run_seed(cfg, seed, only=None):
    if cfg["dataset"] == "synthetic":     # a fresh random DAG per seed; knobs in cfg["synthetic"]
        data, cards, A_true, _ = synthetic(n=cfg["n"], seed=seed, **cfg["synthetic"])
    else:
        data, cards, A_true, _ = load_bnlearn(cfg["dataset"], cfg["n"], seed=seed)
    train, val, test = split(data, cfg["split"], seed=seed)
    k, bic = cfg["max_indegree_k"], BIC(train, cards)

    # Each learner returns {row name: (DAG, extra columns)}.
    def rl(name, make_agent):
        env = RLiGEnv(train, val, cards, cfg, seed=seed)
        agent = make_agent(env)
        agent.train(cfg["episodes"])
        best = agent.best_searched()[0]
        greedy = agent.best_graph()[0]
        extra = {"steps": agent.steps, "gen_evals": env.n_gen_evals}
        return {f"{name}_best": (best, extra),
                f"{name}_greedy": (greedy, dict(extra, unseen=agent.greedy_unseen))}

    qlearn = lambda: rl("qlearn", lambda env: QLearningAgent(
        env, cfg["lr"], cfg["gamma"], cfg["epsilon"], cfg["epsilon_decay"], cfg["epsilon_min"],
        seed=seed))
    dqn = lambda head: lambda: rl(f"dqn_{head}", lambda env: DQNAgent(
        env, cfg["dqn_lr"], cfg["gamma"], cfg["dqn_epsilon"], cfg["dqn_epsilon_decay"],
        cfg["dqn_epsilon_min"], cfg["dqn_hidden"], cfg["dqn_batch"], cfg["dqn_buffer"],
        cfg["dqn_train_every"], cfg["dqn_target_every"], seed=seed, head=head,
        emb=cfg["dqn_emb_dim"]))

    def hc(**kw):
        A, _, hist = hill_climb(train, cards, k, bic=bic, **kw)
        return A, {"steps": len(hist) - 1}

    def rlbayes():
        agent = RLBayesAgent(len(cards), bic, k, cfg["rlbayes_max_len"], cfg["rlbayes_max_iter"],
                             cfg["rlbayes_theta"], seed=seed)
        return agent.train()[0], {"steps": agent.max_iter}

    methods = {
        "true": lambda: {"true": (A_true, {})},
        "hc": lambda: {"hc": hc()},
        "tabu": lambda: {"tabu": hc(tabu_len=cfg["hc_tabu_len"], max_no_improve=cfg["hc_max_no_improve"])},
        "ges": lambda: {"ges": (ges(train, cards), {})},
        "rlbayes": lambda: {"rlbayes": rlbayes()},
        "qlearn": qlearn,
        "dqn_mlp": dqn("mlp"),
        "dqn_emb": dqn("emb"),
    }
    rows = []
    for learn in [methods[m] for m in only or methods]:
        t = time.perf_counter()
        out = learn()
        secs = time.perf_counter() - t
        for name, (A, extra) in out.items():
            row = {"method": name, "seed": seed, "seconds": secs, **extra}
            row.update(evaluate(A, A_true, bic, train, test, cards, cfg, seed))
            rows.append(row)
            print(f"  seed {seed} {name:14s} SHD {row['shd']:2d}  CPDAG SHD {row['shd_cpdag']:2d}  "
                  f"BIC/N {row['bic_per_row']:.4f}  steps {row.get('steps', '-')}  {secs:.1f}s")
    return rows


def main(config_path, n_seeds=None, workers=4, only=None, overrides=()):
    cfg = yaml.safe_load(Path(config_path).read_text())
    sets = dict(o.split("=", 1) for o in overrides)
    cfg.update({k: yaml.safe_load(v) for k, v in sets.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    # An ablation or a subset of methods gets its own file, so it never overwrites the main table.
    tag = "".join(f"_{k}{v}" for k, v in sets.items()) + ("_" + "-".join(only) if only else "")
    path = OUT / f"{cfg['dataset']}{tag}.csv"
    seeds, rows = cfg["seeds"][:n_seeds], []
    with Pool(min(workers, len(seeds))) as pool:
        for seed_rows in pool.imap_unordered(partial(run_seed, cfg, only=only), seeds):
            rows += seed_rows
            rows.sort(key=lambda r: r["seed"])       # stable: method order kept within a seed
            with open(path, "w", newline="") as f:   # after every seed: a kill loses only running seeds
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
    ap.add_argument("--workers", type=int, default=4, help="seeds run at once (RAM: ~0.5 GB each)")
    ap.add_argument("--methods", type=lambda s: s.split(","), default=None,
                    help="comma list, e.g. dqn_mlp,qlearn (default: all)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set beta=0 (repeatable)")
    args = ap.parse_args()
    main(args.config, args.seeds, args.workers, args.methods, args.set)
