"""Required baseline 1: Hill-Climbing (add/delete/reverse) with BIC, optional Tabu.

Plain HC: from the empty graph, take the legal edit with the largest dBIC; stop at
the first local optimum (no edit improves).

Tabu (as in bnlearn's `tabu`): keep moving to the best edit even when it lowers
BIC, but never to one of the last `tabu_len` graphs visited. Return the best graph
seen; stop after `max_no_improve` moves without beating it.
"""

from collections import deque

import numpy as np

from envs.dag import all_actions, apply_action, empty_dag, legal_action_mask
from scoring.bic import BIC


def hill_climb(data, cards, max_indegree, tabu_len=0, max_no_improve=10,
               max_iter=100_000, bic=None):
    """Returns (best DAG, its BIC, BIC after every move) for learning curves.
    Pass a shared `bic` to reuse its local-score cache across runs."""
    bic = bic or BIC(data, cards)
    actions = all_actions(len(cards))
    A = empty_dag(len(cards))
    score = best = bic(A)
    best_A, history, stale = A, [score], 0
    tabu = deque([A.tobytes()], maxlen=tabu_len)   # maxlen 0: plain HC, nothing is tabu

    for _ in range(max_iter):
        legal = [actions[i] for i in np.flatnonzero(legal_action_mask(A, max_indegree))]
        if not legal:
            break
        # ponytail: rescores every legal edit per move (cached, ~4k lookups at d=37);
        # track per-node deltas incrementally if HC ever dominates the runtime.
        deltas = np.array([bic.delta(A, a) for a in legal])
        for i in np.argsort(-deltas, kind="stable"):
            B = apply_action(A, legal[i])
            if B.tobytes() not in tabu:
                break
        else:
            break                                  # every legal move is tabu
        if deltas[i] <= 0 and not tabu_len:
            break                                  # plain HC: local optimum
        A, score = B, bic(B)
        tabu.append(A.tobytes())
        history.append(score)
        if score > best + 1e-9:
            best_A, best, stale = A, score, 0
        else:
            stale += 1
            if stale >= max_no_improve:
                break
    return best_A, best, history


def _demo():
    from data.loaders import synthetic
    from envs.dag import is_acyclic
    from eval.metrics import shd

    # Collider 0 -> 2 <- 1 plus 2 -> 3: a single-member equivalence class, so BIC
    # identifies every orientation and HC should recover it exactly.
    rng = np.random.default_rng(0)
    n = 5000
    x0, x1 = rng.integers(0, 2, n), rng.integers(0, 2, n)
    x2 = np.where(rng.random(n) < 0.9, x0 | x1, 1 - (x0 | x1))   # OR, not XOR: XOR is
    # pairwise independent of each parent, which greedy one-edge search cannot see.
    x3 = np.where(rng.random(n) < 0.85, x2, 1 - x2)
    data = np.stack([x0, x1, x2, x3], axis=1)
    A_true = np.zeros((4, 4), dtype=int)
    A_true[0, 2] = A_true[1, 2] = A_true[2, 3] = 1
    A, s, hist = hill_climb(data, np.array([2, 2, 2, 2]), max_indegree=2)
    assert shd(A, A_true) == 0, A
    assert all(b >= a for a, b in zip(hist, hist[1:]))        # plain HC only climbs

    # Random 12-node network (seed 4 is one where plain HC gets stuck).
    data, cards, A_true, _ = synthetic(d=12, n=5000, max_indegree=2, seed=4)
    bic = BIC(data, cards)
    A_hc, s_hc, _ = hill_climb(data, cards, 2, bic=bic)
    A_tb, s_tb, _ = hill_climb(data, cards, 2, tabu_len=10, bic=bic)
    assert is_acyclic(A_hc) and A_hc.sum(axis=0).max() <= 2
    # Plain HC stops only at a true local optimum: no legal edit improves BIC.
    acts = all_actions(12)
    assert max(bic.delta(A_hc, acts[i]) for i in np.flatnonzero(legal_action_mask(A_hc, 2))) <= 0
    # Tabu never ends worse, and here it escapes the local optimum to the true graph's BIC.
    assert s_tb >= s_hc
    assert s_hc < bic(A_true) - 1 and abs(s_tb - bic(A_true)) < 1e-6
    print(f"baselines.hill_climb self-check passed  (synthetic d=12: BIC hc {s_hc:.1f} "
          f"SHD {shd(A_hc, A_true)}; tabu {s_tb:.1f} SHD {shd(A_tb, A_true)}; true {bic(A_true):.1f})")

if __name__ == "__main__":
    _demo()
