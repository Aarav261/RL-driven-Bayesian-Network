"""Step 7: evaluation metrics — structure quality + generation quality.

Structure (vs. ground-truth DAG): SHD, CPDAG SHD, edge precision/recall/F1, final BIC.
Generation: held-out log-likelihood, MMD/JS across marginals, posterior
predictive checks (empirical vs. simulated moments/histograms).
"""

import numpy as np


def shd(A_pred, A_true):
    """Structural Hamming Distance: # of edge additions/deletions/reversals to
    turn A_pred into A_true. This one is implemented so evaluate.py is usable."""
    A_pred, A_true = np.asarray(A_pred), np.asarray(A_true)
    diff = 0
    d = len(A_true)
    for i in range(d):
        for j in range(i + 1, d):
            p = (A_pred[i, j], A_pred[j, i])
            t = (A_true[i, j], A_true[j, i])
            if p != t:
                diff += 1  # any mismatch on this pair (missing/extra/reversed) = 1
    return diff


def cpdag(A):
    """CPDAG of a DAG: C[i, j] = C[j, i] = 1 for a reversible (undirected) edge,
    C[i, j] = 1 alone for a compelled i -> j. Orient v-structures, then Meek rules 1-3
    until nothing changes (sound and complete for a DAG's equivalence class)."""
    A = np.asarray(A, dtype=bool)
    d = len(A)
    adj = A | A.T
    C = adj.copy()
    for k in range(d):                                   # v-structures i -> k <- j
        pa = np.flatnonzero(A[:, k])
        for i in pa:
            if any(not adj[i, j] for j in pa if j != i):
                C[k, i] = False
    nonadj = ~adj & ~np.eye(d, dtype=bool)
    while True:
        und, dirc = C & C.T, C & ~C.T
        # R1: a -> b - c, a and c non-adjacent  =>  b -> c
        # R2: a -> b -> c and a - c             =>  a -> c
        orient = und & (((dirc.T.astype(int) @ nonadj.astype(int)) > 0)
                        | ((dirc.astype(int) @ dirc.astype(int)) > 0))
        # R3: a - c -> b, a - d -> b, a - b, c and d non-adjacent  =>  a -> b
        for a, b in zip(*np.nonzero(und)):
            cs = np.flatnonzero(und[a] & dirc[:, b])
            if any(nonadj[c, e] for c in cs for e in cs):
                orient[a, b] = True
        if not orient.any():
            return C.astype(int)
        C &= ~orient.T


def shd_cpdag(A_pred, A_true):
    """SHD between CPDAGs: ignores edge directions no score can tell apart."""
    return shd(cpdag(A_pred), cpdag(A_true))


def precision_recall_f1(A_pred, A_true):
    """Directed-edge precision, recall, F1."""
    A_pred, A_true = np.asarray(A_pred), np.asarray(A_true)
    tp = int(((A_pred == 1) & (A_true == 1)).sum())
    pred_pos = int((A_pred == 1).sum())
    true_pos = int((A_true == 1).sum())
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _demo():
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 0->1->2
    perfect = true.copy()
    assert shd(perfect, true) == 0
    assert precision_recall_f1(perfect, true) == (1.0, 1.0, 1.0)

    reversed_edge = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])  # 1->0 instead of 0->1
    assert shd(reversed_edge, true) == 1  # one pair differs

    # A chain and its full reversal are Markov equivalent; a collider is not.
    back = true.T.copy()                                  # 2->1->0
    assert shd(back, true) == 2 and shd_cpdag(back, true) == 0
    collider = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])  # 0->2<-1
    assert (cpdag(collider) == collider).all()

    # ASIA: 5 compelled edges (2 v-structures + either->xray by R1), 3 reversible.
    from data.loaders import read_bif
    A, _, _, names, _ = read_bif("asia")
    C = cpdag(A).astype(bool)
    und = {frozenset((names[i], names[j])) for i, j in zip(*np.nonzero(C & C.T))}
    assert und == {frozenset(p) for p in [("asia", "tub"), ("smoke", "lung"), ("smoke", "bronc")]}
    assert ((C & ~C.T) <= A).all() and int((C & ~C.T).sum()) == 5
    print("eval.metrics self-check passed")


if __name__ == "__main__":
    _demo()
