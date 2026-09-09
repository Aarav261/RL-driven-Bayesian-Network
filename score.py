"""Step 2: CPT learning (MLE + Dirichlet smoothing) and decomposable BIC.

Hand-rolled on numpy instead of pgmpy: BIC on discrete data is a few dozen
lines, avoids a heavy dependency, and gives us the O(1)-per-edit incremental
`delta_bic` the RL reward needs.

Data layout: `data` is an int ndarray (n_samples x d); column v holds values
0..cards[v]-1. `cards` is an int array of per-variable cardinalities.

BIC(G) = sum_node local_score(node | parents), decomposable, so editing one
edge only rescoring the affected node(s):
    local = sum_{parent cfg j, value k} N_jk * log(N_jk / N_j)   (MLE log-lik)
            - 0.5 * log(N) * (cards[node]-1) * (#parent configs)  (complexity)
Higher BIC = better.
"""

import numpy as np

from .dag import ADD, DELETE, REVERSE, apply_action


def _parent_config_index(data, parents, pcards):
    """Map each row's parent values to a single mixed-radix integer in [0, q)."""
    if len(parents) == 0:
        return np.zeros(len(data), dtype=int), 1
    radix = np.ones(len(parents), dtype=int)
    for t in range(len(parents) - 2, -1, -1):
        radix[t] = radix[t + 1] * pcards[t + 1]
    idx = data[:, parents] @ radix
    return idx, int(np.prod(pcards))


def local_score(node, A, data, cards):
    """Decomposable BIC term for `node` given its parents in DAG A."""
    N = len(data)
    r = int(cards[node])
    parents = np.where(A[:, node] == 1)[0]
    pcards = cards[parents]
    pconfig, q = _parent_config_index(data, parents, pcards)

    joint = pconfig * r + data[:, node]          # unique id per (parent cfg, value)
    counts = np.bincount(joint, minlength=q * r).reshape(q, r)

    ll = 0.0
    for row in counts:
        Nj = row.sum()
        if Nj > 0:
            nz = row[row > 0]
            ll += float(np.sum(nz * np.log(nz / Nj)))

    free_params = (r - 1) * q
    return ll - 0.5 * np.log(N) * free_params


def bic(A, data, cards):
    """Full BIC of graph A."""
    return sum(local_score(v, A, data, cards) for v in range(len(A)))


def _affected_nodes(action):
    """Which node's parent set changes under this edit."""
    op, i, j = action
    return [i, j] if op == REVERSE else [j]   # add/delete change child j's parents


def delta_bic(A, action, data, cards):
    """BIC(apply(A, action)) - BIC(A), rescoring only affected node(s)."""
    B = apply_action(A, action)
    nodes = _affected_nodes(action)
    return sum(local_score(v, B, data, cards) - local_score(v, A, data, cards)
               for v in nodes)


def fit_cpts(A, data, cards, dirichlet_alpha=1.0):
    """MLE CPTs with Dirichlet smoothing. Returns {node: {parents, pcards, radix,
    r, table}} where table[j] is P(node | parent-config j). Used by simulate.py."""
    cpts = {}
    for node in range(len(A)):
        r = int(cards[node])
        parents = np.where(A[:, node] == 1)[0]
        pcards = cards[parents]
        pconfig, q = _parent_config_index(data, parents, pcards)
        joint = pconfig * r + data[:, node]
        counts = np.bincount(joint, minlength=q * r).reshape(q, r).astype(float)
        counts += dirichlet_alpha                       # smoothing
        table = counts / counts.sum(axis=1, keepdims=True)
        cpts[node] = {"parents": parents, "pcards": pcards, "r": r, "table": table}
    return cpts


def _demo():
    rng = np.random.default_rng(0)
    n = 4000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)   # x1 strongly depends on x0
    x2 = rng.integers(0, 2, n)                        # x2 independent
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])

    from .dag import empty_dag, apply_action as apply
    A = empty_dag(3)

    # Adding the true edge 0->1 should raise BIC; adding 0->2 should not.
    d_true = delta_bic(A, (ADD, 0, 1), data, cards)
    d_noise = delta_bic(A, (ADD, 0, 2), data, cards)
    assert d_true > 0, d_true
    assert d_noise < 0, d_noise
    assert d_true > d_noise

    # delta_bic must equal the full-BIC difference (decomposability check).
    B = apply(A, (ADD, 0, 1))
    assert abs((bic(B, data, cards) - bic(A, data, cards)) - d_true) < 1e-6

    # CPT rows are valid distributions and recover P(x1=x0) ~ 0.9.
    cpts = fit_cpts(B, data, cards)
    assert np.allclose(cpts[1]["table"].sum(axis=1), 1.0)
    assert cpts[1]["table"][0, 0] > 0.85 and cpts[1]["table"][1, 1] > 0.85
    print("score.py self-check passed")


if __name__ == "__main__":
    _demo()
