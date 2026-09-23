"""Step 2: CPT learning (MLE + Dirichlet smoothing) and decomposable BIC.

Hand-rolled on numpy instead of pgmpy: BIC on discrete data is a few dozen
lines, avoids a heavy dependency, and gives us the O(1)-per-edit incremental
`BIC.delta` the RL reward needs.

Data layout: `data` is an int ndarray (n_samples x d); column v holds values
0..cards[v]-1. `cards` is an int array of per-variable cardinalities.

BIC(G) = sum_node local_score(node | parents), decomposable, so editing one
edge only rescores the affected node(s), and class BIC caches each
(node, parent set) so a family is never scored twice:
    local = sum_{parent cfg j, value k} N_jk * log(N_jk / N_j)   (MLE log-lik)
            - 0.5 * log(N) * (cards[node]-1) * (#parent configs)  (complexity)
Higher BIC = better.
"""

import numpy as np

from envs.dag import ADD, DELETE, REVERSE, apply_action


def _parent_config_index(data, parents, pcards):
    """Map each row's parent values to a single mixed-radix integer in [0, q)."""
    if len(parents) == 0:
        return np.zeros(len(data), dtype=int), 1
    radix = np.ones(len(parents), dtype=int)
    for t in range(len(parents) - 2, -1, -1):
        radix[t] = radix[t + 1] * pcards[t + 1]
    idx = data[:, parents] @ radix
    return idx, int(np.prod(pcards))


def parents_of(A, node):
    """Parent set of `node` as a sorted tuple, the local-score cache key."""
    return tuple(np.flatnonzero(A[:, node]).tolist())


def local_score(node, parents, data, cards):
    """Decomposable BIC term for `node` given a parent tuple. Uncached: one pass
    over the data. Use BIC.local, which memoises it, everywhere else."""
    r = int(cards[node])
    parents = list(parents)                       # a tuple would index as multi-dim
    pconfig, q = _parent_config_index(data, parents, cards[parents])
    joint = pconfig * r + data[:, node]           # unique id per (parent cfg, value)
    counts = np.bincount(joint, minlength=q * r).reshape(q, r)

    Nj = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    nz = counts > 0
    ll = float(np.sum(counts[nz] * np.log((counts / Nj)[nz])))
    return ll - 0.5 * np.log(len(data)) * (r - 1) * q


def _affected_nodes(action):
    """Which node's parent set changes under this edit."""
    op, i, j = action
    return [i, j] if op == REVERSE else [j]   # add/delete change child j's parents


class BIC:
    """Cached decomposable BIC over a fixed dataset.

    Local scores are memoised by (node, parent set), not by graph, so:
      - an edit only ever computes the 1-2 nodes whose parent set changed,
        and only the first time that exact parent set is seen;
      - scoring a whole graph is d dict lookups once its families are cached;
      - revisiting families (RL loops, Q-table rows, hill-climb neighbours)
        costs nothing.
    Call it like a function: BIC(data, cards)(A) -> score, so it drops straight
    into RLBayesAgent(score_fn=...).
    """

    def __init__(self, data, cards):
        self.data = np.asarray(data)
        self.cards = np.asarray(cards)
        # ponytail: unbounded cache. Fine while max_indegree <= 3 (ALARM, k=3:
        # about 37 * 7.8k families); add an LRU bound if memory becomes a problem.
        self.cache = {}

    def local(self, node, parents):
        key = (node, parents)
        s = self.cache.get(key)
        if s is None:
            s = self.cache[key] = local_score(node, parents, self.data, self.cards)
        return s

    def __call__(self, A):
        """Full BIC of graph A."""
        return sum(self.local(v, parents_of(A, v)) for v in range(len(A)))

    def delta(self, A, action):
        """BIC(apply(A, action)) - BIC(A), touching only the affected node(s)."""
        B = apply_action(A, action)
        return sum(self.local(v, parents_of(B, v)) - self.local(v, parents_of(A, v))
                   for v in _affected_nodes(action))


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

    from envs.dag import all_actions, empty_dag, legal_action_mask, apply_action as apply
    A = empty_dag(3)
    bic = BIC(data, cards)

    # Adding the true edge 0->1 should raise BIC; adding 0->2 should not.
    d_true = bic.delta(A, (ADD, 0, 1))
    d_noise = bic.delta(A, (ADD, 0, 2))
    assert d_true > 0, d_true
    assert d_noise < 0, d_noise
    assert d_true > d_noise

    # delta must equal the full-BIC difference (decomposability check).
    B = apply(A, (ADD, 0, 1))
    assert abs((bic(B) - bic(A)) - d_true) < 1e-6

    # Random walk over legal edits: cached delta == uncached full rescore every
    # step, for add, delete and reverse alike.
    d, k = 6, 2
    data6 = rng.integers(0, 3, (500, d))
    cards6 = np.full(d, 3)
    bic6 = BIC(data6, cards6)
    fresh = lambda G: sum(local_score(v, parents_of(G, v), data6, cards6)
                          for v in range(d))
    acts = all_actions(d)
    G = empty_dag(d)
    for _ in range(300):
        legal = np.flatnonzero(legal_action_mask(G, k))
        a = acts[int(rng.choice(legal))]
        H = apply(G, a)
        assert abs(bic6.delta(G, a) - (fresh(H) - fresh(G))) < 1e-6, a
        G = H
    # The cache holds one entry per distinct family, not per step.
    assert len(bic6.cache) <= d * (1 + (d - 1) + (d - 1) * (d - 2) // 2)

    # CPT rows are valid distributions and recover P(x1=x0) ~ 0.9.
    cpts = fit_cpts(B, data, cards)
    assert np.allclose(cpts[1]["table"].sum(axis=1), 1.0)
    assert cpts[1]["table"][0, 0] > 0.85 and cpts[1]["table"][1, 1] > 0.85
    print("scoring.bic self-check passed")


if __name__ == "__main__":
    _demo()
