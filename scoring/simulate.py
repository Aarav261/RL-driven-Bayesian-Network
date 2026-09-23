"""Generate a fake dataset from a learned BN by ancestral sampling.

Process nodes in topological order; each node is sampled from its CPT row for
the parent values already drawn. Vectorised over rows: one pass per node.
"""

import numpy as np

from envs.dag import topological_order
from scoring.bic import _parent_config_index


def ancestral_sample(A, cpts, n, rng):
    """Return n rows (int ndarray, n x d) sampled from the BN (A, cpts).
    `cpts` is scoring.bic.fit_cpts output; `rng` a numpy Generator (seed control)."""
    X = np.zeros((n, len(A)), dtype=int)
    for v in topological_order(A):
        c = cpts[v]
        pconfig, _ = _parent_config_index(X, list(c["parents"]), c["pcards"])
        cdf = np.cumsum(c["table"][pconfig], axis=1)          # n x r, row per sample
        u = rng.random((n, 1))
        X[:, v] = np.minimum((u > cdf).sum(axis=1), c["r"] - 1)  # guard float round-off
    return X


def _demo():
    from envs.dag import ADD, apply_action, empty_dag
    from scoring.bic import fit_cpts

    rng = np.random.default_rng(0)
    n = 20000
    x0 = rng.integers(0, 3, n)
    x1 = np.where(rng.random(n) < 0.8, x0, rng.integers(0, 3, n))   # x1 copies x0 80%
    data = np.stack([x0, x1], axis=1)
    cards = np.array([3, 3])

    A = apply_action(empty_dag(2), (ADD, 0, 1))
    cpts = fit_cpts(A, data, cards)
    fake = ancestral_sample(A, cpts, n, np.random.default_rng(1))
    assert fake.shape == (n, 2) and fake.min() >= 0 and fake[:, 0].max() <= 2
    # The sample reproduces both the marginal of x0 and the x0 -> x1 dependence.
    assert np.allclose(np.bincount(fake[:, 0]) / n, np.bincount(x0) / n, atol=0.02)
    assert abs((fake[:, 0] == fake[:, 1]).mean() - (x0 == x1).mean()) < 0.02
    # Same seed, same sample.
    again = ancestral_sample(A, cpts, n, np.random.default_rng(1))
    assert (fake == again).all()
    print("scoring.simulate self-check passed")


if __name__ == "__main__":
    _demo()
