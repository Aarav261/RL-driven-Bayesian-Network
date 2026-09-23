"""Required baseline 2: score-equivalent Greedy Equivalence Search (GES).

A thin wrapper over pgmpy's GES (Chickering, 2002) with the discrete BIC score.
GES returns a CPDAG; we extend it to one member DAG so it fits the same metrics.
Compare it with eval.metrics.shd_cpdag, since the extension's choice of direction
for reversible edges is arbitrary. pgmpy's GES has no in-degree cap.
"""

import warnings

import numpy as np


def ges(data, cards):
    """Run GES on `data` (n x d ints). Returns our DAG adjacency matrix."""
    with warnings.catch_warnings():               # pgmpy warns on import; not ours to fix
        warnings.simplefilter("ignore")
        import pandas as pd
        from pgmpy.estimators import GES
        d = len(cards)
        # Categorical columns keep states that never appear in the sample.
        df = pd.DataFrame({v: pd.Categorical(data[:, v], categories=range(cards[v]))
                           for v in range(d)})
        dag = GES(df).estimate(scoring_method="bic-d").to_dag()
    A = np.zeros((d, d), dtype=int)
    for u, v in dag.edges():
        A[u, v] = 1
    return A


def _demo():
    from data.loaders import load_bnlearn
    from envs.dag import is_acyclic
    from eval.metrics import shd, shd_cpdag

    data, cards, A_true, _ = load_bnlearn("asia", 5000, seed=0)
    A = ges(data, cards)
    assert is_acyclic(A)
    print(f"baselines.ges self-check passed  (ASIA n=5000: SHD {shd(A, A_true)}, "
          f"CPDAG SHD {shd_cpdag(A, A_true)})")


if __name__ == "__main__":
    _demo()
