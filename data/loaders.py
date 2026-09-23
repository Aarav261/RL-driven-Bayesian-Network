"""Datasets: bnlearn discrete benchmarks + a synthetic generator with known truth.

Every loader returns (data, cards, A_true, names):
    data   int ndarray (n x d), column v in 0..cards[v]-1
    cards  int ndarray (d,)
    A_true ground-truth adjacency matrix (d x d), A[i, j] == 1 means i -> j
    names  variable names, column order
Raw files live in data/raw/ (gitignored).
"""


def load_bnlearn(name, n, seed=0):
    """ASIA / Sachs / Insurance / Child / Alarm: sample n rows from the .bif
    network and read its true DAG."""
    raise NotImplementedError


def synthetic(d, n, max_indegree=2, seed=0):
    """Random DAG + random Dirichlet CPTs, ancestral-sampled n rows."""
    raise NotImplementedError
