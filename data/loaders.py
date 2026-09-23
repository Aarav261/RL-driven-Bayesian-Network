"""Datasets: bnlearn discrete benchmarks + a synthetic generator with known truth.

Every loader returns (data, cards, A_true, names):
    data   int ndarray (n x d), column v in 0..cards[v]-1
    cards  int ndarray (d,)
    A_true ground-truth adjacency matrix (d x d), A[i, j] == 1 means i -> j
    names  variable names, column order
The five bnlearn networks are committed in data/bif/ (from bnlearn.com/bnrepository),
so nothing downloads at run time.

pgmpy is used ONLY here, to parse .bif, and in tests as a reference. read_bif converts
the network once into our own (A, cpts) format; scoring, sampling and RL never see a
pgmpy object, so a pgmpy API change can only break this file.
"""

import warnings
from pathlib import Path

import numpy as np

from scoring.simulate import ancestral_sample

BIF_DIR = Path(__file__).parent / "bif"
BENCHMARKS = ("asia", "sachs", "child", "insurance", "alarm")


def _pgmpy_model(name):
    with warnings.catch_warnings():               # pgmpy warns on import; not ours to fix
        warnings.simplefilter("ignore")
        from pgmpy.readwrite import BIFReader
        return BIFReader(str(BIF_DIR / f"{name}.bif")).get_model()


def read_bif(name):
    """Parse data/bif/<name>.bif and convert it to our format.
    Returns (A, cpts, cards, names, states); cpts matches scoring.bic.fit_cpts output,
    states[v] lists node v's state labels, index = our integer code."""
    model = _pgmpy_model(name)
    names = list(model.nodes())
    idx = {v: i for i, v in enumerate(names)}
    states = [list(model.get_cpds(v).state_names[v]) for v in names]
    cards = np.array([len(s) for s in states])
    A = np.zeros((len(names), len(names)), dtype=int)
    for u, v in model.edges():
        A[idx[u], idx[v]] = 1

    cpts = {}
    for v in names:
        cpd = model.get_cpds(v)
        axes = list(cpd.variables)                # [child, p1, p2, ...]; values in this order
        assert set(axes[1:]) == {names[p] for p in np.flatnonzero(A[:, idx[v]])}, v
        vals = np.asarray(cpd.values, dtype=float)
        # Re-index each axis into the variable's own state order (a CPD may list a
        # parent's states differently from that parent's own CPD).
        for ax, var in enumerate(axes):
            perm = [cpd.state_names[var].index(s) for s in states[idx[var]]]
            vals = np.take(vals, perm, axis=ax)
        # Parents ascending by node index (as fit_cpts does), child axis last, so a
        # C-order reshape puts row j at the mixed-radix parent config _parent_config_index uses.
        parents = sorted(axes[1:], key=idx.get)
        vals = vals.transpose([axes.index(p) for p in parents] + [0])
        pi = np.array([idx[p] for p in parents], dtype=int)
        r = int(cards[idx[v]])
        cpts[idx[v]] = {"parents": pi, "pcards": cards[pi], "r": r,
                        "table": vals.reshape(-1, r)}
    return A, cpts, cards, names, states


def load_bnlearn(name, n, seed=0):
    """Sample n rows from a bnlearn network with our own ancestral sampler."""
    A, cpts, cards, names, _ = read_bif(name)
    data = ancestral_sample(A, cpts, n, np.random.default_rng(seed))
    return data, cards, A, names


def split(data, fracs, seed=0):
    """Shuffle rows, then cut into len(fracs) parts (e.g. train / val / test)."""
    idx = np.random.default_rng(seed).permutation(len(data))
    cuts = np.round(np.cumsum(fracs)[:-1] / np.sum(fracs) * len(data)).astype(int)
    return [data[i] for i in np.split(idx, cuts)]


def synthetic(d, n, max_indegree=2, card=3, edge_prob=0.3, concentration=0.5, seed=0):
    """Random DAG + random CPTs, ancestral-sampled n rows.

    DAG: random topological order; each earlier node becomes a parent with
    probability edge_prob, capped at max_indegree. CPT rows ~ Dirichlet(concentration);
    below 1 the rows are peaked, so dependencies are strong enough to be learnable.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    A = np.zeros((d, d), dtype=int)
    for pos, v in enumerate(order):
        cand = order[:pos][rng.random(pos) < edge_prob]
        A[rng.permutation(cand)[:max_indegree], v] = 1
    cards = np.full(d, card)
    cpts = {}
    for v in range(d):
        parents = np.flatnonzero(A[:, v])
        q = int(np.prod(cards[parents]))
        cpts[v] = {"parents": parents, "pcards": cards[parents], "r": card,
                   "table": rng.dirichlet(np.full(card, concentration), size=q)}
    data = ancestral_sample(A, cpts, n, rng)
    return data, cards, A, [f"X{v}" for v in range(d)]


def _demo():
    from itertools import product

    from scoring.genscore import neg_js

    # bnlearn repository sizes (nodes, arcs): an independent check on the parse.
    known = {"asia": (8, 8), "sachs": (11, 17), "child": (20, 25),
             "insurance": (27, 52), "alarm": (37, 46)}
    for name in BENCHMARKS:
        A, cpts, cards, names, states = read_bif(name)
        model = _pgmpy_model(name)
        assert (len(names), int(A.sum())) == known[name] == (len(model.nodes()), len(model.edges()))
        for v, c in cpts.items():
            assert c["table"].shape == (int(np.prod(c["pcards"])), c["r"])
            assert np.allclose(c["table"].sum(axis=1), 1.0), (name, names[v])
            # Every entry equals pgmpy's own lookup by state LABEL: catches any parent-axis
            # or state-order mix-up, since get_value never goes through our indexing.
            cpd = model.get_cpds(names[v])
            for j, pcfg in enumerate(product(*[range(k) for k in c["pcards"]])):
                labels = {names[p]: states[p][s] for p, s in zip(c["parents"], pcfg)}
                for x in range(c["r"]):
                    want = cpd.get_value(**{names[v]: states[v][x]}, **labels)
                    assert abs(c["table"][j, x] - want) < 1e-9, (name, names[v], labels)

    # ASIA end to end: our sampler vs pgmpy's exact marginals and pgmpy's sampler.
    from pgmpy.inference import VariableElimination
    from pgmpy.sampling import BayesianModelSampling
    A, cpts, cards, names, states = read_bif("asia")
    model = _pgmpy_model("asia")
    ours, _, _, _ = load_bnlearn("asia", 200_000, seed=0)
    ve = VariableElimination(model)
    for v, name in enumerate(names):
        q = ve.query([name], show_progress=False)
        exact = [q.values[q.state_names[name].index(s)] for s in states[v]]
        assert np.allclose(np.bincount(ours[:, v], minlength=cards[v]) / len(ours), exact,
                           atol=0.005), name
    df = BayesianModelSampling(model).forward_sample(size=200_000, seed=1, show_progress=False)
    ref = np.stack([df[n].map(states[v].index).to_numpy() for v, n in enumerate(names)], axis=1)
    js = -neg_js(ref, ours, cards)                 # mean JS over all 1- and 2-way marginals
    assert js < 1e-4, js

    data, _, _, _ = synthetic(d=10, n=1000, seed=0)
    assert data.shape == (1000, 10)
    tr, va, te = split(data, [0.6, 0.2, 0.2])
    assert (len(tr), len(va), len(te)) == (600, 200, 200)
    assert sorted(map(tuple, np.vstack([tr, va, te]))) == sorted(map(tuple, data))
    print(f"data.loaders self-check passed  (5 networks match pgmpy entry by entry; "
          f"ASIA sampler vs pgmpy: mean JS {js:.1e} bits)")


if __name__ == "__main__":
    _demo()
