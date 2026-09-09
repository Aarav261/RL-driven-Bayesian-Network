"""Step 4a: generate a fake dataset from a learned BN by ancestral sampling.

Ancestral sampling: process nodes in topological order; sample each node from
its CPT given already-sampled parent values. pgmpy: BayesianNetwork.simulate(n).

TODO: implement (or wrap pgmpy's simulate).
"""


def ancestral_sample(A, cpts, n):
    """Return n rows sampled from the BN defined by (A, cpts)."""
    raise NotImplementedError
