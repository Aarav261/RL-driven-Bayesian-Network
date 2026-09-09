"""Step 4b: the expensive generative score — how real does simulated data look?

Start with held_out_loglik (simplest, cheapest). Add js_divergence, then mmd
only once the tabular agent learns on ASIA. GenScore feeds the hybrid reward:
    reward = alpha * delta_bic + beta * gen_score(G')
"""


def held_out_loglik(A, cpts, held_out_data):
    """Average log-likelihood of held-out real rows under the BN. Higher = better.
    START HERE."""
    raise NotImplementedError


def js_divergence(real_data, fake_data, marginals):
    """Mean Jensen-Shannon divergence over chosen 1- and 2-way marginals.
    Lower = better (negate when using as reward)."""
    raise NotImplementedError


def mmd(real_data, fake_data):
    """Maximum Mean Discrepancy with a Hamming kernel (categorical). Later."""
    raise NotImplementedError
