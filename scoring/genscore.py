"""The expensive generative score: how real does the learned BN's data look?

All scores are "higher = better" so they drop straight into the reward:
    held_out_loglik  mean log p(row) of held-out real rows under the BN, in nats
                     per row (same unit as BIC / N). Exact: needs no sampling.
    neg_js           minus the mean Jensen-Shannon divergence (bits, in [0, 1])
                     between real and simulated 1- and 2-way marginals.
    mmd              Hamming-kernel MMD. Later, if JS proves too coarse.
"""

from itertools import combinations

import numpy as np

from scoring.bic import _parent_config_index


def held_out_loglik(cpts, held_out):
    """Average log-likelihood of held-out rows under the BN. Higher = better."""
    ll = np.zeros(len(held_out))
    for v, c in cpts.items():
        pconfig, _ = _parent_config_index(held_out, list(c["parents"]), c["pcards"])
        ll += np.log(c["table"][pconfig, held_out[:, v]])   # Dirichlet smoothing: no log 0
    return float(ll.mean())


def _js(p, q):
    """Jensen-Shannon divergence in bits between two histograms (same bins)."""
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a: float(np.sum(a[a > 0] * np.log2(a[a > 0] / m[a > 0])))
    return 0.5 * kl(p) + 0.5 * kl(q)


def neg_js(real, fake, cards):
    """Minus the mean JS divergence over every 1-way and 2-way marginal.
    Higher = better; 0 means every low-order marginal matches exactly."""
    # ponytail: all d(d-1)/2 pairs, ~666 bincounts for Alarm; sample pairs if it shows up in profiles.
    cards = np.asarray(cards)
    divs = [_js(np.bincount(real[:, v], minlength=cards[v]),
                np.bincount(fake[:, v], minlength=cards[v]))
            for v in range(len(cards))]
    for u, v in combinations(range(len(cards)), 2):
        size = cards[u] * cards[v]
        divs.append(_js(np.bincount(real[:, u] * cards[v] + real[:, v], minlength=size),
                        np.bincount(fake[:, u] * cards[v] + fake[:, v], minlength=size)))
    return -float(np.mean(divs))


def mmd(real_data, fake_data):
    """Maximum Mean Discrepancy with a Hamming kernel (categorical). Later."""
    raise NotImplementedError


def _demo():
    from envs.dag import ADD, apply_action, empty_dag
    from scoring.bic import fit_cpts
    from scoring.simulate import ancestral_sample

    rng = np.random.default_rng(0)
    n = 6000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)
    x2 = rng.integers(0, 2, n)
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])
    train, held = data[:4000], data[4000:]

    empty = empty_dag(3)
    true = apply_action(empty, (ADD, 0, 1))
    c_empty, c_true = fit_cpts(empty, train, cards), fit_cpts(true, train, cards)

    # The true structure explains unseen rows better than independence.
    assert held_out_loglik(c_true, held) > held_out_loglik(c_empty, held)
    # Cross-check against a direct product of CPT entries for one row.
    row = held[0]
    direct = (np.log(c_true[0]["table"][0, row[0]]) + np.log(c_true[1]["table"][row[0], row[1]])
              + np.log(c_true[2]["table"][0, row[2]]))
    assert abs(held_out_loglik(c_true, held[:1]) - direct) < 1e-12

    # Data simulated from the true structure matches the real 2-way marginals better.
    f_true = ancestral_sample(true, c_true, 4000, np.random.default_rng(1))
    f_empty = ancestral_sample(empty, c_empty, 4000, np.random.default_rng(1))
    assert neg_js(held, f_true, cards) > neg_js(held, f_empty, cards)
    assert neg_js(held, held, cards) == 0.0
    print("scoring.genscore self-check passed")


if __name__ == "__main__":
    _demo()
