"""The RL environment: DAG editing as a sequential decision process, with tiling.

    state      (A, tau, bic, budget): DAG adjacency matrix, step index within the
               tile (0..L-1), current BIC, edits left. (A, tau) is the Markov part;
               bic and budget are the optional features the spec allows.
    action     index into envs.dag.all_actions(d): add / delete / reverse (i, j)
    transition apply the edit if feasible, else stay put and pay `penalty`
    reward     alpha * dBIC/N  +  [tau in I_g] * beta * G
    done       t == budget_T, or L_stall consecutive steps with dBIC <= 0

dBIC is divided by N so it is in nats per row, which keeps alpha comparable across
datasets. Only held-out log-lik shares that unit; neg_js is in bits and its step
changes are about 4x smaller than dBIC/N on ASIA, so beta has to absorb the scale.

The stall counter follows dBIC, not the full reward: a noisy generative term could
otherwise push reward above 0 by chance and keep a going-nowhere episode alive.

G depends on reward_mode (the spec allows either):
    "absolute"    G = GenScore(G')
    "difference"  G = GenScore(G') - GenScore at the previous generative step,
                  so generative rewards telescope to GenScore(last) - GenScore(G0)
                  and never reward standing still.

Tiling: tau = t mod L. Only steps with tau in I_g pay for fit_cpts (+ sampling
for neg_js); every other step costs a couple of cached BIC lookups.
"""

import numpy as np

from envs.dag import all_actions, apply_action, empty_dag, is_legal, legal_action_mask
from scoring.bic import BIC, fit_cpts
from scoring.genscore import held_out_loglik, neg_js
from scoring.simulate import ancestral_sample


class RLiGEnv:
    def __init__(self, train, held_out, cards, cfg, seed=0):
        self.train, self.held_out = np.asarray(train), np.asarray(held_out)
        self.cards = np.asarray(cards)
        self.d = len(self.cards)
        self.cfg = cfg
        self.k = cfg["max_indegree_k"]
        self.L, self.I_g = cfg["tile_L"], set(cfg["I_g"])
        assert self.I_g <= set(range(self.L)), "I_g must be a subset of 0..L-1"
        self.bic = BIC(self.train, self.cards)
        self.actions = all_actions(self.d)
        self.rng = np.random.default_rng(seed)
        self.n_gen_calls = 0                      # compute cost, for the tiling plots
        self.reset()

    # ---- spec API ---------------------------------------------------------
    def reset(self):
        """Empty DAG, step 0."""
        self.A = empty_dag(self.d)
        self.t = 0
        self.stall = 0
        self.gen_prev = None
        if self.cfg["reward_mode"] == "difference" and self._uses_gen():
            self.gen_prev = self.gen_score(self.A)   # baseline for the first difference
        return self.state()

    def state(self):
        return {"A": self.A, "tau": self.t % self.L,
                "bic": self.bic(self.A), "budget": self.cfg["budget_T"] - self.t}

    def legal_mask(self):
        """Boolean mask over self.actions for the current DAG."""
        return legal_action_mask(self.A, self.k)

    def step(self, a):
        """Take action index a. Returns (state, reward, done, info)."""
        cfg = self.cfg
        action = self.actions[a]
        tau = self.t % self.L
        self.t += 1
        info = {"legal": is_legal(self.A, action, self.k), "generative": False,
                "delta_bic": 0.0, "gen": None}

        if not info["legal"]:
            reward = cfg["penalty"]                               # state unchanged
        else:
            info["delta_bic"] = self.bic.delta(self.A, action) / len(self.train)
            self.A = apply_action(self.A, action)
            reward = cfg["alpha"] * info["delta_bic"]
            if tau in self.I_g and self._uses_gen():
                g = self.gen_score(self.A)
                info.update(generative=True, gen=g)
                if cfg["reward_mode"] == "difference":
                    g, self.gen_prev = g - self.gen_prev, g
                reward += cfg["beta"] * g

        self.stall = 0 if info["delta_bic"] > 0 else self.stall + 1
        done = self.t >= cfg["budget_T"] or self.stall >= cfg["l_stall"]
        return self.state(), reward, done, info

    # ---- generative score -------------------------------------------------
    def _uses_gen(self):
        return self.cfg["beta"] != 0 and len(self.I_g) > 0

    def gen_score(self, A):
        """GenScore(A): fit Dirichlet-MLE CPTs on train, score against held-out."""
        self.n_gen_calls += 1
        cfg = self.cfg
        cpts = fit_cpts(A, self.train, self.cards, cfg["dirichlet_alpha"])
        if cfg["gen_score"] == "held_out_loglik":
            return held_out_loglik(cpts, self.held_out)
        if cfg["gen_score"] == "js":
            fake = ancestral_sample(A, cpts, cfg["N_s"], self.rng)
            return neg_js(self.held_out, fake, self.cards)
        raise ValueError(f"unknown gen_score {cfg['gen_score']}")


def _demo():
    from envs.dag import ADD, DELETE

    rng = np.random.default_rng(0)
    n = 5000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)
    x2 = rng.integers(0, 2, n)
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])
    train, held = data[:4000], data[4000:]
    cfg = dict(max_indegree_k=2, budget_T=12, l_stall=100, tile_L=3, I_g=[2], N_s=2000,
               alpha=1.0, beta=1.0, gen_score="held_out_loglik", reward_mode="difference",
               dirichlet_alpha=1.0, penalty=-0.05)

    env = RLiGEnv(train, held, cards, cfg)
    idx = {a: n for n, a in enumerate(env.actions)}
    s = env.state()
    assert s["A"].sum() == 0 and s["tau"] == 0 and s["budget"] == 12
    assert env.n_gen_calls == 1                                # difference-mode baseline

    # Non-generative step (tau 0): reward is exactly alpha * dBIC / N.
    s, r, done, info = env.step(idx[(ADD, 0, 1)])
    assert not info["generative"] and abs(r - info["delta_bic"]) < 1e-12 and r > 0
    assert s["tau"] == 1 and s["A"][0, 1] == 1

    # Illegal action (edge already there): penalty, DAG unchanged, still costs a step.
    s, r, done, info = env.step(idx[(ADD, 0, 1)])
    assert not info["legal"] and r == -0.05 and s["A"].sum() == 1 and s["tau"] == 2

    # Tiling: 12 steps with L=3, I_g={2} -> generative at t = 2, 5, 8, 11, all legal here.
    gens = 0
    for a in [(ADD, 0, 2), (DELETE, 0, 2)] * 5:
        s, r, done, info = env.step(idx[a])
        gens += info["generative"]
    assert done and env.t == 12
    assert env.n_gen_calls == 1 + 4 and gens == 4

    # Difference mode telescopes: sum of generative parts == Gen(last) - Gen(G0).
    env = RLiGEnv(train, held, cards, cfg)
    g0, total = env.gen_prev, 0.0
    for a in [(ADD, 0, 1), (ADD, 2, 1), (ADD, 0, 2), (DELETE, 2, 1), (DELETE, 0, 2), (ADD, 1, 2)]:
        s, r, done, info = env.step(idx[a])
        total += r - cfg["alpha"] * info["delta_bic"]
    assert abs(total - (env.gen_prev - g0)) < 1e-9

    # Stall termination: 3 non-improving steps in a row ends the episode.
    env = RLiGEnv(train, held, cards, dict(cfg, l_stall=3, beta=0.0))
    _, _, done, _ = env.step(idx[(ADD, 0, 1)])                 # improving: count stays 0
    assert not done
    dones = [env.step(idx[(ADD, 0, 1)])[2] for _ in range(3)]  # illegal x3
    assert dones == [False, False, True]
    assert env.n_gen_calls == 0                                # beta = 0: never simulates

    # JS mode runs end to end and is seeded.
    js = lambda: RLiGEnv(train, held, cards, dict(cfg, gen_score="js"), seed=7).gen_prev
    assert js() == js() and js() <= 0
    print("envs.env self-check passed")


if __name__ == "__main__":
    _demo()
