"""Tabular Q-learning on RLiGEnv: the RL agent the hybrid, tiled reward is built for.

    state   (A.tobytes(), tau). tau is part of the state because the same graph is
            worth more or less depending on how far it is from the next generative step.
    update  Q(s, a) += lr * (r + gamma * max over legal a' of Q(s', a') - Q(s, a)),
            with no bootstrap on the terminal step.
    policy  epsilon-greedy over legal actions only (envs.dag.legal_action_mask), so the
            infeasible-action penalty is never paid. epsilon decays once per episode.

Bootstrapping is what RLBayes lacks. In difference mode the generative reward at a
step in I_g covers the last L edits, and gamma carries it back to the edits that
earned it.

Two outputs, reported separately because the gap between them is itself a result:
    best_searched()  the search result. Keeps the top_k graphs by BIC seen during
                     training and picks by alpha*BIC/N + beta*GenScore at the end. Not
                     by cumulative return: in difference mode that only refreshes
                     GenScore at generative steps, so it ranks graphs on a stale one.
    best_graph()     the learned policy. A greedy rollout (no stop action, so it keeps
                     editing past its best graph); returns the best-return prefix.
"""

import heapq

import numpy as np


class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.999,
                 epsilon_min=0.01, top_k=20, seed=0):
        self.env = env
        self.lr, self.gamma = lr, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.top_k = top_k
        self._top, self._top_keys = [], set()   # min-heap of (bic, key, A) seen in training
        self.steps = 0                           # env steps taken in training
        self.greedy_unseen = 0                   # greedy moves made from states not in Q
        self.rng = np.random.default_rng(seed)
        # ponytail: one dense row of len(actions) floats per visited state; switch to
        # sparse rows if the table outgrows memory on ALARM.
        self.Q = {}

    @staticmethod
    def _key(s):
        return s["A"].tobytes(), s["tau"]

    def _q(self, s):
        """Q-values of every action in s, or None if s was never visited."""
        return self.Q.get(self._key(s))

    def _act(self, q, legal, eps):
        if q is None or self.rng.random() < eps:
            return int(self.rng.choice(legal))
        best = legal[q[legal] == q[legal].max()]         # random tie-break
        return int(self.rng.choice(best))

    def train(self, episodes):
        """Returns the total reward of every episode (the learning curve)."""
        env, returns, eps = self.env, [], self.epsilon
        for _ in range(episodes):
            s, done, total = env.reset(), False, 0.0
            legal = np.flatnonzero(env.legal_mask())
            while not done and legal.size:
                q = self.Q.setdefault(self._key(s), np.zeros(len(env.actions)))
                a = self._act(q, legal, eps)
                s, r, done, _ = env.step(a)
                legal = np.flatnonzero(env.legal_mask())
                target = r
                q_next = self.Q.get(self._key(s))          # unseen state: Q = 0
                if not done and legal.size and q_next is not None:
                    target += self.gamma * q_next[legal].max()
                q[a] += self.lr * (target - q[a])
                total += r
                self.steps += 1
                self._remember(s)
            returns.append(total)
            eps = max(eps * self.epsilon_decay, self.epsilon_min)
        return returns

    def _remember(self, s):
        key = s["A"].tobytes()
        if key in self._top_keys:
            return
        item = (s["bic"], key, s["A"])
        if len(self._top) < self.top_k:
            heapq.heappush(self._top, item)
        elif item[0] > self._top[0][0]:
            self._top_keys.discard(heapq.heappushpop(self._top, item)[1])
        else:
            return
        self._top_keys.add(key)

    def best_searched(self):
        """Best of the top_k-by-BIC training graphs under alpha*BIC/N + beta*GenScore.
        GenScore is cached per graph, so this costs at most top_k simulations."""
        env, cfg = self.env, self.env.cfg
        score = lambda A, bic: (cfg["alpha"] * bic / len(env.train)
                                + (cfg["beta"] * env.gen_score(A) if cfg["beta"] else 0.0))
        return max(((A, score(A, bic)) for bic, _, A in self._top), key=lambda t: t[1])

    def best_graph(self):
        """Greedy rollout; returns (graph with the best cumulative return, that return).
        Sets greedy_unseen: moves where the state had no Q row, so _act picked at random."""
        env, self.greedy_unseen = self.env, 0
        s, done, total = env.reset(), False, 0.0
        best_A, best = s["A"], 0.0
        legal = np.flatnonzero(env.legal_mask())
        while not done and legal.size:
            q = self._q(s)
            self.greedy_unseen += q is None
            s, r, done, _ = env.step(self._act(q, legal, 0.0))
            legal = np.flatnonzero(env.legal_mask())
            total += r
            if total > best:
                best_A, best = s["A"], total
        return best_A, best


def _demo():
    from envs.env import RLiGEnv

    # x1 copies x0 90% of the time, x2 is independent noise: the answer is one 0-1 edge.
    rng = np.random.default_rng(0)
    n = 5000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)
    x2 = rng.integers(0, 2, n)
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])
    cfg = dict(max_indegree_k=2, budget_T=6, l_stall=6, tile_L=3, I_g=[2], N_s=2000,
               alpha=1.0, beta=30.0, gen_score="js", reward_mode="difference",
               dirichlet_alpha=1.0, penalty=-0.05)
    env = RLiGEnv(data[:4000], data[4000:], cards, cfg)
    agent = QLearningAgent(env, lr=0.1, gamma=0.95, epsilon=0.3, epsilon_decay=0.99, seed=0)
    returns = agent.train(300)

    A, ret = agent.best_graph()
    assert A[0, 1] + A[1, 0] == 1 and A.sum() == 1, A      # exactly the 0-1 link
    assert ret > 0
    assert np.mean(returns[-50:]) > np.mean(returns[:50])  # the policy improved
    # best_searched is the hybrid-score argmax over the kept candidates, and the true
    # link is among them. It need not *be* the argmax: at beta = 30 GenScore noise
    # (~6e-3 across these graphs) outweighs one spurious edge's BIC cost (~7e-4).
    B, sB = agent.best_searched()
    hybrid = lambda A: env.bic(A) / 4000 + 30 * env.gen_score(A)
    kept = [A for _, _, A in agent._top]
    assert len(kept) == agent.top_k and abs(sB - max(map(hybrid, kept))) < 1e-12
    assert any(A[0, 1] + A[1, 0] == 1 and A.sum() == 1 for A in kept)

    # beta = 0: reward is pure BIC gain, and actions are always legal, so the greedy
    # return must equal (BIC(A) - BIC(empty)) / N exactly (dBIC telescopes).
    env0 = RLiGEnv(data[:4000], data[4000:], cards, dict(cfg, beta=0.0))
    agent0 = QLearningAgent(env0, lr=0.1, gamma=0.95, epsilon=0.3, epsilon_decay=0.99, seed=0)
    agent0.train(100)
    A0, ret0 = agent0.best_graph()
    gain = (env0.bic(A0) - env0.bic(np.zeros_like(A0))) / 4000
    assert abs(ret0 - gain) < 1e-9, (ret0, gain)
    # ... and best_searched is then just the highest-BIC graph seen.
    B0, s0 = agent0.best_searched()
    assert abs(s0 - env0.bic(B0) / 4000) < 1e-12 and env0.n_gen_calls == 0
    print(f"agents.qlearn self-check passed  (greedy return {ret:.4f}; "
          f"mean return first 50 eps {np.mean(returns[:50]):.4f}, last 50 {np.mean(returns[-50:]):.4f})")


if __name__ == "__main__":
    _demo()
