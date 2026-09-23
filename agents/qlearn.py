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

There is no stop action, so an episode keeps editing after it reaches its best graph.
The learned structure is therefore the graph with the highest cumulative return along
one greedy rollout, not wherever that rollout happens to end.
"""

import numpy as np


class QLearningAgent:
    def __init__(self, env, lr=0.1, gamma=0.95, epsilon=0.2, epsilon_decay=0.999, seed=0):
        self.env = env
        self.lr, self.gamma = lr, gamma
        self.epsilon, self.epsilon_decay = epsilon, epsilon_decay
        self.rng = np.random.default_rng(seed)
        # ponytail: one dense row of len(actions) floats per visited state; switch to
        # sparse rows if the table outgrows memory on ALARM.
        self.Q = {}

    @staticmethod
    def _key(s):
        return s["A"].tobytes(), s["tau"]

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
            returns.append(total)
            eps *= self.epsilon_decay
        return returns

    def best_graph(self):
        """Greedy rollout; returns (graph with the best cumulative return, that return)."""
        env = self.env
        s, done, total = env.reset(), False, 0.0
        best_A, best = s["A"], 0.0
        legal = np.flatnonzero(env.legal_mask())
        while not done and legal.size:
            s, r, done, _ = env.step(self._act(self.Q.get(self._key(s)), legal, 0.0))
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
    print(f"agents.qlearn self-check passed  (greedy return {ret:.4f}; "
          f"mean return first 50 eps {np.mean(returns[:50]):.4f}, last 50 {np.mean(returns[-50:]):.4f})")


if __name__ == "__main__":
    _demo()
