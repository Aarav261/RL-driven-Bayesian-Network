"""HD extension 1: masked DQN on RLiGEnv. Does generalisation fix tabular Q-learning?

Tabular Q-learning searches well but its greedy policy lands in states it never
visited and picks at random there (`greedy_unseen`, ~12 moves per ASIA rollout).
A network shares what it learns across graphs, so every state gets a real Q-value.

    features  [A flattened (d*d), one-hot tau (L), edits left / T]. Same Markov part
              as the tabular key (A, tau), plus the budget, which termination uses.
              Gen_prev and the stall count are still dropped (plan.md D2).
    network   MLP: features -> one Q-value per action in envs.dag.all_actions(d).
    target    Double DQN: r + gamma * Q_target(s', argmax over LEGAL a' of Q(s', a')); no
              bootstrap on the terminal step. The legal mask of s' is in the replay buffer.
    policy    epsilon-greedy over legal actions only, as in agents.qlearn.

Everything else (top-k search memory, best_searched, greedy best_graph, _act) is
inherited from QLearningAgent, so the two agents are compared on the same outputs.
"""

import copy

import numpy as np
import torch
from torch import nn

from agents.qlearn import QLearningAgent


class DQNAgent(QLearningAgent):
    def __init__(self, env, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998,
                 epsilon_min=0.05, hidden=256, batch=64, buffer=50_000, train_every=4,
                 target_every=1000, top_k=20, seed=0):
        super().__init__(env, lr, gamma, epsilon, epsilon_decay, epsilon_min, top_k, seed)
        torch.manual_seed(seed)
        d, self.n_act = env.d, len(env.actions)
        n_in = d * d + env.L + 1
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.n_act))
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.batch, self.train_every, self.target_every = batch, train_every, target_every
        # ponytail: dense replay (features + next-state mask per transition), ~5 MB on ASIA
        # but ~1 GB on ALARM at 50k; store A as bits and rebuild masks if memory bites.
        self.cap, self.n = buffer, 0
        self.X = np.zeros((buffer, n_in), np.float32)
        self.X2 = np.zeros((buffer, n_in), np.float32)
        self.a = np.zeros(buffer, np.int64)
        self.r = np.zeros(buffer, np.float32)
        self.M2 = np.zeros((buffer, self.n_act), bool)
        self.term = np.zeros(buffer, bool)

    def _feat(self, s):
        env = self.env
        x = np.zeros(env.d * env.d + env.L + 1, np.float32)
        x[:env.d * env.d] = s["A"].ravel()
        x[env.d * env.d + s["tau"]] = 1.0
        x[-1] = s["budget"] / env.cfg["budget_T"]
        return x

    def _q(self, s):
        with torch.no_grad():
            return self.net(torch.from_numpy(self._feat(s))).numpy()

    def _update(self):
        idx = self.rng.integers(0, min(self.n, self.cap), self.batch)
        X, X2 = torch.from_numpy(self.X[idx]), torch.from_numpy(self.X2[idx])
        a, r = torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx])
        M2, term = torch.from_numpy(self.M2[idx]), torch.from_numpy(self.term[idx])
        with torch.no_grad():
            # Double DQN: the online net picks a', the target net values it. Plain DQN's
            # max over its own noisy estimates overestimated ~2.5x on ASIA, and the greedy
            # policy looped add/delete of one edge for 30 steps.
            a_next = self.net(X2).masked_fill(~M2, -torch.inf).argmax(1, keepdim=True)
            q_next = self.target(X2).gather(1, a_next).squeeze(1)
            y = r + self.gamma * torch.where(term, 0.0, q_next)
        q = self.net(X).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, episodes):
        """Returns the total reward of every episode (the learning curve)."""
        env, returns, eps = self.env, [], self.epsilon
        for _ in range(episodes):
            s, done, total = env.reset(), False, 0.0
            mask = env.legal_mask()
            while not done and mask.any():
                x = self._feat(s)
                a = self._act(self._q(s), np.flatnonzero(mask), eps)
                s, r, done, _ = env.step(a)
                mask = env.legal_mask()
                i = self.n % self.cap
                self.X[i], self.a[i], self.r[i] = x, a, r
                self.X2[i], self.M2[i], self.term[i] = self._feat(s), mask, done or not mask.any()
                self.n += 1
                total += r
                self.steps += 1
                self._remember(s)
                if self.n >= self.batch and self.steps % self.train_every == 0:
                    self._update()
                if self.steps % self.target_every == 0:
                    self.target.load_state_dict(self.net.state_dict())
            returns.append(total)
            eps = max(eps * self.epsilon_decay, self.epsilon_min)
        return returns


def _demo():
    from envs.env import RLiGEnv

    # Same toy as agents.qlearn: x1 copies x0 90% of the time, x2 is noise.
    rng = np.random.default_rng(0)
    n = 5000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)
    x2 = rng.integers(0, 2, n)
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])
    # beta = 0 so this tests the learner: at beta = 30 GenScore noise can make a spurious
    # edge earn more return (see agents.qlearn), and the DQN is right to take it.
    cfg = dict(max_indegree_k=2, budget_T=6, l_stall=6, tile_L=3, I_g=[2], N_s=2000,
               alpha=1.0, beta=0.0, gen_score="js", reward_mode="difference",
               dirichlet_alpha=1.0, penalty=-0.05)
    env = RLiGEnv(data[:4000], data[4000:], cards, cfg)
    # 600 episodes: at 300, 1 seed in 6 still orders two edits wrong (added 1->2 first).
    agent = DQNAgent(env, epsilon=1.0, epsilon_decay=0.99, hidden=64, batch=32,
                     target_every=200, train_every=1, seed=0)
    returns = agent.train(600)

    A, ret = agent.best_graph()
    assert A[0, 1] + A[1, 0] == 1 and A.sum() == 1, A      # exactly the 0-1 link
    assert ret > 0 and agent.greedy_unseen == 0            # a network never has "unseen"
    assert np.mean(returns[-50:]) > np.mean(returns[:50])  # the policy improved
    print(f"agents.dqn self-check passed  (greedy return {ret:.4f}; "
          f"mean return first 50 eps {np.mean(returns[:50]):.4f}, last 50 {np.mean(returns[-50:]):.4f})")


if __name__ == "__main__":
    _demo()
