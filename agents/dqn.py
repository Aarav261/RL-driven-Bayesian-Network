"""HD extension 1: masked DQN on RLiGEnv. Does generalisation fix tabular Q-learning?

Tabular Q-learning searches well but its greedy policy lands in states it never
visited and picks at random there (`greedy_unseen`, ~12 moves per ASIA rollout).
A network shares what it learns across graphs, so every state gets a real Q-value.

    features  [A flattened (d*d), one-hot tau (L), edits left / T]. Same Markov part
              as the tabular key (A, tau), plus the budget, which termination uses.
              Gen_prev and the stall count are still dropped (plan.md D2).
    network   head="mlp": features -> one Q-value per action in envs.dag.all_actions(d).
              head="emb": Q(s, a) = phi(s) . psi(a), psi an MLP over learned embeddings
              of a's (op, i, j). Edits that share a node or an op share parameters, and
              the size no longer grows with the action count (d^2 x 3 outputs for "mlp").
              psi needs the hidden layer: with a plain sum of embeddings, the value of
              edge i->j would be an i-term plus a j-term, with no way to prefer one pair.
    target    Double DQN: r + gamma * Q_target(s', argmax over LEGAL a' of Q(s', a')); no
              bootstrap on the terminal step.
    replay    (A, tau, budget) of s and s' as small ints; features and the legal mask of s'
              are rebuilt per batch (one batched legal_action_mask call). Dense float
              features + masks would be ~750 MB on ALARM at 50k transitions; this is ~140 MB.
    policy    epsilon-greedy over legal actions only, as in agents.qlearn.

Everything else (top-k search memory, best_searched, greedy best_graph, _act) is
inherited from QLearningAgent, so the two agents are compared on the same outputs.
"""

import copy

import numpy as np
import torch
from torch import nn

from agents.qlearn import QLearningAgent
from envs.dag import ADD, DELETE, REVERSE, legal_action_mask


class EmbQ(nn.Module):
    """Q(s, a) = phi(s) . psi(a), psi from embeddings of the action's (op, i, j)."""

    def __init__(self, n_in, hidden, actions, d, emb=16):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.op, self.src, self.dst = nn.Embedding(3, emb), nn.Embedding(d, emb), nn.Embedding(d, emb)
        self.psi = nn.Sequential(nn.Linear(3 * emb, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        ops = {ADD: 0, DELETE: 1, REVERSE: 2}
        self.register_buffer("acts", torch.tensor([(ops[o], i, j) for o, i, j in actions]))

    def forward(self, x):
        a = self.acts
        psi = self.psi(torch.cat([self.op(a[:, 0]), self.src(a[:, 1]), self.dst(a[:, 2])], -1))
        return self.phi(x) @ psi.T


class DQNAgent(QLearningAgent):
    def __init__(self, env, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.998,
                 epsilon_min=0.05, hidden=256, batch=64, buffer=50_000, train_every=4,
                 target_every=1000, top_k=20, seed=0, head="mlp", emb=16):
        super().__init__(env, lr, gamma, epsilon, epsilon_decay, epsilon_min, top_k, seed)
        torch.manual_seed(seed)
        d, self.n_act = env.d, len(env.actions)
        n_in = d * d + env.L + 1
        if head == "mlp":
            self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, self.n_act))
        else:
            self.net = EmbQ(n_in, hidden, env.actions, d, emb)
        self.target = copy.deepcopy(self.net)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.batch, self.train_every, self.target_every = batch, train_every, target_every
        # ponytail: A as uint8 (d*d bytes per state, ~140 MB for s and s' on ALARM at 50k);
        # np.packbits would cut that 8x if it ever matters.
        self.cap, self.n = buffer, 0
        self.A, self.A2 = (np.zeros((buffer, d, d), np.uint8) for _ in range(2))
        self.tau, self.tau2 = (np.zeros(buffer, np.int16) for _ in range(2))
        self.bud, self.bud2 = (np.zeros(buffer, np.int16) for _ in range(2))
        self.a = np.zeros(buffer, np.int64)
        self.r = np.zeros(buffer, np.float32)
        self.term = np.zeros(buffer, bool)

    def _feat(self, A, tau, budget):
        """Features of a batch: A (b, d, d), tau (b,), budget (b,) -> (b, n_in) float32."""
        env, b = self.env, len(A)
        x = np.zeros((b, env.d * env.d + env.L + 1), np.float32)
        x[:, :env.d * env.d] = A.reshape(b, -1)
        x[np.arange(b), env.d * env.d + tau] = 1.0
        x[:, -1] = budget / env.cfg["budget_T"]
        return torch.from_numpy(x)

    def _q(self, s):
        with torch.no_grad():
            return self.net(self._feat(s["A"][None], np.array([s["tau"]]),
                                       np.array([s["budget"]])))[0].numpy()

    def _update(self):
        idx = self.rng.integers(0, min(self.n, self.cap), self.batch)
        X = self._feat(self.A[idx], self.tau[idx], self.bud[idx])
        X2 = self._feat(self.A2[idx], self.tau2[idx], self.bud2[idx])
        a, r = torch.from_numpy(self.a[idx]), torch.from_numpy(self.r[idx])
        M2 = torch.from_numpy(legal_action_mask(self.A2[idx], self.env.k))
        term = torch.from_numpy(self.term[idx])
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
                i = self.n % self.cap
                self.A[i], self.tau[i], self.bud[i] = s["A"], s["tau"], s["budget"]
                a = self._act(self._q(s), np.flatnonzero(mask), eps)
                s, r, done, _ = env.step(a)
                mask = env.legal_mask()
                self.A2[i], self.tau2[i], self.bud2[i] = s["A"], s["tau"], s["budget"]
                self.a[i], self.r[i], self.term[i] = a, r, done or not mask.any()
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
    for head in ["mlp", "emb"]:
        # 600 episodes: at 300, 1 seed in 6 still orders two edits wrong (added 1->2 first).
        agent = DQNAgent(env, epsilon=1.0, epsilon_decay=0.99, hidden=64, batch=32,
                         target_every=200, train_every=1, seed=0, head=head)
        returns = agent.train(600)

        A, ret = agent.best_graph()
        assert A[0, 1] + A[1, 0] == 1 and A.sum() == 1, (head, A)   # exactly the 0-1 link
        assert ret > 0 and agent.greedy_unseen == 0            # a network never has "unseen"
        assert np.mean(returns[-50:]) > np.mean(returns[:50])  # the policy improved
        print(f"  {head}: greedy return {ret:.4f}; mean return first 50 eps "
              f"{np.mean(returns[:50]):.4f}, last 50 {np.mean(returns[-50:]):.4f}")
    print("agents.dqn self-check passed")


if __name__ == "__main__":
    _demo()
