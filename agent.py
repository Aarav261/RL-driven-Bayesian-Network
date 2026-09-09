"""Step 6: RL search agent — the RLBayes dynamic Q-table (Wang et al., 2025, Alg. 2).

Standard Q-learning can't table a super-exponential DAG space. RLBayes' fix:
grow the table only over BNs actually visited, and cap it — once it holds
MAX_LEN rows, drop the worst-scored BN. Bounded memory, unbounded search.

    row    = a visited BN (adjacency matrix)
    column = an operation add/del/rev(i,j)   (fixed action space)
    cell   = benefit of that op = score(BN') - score(BN)   (0 if untried, -inf if illegal)

The reward is a *score difference*, so this agent is reward-agnostic: pass
score_fn = bic for the classic baseline, or the hybrid alpha*BIC + beta*GenScore
for RLiG. Illegal ops are pre-masked to -inf (dag.legal_action_mask) rather than
discovered by trial as in the paper — same effect, fewer wasted iterations.
"""

import numpy as np

from .dag import (ADD, DELETE, REVERSE, all_actions, apply_action,
                  empty_dag, legal_action_mask)
from .score import bic


def _reverse_action(action):
    op, i, j = action
    if op == ADD:
        return (DELETE, i, j)
    if op == DELETE:
        return (ADD, i, j)
    return (REVERSE, j, i)          # reverse of a reversal


class RLBayesAgent:
    def __init__(self, d, score_fn, max_indegree=2, max_len=500,
                 max_iter=5000, theta=0.1, seed=0):
        self.d = d
        self.score_fn = score_fn            # score_fn(A) -> float, higher is better
        self.k = max_indegree
        self.max_len = max_len
        self.max_iter = max_iter
        self.theta = theta
        self.rng = np.random.default_rng(seed)

        self.actions = all_actions(d)
        self.rev_idx = np.array([self.actions.index(_reverse_action(a))
                                 for a in self.actions])

        # Parallel table: rows[i] BN, Q[i] benefits, scores[i] absolute score.
        self.rows, self.Q, self.scores, self.index_of = [], [], [], {}
        self._add_row(empty_dag(d))
        self.cur = 0

    # ---- table maintenance -------------------------------------------------
    def _key(self, A):
        return A.tobytes()

    def _add_row(self, A):
        q = np.zeros(len(self.actions))
        mask = legal_action_mask(A, self.k)
        q[[not m for m in mask]] = -np.inf          # illegal ops never chosen
        self.rows.append(A)
        self.Q.append(q)
        self.scores.append(self.score_fn(A))
        self.index_of[self._key(A)] = len(self.rows) - 1
        return len(self.rows) - 1

    def _prune(self):
        cur_key = self._key(self.rows[self.cur])
        while len(self.rows) > self.max_len:
            w = int(np.argmin(self.scores))
            del self.rows[w]; del self.Q[w]; del self.scores[w]
        self.index_of = {self._key(A): i for i, A in enumerate(self.rows)}
        # current row may have been the one dropped -> jump to the best so far
        self.cur = self.index_of.get(cur_key, int(np.argmax(self.scores)))

    # ---- Algorithm 1: operation choosing -----------------------------------
    def _choose(self, row):
        q = self.Q[row]
        legal = np.flatnonzero(np.isfinite(q))       # anything not -inf
        if legal.size == 0:
            return None
        if self.rng.random() >= 0.5:                 # explore: random legal op
            return int(self.rng.choice(legal))
        best = legal[q[legal] == q[legal].max()]     # exploit: best benefit
        return int(self.rng.choice(best))

    # ---- Algorithm 2: the search loop --------------------------------------
    def train(self):
        history = []                                 # best score per iter (learning curve)
        for _ in range(self.max_iter):
            p = self.cur
            op_idx = self._choose(p)
            if op_idx is not None:
                action = self.actions[op_idx]
                B = apply_action(self.rows[p], action)
                key = self._key(B)
                if key in self.index_of:
                    n = self.index_of[key]
                else:
                    n = self._add_row(B)
                benefit = self.scores[n] - self.scores[p]
                self.Q[p][op_idx] = benefit
                self.Q[n][self.rev_idx[op_idx]] = -benefit   # reverse op = opposite benefit
                self.cur = n
                self._prune()

            if self.rng.random() <= self.theta:      # state transfer -> best-so-far
                self.cur = int(np.argmax(self.scores))
            history.append(max(self.scores))

        best = int(np.argmax(self.scores))
        return self.rows[best], self.scores[best], history


def _demo():
    # Same synthetic data as score.py: x1 strongly depends on x0, x2 is noise.
    rng = np.random.default_rng(1)
    n = 4000
    x0 = rng.integers(0, 2, n)
    x1 = np.where(rng.random(n) < 0.9, x0, 1 - x0)
    x2 = rng.integers(0, 2, n)
    data = np.stack([x0, x1, x2], axis=1)
    cards = np.array([2, 2, 2])

    score_fn = lambda A: bic(A, data, cards)
    agent = RLBayesAgent(d=3, score_fn=score_fn, max_indegree=2,
                         max_len=30, max_iter=1500, theta=0.1, seed=0)
    A, s, history = agent.train()

    empty_score = bic(empty_dag(3), data, cards)
    assert s > empty_score, (s, empty_score)          # learned something
    assert A[0, 1] == 1 or A[1, 0] == 1               # recovered the 0-1 link
    assert A[0, 2] == 0 and A[2, 0] == 0              # no spurious 0-2 edge
    assert A[1, 2] == 0 and A[2, 1] == 0              # no spurious 1-2 edge
    assert history[-1] >= history[0]                  # learning curve non-decreasing overall
    print(f"agent.py self-check passed  (best BIC {s:.2f} vs empty {empty_score:.2f})")


if __name__ == "__main__":
    _demo()
