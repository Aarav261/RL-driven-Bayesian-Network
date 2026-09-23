"""Step 5: the RL environment — DAG editing as a sequential decision process.

Wraps envs.dag + scoring.bic + scoring.genscore and adds TILING: within a tile of length
L, only steps whose index is in I_g pay the expensive generative score;
everywhere else the reward is the cheap delta_bic.

    state  = (adjacency matrix, step index within tile)
    step(action) -> (next_state, reward, done)
    reward = alpha*delta_bic + (beta*gen_score if step in I_g else 0)
    done   = edit budget T reached, or no improvement for L_stall steps
"""


class RLiGEnv:
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg  # alpha, beta, tile_L, I_g, N_s, budget_T, max_indegree_k
        raise NotImplementedError

    def reset(self):
        """Return the initial state (empty DAG, step 0)."""
        raise NotImplementedError

    def step(self, action):
        """Apply a legal edit, return (state, reward, done). Illegal -> penalty."""
        raise NotImplementedError

    def legal_mask(self):
        """Boolean mask over the action space for the current DAG."""
        raise NotImplementedError
