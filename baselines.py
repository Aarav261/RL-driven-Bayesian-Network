"""Step 3 + required baselines: Hill-Climbing (BIC) and GES.

Build hill_climb FIRST — it validates dag.py + score.py (it should roughly
recover ASIA's true structure) and is a required comparison in the report.
GES = score-equivalent greedy search; wrap an existing implementation.
"""


def hill_climb(data, cfg, tabu=False):
    """Greedily apply the legal edit with best positive delta_bic until none
    improves (optional Tabu list to escape plateaus). Returns final DAG."""
    raise NotImplementedError


def ges(data, cfg):
    """Greedy Equivalence Search. Wrap pgmpy / causal-learn rather than reimplement."""
    raise NotImplementedError
