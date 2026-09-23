"""Required baseline 1: Hill-Climbing (add/delete/reverse) with BIC, optional Tabu.

Build this FIRST: it validates envs.dag + scoring.bic (it should roughly
recover ASIA's true structure) and is a required comparison in the report.
"""


def hill_climb(data, cfg, tabu=False):
    """Greedily apply the legal edit with best positive delta_bic until none
    improves (optional Tabu list to escape plateaus). Returns final DAG."""
    raise NotImplementedError
