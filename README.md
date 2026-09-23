# RLiG — Reinforcement Learning for Bayesian Network Structure Learning

An exploration into whether reinforcement learning can learn the *structure* of a
Bayesian Network more effectively than the search heuristics the field usually
leans on.

## What I'm trying to do

Learning the graph of a Bayesian Network — which variable causes which — is an
NP-hard search: the number of possible DAGs grows super-exponentially with the
number of variables, so we can't enumerate them. The established approaches
(MCMC sampling, and more recently GFlowNets à la Deleu et al., 2022) are either
slow on large distributions or short on evidence that they beat the classics.

I'm taking a different angle: **treat structure learning as a sequential
decision process and let an RL agent search it.** The agent starts from an empty
graph and edits it one edge at a time (add / delete / reverse), guided by a
reward that says whether each edit made the graph a better explanation of the
data. Over many edits it converges on a high-scoring network without ever
touching most of the search space.

Two ideas sit at the centre of this:

1. **A dynamic Q-table search** (adapted from RLBayes, Wang et al., 2025). Rather
   than tabling an impossibly large state space, the agent only remembers the
   networks it has actually visited and caps that memory, dropping the worst
   ones. This is what makes tabular RL feasible on a combinatorial graph space.

2. **A hybrid reward.** Classical learners score a structure purely on how well
   its shape fits the data (BIC). I add a second term: *simulate* data from the
   learned network and measure how close the synthetic data is to the real data.
   The reward becomes `α · ΔBIC + β · generative-fidelity`. To keep the expensive
   simulation affordable, it's only evaluated at selected steps (a *tiling*
   schedule); every other step uses the cheap structural score.

The hypothesis is that pairing an RL search — particularly a lightweight,
single-state metaheuristic like greedy hill climbing — with this hybrid,
generation-aware objective yields structure learning that is both efficient and
faithful to the data-generating process, especially where data is scarce.

## Why it matters

Bayesian Networks model causal relationships under uncertainty, and they're
useful precisely where data is incomplete and the causal factor space is large —
medicine, economics, and the domain I'm most drawn to, **agriculture**. Crop
yield depends on a high-dimensional tangle of environmental and meteorological
factors, and better structure learning could make yield and food-supply
prediction more accessible and more accurate — an increasingly important problem
as climate change widens the swings. RL is attractive here because it can learn
from less data than traditional methods need.

## Approach in code

```
rlig/
  envs/        dag.py (adjacency matrix, edits, cycle check, legal-edit mask), env.py (tiling env)
  scoring/     bic.py (cached decomposable BIC + Dirichlet MLE CPTs), simulate.py, genscore.py
  agents/      rlbayes.py (dynamic Q-table search); DQN / actor-critic go here
  baselines/   hill_climb.py, ges.py
  data/        loaders.py (bnlearn benchmarks + synthetic); raw files in data/raw/ (gitignored)
  eval/        metrics.py (SHD, precision/recall/F1; CPDAG + log-lik to come)
  configs/     one YAML per experiment
  scripts/     repro.py (runs everything, writes report/tables + report/figures)
  report/      report.typ (the deliverable), refs.bib, figures/, tables/
  plan.md      assignment spec, build order, HD extensions
  HANDOFF.md   where the work is right now
```

The agent is deliberately reward-agnostic: the same code runs the pure-BIC
baseline today and the hybrid objective by swapping the `score_fn` — no changes
to the search itself.

## Run

Every folder is a package; run modules with `-m` from inside `rlig/`:

```
python -m envs.dag          # self-check: fast mask == brute-force legality
python -m scoring.bic       # self-check: cached delta == full rescore
python -m agents.rlbayes    # RLBayes Q-table recovering structure on synthetic data
python -m eval.metrics      # self-check
python -m scripts.repro --config configs/asia.yaml   # (full experiments, WIP)
typst compile report/report.typ                        # build the report PDF
```

Setup:
```
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
pip install -r requirements.txt
```

Experiment knobs (α, β, tile length, generative-step indices, sample count,
edit budget, max in-degree): `configs/*.yaml`.

## References

- Deleu et al. (2022). *Bayesian Structure Learning with Generative Flow Networks.* UAI.
- Wang et al. (2025). *RLBayes: a Bayesian Network Structure Learning Algorithm via Reinforcement Learning-Based Search Strategy.*
- Kuipers & Moffa. *Partition MCMC* for inference on acyclic digraphs.
- Constantinou et al. (2022). *Effective and efficient structure learning with pruning and model averaging strategies.*
