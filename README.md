# RLiG: Reinforcement Learning for Bayesian Network Structure Learning

An exploration into whether reinforcement learning can learn the *structure* of a
Bayesian Network more effectively than the search heuristics the field usually
leans on.

## What I'm trying to do

Learning the graph of a Bayesian Network (which variable causes which) is an
NP-hard search: the number of possible DAGs grows super-exponentially with the
number of variables, so we can't enumerate them. Score-based search such as hill
climbing and GES (Chickering, 2002) is the standard approach. Sampling methods
such as MCMC and GFlowNets (Deleu et al., 2022) give a posterior over graphs rather
than a single graph, but they get expensive as the number of variables grows.

I'm taking a different angle: **treat structure learning as a sequential
decision process and let an RL agent search it.** The agent starts from an empty
graph and edits it one edge at a time (add / delete / reverse), guided by a
reward that says whether each edit made the graph a better explanation of the
data. The hope is that over many edits it finds a high-scoring network without ever
touching most of the search space.

RL for structure learning has been tried before with a whole-graph generator
trained by policy gradient (Zhu et al., 2020). Here the agent edits the graph one
step at a time instead. Two ideas sit at the centre of this:

1. **Q-learning on a tiled environment.** The state is the DAG plus its step index
   within a tile, and a Q-learning agent learns on `env.step()`, so a discount
   factor can carry a delayed generative reward back to the edits that earned it.
   RLBayes (Wang et al., 2025) is the RL baseline: a dynamic Q-table that only
   stores the networks it has visited and caps that memory by dropping the worst.
   Its cells are one-step score changes with no bootstrapping, so it acts as a
   strong search heuristic rather than the full RL agent.

2. **A hybrid reward.** Classical learners score a structure purely on how well
   its shape fits the data (BIC). I add a second term: *simulate* data from the
   learned network and measure how close the synthetic data is to the real data.
   The reward becomes `α · ΔBIC + β · generative-fidelity`. To keep the expensive
   simulation affordable, it's only evaluated at selected steps (a *tiling*
   schedule). Every other step uses the cheap structural score.

The hypothesis is that an RL search driven by this hybrid, generation-aware
objective learns structures that score as well as hill climbing and GES while
reproducing the data-generating distribution more faithfully.

## Why it matters

Bayesian Networks model causal relationships under uncertainty, and they're
useful precisely where data is incomplete and the causal factor space is large:
medicine, economics, and the domain I'm most drawn to, **agriculture**. Crop
yield depends on a high-dimensional tangle of environmental and meteorological
factors, and better structure learning could make yield and food-supply
prediction more accessible and more accurate. That matters more as climate
change widens the swings.

## Approach in code

```
HD/
  plan.md      assignment spec, build order, HD extensions
  HANDOFF.md   where the work is right now
  report.typ   the deliverable, with refs.bib
  rlig/        this repo
    envs/        dag.py (adjacency matrix, edits, cycle check, legal-edit mask), env.py (tiling env)
    scoring/     bic.py (cached decomposable BIC + Dirichlet MLE CPTs), simulate.py, genscore.py
    agents/      qlearn.py (tabular Q-learning on the env), dqn.py (masked Double DQN, MLP or (op, i, j)-embedding Q-head, HD ext 1),
                 rlbayes.py (RLBayes baseline)
    baselines/   hill_climb.py (HC + Tabu), ges.py (pgmpy GES wrapper)
    data/        loaders.py (samples from the bnlearn networks in data/bif/, train/val/test split)
    eval/        metrics.py (SHD, CPDAG SHD, precision/recall/F1)
    configs/     one YAML per experiment
    scripts/     repro.py (runs every method on every seed, writes report/tables)
    report/      tables/ (generated)
```

RLBayes takes any `score_fn`, so it runs the pure-BIC baseline as is. The catch
is that a hybrid `score_fn` would compute GenScore on every new graph and skip
tiling, so the hybrid objective belongs to the Q-learning agent.

Data is split three ways. BIC and the CPTs come from train, the generative reward
is scored on val, and every reported metric is computed on test, which no method
sees while learning.

## Run

Every folder is a package; run modules with `-m` from inside `rlig/`:

```
python -m envs.dag          # self-check: fast mask == brute-force legality
python -m scoring.bic       # self-check: cached delta == full rescore
python -m agents.rlbayes    # RLBayes Q-table recovering structure on synthetic data
python -m agents.qlearn     # Q-learning on the tiled env recovering structure on synthetic data
python -m agents.dqn        # masked Double DQN (both Q-heads), same env and outputs as Q-learning
python -m eval.metrics      # self-check: CPDAG of ASIA, SHD vs CPDAG SHD
python -m baselines.ges     # GES on ASIA
python -m scripts.repro --config configs/asia.yaml   # all methods x all seeds (4 in parallel) -> report/tables/asia.csv
```

Setup:
```
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
pip install -r requirements.txt
```

Experiment knobs (seeds, n, train/val/test split, α, β, tile length,
generative-step indices, sample count, edit budget, max in-degree): `configs/*.yaml`.

## References

- Chickering (2002). *Optimal Structure Identification with Greedy Search.* JMLR.
- Deleu et al. (2022). *Bayesian Structure Learning with Generative Flow Networks.* UAI.
- Wang et al. (2025). *RLBayes: a Bayesian Network Structure Learning Algorithm via Reinforcement Learning-Based Search Strategy.*
- Kuipers & Moffa. *Partition MCMC* for inference on acyclic digraphs.
- Zhu, Ng & Chen (2020). *Causal Discovery with Reinforcement Learning.* ICLR.
- Constantinou et al. (2022). *Effective and efficient structure learning with pruning and model averaging strategies.*
