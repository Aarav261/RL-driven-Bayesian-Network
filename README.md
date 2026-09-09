# RLiG — RL-driven Bayesian Network structure learning (SIT320 HD, Option A)

An RL agent edits a DAG (`add`/`delete`/`reverse`) to learn a Bayesian Network.
Reward is a **hybrid**: cheap structural score (ΔBIC) most steps, plus an
expensive **generative** score (does data simulated from the BN look like the
real data?) at selected steps inside fixed-length *tiles*.

## Build order (do these in order; get end-to-end dumb, then smart)

1. `rlig/dag.py` — DAG + legal edit mask. ✅ done, `python rlig/dag.py` self-checks.
2. `rlig/score.py` — CPT learning (MLE + Dirichlet) + decomposable BIC.
3. `rlig/baselines.py` — Hill-Climbing (validates 1–2, required baseline).
4. `rlig/simulate.py` + `rlig/genscore.py` — sample from BN; start with held-out log-lik.
5. `rlig/env.py` — RL environment + tiling.
6. `rlig/agent.py` — tabular Q-learning on ASIA.
7. `rlig/evaluate.py` — SHD, P/R/F1 ✅ (BIC + generation metrics todo).
8. `rlig/run.py` + ablations → `report.md` (6–8 pages).
9. HD extension (pick 1–2): DQN / learned tiling / theory note / discriminator score.

## Reuse, don't reinvent
`pgmpy` gives BN structures, CPT fitting, BIC, and sampling. Wrap it; only
hand-roll where the hybrid reward needs something its API can't express.
Ground-truth structures + datasets: the **bnlearn** repository.

## Run
```
python -m rlig.dag         # self-check
python -m rlig.score       # self-check
python -m rlig.evaluate    # self-check
python run.py --config config.yaml   # (once steps 2-6 are implemented)
```
(Run as modules — `rlig/` is a package; `score.py` imports `dag.py`.)
Config knobs (α, β, tile L, I_g, N_s, budget T, k): `config.yaml`.
