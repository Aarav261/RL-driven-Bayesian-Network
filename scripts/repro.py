"""The repro script: read a config, run all experiments, write tables/figures.

    python -m scripts.repro --config configs/asia.yaml     (from rlig/)

Ties everything together: data.loaders -> train RLiG agent + run baselines ->
eval.metrics -> results tables in report/tables, figures in report/figures.
TODO: fill in once env.py / agent.py / baselines.py exist.
"""

import argparse


def main(config_path):
    raise NotImplementedError("wire up once env/agent/baselines are implemented")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/asia.yaml")
    main(ap.parse_args().config)
