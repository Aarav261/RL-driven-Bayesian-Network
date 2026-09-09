"""The repro script: read config.yaml, run all experiments, write tables/figures.

    python run.py --config config.yaml

Ties everything together: load dataset -> train RLiG agent + run baselines ->
evaluate.py metrics -> dump a results table and learning-curve figures.
TODO: fill in once env.py / agent.py / baselines.py exist.
"""

import argparse


def main(config_path):
    raise NotImplementedError("wire up once env/agent/baselines are implemented")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
