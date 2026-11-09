"""
Run models on every game, or those given with --games, loading each model once.

Usage: python src/experiments/mei_et_al/run.py laya jev ...
       python src/experiments/mei_et_al/run.py laya jev ... --games dictator trust --repeats 30
"""

import argparse

from common import models
from experiments.mei_et_al.registry import NAMES, results_dir
from experiments.mei_et_al.runner import add_repeats_argument, run_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=models.MODELS)
    parser.add_argument("--games", nargs="+", choices=NAMES, default=list(NAMES), help="games to run (default: all)")
    add_repeats_argument(parser)
    args = parser.parse_args()

    for model_name in args.models:
        model = models.load(model_name)
        for name in args.games:
            print(f"\n{model_name}: {name}")
            run_model(model_name, model, NAMES[name].GAMES, results_dir(NAMES[name]), args.repeats)
