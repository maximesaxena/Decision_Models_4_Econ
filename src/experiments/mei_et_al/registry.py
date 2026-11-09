"""
Every game's module (with GAMES and human_choices()), for scripts that work across them.
"""

from experiments.mei_et_al import RESULTS_DIR
from experiments.mei_et_al.game_specs import bomb_risk, dictator, public_goods, trust, ultimatum

EXPERIMENTS = [dictator, ultimatum, trust, public_goods, bomb_risk]
NAMES = {experiment.__name__.split(".")[-1]: experiment for experiment in EXPERIMENTS}


def results_dir(experiment):
    return RESULTS_DIR / experiment.__name__.split(".")[-1]
