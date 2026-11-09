"""
The bomb risk game in Mei, Xie, Yuan & Jackson (2024, PNAS): how many boxes to open in the first round. Its prompts and human answers.

Wording from Mei et al. (github.com/yutxie/ChatGPT-Behavioral).
"""

import pandas as pd

from experiments.mei_et_al import HUMAN_DATA_DIR
from experiments.mei_et_al.games import Game

STATE = (
    "There are 100 boxes, and one bomb has been randomly placed in 1 of 100 boxes. "
    "You can choose to open 0-100 boxes at the same time. "
    "If none of the boxes you open has the bomb, you earn points that are equal to the number of boxes you open. "
    "If one of the boxes you open has the bomb, you earn zero points."
)
QUESTION = "How many boxes would you open?"

BOMB_RISK = Game(
    name="bomb_risk",
    state=STATE,
    question=QUESTION,
    original=f"{STATE} {QUESTION} Please give one concrete number and highlight it with [] (such as [x]).",
    top=100,
    unit="{x:g}",  # a number of boxes
)

GAMES = [BOMB_RISK]


def human_choices():
    """
    Each player's first valid number of boxes opened, as in Mei et al.'s plot.ipynb.
    """
    bomb = pd.read_csv(HUMAN_DATA_DIR / "bomb_risk.csv")
    bomb = bomb[(bomb["Role"] == "player") & (bomb["gameType"] == "bomb_risk")]
    player_block = (bomb["UserID"] != bomb["UserID"].shift()).cumsum()
    valid = bomb["move"].between(0, 100)
    return {"bomb_risk": bomb[valid].groupby(player_block[valid]).head(1)["move"]}
