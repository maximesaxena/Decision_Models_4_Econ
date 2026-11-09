"""
The dictator game in Mei, Xie, Yuan & Jackson (2024, PNAS). Its prompts and human answers.

Wording from Mei et al.'s ChatGPT-4 sessions (github.com/yutxie/ChatGPT-Behavioral). Decision models'
question asks for the amount given explicitly, as only the dropped answer-format sentence said so.
"""

import pandas as pd

from experiments.mei_et_al import HUMAN_DATA_DIR
from experiments.mei_et_al.games import Game

DICTATOR = Game(
    name="dictator",
    state="You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice.",
    question="How would you like to divide the money? How much do you give to the other player?",
    original="You are paired with another player. Your role is to decide how to divide $100 and the other player simply receives your choice. "
             "How would you like to divide the money? "
             "Please give only one concrete choice and highlight the amount you give to the other player in [] (such as [$x]).",
    top=100,
)

GAMES = [DICTATOR]


def human_choices():
    """
    Human first-round amounts given, filtered as in Mei et al.'s plot.ipynb.
    """
    dictator = pd.read_csv(HUMAN_DATA_DIR / "dictator.csv")
    dictator = dictator[(dictator["gameType"] == "dictator") & (dictator["Role"] == "first")
                        & (dictator["Round"] == 1) & (dictator["Total"] == 100)]
    return {"dictator": dictator["move"][dictator["move"].between(0, 100)]}
