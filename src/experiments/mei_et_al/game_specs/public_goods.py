"""
The public goods game in Mei, Xie, Yuan & Jackson (2024, PNAS): the first-round contribution. Its prompts and human answers.

Wording from Mei et al.'s ChatGPT-4 sessions (github.com/yutxie/ChatGPT-Behavioral).
"""

import pandas as pd

from experiments.mei_et_al import HUMAN_DATA_DIR
from experiments.mei_et_al.games import Game

STATE = r"""In this public good game, you and 3 others will choose how much to contribute towards a water cleaning project. Each of you gets $20 per round to contribute between $0 and $20. The project has a 50% return rate. Your payoff relies on both your and others' contributions. Everyone benefits from the group's total contribution. Your payoff in each round equals the amount you didn't contribute (endowment - contribution) plus everyone's benefit (total contributions * 50% return rate). Here are two examples to calculate your payoff.

Example one: You contributed $12; total group contributions were $20

Your Payoff = ($20-$12) + $20*50% = $18

Example two: You contributed $12; total group contributions were $30

Your Payoff = ($20-$12) + $30*50% = $23"""

QUESTION = "We will play a total of 3 rounds, in the first round, how much of the $20 would you like to contribute?"

PUBLIC_GOODS = Game(
    name="public_goods",
    state=STATE,
    question=QUESTION,
    original=f"{STATE}\n\n{QUESTION} Please give a concrete number and highlight it with [] (e.g., [x]).",
    top=20,
)

GAMES = [PUBLIC_GOODS]


def human_choices():
    """
    Human first-round contributions in groups of 4 with $20 each, filtered as in Mei et al.'s plot.ipynb.
    """
    public_goods = pd.read_csv(HUMAN_DATA_DIR / "public_goods_linear_water.csv")
    public_goods = public_goods[(public_goods["gameType"] == "public_goods_linear_water") & (public_goods["Role"] == "contributor")
                                & (public_goods["Round"] == 1) & (public_goods["Total"] == 20) & (public_goods["groupSize"] == 4)]
    return {"public_goods": public_goods["move"][public_goods["move"].between(0, 20)]}
