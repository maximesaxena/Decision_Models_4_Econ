"""
The trust game in Mei, Xie, Yuan & Jackson (2024, PNAS), as investor and as banker. Its prompts and human answers.

Wording from Mei et al.'s ChatGPT-4 sessions (github.com/yutxie/ChatGPT-Behavioral).
"""

import json

import pandas as pd

from experiments.mei_et_al import HUMAN_DATA_DIR
from experiments.mei_et_al.games import Game

ANSWER_FORMAT = "Please give only one concrete choice, and highlight the number with [] (such as [$x])."

INVESTOR_STATE = (
    "This is a two-player game. You are an Investor and the other player is a Banker. "
    "You have $100 to invest and you choose how much of your money to invest with the Banker. "
    "The amount you choose to invest will grow by 3x with the Banker. "
    "For example, if you invest $10, it will grow to $30 with the Banker. "
    "The Banker then decides how much of the money ($0-$30) to return to you, the Investor."
)
INVESTOR_QUESTION = "How much of the $100 would you like to invest with the Banker?"

INVESTOR = Game(
    name="trust_investor",
    state=INVESTOR_STATE,
    question=INVESTOR_QUESTION,
    original=f"{INVESTOR_STATE} {INVESTOR_QUESTION} {ANSWER_FORMAT}",
    top=100,
)

BANKER_STATE = (
    "This is a two-player game. You are a Banker and the other player is an Investor, and the goal for each player is to earn more. "
    "The Investor chooses how much of the money (up to $100) to invest with you. "
    "The amount the Investor invests will generate a 2x return with you (the current value of investment becomes 3x). "
    "To settle the investment, as the Banker, you get to decide how much of this total amount to return to the Investor and the rest will be kept as your profit. "
    "For example, you can choose to return $0 (therefore the investor will lose their investment), "
    "or you can return the entire 3x (initial investment + 2x profit) to the investor, or you can return any amount in between."
)
BANKER_QUESTION = "Now, if the investor has invested $50 with you and the current value became $150, how much of the $150 would you like to return to the Investor?"

BANKER = Game(
    name="trust_banker",
    state=BANKER_STATE,
    question=BANKER_QUESTION,
    original=f"{BANKER_STATE} {BANKER_QUESTION} {ANSWER_FORMAT}",
    top=150,
)

GAMES = [INVESTOR, BANKER]


def human_choices():
    """
    First-round investments, and returns by bankers given $50, filtered as in Mei et al.'s plot.ipynb.
    """
    trust = pd.read_csv(HUMAN_DATA_DIR / "trust_investment.csv")
    trust = trust[(trust["gameType"] == "trust_investment") & (trust["Round"] == 1)]
    banker_results = trust.loc[trust["Role"] == "second", "roundResult"].map(json.loads)
    investments = trust.loc[trust["Role"] == "first", "move"]
    returns = pd.Series([returned for invested, returned in banker_results if invested == 50])
    return {"trust_investor": investments[investments.between(0, 100)], "trust_banker": returns[returns.between(0, 150)]}
