"""
The ultimatum game in Mei, Xie, Yuan & Jackson (2024, PNAS), as proposer and as responder. Its prompts and human answers.

Wording from Mei et al.'s ChatGPT-4 sessions (github.com/yutxie/ChatGPT-Behavioral).
"""

import pandas as pd

from experiments.mei_et_al import HUMAN_DATA_DIR
from experiments.mei_et_al.games import Game

ANSWER_FORMAT = "Please give only one concrete choice, and highlight the amount with [] (such as [$x])."

PROPOSER_STATE = (
    "This is a two-player game. You are the Proposer, and the other player is the Responder. "
    "As the proposer, you propose how to divide $100 and the Responder chooses either Accept or Reject. "
    "If accepted, the two of you will earn as described by the accepted proposal accordingly. "
    "If rejected, then both of you will earn $0."
)
PROPOSER_QUESTION = "How much would you like to propose to give to the Responder?"

PROPOSER = Game(
    name="ultimatum_proposer",
    state=PROPOSER_STATE,
    question=PROPOSER_QUESTION,
    original=f"{PROPOSER_STATE} {PROPOSER_QUESTION} {ANSWER_FORMAT}",
    top=100,
)

RESPONDER_STATE = (
    "This is a two-player game. You are the Responder, and the other player is the Proposer. "
    "The proposer proposes how to divide $100 and you, as the Responder, choose either Accept or Reject. "
    "If accepted, the two of you will earn as described by the accepted proposal accordingly. "
    "If rejected, then both of you will earn $0."
)
RESPONDER_QUESTION = "As the Responder, what is the minimal amount in the proposal that you would accept?"

RESPONDER = Game(
    name="ultimatum_responder",
    state=RESPONDER_STATE,
    question=RESPONDER_QUESTION,
    original=f"{RESPONDER_STATE} {RESPONDER_QUESTION} {ANSWER_FORMAT}",
    top=100,
)

GAMES = [PROPOSER, RESPONDER]


def human_choices():
    """
    First-round proposals and minimum acceptable offers, filtered as in Mei et al.'s plot.ipynb.
    """
    ultimatum = pd.read_csv(HUMAN_DATA_DIR / "ultimatum_strategy.csv")
    ultimatum = ultimatum[(ultimatum["gameType"] == "ultimatum_strategy") & (ultimatum["Role"] == "player")
                          & (ultimatum["Round"] == 1) & (ultimatum["Total"] == 100)
                          & ultimatum["propose"].between(0, 100) & ultimatum["accept"].between(0, 100)]
    return {"ultimatum_proposer": ultimatum["propose"], "ultimatum_responder": ultimatum["accept"]}
