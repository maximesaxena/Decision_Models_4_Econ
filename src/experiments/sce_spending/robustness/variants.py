"""
The robustness variants, each a change to the baseline prompts of prompts.py:
- human: "Act like a human." starts every prompt (the state for decision models, the prompt for LLMs), and the LLMs
  have no system prompt, as "You are a helpful assistant." would contradict it
- nosystem: the baseline prompts, the LLMs without their system prompt (so human vs nosystem is the cue alone)
- wording: QSP13new only, reworded to mirror QSP12n
"""

from common.models import SYSTEM_PROMPT
from experiments.sce_spending.prompts import QUESTIONS, prompts
from experiments.sce_spending.robustness import DMS, LLMS

HUMAN = "Act like a human."
QSP13_REWORDED = ("Suppose next year you were to find your household with 10% less income than you currently expect. "
                  "What would you do?")

# each variant's LLM system prompt (None for none) and models
VARIANTS = {
    "human": (None, [*DMS, *LLMS]),
    "nosystem": (None, LLMS),
    "wording": (SYSTEM_PROMPT, [*DMS, *LLMS]),
}


def rows(people, variant):
    """
    The variant's prompts, in the format of prompts.prompts.
    """
    every = prompts(people)
    if variant == "human":
        return [{**row, "state": f"{HUMAN} {row['state']}", "original": f"{HUMAN}\n\n{row['original']}"} for row in every]
    if variant == "wording":
        stem = QUESTIONS["qsp13new"][0]
        reworded = []
        for row in every:
            if row["item"] == "qsp13new":
                assert row["original"].count(stem) == 1, f"QSP13new's stem is not in {row['userid']}'s prompt exactly once"
                reworded.append({**row, "question": {**row["question"], "instructions": QSP13_REWORDED},
                                 "original": row["original"].replace(stem, QSP13_REWORDED)})
        return reworded
    return every
