"""
Check that every game's prompts fit PROMPT_TOKEN_LIMIT (Jev's character limits are checked in games.py).
"""

import os

import pandas as pd
from transformers import AutoTokenizer

from common import models
from common.models import PROMPT_TOKEN_LIMIT
from experiments.mei_et_al.runner import decision_questions
from experiments.mei_et_al.registry import EXPERIMENTS

GAMES = [game for experiment in EXPERIMENTS for game in experiment.GAMES]

laya = models.Laya()
qwen_tokenizer = AutoTokenizer.from_pretrained(models.QWEN_MODEL)
gemini = None
if os.getenv("GEMINI_API_KEY"):
    from google import genai
    gemini = genai.Client()

rows = []
for game in GAMES:
    for elicitation, question in decision_questions(game).items():
        rows.append({"game": game.name, "prompt": elicitation,
                     "laya": laya.ask(game.state, question)[1]["input_tokens"],
                     "qwen tokenizer (anyjev, kev)": models.prompt_tokens(qwen_tokenizer, game.state, question)})
    rows.append({"game": game.name, "prompt": "original (LLMs)",
                 "qwen tokenizer (qwen)": models.prompt_tokens(qwen_tokenizer, game.original),
                 "gemini": gemini.models.count_tokens(model=models.GEMINI_MODEL, contents=game.original).total_tokens if gemini else None})

table = pd.DataFrame(rows).set_index(["game", "prompt"])
print(table.to_string(na_rep="-", float_format="{:.0f}".format))
largest = table.max().max()
print(f"\nLargest prompt: {largest:.0f} tokens; limit: {PROMPT_TOKEN_LIMIT}")
assert largest <= PROMPT_TOKEN_LIMIT, "Some prompts are over the limit"
