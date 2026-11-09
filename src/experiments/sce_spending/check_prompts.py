"""
Check every prompt fits Jev's character limits, and the LONGEST prompts of each question the token limits.
"""

import json
import os

import pandas as pd
from transformers import AutoTokenizer

from common import models
from experiments.mei_et_al.games import MAX_CRITERIA_CHARACTERS, MAX_INSTRUCTIONS_CHARACTERS, MAX_OPTIONS, MAX_STATE_CHARACTERS
from experiments.sce_spending.data import personas
from experiments.sce_spending.prompts import prompts

LONGEST = 5

people = personas()
laya = models.Laya()
qwen_tokenizer = AutoTokenizer.from_pretrained(models.QWEN_MODEL)
gemini = None
if os.getenv("GEMINI_API_KEY"):
    from google import genai
    gemini = genai.Client()

rows = []
every = prompts(people)
for row in every:
    question = row["question"]
    assert len(row["state"]) <= MAX_STATE_CHARACTERS and len(row["original"]) <= MAX_STATE_CHARACTERS
    assert len(question["instructions"]) <= MAX_INSTRUCTIONS_CHARACTERS
    assert len(question["criteria"]) <= MAX_OPTIONS and len(json.dumps(question["criteria"])) <= MAX_CRITERIA_CHARACTERS
for item in ["qsp12n", "qsp13new"]:
    longest = sorted((row for row in every if row["item"] == item), key=lambda row: len(row["original"]))[-LONGEST:]
    for row in longest:
        rows.append({"item": item, "userid": row["userid"],
                     "laya": laya.ask(row["state"], row["question"])[1]["input_tokens"],
                     "qwen tokenizer (anyjev, kev)": models.prompt_tokens(qwen_tokenizer, row["state"], row["question"]),
                     "original, qwen tokenizer": models.prompt_tokens(qwen_tokenizer, row["original"]),
                     "original, gemini": gemini.models.count_tokens(model=models.GEMINI_MODEL, contents=row["original"]).total_tokens if gemini else None})

table = pd.DataFrame(rows).set_index(["item", "userid"])
print(table.to_string(na_rep="-", float_format="{:.0f}".format))
largest = table.max().max()
print(f"\nEvery prompt is within Jev's character limits. Largest prompt: {largest:.0f} tokens; limit: {models.PROMPT_TOKEN_LIMIT}")
assert largest <= models.PROMPT_TOKEN_LIMIT, "Some prompts are over the limit"
