"""
Ask models every Mas & Pallais prompt --repeats times, one call at a time, appending each call to
results/mas_pallais/<model>.csv (p_1, p_2: probabilities of the positions as shown). A rerun resumes.

Usage: python src/experiments/mas_pallais/run.py jev laya ...
       python src/experiments/mas_pallais/run.py jev --repeats 10
       python src/experiments/mas_pallais/run.py gemini --pilot   # work from home, $16, once each: 58 calls"""

import argparse
import time
from collections import Counter

import pandas as pd
from tqdm import tqdm

from common import models
from experiments.mas_pallais import RESULTS_DIR
from experiments.mas_pallais.prompts import parse, prompts

ATTEMPTS = 10
LLMS = ["gemini", "qwen", "gemini-lowthink", "qwen-nothink", "gpt", "gpt-sol", "gemma", "glm-flash", "gemini-argon", "haiku"]
REPEATS = 6  
PILOT = {"treatment": "work_from_home", "max_wage": 16}
COST_PER_CALL = {"gemini": 0.0027} #my guesstimate of cost per call for Gemini
KEY = ["treatment", "gap", "max_wage", "a_first"]


def timed(call, *args):
    """
    The call's result, the seconds taken by the attempt that worked, and the number of attempts.
    """
    for attempt in range(1, ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            result = call(*args)
        except AssertionError:  # a prompt over a model's limits: retrying will not help
            raise
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            tqdm.write(f"  attempt {attempt} failed ({type(error).__name__}: {error}); retrying in {2 ** attempt} s")
            time.sleep(2 ** attempt)
            continue
        return result, time.perf_counter() - start, attempt


def append(path, rows):
    """
    Append rows to a CSV, refusing if its columns differ.
    """
    if path.exists():
        columns = list(pd.read_csv(path, nrows=0).columns)
        assert columns == list(rows[0]), f"{path} has columns {columns}, not {list(rows[0])}; move it to start again"
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_model(model_name, model, rows, repeats):
    """
    Ask the model every prompt repeats times (each prompt once per pass), skipping calls already saved.
    """
    path = RESULTS_DIR / f"{model_name}.csv"
    done = set(pd.read_csv(path)[[*KEY, "repeat"]].itertuples(index=False, name=None)) if path.exists() else set()
    per_prompt = Counter(key[:-1] for key in done)
    assert max(per_prompt.values(), default=0) <= repeats, (
        f"{path} has up to {max(per_prompt.values())} calls per prompt, more than --repeats {repeats}; move it to start again")
    todo = [(row, repeat) for repeat in range(repeats) for row in rows
            if tuple(row[k] for k in KEY) + (repeat,) not in done]
    cost = f", about ${len(todo) * COST_PER_CALL[model_name]:.2f}" if model_name in COST_PER_CALL else ""
    print(f"{model_name}: {len(todo)} calls to make ({len(done)} already saved){cost}")
    if not todo:
        return
    llm = model_name in LLMS
    first = todo[0][0]
    if llm:
        timed(model.respond, first["original"])  # warm-up
    else:
        timed(model.ask, first["state"], first["question"])  # warm-up
    for row, repeat in tqdm(todo, desc=model_name, unit="call"):
        options = list(row["question"]["criteria"])
        if llm:
            reply, seconds, attempts = timed(model.respond, row["original"])
            chose = parse(reply["response"])
            output = {**{f"p_{option}": None if chose is None else float(chose == option) for option in options},
                      "chose": chose, **reply}
        else:
            (probs, usage), seconds, attempts = timed(model.ask, row["state"], row["question"])
            # a refusal has no probabilities
            output = {**{f"p_{option}": probs.get(option) for option in options}, **usage}
        result = {"model": model_name, **{k: row[k] for k in KEY}, "repeat": repeat, **output,
                  "seconds": seconds, "attempts": attempts, "device": model.device}
        append(path, [result])
    if llm:
        valid = pd.read_csv(path)["chose"].notna().mean()
        print(f"{100 * valid:.1f}% of {model_name}'s saved calls have a valid answer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", choices=[*models.DECISION_MODELS, *LLMS])
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"calls per prompt (default {REPEATS})")
    parser.add_argument("--pilot", action="store_true", help="LLMs only: work from home at $16, once per prompt")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = prompts()
    for model_name in args.models:
        pilot = args.pilot and model_name in LLMS
        model_rows = [row for row in rows if all(row[k] == v for k, v in PILOT.items())] if pilot else rows
        run_model(model_name, models.load(model_name), model_rows, 1 if pilot else args.repeats)
        print(f"Saved results to {RESULTS_DIR / f'{model_name}.csv'}")
