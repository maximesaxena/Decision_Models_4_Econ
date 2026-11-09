"""
Run models on an experiment's games, saving every call's raw output, one call at a time.

Each model is called --repeats times per game and way of asking (decision models: choice and score;
LLMs: sampled). Saves to the experiment's results folder:
- <model>.csv: one row per call and answer, with its probability (an LLM's answer has probability 1;
  a call with no answer has an empty one)
- calls/<model>.csv: one row per call, with its time, attempts, tokens and any LLM response

Both are appended after every call, so a rerun resumes where it stopped.
"""

import time
from collections import Counter

import pandas as pd
from tqdm import tqdm

from common import models
from common.models import PROMPT_TOKEN_LIMIT

REPEATS = 50  
ATTEMPTS = 5  # waiting 2, 4, 8 and 16 seconds between them
KEY = ["model", "game", "elicitation", "repeat"]  # one call


def timed(call, *args):
    """
    The call's result, the seconds taken by the attempt that worked, and the number of attempts.
    """
    for attempt in range(1, ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            result = call(*args)
        except AssertionError:  # a prompt over the limits: retrying will not help
            raise
        except Exception as error:
            if attempt == ATTEMPTS:
                raise
            tqdm.write(f"  attempt {attempt} failed ({type(error).__name__}: {error}); retrying in {2 ** attempt} s")
            time.sleep(2 ** attempt)
            continue
        return result, time.perf_counter() - start, attempt


def check_tokens(model_name, game, tokens):
    assert tokens <= PROMPT_TOKEN_LIMIT, f"{model_name} {game.name}: prompt is {tokens} tokens, over the {PROMPT_TOKEN_LIMIT} limit"


def decision_questions(game):
    """
    The game asked as a choice question and as a score question.
    """
    return {"choice": {"type": "choice", "instructions": game.question, "criteria": game.options()},
            "score": {"type": "score", "instructions": game.question, "criteria": list(game.levels().values())}}


def elicitations(model_name, game):
    """
    The ways the model is asked the game: decision questions, or "sampled" for an LLM.
    """
    return {"sampled": None} if model_name in models.LLMS else decision_questions(game)


def call(model_name, model, game, elicitation, question):
    """
    Ask the model the game once, returning its answer rows and its log row.
    """
    if question is None:  # an LLM
        reply, seconds, attempts = timed(model.respond, game.original)
        check_tokens(model_name, game, reply["input_tokens"])
        answer = game.parse(reply["response"])
        return ([{"answer": answer, "probability": None if answer is None else 1.0}],
                {"seconds": seconds, "attempts": attempts, **reply, "answer": answer})
    (probs, usage), seconds, attempts = timed(model.ask, game.state, question)
    # Jev's input_tokens include its own template, so its limits are checked in characters (see games.py)
    if model.device != "api":
        check_tokens(model_name, game, usage["input_tokens"])
    if not probs:  # a refusal
        return [{"answer": None, "probability": None}], {"seconds": seconds, "attempts": attempts, **usage, "answer": None}
    if elicitation == "score":
        probs = {answer: probs[str(level)] for level, answer in enumerate(game.levels())}
    return ([{"answer": answer, "probability": p} for answer, p in probs.items()],
            {"seconds": seconds, "attempts": attempts, **usage, "answer": max(probs, key=probs.get)})


def warm_up(model_name, model, game):
    """
    One untimed call per loaded model, so start-up is not timed.
    """
    if getattr(model, "warmed_up", False):
        return
    if model_name in models.LLMS:
        timed(model.respond, game.original)
    else:
        timed(model.ask, game.state, decision_questions(game)["choice"])
    model.warmed_up = True


def add_repeats_argument(parser):
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"calls per game and way of asking (default {REPEATS})")


def saved(path):
    return pd.read_csv(path, dtype={"answer": str}) if path.exists() else None


def append(path, rows):
    """
    Append rows to a CSV, refusing if its columns differ.
    """
    if path.exists():
        columns = list(pd.read_csv(path, nrows=0).columns)
        assert columns == list(rows[0]), f"{path} has columns {columns}, not {list(rows[0])}; move it to start again"
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_model(model_name, model, games, results_dir, repeats=REPEATS):
    """
    Call one loaded model repeats times on every game and way of asking (each once per pass), skipping
    calls already saved.
    """
    answers_path, calls_path = results_dir / f"{model_name}.csv", results_dir / "calls" / f"{model_name}.csv"
    calls, answers = saved(calls_path), saved(answers_path)
    done = set() if calls is None else set(calls[KEY].itertuples(index=False, name=None))
    if answers is not None:
        # drop answers whose call was never logged (a crash between the two writes)
        logged = pd.Series([tuple(key) in done for key in answers[KEY].itertuples(index=False, name=None)], index=answers.index)
        if not logged.all():
            answers[logged].to_csv(answers_path, index=False)

    per_question = Counter((game, elicitation) for _, game, elicitation, _ in done)
    assert max(per_question.values(), default=0) <= repeats, (
        f"{calls_path} has up to {max(per_question.values())} calls per question, more than --repeats {repeats}; "
        f"move it and {answers_path} to start again")
    todo = [(game, elicitation, question, repeat) for repeat in range(repeats) for game in games
            for elicitation, question in elicitations(model_name, game).items()
            if (model_name, game.name, elicitation, repeat) not in done]
    print(f"{model_name}: {len(todo)} calls to make ({len(done)} already saved in {results_dir})")
    if not todo:
        return

    warm_up(model_name, model, todo[0][0])
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    for game, elicitation, question, repeat in tqdm(todo, desc=model_name, unit="call"):
        answer_rows, log_row = call(model_name, model, game, elicitation, question)
        key = dict(zip(KEY, [model_name, game.name, elicitation, repeat]))
        append(answers_path, [{**key, **row} for row in answer_rows])
        append(calls_path, [{**key, "device": model.device, **log_row}])
        #time.sleep(3)
    print(f"Saved answers to {answers_path} and calls to {calls_path}")
