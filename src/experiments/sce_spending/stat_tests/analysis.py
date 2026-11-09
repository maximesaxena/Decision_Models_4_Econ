"""
Compare models with the SCE respondents they were asked to be, each of the 7 options a bin: Mei et al.'s (2024)
Turing test (win/tie/lose) and predictive fit (cross-entropy and KL divergence, in bits per answer).
"""

import numpy as np
import pandas as pd

from common import models
from common.plotting import REFERENCES
from experiments.sce_spending import RESULTS_DIR
from experiments.sce_spending.data import personas
from experiments.sce_spending.prompts import QUESTIONS

ANSWERS = list(QUESTIONS["qsp12n"][1])  # "1" to "7", for both questions
P = [f"p_{answer}" for answer in ANSWERS]
SMOOTHING = 0.01  # keeps the KL divergence finite


def model_results():
    """
    Every model's probabilities per persona and question, averaged over its valid calls, and its calls
    per question; None if no model has been run.
    """
    paths = sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        return None
    results = pd.concat([pd.read_csv(path) for path in paths])
    valid = results.dropna(subset=P)
    by_persona = valid.groupby(["model", "userid", "date", "item"], sort=False)[P].mean()
    by_persona["answers"] = valid.groupby(["model", "userid", "date", "item"], sort=False).size()
    calls = results.groupby(["model", "item"], sort=False).size().rename("calls")
    return by_persona.reset_index(), calls


def smooth(p):
    return (1 - SMOOTHING) * p + SMOOTHING / len(ANSWERS)


def kl_bits(target, shares):
    """
    The KL divergence of the smoothed shares from the target, in bits. Either can also be one distribution per row.
    """
    target, shares = np.asarray(target), smooth(np.asarray(shares))
    with np.errstate(divide="ignore", invalid="ignore"):  # options no one chose add nothing
        return np.where(target > 0, target * np.log2(target / shares), 0).sum(axis=-1)


def entropy(target):
    target = np.asarray(target)
    target = target[target > 0]
    return float(-np.sum(target * np.log2(target)))


def turing_test(human_share, model_share):
    """
    Chances that a random model answer is more, equally or less typical of humans than a random human one. Either
    can also be one distribution per row.
    """
    more = human_share[..., :, None] > human_share[..., None, :]
    equal = human_share[..., :, None] == human_share[..., None, :]
    win = np.einsum("...i,...ij,...j->...", model_share, more, human_share)
    tie = np.einsum("...i,...ij,...j->...", model_share, equal, human_share)
    return win, tie, 1 - win - tie


def compare():
    """
    One row per question and source: humans, uniform, then every model run.
    """
    people = personas()
    results = model_results()
    by_persona, calls = results if results is not None else (None, None)
    rows = []
    for item in QUESTIONS:
        answers = people[item].astype(int).astype(str)
        target = np.array([(answers == answer).mean() for answer in ANSWERS])

        def row(source, shares, extra=None):
            win, tie, lose = turing_test(target, shares)
            kl = 0.0 if source == "humans" else kl_bits(target, shares)
            return {"item": item, "model": source, "win_%": 100 * win, "tie_%": 100 * tie, "lose_%": 100 * lose,
                    "cross_entropy_bits": entropy(target) + kl, "kl_bits": kl, **(extra or {})}

        rows += [row("humans", target, {"personas": len(people)}), row("uniform", np.full(len(ANSWERS), 1 / len(ANSWERS)))]
        if by_persona is not None:
            for model, r in by_persona[by_persona["item"] == item].groupby("model", sort=False):
                r = r.merge(people[["userid", "date"]])  # only the personas
                rows.append(row(model, r[P].mean().to_numpy(), {
                    "personas": len(r), "calls": calls[model, item],
                    "valid_answer_pct": 100 * r["answers"].sum() / calls[model, item]}))
    return in_order(pd.DataFrame(rows))


def summarise(comparison):
    """
    The comparison averaged over the two questions.
    """
    numbers = [column for column in comparison.columns if column not in ["item", "model"]]
    return in_order(comparison.groupby("model", as_index=False, sort=False).agg(
        questions=("item", "size"), **{column: (column, "mean") for column in numbers}))


def in_order(table):
    """
    Rows ordered by question, then humans, uniform and the models in models.MODELS's order.
    """
    rank = {reference: i - len(REFERENCES) for i, reference in enumerate(REFERENCES)}
    rank |= {model: i for i, model in enumerate(models.MODELS)}
    keys = [table["model"].map(rank).fillna(len(rank))]
    if "item" in table:
        keys.insert(0, table["item"].map(list(QUESTIONS).index))
    return table.iloc[np.lexsort(keys[::-1])].reset_index(drop=True)
