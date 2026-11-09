"""
Compare models' answers with Mei et al.'s (2024) human data, in the paper's 10 bins: the paper's Turing
test (win/tie/lose), predictive fit (cross-entropy and KL divergence, in bits per human answer) and the
Wasserstein-1 (earth mover's) distance, as a percentage of the game's range.
"""

import numpy as np
import pandas as pd

HUMAN_COLOR = "#898781"

SMOOTHING = 0.01  # keeps the KL divergence finite


def load_results(results_dir):
    """
    Every model's distribution of answers in an experiment, or None if none has been run.
    """
    paths = sorted(results_dir.glob("*.csv"))
    return distributions(pd.concat([pd.read_csv(path, dtype={"answer": str}) for path in paths])) if paths else None


def distributions(raw):
    """
    One distribution per model, game and way of asking: probabilities averaged over the calls with an
    answer (for an LLM, the share of valid responses giving each answer).
    """
    valid = raw.dropna(subset=["answer"])
    by = ["model", "game", "elicitation"]
    calls = valid.groupby(by, sort=False)["repeat"].nunique().rename("calls").reset_index()
    summed = valid.groupby([*by, "answer"], sort=False)["probability"].sum().reset_index().merge(calls)
    summed["probability"] /= summed.pop("calls")
    return summed


def load_calls(results_dir):
    """
    Every call logged in an experiment, or None.
    """
    paths = sorted((results_dir / "calls").glob("*.csv"))
    return pd.concat([pd.read_csv(path, dtype={"answer": str}) for path in paths]) if paths else None


def shares(game, answers, weights=None):
    """
    Share of answers in each of the game's bins.
    """
    counts = np.histogram(answers.astype(float), game.bin_edges(), weights=weights)[0]
    return counts / counts.sum()


def turing_test(human_share, model_share):
    """
    Chances that a random model answer is more, equally or less typical of humans than a random human one.
    """
    more = human_share[:, None] > human_share[None, :]  # bin i more typical than bin j
    equal = human_share[:, None] == human_share[None, :]
    win, tie = model_share @ more @ human_share, model_share @ equal @ human_share
    return win, tie, 1 - win - tie


def cross_entropy(human_share, model_share):
    """
    -sum(human * log2(model)), in bits per human answer.
    """
    answered = human_share > 0
    return -(human_share[answered] * np.log2(model_share[answered])).sum()


def predictive_fit(human_share, model_share):
    """
    The smoothed cross-entropy and the KL divergence (cross-entropy less the humans' entropy).
    """
    smoothed = (1 - SMOOTHING) * model_share + SMOOTHING / len(model_share)
    model_cross_entropy = cross_entropy(human_share, smoothed)
    return model_cross_entropy, model_cross_entropy - cross_entropy(human_share, human_share)


def wasserstein(game, human_share, model_share):
    """
    The Wasserstein-1 (earth mover's) distance between the two distributions over the bins' middles, as a
    percentage of the game's range: how far, on average, answers must move to turn one into the other. model_share
    can also be one distribution per row.
    """
    gaps = np.abs(np.cumsum(human_share - model_share, axis=-1))[..., :-1]  # the gap between the CDFs, between each bin and the next
    return 100 * gaps @ np.diff(game.bin_middles()) / game.top


def compare_game(game, human_answers, results):
    """
    One game's rows (humans, uniform, then every model and way of asking) and distributions. Means are
    taken from the bins' middles, as decision models answer with a bin.
    """
    human_share = shares(game, human_answers)

    def row(model, elicitation, share, fit):
        win, tie, lose = turing_test(human_share, share)
        model_cross_entropy, kl = fit
        return {"game": game.name, "model": model, "elicitation": elicitation, "mean": share @ game.bin_middles(),
                "win_%": 100 * win, "tie_%": 100 * tie, "lose_%": 100 * lose,
                "cross_entropy_bits": model_cross_entropy, "kl_bits": kl,
                "wasserstein_%": wasserstein(game, human_share, share)}

    uniform = np.full(len(human_share), 1 / len(human_share))
    distributions = {"humans": human_share}
    rows = [
        row("humans", "humans", human_share, (cross_entropy(human_share, human_share), 0.0)),
        row("uniform", "reference", uniform, predictive_fit(human_share, uniform)),
    ]
    for (model, elicitation), answers in results[results["game"] == game.name].groupby(["model", "elicitation"], sort=False):
        share = distributions[model, elicitation] = shares(game, answers["answer"], weights=answers["probability"].to_numpy())
        rows.append(row(model, elicitation, share, predictive_fit(human_share, share)))
    return rows, distributions


def samples_label(calls):
    """
    A label for the LLMs, with their samples per game as logged.
    """
    counts = set() if calls is None else set(calls[calls["elicitation"] == "sampled"].groupby(["game", "model"]).size())
    return f"LLMs, {counts.pop()} samples" if len(counts) == 1 else "LLMs, sampled"
