"""
Bootstrap tests of every decision model against every LLM, averaged over Mei et al.'s games, on the KL divergence
from humans and the Turing test win rate. Humans' distribution is held fixed. In each draw, each model's calls in each
game are resampled with replacement, averaged into a distribution and scored; a pair's difference is the average over
the games both models have results for. Each test's two-sided p-value,
p = 2 * min(share of draws <= 0, share of draws >= 0), is Holm-corrected over each metric's tests; a test is
significant if its corrected p-value is below LEVEL.

Writes analysis_results/mei_et_al/bootstrap_tests_kl.tex, one table per way the decision models are asked (choice and
score), and prints how many tests the decision models and the LLMs win.
With --best, tests only the best decision models (common.plotting.GROUPS["Frontier DMs"]), so the Holm correction is
over fewer tests, and writes bootstrap_tests_kl_best.tex.

Usage: python src/experiments/mei_et_al/stat_tests/bootstrap_tests_kl.py [--best]
"""

import argparse

import numpy as np
import pandas as pd

from common import models
from common.plotting import GROUPS, NAMES
from experiments.mei_et_al import ANALYSIS_DIR
from experiments.mei_et_al.registry import EXPERIMENTS, results_dir
from experiments.mei_et_al.stat_tests.analysis import SMOOTHING, turing_test

OUTPUT = ANALYSIS_DIR / "bootstrap_tests_kl.tex"
BEST_OUTPUT = ANALYSIS_DIR / "bootstrap_tests_kl_best.tex"
BEST = GROUPS["Frontier DMs"]
DRAWS = 10_000
LEVEL = 0.01
SEED = 0
ELICITATIONS = ["choice", "score", "sampled"]
WAYS = ["choice", "score"]  # the decision models' ways of asking, a table each


def kl_divergence(human_share, model_share):
    """
    The KL divergence of the smoothed model distribution from humans', in bits per human answer, as in
    analysis.predictive_fit. model_share can also be one distribution per row.
    """
    smoothed = (1 - SMOOTHING) * model_share + SMOOTHING / model_share.shape[-1]
    answered = human_share > 0
    return (human_share[answered] * np.log2(human_share[answered] / smoothed[..., answered])).sum(axis=-1)


# each metric's panel title, its score for one distribution per row, +1 if higher is closer to humans or -1 if lower,
# and its cells' format
METRICS = {
    "KL divergence (bits)": (lambda game, human, model: kl_divergence(human, model), -1, ".2f"),
    "Turing test win rate (\\%)": (lambda game, human, model: 100 * turing_test(human, model)[0], 1, ".1f"),
}


def call_shares(game, calls):
    """
    Each call's probabilities in the game's bins, one row per call with an answer.
    """
    return np.array([np.histogram(call["answer"].astype(float), game.bin_edges(), weights=call["probability"])[0]
                     for _, call in calls.groupby("repeat")])


def scores(game, human_share, per_call, rng):
    """
    Each metric's score for the calls as they are, and for DRAWS resamples of them.
    """
    observed = per_call.sum(axis=0)
    draws = per_call[rng.integers(len(per_call), size=(DRAWS, len(per_call)))].sum(axis=1)
    return {name: (metric(game, human_share, observed / observed.sum()),
                   metric(game, human_share, draws / draws.sum(axis=1, keepdims=True)))
            for name, (metric, _, _) in METRICS.items()}


def collect():
    """
    {(model, elicitation): {game: scores}} for every model with results, and the number of games.
    """
    rng = np.random.default_rng(SEED)
    scored, games = {}, 0
    for experiment in EXPERIMENTS:
        paths = sorted(results_dir(experiment).glob("*.csv"))
        if not paths:
            continue
        raw = pd.concat([pd.read_csv(path, dtype={"answer": str}) for path in paths]).dropna(subset=["answer"])
        humans = experiment.human_choices()
        for game in experiment.GAMES:
            games += 1
            human_share = np.histogram(humans[game.name].astype(float), game.bin_edges())[0]
            human_share = human_share / human_share.sum()
            for key, calls in raw[raw["game"] == game.name].groupby(["model", "elicitation"], sort=False):
                scored.setdefault(key, {})[game.name] = scores(game, human_share, call_shares(game, calls), rng)
    return scored, games


def test(dm, llm, metric):
    """
    How much closer to humans the decision model is than the LLM, averaged over their common games (positive if the
    decision model is closer), and its two-sided bootstrap p-value.
    """
    sign = METRICS[metric][1]
    games = [game for game in dm if game in llm]
    observed = sign * np.mean([dm[game][metric][0] - llm[game][metric][0] for game in games])
    draws = sign * np.mean([dm[game][metric][1] - llm[game][metric][1] for game in games], axis=0)
    return observed, min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))


def holm(p_values):
    """
    Holm-corrected p-values: the k-th smallest of m is multiplied by m - k + 1, and none is below a smaller one's.
    """
    order = np.argsort(p_values)
    m = len(p_values)
    corrected = np.empty(m)
    corrected[order] = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * np.asarray(p_values)[order]))
    return corrected


def ordered(scored, family):
    """
    The family's (model, elicitation) keys with results, in models.MODELS order then way of asking.
    """
    keys = [key for key in scored if key[0] in family]
    return sorted(keys, key=lambda key: (list(family).index(key[0]), ELICITATIONS.index(key[1])))


def name(key, scored):
    model, elicitation = key
    ways = sum(other[0] == model for other in scored)
    return NAMES.get(model, model).strip() + ("" if ways == 1 else f" ({elicitation})")


def wins(results, metric, dms, llms):
    """
    How many of the metric's tests between dms and llms the decision model wins, the LLM wins, and neither.
    """
    tests = [results[metric, dm, llm] for dm in dms for llm in llms]
    dm_wins = sum(p < LEVEL and observed > 0 for observed, p in tests)
    llm_wins = sum(p < LEVEL and observed < 0 for observed, p in tests)
    return dm_wins, llm_wins, len(tests) - dm_wins - llm_wins


def latex(scored, way, dms, llms, results, counts, games, best, family):
    """
    One panel per metric: a row per LLM, a column per decision model asked this way, each cell how much closer the
    decision model is. family is how many tests each metric's Holm correction is over, across all ways.
    """
    columns = len(dms) + 1
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Bootstrap tests of each {'of the best decision models' if best else 'decision model'}, "
             rf"asked for a {way}, against each LLM on the KL divergence, averaged over Mei et al.'s games}}",
             rf"\label{{tab:mei_bootstrap_kl{'_best' if best else ''}_{way}}}",
             rf"\begin{{tabular}}{{l{'r' * len(dms)}}}", r"\toprule",
             " & ".join(["LLM", *[NAMES.get(dm[0], dm[0]).strip() for dm in dms]]) + r" \\"]
    for panel, metric in zip("AB", METRICS):
        lines += [r"\midrule", rf"\multicolumn{{{columns}}}{{l}}{{\textit{{Panel {panel}: {metric}}}}} \\", r"\midrule"]
        for llm in llms:
            cells = []
            for dm in dms:
                observed, p = results[metric, dm, llm]
                cells.append(f"${observed:{METRICS[metric][2]}}{'^{*}' if p < LEVEL else ''}$")
            lines.append(" & ".join([name(llm, scored), *cells]) + r" \\")
        dm_wins, llm_wins, ties = counts[metric]
        lines.append(rf"\multicolumn{{{columns}}}{{l}}{{\footnotesize Decision model closer: {dm_wins}; "
                     rf"LLM closer: {llm_wins}; no significant difference: {ties}.}} \\")
    short = [f"{name(key, scored)} ({len(scored[key])})" for key in [*dms, *llms] if len(scored[key]) < games]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to humans the decision model is than "
              rf"the LLM, in bits per human answer in Panel A and percentage points in Panel B, averaged over the "
              rf"{games} games with equal weights: positive if the decision model is closer. Model distributions are "
              rf"smoothed with {100 * SMOOTHING:g}\% uniform weight to keep the KL divergence finite. "
              rf"$^{{*}}$ Significant at the {100 * LEVEL:g}\% level (two-sided bootstrap test, "
              rf"{DRAWS:,} draws, Holm-corrected over each metric's {family} tests across the choice and score tables). "
              rf"The human distribution is held fixed; each model's calls are resampled with "
              rf"replacement in each game."
              + (f" Averaged over fewer games, the ones both models have results for: {', '.join(short)}." if short else "")
              + "}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true", help=f"test only {', '.join(BEST)}")
    best = parser.parse_args().best
    output = BEST_OUTPUT if best else OUTPUT
    scored, games = collect()
    dms = [key for key in ordered(scored, models.DECISION_MODELS) if not best or key[0] in BEST]
    llms = ordered(scored, models.LLMS)
    pairs = [(dm, llm) for dm in dms for llm in llms]
    results = {}
    for metric in METRICS:
        observed, p_values = zip(*[test(scored[dm], scored[llm], metric) for dm, llm in pairs])
        results |= {(metric, *pair): result for pair, result in zip(pairs, zip(observed, holm(p_values)))}
    tables, summary = [], []
    for way in WAYS:
        way_dms = [dm for dm in dms if dm[1] == way]
        if not way_dms:
            continue
        counts = {metric: wins(results, metric, way_dms, llms) for metric in METRICS}
        tables.append(latex(scored, way, way_dms, llms, results, counts, games, best, len(pairs)))
        summary.append(f"\nTests at the {LEVEL:.0%} level, {len(way_dms)} decision models asked for a {way} x "
                       f"{len(llms)} LLMs:")
        summary += [f"  {metric.replace(chr(92), '')}: decision model closer {dm_wins}, LLM closer {llm_wins}, "
                    f"no significant difference {ties}" for metric, (dm_wins, llm_wins, ties) in counts.items()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(tables) + "\n")
    print("\n\n".join(tables))
    print("\n".join(summary))
    print(f"\nSaved {output.name} to {output.parent}")


if __name__ == "__main__":
    main()
