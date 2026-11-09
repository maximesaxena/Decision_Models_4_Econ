"""
Paired bootstrap tests of every decision model against every LLM, averaged over the two SCE questions, on the
KL divergence from humans and the Turing test win rate. Every model answered as the same personas, so each
draw resamples the personas with replacement, the same ones for humans, every model and both questions; a
distribution is the average over the draw's personas (for a model, those it gave a valid answer as), and a pair's
difference is the average over the questions both models have results for. Each test's two-sided p-value,
p = 2 * min(share of draws <= 0, share of draws >= 0), is Holm-corrected over each metric's tests; a test is
significant if its corrected p-value is below LEVEL.

Writes analysis_results/sce_spending/bootstrap_tests.tex and prints how many tests the decision models and the LLMs
win. With --best, tests only the best decision models (common.plotting.GROUPS["Best DMs"]), so the Holm correction
is over fewer tests, and writes bootstrap_tests_best.tex.

Usage: python src/experiments/sce_spending/stat_tests/bootstrap_tests.py [--best]
"""

import argparse

import numpy as np

from common import models
from common.plotting import GROUPS, NAMES
from experiments.mei_et_al.stat_tests.bootstrap_tests import holm
from experiments.sce_spending import ANALYSIS_DIR
from experiments.sce_spending.data import personas
from experiments.sce_spending.prompts import QUESTIONS
from experiments.sce_spending.stat_tests.analysis import ANSWERS, P, kl_bits, model_results, turing_test

OUTPUT = ANALYSIS_DIR / "bootstrap_tests.tex"
BEST_OUTPUT = ANALYSIS_DIR / "bootstrap_tests_best.tex"
BEST = GROUPS["Frontier DMs"]
DRAWS = 10_000
LEVEL = 0.01
SEED = 0
#EXCLUDED = {"gpt-sol", "gemma"}  # TEMPORARY: left out until their runs finish
EXCLUDED = {}
# each metric's panel title, its score for one distribution per row, +1 if higher is closer to humans or -1 if lower,
# and the decimals its cells are shown to
METRICS = {
    "KL divergence from humans (bits)": (lambda human, model: kl_bits(human, model), -1, 2),
    "Turing test win rate (\\%)": (lambda human, model: 100 * turing_test(human, model)[0], 1, 1),
}


def shares(probabilities, weights):
    """
    The distribution over the options averaged over the personas with an answer (rows without NaN), as they are and
    for each draw's weights (how many times it drew each persona).
    """
    valid = ~np.isnan(probabilities).any(axis=1)
    probabilities = np.nan_to_num(probabilities)
    return probabilities[valid].mean(axis=0), (weights @ probabilities) / (weights @ valid)[:, None]


def collect():
    """
    {model: {question: scores}} for every model with results, each metric's score as it is and for DRAWS resamples
    of the personas, and the number of personas.
    """
    people = personas()
    by_persona, _ = model_results()
    weights = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=DRAWS)
    scored = {}
    for item in QUESTIONS:
        human = (people[item].astype(int).astype(str).to_numpy()[:, None] == np.array(ANSWERS)).astype(float)
        human_observed, human_draws = shares(human, weights)
        for model, rows in by_persona[by_persona["item"] == item].groupby("model", sort=False):
            model_observed, model_draws = shares(people[["userid", "date"]].merge(rows, how="left")[P].to_numpy(), weights)
            scored.setdefault(model, {})[item] = {
                name: (metric(human_observed, model_observed), metric(human_draws, model_draws))
                for name, (metric, *_) in METRICS.items()}
    return scored, len(people)


def test(dm, llm, metric):
    """
    How much closer to humans the decision model is than the LLM, averaged over their common questions (positive if
    the decision model is closer), and its two-sided bootstrap p-value.
    """
    sign = METRICS[metric][1]
    items = [item for item in dm if item in llm]
    observed = sign * np.mean([dm[item][metric][0] - llm[item][metric][0] for item in items])
    draws = sign * np.mean([dm[item][metric][1] - llm[item][metric][1] for item in items], axis=0)
    return observed, min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))


def name(model):
    return NAMES.get(model, model).strip()


def latex(scored, dms, llms, results, counts, people, best):
    """
    One panel per metric: a row per LLM, a column per decision model, each cell how much closer the decision model is.
    """
    columns = len(dms) + 1
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Paired bootstrap tests of each {'of the best decision models' if best else 'decision model'} "
             r"against each LLM, averaged over the two SCE spending questions}",
             rf"\label{{tab:sce_bootstrap{'_best' if best else ''}}}", rf"\begin{{tabular}}{{l{'r' * len(dms)}}}", r"\toprule",
             " & ".join(["LLM", *[name(dm) for dm in dms]]) + r" \\"]
    for panel, metric in zip("AB", METRICS):
        lines += [r"\midrule", rf"\multicolumn{{{columns}}}{{l}}{{\textit{{Panel {panel}: {metric}}}}} \\", r"\midrule"]
        decimals = METRICS[metric][2]
        for llm in llms:
            cells = []
            for dm in dms:
                observed, p = results[metric, dm, llm]
                cells.append(f"${observed:.{decimals}f}{'^{*}' if p < LEVEL else ''}$")
            lines.append(" & ".join([name(llm), *cells]) + r" \\")
        dm_wins, llm_wins, ties = counts[metric]
        lines.append(rf"\multicolumn{{{columns}}}{{l}}{{\footnotesize Decision model closer: {dm_wins}; "
                     rf"LLM closer: {llm_wins}; no significant difference: {ties}.}} \\")
    short = [f"{name(model)} ({len(scored[model])})" for model in [*dms, *llms] if len(scored[model]) < len(QUESTIONS)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to humans the decision model is than "
              rf"the LLM, in bits (Panel A) or percentage points (Panel B), averaged over the {len(QUESTIONS)} questions "
              rf"with equal weights: positive if the decision model is closer. $^{{*}}$ Significant at the "
              rf"{100 * LEVEL:g}\% level (two-sided paired bootstrap test, {DRAWS:,} draws, Holm-corrected over each "
              rf"panel's {len(dms) * len(llms)} tests). Each draw resamples the {people:,} personas with replacement, "
              rf"the same ones for humans and every model."
              + (f" Averaged over fewer questions, the ones both models have results for: {', '.join(short)}." if short else "")
              + "}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true", help=f"test only {', '.join(BEST)}")
    best = parser.parse_args().best
    output = BEST_OUTPUT if best else OUTPUT
    scored, people = collect()
    dms = [model for model in models.DECISION_MODELS if model in scored and (not best or model in BEST)]
    llms = [model for model in models.LLMS if model in scored and model not in EXCLUDED]
    pairs = [(dm, llm) for dm in dms for llm in llms]
    results, counts = {}, {}
    for metric in METRICS:
        observed, p_values = zip(*[test(scored[dm], scored[llm], metric) for dm, llm in pairs])
        tests = list(zip(observed, holm(p_values)))
        results |= {(metric, *pair): result for pair, result in zip(pairs, tests)}
        dm_wins = sum(p < LEVEL and observed > 0 for observed, p in tests)
        llm_wins = sum(p < LEVEL and observed < 0 for observed, p in tests)
        counts[metric] = dm_wins, llm_wins, len(tests) - dm_wins - llm_wins
    table = latex(scored, dms, llms, results, counts, people, best)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n")
    print(table)
    print(f"\nTests at the {LEVEL:.0%} level, {len(dms)} decision models x {len(llms)} LLMs:")
    for metric, (dm_wins, llm_wins, ties) in counts.items():
        print(f"  {metric.replace(chr(92), '')}: decision model closer {dm_wins}, LLM closer {llm_wins}, "
              f"no significant difference {ties}")
    print(f"\nSaved {output.name} to {output.parent}")


if __name__ == "__main__":
    main()
