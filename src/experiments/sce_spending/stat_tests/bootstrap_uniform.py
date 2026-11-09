"""
Paired bootstrap tests of the best decision models (common.plotting.GROUPS["Best DMs"]) against the uniform
distribution, averaged over the two SCE questions, on the KL divergence from humans and the Turing test win rate
(higher is closer), as in bootstrap_tests.py and with its draws: each draw resamples the personas, the same ones for
humans and every model, and scores uniform against the draw's humans. Each test's two-sided p-value is Holm-corrected
over each metric's tests.

Writes analysis_results/sce_spending/bootstrap_uniform.tex.

Usage: python src/experiments/sce_spending/stat_tests/bootstrap_uniform.py
"""

import numpy as np

from experiments.sce_spending import ANALYSIS_DIR
from experiments.sce_spending.data import personas
from experiments.sce_spending.prompts import QUESTIONS
from experiments.sce_spending.stat_tests.analysis import ANSWERS
from experiments.sce_spending.stat_tests.bootstrap_tests import (BEST, DRAWS, LEVEL, METRICS, SEED, collect, holm, name,
                                                                 shares)

OUTPUT = ANALYSIS_DIR / "bootstrap_uniform.tex"


def uniform_scores():
    """
    {question: {metric: the uniform distribution's score as it is and for each draw}}.
    """
    people = personas()
    weights = np.random.default_rng(SEED).multinomial(len(people), np.full(len(people), 1 / len(people)), size=DRAWS)
    uniform = np.full(len(ANSWERS), 1 / len(ANSWERS))
    scored = {}
    for item in QUESTIONS:
        human = (people[item].astype(int).astype(str).to_numpy()[:, None] == np.array(ANSWERS)).astype(float)
        human_observed, human_draws = shares(human, weights)
        scored[item] = {metric: (score(human_observed, uniform), score(human_draws, np.tile(uniform, (DRAWS, 1))))
                        for metric, (score, *_) in METRICS.items()}
    return scored


def test(dm, uniform, metric):
    """
    How much closer to humans the decision model is than uniform, averaged over its questions (positive if the decision
    model is closer), and its two-sided bootstrap p-value.
    """
    sign = METRICS[metric][1]
    observed = sign * np.mean([dm[item][metric][0] - uniform[item][metric][0] for item in dm])
    draws = sign * np.mean([dm[item][metric][1] - uniform[item][metric][1] for item in dm], axis=0)
    return observed, min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))


def latex(scored, dms, results, people):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Paired bootstrap tests of the best decision models against the uniform distribution, averaged "
             r"over the two SCE spending questions}", r"\label{tab:sce_bootstrap_uniform}", r"\begin{tabular}{lrr}",
             r"\toprule", " & ".join(["Decision model", *[metric.replace("Turing test win rate", "Turing test")
                                                          for metric in METRICS]]) + r" \\", r"\midrule"]
    for dm in dms:
        cells = [f"${results[metric, dm][0]:.{METRICS[metric][2]}f}{'^{*}' if results[metric, dm][1] < LEVEL else ''}$"
                 for metric in METRICS]
        lines.append(" & ".join([name(dm), *cells]) + r" \\")
    short = [f"{name(dm)} ({len(scored[dm])})" for dm in dms if len(scored[dm]) < len(QUESTIONS)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to humans the decision model is than "
              rf"the uniform distribution, averaged over the {len(QUESTIONS)} questions with equal weights: positive if "
              rf"the decision model is closer. $^{{*}}$ Significant at the {100 * LEVEL:g}\% level (two-sided paired "
              rf"bootstrap test, {DRAWS:,} draws, Holm-corrected over each column's {len(dms)} tests). Each draw "
              rf"resamples the {people:,} personas with replacement, the same ones for humans and every model."
              + (f" Averaged over fewer questions, the ones the model has results for: {', '.join(short)}." if short else "")
              + "}", r"\end{table}"]
    return "\n".join(lines)


def main():
    scored, people = collect()
    uniform = uniform_scores()
    dms = [dm for dm in BEST if dm in scored]
    results = {}
    for metric in METRICS:
        observed, p_values = zip(*[test(scored[dm], uniform, metric) for dm in dms])
        results |= {(metric, dm): result for dm, result in zip(dms, zip(observed, holm(p_values)))}
    table = latex(scored, dms, results, people)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table + "\n")
    print(table)
    print(f"\nSaved {OUTPUT.name} to {OUTPUT.parent}")


if __name__ == "__main__":
    main()
