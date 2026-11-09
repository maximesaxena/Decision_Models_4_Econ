"""
Bootstrap tests of the best decision models (common.plotting.GROUPS["Best DMs"]), each way of asking, against the
uniform distribution, averaged over Mei et al.'s games, on the Wasserstein-1 distance from humans and the Turing test
win rate (higher is closer), as in bootstrap_tests.py and with its draws. Humans' and the uniform distributions are
fixed, so only the decision model's calls are resampled. Each test's two-sided p-value is Holm-corrected over each
metric's tests.

Writes analysis_results/mei_et_al/bootstrap_uniform.tex.

Usage: python src/experiments/mei_et_al/stat_tests/bootstrap_uniform.py
"""

import numpy as np

from common.plotting import GROUPS
from experiments.mei_et_al import ANALYSIS_DIR
from experiments.mei_et_al.registry import EXPERIMENTS
from experiments.mei_et_al.stat_tests.bootstrap_tests import DRAWS, LEVEL, METRICS, collect, holm, name, ordered

OUTPUT = ANALYSIS_DIR / "bootstrap_uniform.tex"


def uniform_scores():
    """
    {game: {metric: the uniform distribution's score}}.
    """
    scored = {}
    for experiment in EXPERIMENTS:
        humans = experiment.human_choices()
        for game in experiment.GAMES:
            human_share = np.histogram(humans[game.name].astype(float), game.bin_edges())[0]
            human_share = human_share / human_share.sum()
            uniform = np.full(len(human_share), 1 / len(human_share))
            scored[game.name] = {metric: score(game, human_share, uniform) for metric, (score, _) in METRICS.items()}
    return scored


def test(dm, uniform, metric):
    """
    How much closer to humans the decision model is than uniform, averaged over its games (positive if the decision
    model is closer), and its two-sided bootstrap p-value.
    """
    sign = METRICS[metric][1]
    observed = sign * np.mean([dm[game][metric][0] - uniform[game][metric] for game in dm])
    draws = sign * np.mean([dm[game][metric][1] - uniform[game][metric] for game in dm], axis=0)
    return observed, min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))


def latex(scored, dms, results, games):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Bootstrap tests of the best decision models against the uniform distribution, averaged over "
             r"Mei et al.'s games}", r"\label{tab:mei_bootstrap_uniform}", r"\begin{tabular}{lrr}", r"\toprule",
             " & ".join(["Decision model", *[metric.replace("Turing test win rate", "Turing test") for metric in METRICS]])
             + r" \\", r"\midrule"]
    for dm in dms:
        cells = [f"${results[metric, dm][0]:.1f}{'^{*}' if results[metric, dm][1] < LEVEL else ''}$"
                 for metric in METRICS]
        lines.append(" & ".join([name(dm, scored), *cells]) + r" \\")
    short = [f"{name(dm, scored)} ({len(scored[dm])})" for dm in dms if len(scored[dm]) < games]
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to humans the decision model is than "
              rf"the uniform distribution, averaged over the {games} games with equal weights: positive if the decision "
              rf"model is closer, in percentage points. $^{{*}}$ Significant at the {100 * LEVEL:g}\% level (two-sided bootstrap test, "
              rf"{DRAWS:,} draws, Holm-corrected over each column's {len(dms)} tests). The human and uniform "
              rf"distributions are fixed; each decision model's calls are resampled with replacement in each game."
              + (f" Averaged over fewer games, the ones the model has results for: {', '.join(short)}." if short else "")
              + "}", r"\end{table}"]
    return "\n".join(lines)


def main():
    scored, games = collect()
    uniform = uniform_scores()
    dms = ordered(scored, GROUPS["Frontier DMs"])
    results = {}
    for metric in METRICS:
        observed, p_values = zip(*[test(scored[dm], uniform, metric) for dm in dms])
        results |= {(metric, dm): result for dm, result in zip(dms, zip(observed, holm(p_values)))}
    table = latex(scored, dms, results, games)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table + "\n")
    print(table)
    print(f"\nSaved {OUTPUT.name} to {OUTPUT.parent}")


if __name__ == "__main__":
    main()
