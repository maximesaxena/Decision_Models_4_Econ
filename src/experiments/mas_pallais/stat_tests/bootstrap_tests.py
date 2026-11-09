"""
Bootstrap tests of every decision model against every LLM, averaged over Mas & Pallais's five treatments, on the
Wasserstein-1 distance between the model's fitted WTP distribution and the applicants', in $ per hour: the area
between their fitted choice curves without inattention. The applicants' fit is held fixed. In each draw, each
prompt's answers are redrawn with replacement from its own (its repeats), the model's logit is refitted and its
distance recomputed; a pair's difference is averaged over the treatments. Each test's two-sided p-value,
p = 2 * min(share of draws <= 0, share of draws >= 0), is Holm-corrected over the tests; a test is significant if its
corrected p-value is below LEVEL. Models with no fitted WTP in some treatment (inattention above MAX_ALPHA, or b <= 0)
are left out; draws in which a model has no fit are left out of its tests and counted.

Writes analysis_results/mas_pallais/bootstrap_tests.tex and prints how many tests the decision models and the LLMs
win. With --best, tests only the best decision models (common.plotting.GROUPS["Best DMs"]), so the Holm correction
is over fewer tests, and writes bootstrap_tests_best.tex.

Usage: python src/experiments/mas_pallais/stat_tests/bootstrap_tests.py [--best]
"""

import argparse

import numpy as np

from common import models
from common.plotting import GROUPS, NAMES
from experiments.mas_pallais import ANALYSIS_DIR
from experiments.mas_pallais.prompts import TREATMENTS
from experiments.mas_pallais.stat_tests.analysis import compare, fit_logit, inattention, model_results, quantiles, wtp
from experiments.mei_et_al.stat_tests.bootstrap_tests import holm

BEST = GROUPS["Frontier DMs"]
DRAWS = 10_000
LEVEL = 0.01
SEED = 0
PROMPT = ["gap", "max_wage", "a_first"]


def distance(gap, p_a, human):
    """
    The Wasserstein-1 distance between the WTP fitted to the calls, as in analysis.compare, and the applicants'
    (their quantiles); NaN if the calls have no fit.
    """
    alpha = inattention(gap, p_a)
    return np.abs(quantiles(wtp(fit_logit(gap, p_a, alpha), alpha)) - human).mean()


def redraws(sizes, rng):
    """
    DRAWS rows of indices into rows sorted into groups of these sizes, each row redrawn with replacement from its own
    group.
    """
    first = np.repeat(np.cumsum(sizes) - sizes, sizes)  # each row's group's first row
    return first + (rng.random((DRAWS, sizes.sum())) * np.repeat(sizes, sizes)).astype(int)


def resampled_distances(calls, human, rng):
    """
    The distance for the calls as they are, and for DRAWS resamples, each prompt's answers redrawn from its own. The
    fit depends only on the sum of answers at each gap, so each distinct set of sums is fitted once.
    """
    calls = calls.sort_values(PROMPT)  # each prompt's calls together
    p_a, gap = calls["p_a"].to_numpy(), calls["gap"].to_numpy(float)
    picks = redraws(calls.groupby(PROMPT).size().to_numpy(), rng)
    gaps, at_gap = np.unique(gap, return_inverse=True)
    sums = (p_a[picks] @ np.eye(len(gaps))[at_gap]).round(9)
    _, draw, which = np.unique(sums, axis=0, return_index=True, return_inverse=True)
    fitted = np.array([distance(gap, p_a[picks[i]], human) for i in draw])
    return distance(gap, p_a, human), fitted[which.reshape(-1)]


def collect():
    """
    {model: {treatment: (distance, distances in DRAWS resamples)}} for every model with results.
    """
    table, _ = compare()
    results = model_results().dropna(subset=["p_a"])
    rng = np.random.default_rng(SEED)
    scored = {}
    for treatment in TREATMENTS:
        human = quantiles(table[(table["treatment"] == treatment) & (table["source"] == "humans")].iloc[0])
        for model, calls in results[results["treatment"] == treatment].groupby("model", sort=False):
            scored.setdefault(model, {})[treatment] = resampled_distances(calls, human, rng)
    return scored


def test(dm, llm):
    """
    How much closer to the applicants the decision model is than the LLM, in $ per hour, averaged over the treatments
    (positive if the decision model is closer), its two-sided bootstrap p-value, and the draws left out for want of
    a fit.
    """
    observed = np.mean([llm[treatment][0] - dm[treatment][0] for treatment in TREATMENTS])
    draws = np.mean([llm[treatment][1] - dm[treatment][1] for treatment in TREATMENTS], axis=0)
    fitted = draws[~np.isnan(draws)]
    return observed, min(1.0, 2 * min((fitted <= 0).mean(), (fitted >= 0).mean())), len(draws) - len(fitted)


def name(model):
    return NAMES.get(model, model).strip()


def latex(dms, llms, results, counts, unfitted, best, humans_redrawn=False):
    """
    A row per LLM, a column per decision model, each cell how much closer the decision model is.
    """
    columns = len(dms) + 1
    dm_wins, llm_wins, ties = counts
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{Bootstrap tests of each {'of the best decision models' if best else 'decision model'} "
             rf"against each LLM, averaged over Mas \& Pallais's treatments{', redrawing the applicants' if humans_redrawn else ''}}}",
             rf"\label{{tab:mas_pallais_bootstrap{'_humans' if humans_redrawn else ''}{'_best' if best else ''}}}",
             rf"\begin{{tabular}}{{l{'r' * len(dms)}}}", r"\toprule",
             " & ".join(["LLM", *[name(dm) for dm in dms]]) + r" \\", r"\midrule"]
    for llm in llms:
        cells = []
        for dm in dms:
            observed, p, _ = results[dm, llm]
            cells.append(f"${observed:.2f}{'^{*}' if p < LEVEL else ''}$")
        lines.append(" & ".join([name(llm), *cells]) + r" \\")
    lines += [r"\midrule", rf"\multicolumn{{{columns}}}{{l}}{{\footnotesize Decision model closer: {dm_wins}; "
                           rf"LLM closer: {llm_wins}; no significant difference: {ties}.}} \\",
              r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to the applicants' fitted WTP "
              rf"distribution the column's decision model's is than the row's LLM's, in \$ per hour of Wasserstein-1 distance (the "
              rf"area between the fitted choice curves without inattention), averaged over the {len(TREATMENTS)} "
              rf"treatments with equal weights: positive if the decision model is closer. $^{{*}}$ Significant at the "
              rf"{100 * LEVEL:g}\% level (two-sided bootstrap test, {DRAWS:,} draws, Holm-corrected over the "
              rf"{len(dms) * len(llms)} tests). "
              + (r"In each draw, the applicants at each wage gap are redrawn with replacement and their logit is "
                 r"refitted; " if humans_redrawn else r"The applicants' fit is held fixed; in each draw, ")
              + r"each prompt's answers are redrawn with replacement from its own repeats and the model's logit is "
                r"refitted."
              + (r" Draws with no fitted WTP for the applicants (their inattention above 0.4) are left out."
                 if humans_redrawn else "")
              + (f" Left out, with no fitted WTP in some treatment: {', '.join(map(name, unfitted))}." if unfitted else "")
              + "}", r"\end{table}"]
    return "\n".join(lines)


def main(collect=collect, humans_redrawn=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", action="store_true", help=f"test only {', '.join(BEST)}")
    best = parser.parse_args().best
    output = ANALYSIS_DIR / f"bootstrap_tests{'_humans' if humans_redrawn else ''}{'_best' if best else ''}.tex"
    scored = collect()
    tested = [model for model in models.MODELS if model in scored and (not best or model in models.LLMS or model in BEST)]
    unfitted = [model for model in tested if any(np.isnan(scored[model][treatment][0]) for treatment in TREATMENTS)]
    dms = [model for model in tested if model in models.DECISION_MODELS and model not in unfitted]
    llms = [model for model in tested if model in models.LLMS and model not in unfitted]
    pairs = [(dm, llm) for dm in dms for llm in llms]
    observed, p_values, dropped = zip(*[test(scored[dm], scored[llm]) for dm, llm in pairs])
    results = {pair: result for pair, result in zip(pairs, zip(observed, holm(p_values), dropped))}
    dm_wins = sum(p < LEVEL and difference > 0 for difference, p, _ in results.values())
    llm_wins = sum(p < LEVEL and difference < 0 for difference, p, _ in results.values())
    counts = dm_wins, llm_wins, len(pairs) - dm_wins - llm_wins
    table = latex(dms, llms, results, counts, unfitted, best, humans_redrawn)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table + "\n")
    print(table)
    print(f"\nTests at the {LEVEL:.0%} level, {len(dms)} decision models x {len(llms)} LLMs: decision model closer "
          f"{dm_wins}, LLM closer {llm_wins}, no significant difference {counts[2]}")
    for model in [*dms, *llms]:
        missing = sum(np.isnan(scored[model][treatment][1]).sum() for treatment in TREATMENTS)
        if missing:
            print(f"  {name(model)}: {missing} resampled distances with no fit (the model's or the applicants'), "
                  f"left out of its tests")
    print(f"\nSaved {output.name} to {output.parent}")


if __name__ == "__main__":
    main()
