"""
Leave-one-out robustness check of bootstrap_tests.py --best: its tests of the best decision models
(common.plotting.GROUPS["Frontier DMs"]), asked for a choice and a score, against every LLM, rerun once per game with
that game left out of each pair's average. Each game's bootstrap draws are computed once, as in bootstrap_tests.py, so
leaving a game out only changes which games are averaged; the Holm correction is redone over each metric's tests.

Writes analysis_results/mei_et_al/leave_one_out.tex, a row per game left out and, for each metric, how many tests the
decision model wins, neither wins and the LLM wins.

Usage: python src/experiments/mei_et_al/stat_tests/leave_one_out.py
"""

from common import models
from experiments.mei_et_al import ANALYSIS_DIR
from experiments.mei_et_al.stat_tests.bootstrap_tests import (BEST, DRAWS, EXCLUDED, LEVEL, METRICS, WAYS, collect,
                                                              holm, ordered, test, wins)

OUTPUT = ANALYSIS_DIR / "leave_one_out.tex"


def counts(scored, dms, llms, left_out=None):
    """
    For each metric, how many tests between dms and llms the decision model wins, neither wins and the LLM wins, with
    the game left_out dropped from every model's average.
    """
    kept = {key: {game: s for game, s in per_game.items() if game != left_out} for key, per_game in scored.items()}
    pairs = [(dm, llm) for dm in dms for llm in llms]
    result = {}
    for metric in METRICS:
        observed, p_values = zip(*[test(kept[dm], kept[llm], metric) for dm, llm in pairs])
        results = {(metric, *pair): test_result for pair, test_result in zip(pairs, zip(observed, holm(p_values)))}
        dm_wins, llm_wins, ties = wins(results, metric, dms, llms)
        result[metric] = dm_wins, ties, llm_wins
    return result


def latex(rows, full, tests):
    """
    A row per game left out, three columns per metric.
    """
    columns = 1 + 3 * len(METRICS)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Leave-one-out tests of the best decision models against each LLM, averaged over Mei et al.'s "
             r"games but one}", r"\label{tab:mei_leave_one_out}", rf"\begin{{tabular}}{{l{'rrr' * len(METRICS)}}}",
             r"\toprule",
             " & ".join(["", *[rf"\multicolumn{{3}}{{c}}{{{metric}}}" for metric in METRICS]]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(METRICS))),
             " & ".join(["Game left out", *["DM", "Neither", "LLM"] * len(METRICS)]) + r" \\", r"\midrule"]
    for game, result in rows.items():
        lines.append(" & ".join([game.replace("_", " ").capitalize(),
                                 *[str(n) for metric in METRICS for n in result[metric]]]) + r" \\")
    with_all = "; ".join(f"{metric}: {', '.join(map(str, full[metric]))}" for metric in METRICS)
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each row reruns the bootstrap tests of the {len(BEST)} best "
              rf"decision models, asked for a choice and a score, against each LLM with one game left out of the "
              rf"average: the number of tests in which the decision model is significantly closer to humans (DM), "
              rf"neither is (Neither) and the LLM is (LLM), at the {100 * LEVEL:g}\% level (two-sided bootstrap test, "
              rf"{DRAWS:,} draws, Holm-corrected over each metric's {tests} tests). With all games: {with_all}.}}",
              r"\end{table}"]
    return "\n".join(lines)


def main():
    scored, _ = collect()
    games = list(dict.fromkeys(game for per_game in scored.values() for game in per_game))
    dms = [key for key in ordered(scored, models.DECISION_MODELS) if key[0] in BEST and key[1] in WAYS]
    llms = [key for key in ordered(scored, models.LLMS) if key[0] not in EXCLUDED]
    rows = {game: counts(scored, dms, llms, game) for game in games}
    table = latex(rows, counts(scored, dms, llms), len(dms) * len(llms))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table + "\n")
    print(table)
    print(f"\nSaved {OUTPUT.name} to {OUTPUT.parent}")


if __name__ == "__main__":
    main()
