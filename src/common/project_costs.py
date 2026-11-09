"""
Total time and cost of running chosen models on ABMs - basically the "so what" bit of the speed and cost seciton in the paper
Usage: python src/common/project_costs.py
"""

import pandas as pd

from common import models
from common.speed_cost import name, speed_cost

PLANNED = {  # total LLM calls, calls run at once, and decision model calls per LLM call (1 if not given)
    "Purchase Intent": {"calls": 18600, "concurrent": 10},
    "Occupational mobility": {"calls": 500000, "concurrent": 10, "dm_multiple": 3},
    "Rationing": {"calls": 500000, "concurrent": 10},
}
MODELS = ["pplx", "d1-ultra", "jev", "mammoth", "gpt", "haiku", "gpt-sol", "gemini-argon"]
AVERAGE = "Average"
MEASURES = {"cost": "Cost", "time": "Time"}
DIGITS = {AVERAGE: (3, 2)}  


def project():
    """
    A row per model in models.MODELS order, and the time and cost of a call on average and in total for each planned
    experiment.
    """
    means = speed_cost().reindex([model for model in models.MODELS if model in MODELS])
    seconds, dollars = means[("seconds", "mean")], means[("dollars", "mean")]
    table = {(AVERAGE, "cost"): dollars, (AVERAGE, "time"): seconds}
    dm = means.index.isin(models.DECISION_MODELS)
    for experiment, plan in PLANNED.items():
        calls = plan["calls"] * pd.Series(1, index=means.index).mask(dm, plan.get("dm_multiple", 1))
        table[(experiment, "cost")] = calls * dollars / 1000
        table[(experiment, "time")] = calls * seconds / plan["concurrent"] / 3600
    return pd.DataFrame(table)


def cell(value, digits):
    return "--" if pd.isna(value) else f"${value:,.{digits}f}$".replace(",", "{,}")


def latex(table):
    """
    A cost and time column under the average and each experiment; the decision models and the LLMs, each block under a
    rule and an italic label.
    """
    groups = [AVERAGE, *PLANNED]
    plans = []
    for experiment, plan in PLANNED.items():
        multiple = plan.get("dm_multiple", 1)
        per_dm = f" ({multiple} for each by a decision model)" if multiple != 1 else ""
        plans.append(f"{experiment}, {plan['calls']:,} calls{per_dm}, {plan['concurrent']} at once")
    plans = "; ".join(plans)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Projected total cost and time of each experiment}",
             r"\label{tab:project_costs}",
             rf"\begin{{tabular}}{{l{'r' * 2 * len(groups)}}}", r"\toprule",
             " & ".join(["", *[rf"\multicolumn{{2}}{{c}}{{{group}}}" for group in groups]]) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(groups))),
             " & ".join(["", *list(MEASURES.values()) * len(groups)]) + r" \\"]
    blocks = [("Frontier DM", [model for model in table.index if model in models.DECISION_MODELS]),
              ("LLM", [model for model in table.index if model in models.LLMS])]
    for label, block in [(label, block) for label, block in blocks if block]:
        lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * len(groups)}}}{{l}}{{\textit{{{label}}}}} \\"]
        for model in block:
            cells = [cell(table.loc[model, (group, measure)], digits)
                     for group in groups for measure, digits in zip(MEASURES, DIGITS.get(group, (2, 2)))]
            lines.append(" & ".join([name(model), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Average is the mean cost (\$ per 1,000 calls) and time (seconds "
              rf"per call) over the three experiments of Table~\ref{{tab:speed_cost}}. Each experiment's cost (\$) is "
              rf"its calls times the mean cost per call, and its time (hours) its calls times the mean seconds per "
              rf"call divided by the calls run at once. {plans}.}}",
              r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(latex(project()))
