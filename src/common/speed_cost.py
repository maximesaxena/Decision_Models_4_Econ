"""
Mean seconds and mean dollars per 1,000 calls for every model in each experiment (SCE spending, Mei et al., Mas &
Pallais) and averaged over them, printed as a LaTeX table, and drawn to analysis_results/speed_cost.png as in
mei_et_al's compare_all_wasserstein.py right panel, each model at its mean over the experiments it was run in
(Qwen with thinking left out of the chart, not the table).

Usage: python src/common/speed_cost.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ANALYSIS_ROOT, RESULTS_ROOT, models
from common.plotting import NAMES, add_legend, figure_style, save
from experiments.mei_et_al.plotting.compare_all_2 import cost, draw_divide, plot_time_vs_cost

EXPERIMENTS = {"sce_spending": "SCE spending", "mei_et_al": "Mei et al.", "mas_pallais": "Mas \\& Pallais"}
CALL_FILES = {"sce_spending": "sce_spending/*.csv", "mei_et_al": "mei_et_al/*/calls/*.csv", "mas_pallais": "mas_pallais/*.csv"}
MEASURES = {"seconds": "Mean seconds per call", "dollars": "Mean \\$ per 1,000 calls"}
FIGURE = ANALYSIS_ROOT / "speed_cost.png"
NOT_PLOTTED = {"qwen"} 


def name(model):
    return NAMES.get(model, model).strip()


def load_calls():
    """
    Every call logged in every experiment, with its time (blank if Laya retried it internally) and cost.
    """
    calls = pd.concat([pd.read_csv(path, low_memory=False).assign(experiment=experiment)
                       for experiment, pattern in CALL_FILES.items() for path in sorted(RESULTS_ROOT.glob(pattern))],
                      ignore_index=True)
    retried = calls["internal_retries"].fillna(0) > 0
    return calls.assign(seconds=calls["seconds"].mask(retried), dollars=1000 * cost(calls))


def speed_cost():
    """
    A row per model in models.MODELS order, and for each measure a column per experiment and their mean (blank
    unless the model was run in every experiment).
    """
    means = load_calls().groupby(["model", "experiment"])[list(MEASURES)].mean().unstack("experiment")
    table = {}
    for measure in MEASURES:
        grid = means[measure].reindex(columns=list(EXPERIMENTS))
        grid["mean"] = grid.mean(axis=1, skipna=False)
        table[measure] = grid
    table = pd.concat(table, axis=1)
    return table.loc[[model for model in models.MODELS if model in table.index]]


def cell(value, digits):
    return "--" if pd.isna(value) else f"${value:.{digits}f}$"


def latex(table):
    """
    The decision models and the LLMs, each block under a rule.
    """
    columns = [*EXPERIMENTS, "mean"]
    headers = [*EXPERIMENTS.values(), "Mean"]
    unpriced = [name(model) for model in table.index if table.loc[model, "dollars"].isna().all()]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Mean time and cost per call in each experiment}",
             r"\label{tab:speed_cost}",
             rf"\begin{{tabular}}{{l{'r' * 2 * len(columns)}}}", r"\toprule",
             " & ".join(["", *[rf"\multicolumn{{{len(columns)}}}{{c}}{{{title}}}" for title in MEASURES.values()]]) + r" \\",
             rf"\cmidrule(lr){{2-{1 + len(columns)}}} \cmidrule(lr){{{2 + len(columns)}-{1 + 2 * len(columns)}}}",
             " & ".join(["", *headers, *headers]) + r" \\"]
    blocks = [[model for model in table.index if model in models.DECISION_MODELS],
              [model for model in table.index if model in models.LLMS]]
    for block in [block for block in blocks if block]:
        lines.append(r"\midrule")
        for model in block:
            row = table.loc[model]
            cells = [cell(row[("seconds", column)], 2) for column in columns] + [cell(row[("dollars", column)], 3) for column in columns]
            lines.append(" & ".join([name(model), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Seconds exclude model loading and a warm-up call. Cost is worked "
              rf"out from each call's tokens and the API's list prices; models run locally cost 0"
              + (f", and {' and '.join(unpriced)} have no price" if unpriced else "")
              + rf". Mean is the average over the {len(EXPERIMENTS)} experiments with equal weights, blank for a model "
              rf"not run in all of them.}}",
              r"\end{table}"]
    return "\n".join(lines)


def plot(table, path):
    """
    Mean seconds against mean dollars per 1,000 calls, averaged over the experiments each model was run in.
    """
    table = table.drop(index=NOT_PLOTTED, errors="ignore")
    summary = pd.DataFrame({"model": table.index,
                            "mean_seconds": table["seconds"][list(EXPERIMENTS)].mean(axis=1).to_numpy(),
                            "dollars_per_1000_calls": table["dollars"][list(EXPERIMENTS)].mean(axis=1).to_numpy()})
    with figure_style():
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=100)
        families = plot_time_vs_cost(ax, summary)
        ax.set_title("")
        right = 1.15 * summary.loc[summary["dollars_per_1000_calls"].notna(), "mean_seconds"].max()
        decade = 10.0 ** np.floor(np.log10(right))
        ax.set_xlim(ax.get_xlim()[0], min(tick for tick in (decade, 3 * decade, 10 * decade) if tick >= right))  # to a labelled tick
        fig.tight_layout()
        draw_divide(ax, families)  # after the layout, which moves the axes
        add_legend(fig, summary[summary["dollars_per_1000_calls"].notna()])
        save(fig, path, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


if __name__ == "__main__":
    table = speed_cost()
    print(latex(table))
    plot(table, FIGURE)
    print(f"\nSaved {FIGURE.name} to {ANALYSIS_ROOT}")
