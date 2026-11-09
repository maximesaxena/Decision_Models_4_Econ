"""
Compare each robustness variant with the baseline, for the frontier decision models and the three LLMs, using the
baseline's analysis (stat_tests) on each condition's results. Writes to analysis_results/sce_spending/robustness/:
- comparison.csv: each condition's Turing test and fit per question and model, and the share of personas whose most
  likely answer is the baseline's
- vs_baseline.tex: how much closer to humans each model is under each variant than in the baseline, averaged over the
  variant's questions, with paired bootstrap tests Holm-corrected over each metric's tests
- headline.tex: within each condition, how many decision model vs LLM tests each side wins; nosystem's decision
  models are the baseline's
- conditions.png (and .pdf): Turing test against KL divergence, as in turing_vs_fit.png, with each checked model's
  point under each variant and the other models faded

Usage: python src/experiments/sce_spending/robustness/analysis.py
"""

import contextlib
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.text import Text

from common.plotting import (AXIS_SIZE, GROUPS, HOLLOW_EDGE, INK, MUTED, NAMES, REFERENCE_STYLES, REFERENCES, TITLE_SIZE,
                             add_legend, figure_style, grouped_points, model_marker, place_labels, save, style_axes)
from experiments.mei_et_al.stat_tests.bootstrap_tests import holm
from experiments.sce_spending import ANALYSIS_DIR as BASELINE_ANALYSIS_DIR, RESULTS_DIR as BASELINE_DIR
from experiments.sce_spending.robustness import ANALYSIS_DIR, DMS, LLMS, RESULTS_DIR
from experiments.sce_spending.robustness.variants import VARIANTS
from experiments.sce_spending.stat_tests import analysis
from experiments.sce_spending.stat_tests.bootstrap_tests import DRAWS, LEVEL, METRICS, collect, name, test

CONDITIONS = {"baseline": BASELINE_DIR, **{variant: RESULTS_DIR / variant for variant in VARIANTS}}
MODELS = [*DMS, *LLMS]
# each variant's marker on conditions.png; the baseline has each model's usual one
CONDITION_MARKERS = {"human": "^", "nosystem": "D", "wording": "P"}
FADED = 0.15  # opacity of the models without robustness checks on conditions.png
POSITIONS = BASELINE_ANALYSIS_DIR / "label_positions.json"  # turing_vs_fit.png's label offsets, read only


class Header(str):
    """
    A legend column's header, drawn where the column's markers are.
    """


class HandleHeader(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [Text(width / 2 - xdescent, height / 2 - ydescent, orig_handle, ha="center", va="center",
                     fontsize=fontsize, color=MUTED, fontweight="bold", transform=trans)]


@contextlib.contextmanager
def results_from(directory):
    """
    stat_tests.analysis (and so bootstrap_tests) reading its results from directory.
    """
    saved = analysis.RESULTS_DIR
    analysis.RESULTS_DIR = directory
    try:
        yield
    finally:
        analysis.RESULTS_DIR = saved


def load():
    """
    {condition: (comparison, per-persona probabilities, bootstrap scores)} for each condition with results, MODELS only.
    """
    conditions = {}
    for condition, directory in CONDITIONS.items():
        if not any(directory.glob("*.csv")):
            continue
        with results_from(directory):
            comparison = analysis.compare()
            by_persona, _ = analysis.model_results()
            scored, _ = collect()
        conditions[condition] = (comparison[comparison["model"].isin(MODELS)], by_persona[by_persona["model"].isin(MODELS)],
                                 {model: scores for model, scores in scored.items() if model in MODELS})
    return conditions


def same_answer(by_persona, baseline):
    """
    Per model and question, the % of personas (with a valid answer in both) whose most likely answer is the baseline's.
    """
    merged = by_persona.merge(baseline, on=["model", "userid", "date", "item"], suffixes=("", "_baseline"))
    same = merged[analysis.P].to_numpy().argmax(axis=1) == merged[[f"{p}_baseline" for p in analysis.P]].to_numpy().argmax(axis=1)
    return 100 * merged.assign(same=same).groupby(["model", "item"])["same"].mean().rename("same_answer_%")


def holm_tests(pairs):
    """
    {metric: [(observed, Holm-corrected p)]}: how much closer to humans a is than b, for each pair of scores (a, b).
    """
    results = {}
    for metric in METRICS:
        observed, p_values = zip(*[test(a, b, metric) for a, b in pairs])
        results[metric] = list(zip(observed, holm(p_values)))
    return results


def vs_baseline(conditions):
    """
    The LaTeX table of each model's change from the baseline under each variant.
    """
    scored = {condition: scores for condition, (_, _, scores) in conditions.items()}
    variants = [variant for variant in VARIANTS if variant in scored]
    pairs = [(model, variant) for variant in variants for model in VARIANTS[variant][1]
             if model in scored[variant] and model in scored["baseline"]]
    tests = holm_tests([(scored[variant][model], scored["baseline"][model]) for model, variant in pairs])
    results = {(metric, *pair): result for metric in METRICS for pair, result in zip(pairs, tests[metric])}
    columns = len(variants) + 1
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{How much closer to humans each model is under each robustness variant than in the baseline, "
             r"SCE spending questions}", r"\label{tab:sce_robustness}", rf"\begin{{tabular}}{{l{'r' * len(variants)}}}",
             r"\toprule", " & ".join(["Model", *variants]) + r" \\"]
    for panel, metric in zip("AB", METRICS):
        decimals = METRICS[metric][2]
        lines += [r"\midrule", rf"\multicolumn{{{columns}}}{{l}}{{\textit{{Panel {panel}: {metric}}}}} \\", r"\midrule"]
        for model in MODELS:
            cells = []
            for variant in variants:
                observed, p = results.get((metric, model, variant), (None, None))
                cells.append("--" if observed is None else f"${observed + 0:.{decimals}f}{'^{*}' if p < LEVEL else ''}$")  # + 0: no -0
            lines.append(" & ".join([name(model), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Each cell is how much closer to humans the model is under the "
              rf"variant than in the baseline, in bits (Panel A) or percentage points (Panel B): positive if the variant "
              rf"is closer. human: ``Act like a human.'' starts every prompt, and the LLMs have no system prompt. nosystem: the LLMs without their system "
              rf"prompt. wording: QSP13new only, reworded to mirror QSP12n, compared with the baseline's QSP13new. "
              rf"$^{{*}}$ Significant at the {100 * LEVEL:g}\% level (two-sided paired bootstrap test, {DRAWS:,} draws "
              rf"of the personas, Holm-corrected over each panel's {len(pairs)} tests).}}", r"\end{table}"]
    return "\n".join(lines)


def headline(conditions):
    """
    The LaTeX table of decision model vs LLM tests won by each side within each condition, and the counts.
    """
    scored = {condition: scores for condition, (_, _, scores) in conditions.items()}
    counts = {}
    for condition, scores in scored.items():
        runs_dms = condition == "baseline" or set(DMS) <= set(VARIANTS[condition][1])
        dms = scores if runs_dms else scored["baseline"]
        pairs = [(dm, llm) for dm in DMS if dm in dms for llm in LLMS if llm in scores]
        if not pairs:
            continue
        for metric, tests in holm_tests([(dms[dm], scores[llm]) for dm, llm in pairs]).items():
            dm_wins = sum(p < LEVEL and observed > 0 for observed, p in tests)
            llm_wins = sum(p < LEVEL and observed < 0 for observed, p in tests)
            counts[condition, metric] = dm_wins, llm_wins, len(tests) - dm_wins - llm_wins
    conditions_run = list(dict.fromkeys(condition for condition, _ in counts))
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Frontier decision models against LLMs within each robustness condition, SCE spending questions}",
             r"\label{tab:sce_robustness_headline}", r"\begin{tabular}{l" + "rrr" * len(METRICS) + "}", r"\toprule",
             " & ".join(["", *[rf"\multicolumn{{3}}{{c}}{{{metric}}}" for metric in METRICS]]) + r" \\",
             " & ".join(["Condition", *["DM closer", "LLM closer", "No difference"] * len(METRICS)]) + r" \\", r"\midrule"]
    for condition in conditions_run:
        lines.append(" & ".join([condition, *[str(n) for metric in METRICS for n in counts[condition, metric]]]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Counts of paired bootstrap tests of each of {', '.join(map(name, DMS))} "
              rf"against each of {', '.join(map(name, LLMS))}, significant at the {100 * LEVEL:g}\% level (two-sided, "
              rf"{DRAWS:,} draws, Holm-corrected within each condition and metric). nosystem's decision models are the "
              rf"baseline's; wording is QSP13new only.}}", r"\end{table}"]
    return "\n".join(lines), counts


def plot(comparison):
    """
    conditions.png: Turing test against KL divergence, averaged over the two questions (a variant that changes only
    one question keeps the baseline's other), as in turing_vs_fit.png: every baseline model, those without
    robustness checks faded, and each checked model's point under each variant, marked by the variant. The grey
    outlines also take in the variants' points.
    """
    with results_from(BASELINE_DIR):
        summary = analysis.summarise(analysis.compare())
    items = comparison.loc[comparison["condition"] == "baseline", "item"].unique()
    full = []
    for condition, table in comparison[comparison["condition"] != "baseline"].groupby("condition", sort=False):
        missing = comparison[(comparison["condition"] == "baseline") & ~comparison["item"].isin(table["item"])
                             & comparison["model"].isin(table["model"])]
        full.append(pd.concat([table, missing.assign(condition=condition)]))
    full = pd.concat(full)
    full = full[full.groupby(["condition", "model"])["item"].transform("nunique") == len(items)]
    variants = full.groupby(["condition", "model"], as_index=False)[["win_%", "kl_bits"]].mean()
    with figure_style():
        fig, ax = plt.subplots(figsize=(9, 7.5))
        for model, win, kl in summary[["model", "win_%", "kl_bits"]].itertuples(index=False):
            if model in REFERENCES:
                color, marker = REFERENCE_STYLES[model]
                ax.scatter(win, kl, color=color, marker=marker, s=220, zorder=3)
                continue
            marker, face, edge, width = model_marker(model)
            checked = model in MODELS
            ax.scatter(win, kl, marker=marker, s=150 if checked else 110, zorder=4 if checked else 2, linewidths=width, facecolors=face,
                       edgecolors=edge, alpha=1 if checked else FADED)
        for condition, model, win, kl in variants[["condition", "model", "win_%", "kl_bits"]].itertuples(index=False):
            _, face, edge, width = model_marker(model)
            ax.scatter(win, kl, marker=CONDITION_MARKERS[condition], s=170, zorder=4, linewidths=width,
                       facecolors=face, edgecolors=edge)
        ax.set_xlabel("Turing test: comparisons won (%)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
        ax.set_ylabel("KL divergence from humans", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
        ax.set_title("Turing test vs KL divergence, by condition", fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
        style_axes(ax)
        fig.tight_layout()
        groups = grouped_points(summary, "win_%", "kl_bits")
        for group, members in GROUPS.items():
            extra = variants.loc[variants["model"].isin(members), ["win_%", "kl_bits"]].itertuples(index=False)
            groups[group] = groups.get(group, []) + [tuple(point) for point in extra]
        saved = json.loads(POSITIONS.read_text())["human_likeness"] if POSITIONS.exists() else {}
        place_labels(ax, groups, saved)  # after the layout, which moves the axes
        # one column of filled decision model markers and one of hollow LLM markers, blank where a condition has none
        shown = [condition for condition in CONDITION_MARKERS if condition in set(variants["condition"])]
        markers = {"baseline": ("o", "s"), **{condition: (CONDITION_MARKERS[condition],) * 2 for condition in shown}}
        has = {"baseline": (True, True), **{condition: (variants.loc[variants["condition"] == condition, "model"].isin(DMS).any(),
                                                        variants.loc[variants["condition"] == condition, "model"].isin(LLMS).any())
                                            for condition in shown}}
        dm_column = [Line2D([], [], linestyle="none", marker=markers[condition][0] if has[condition][0] else None,
                            markersize=12, color=MUTED) for condition in markers]
        llm_column = [Line2D([], [], linestyle="none", marker=markers[condition][1] if has[condition][1] else None,
                             markersize=12, markerfacecolor="white", markeredgecolor=MUTED,
                             markeredgewidth=HOLLOW_EDGE) for condition in markers]
        ax.legend(handles=[Header("DMs"), *dm_column, Header("LLMs"), *llm_column],
                  labels=[""] * (len(markers) + 2) + list(markers), ncol=2, columnspacing=0.6,
                  handler_map={Header: HandleHeader()}, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=AXIS_SIZE,
                  labelcolor=INK, frameon=False)
        add_legend(fig, summary)
        faded = {NAMES.get(model, model) for model in summary["model"] if model not in MODELS and model not in REFERENCES}
        for legend in fig.legends:  # fade the same models in the legend
            for handle, text in zip(legend.legend_handles, legend.get_texts()):
                if text.get_text() in faded:
                    handle.set_alpha(FADED)
                    text.set_alpha(FADED + 0.25)
        save(fig, ANALYSIS_DIR / "conditions.png", bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


def main():
    conditions = load()
    assert "baseline" in conditions and len(conditions) > 1, "no robustness results yet"
    _, baseline, _ = conditions["baseline"]
    tables = []
    for condition, (comparison, by_persona, _) in conditions.items():
        table = comparison.assign(condition=condition)
        if condition != "baseline":
            table = table.merge(same_answer(by_persona, baseline).reset_index(), on=["model", "item"], how="left")
        tables.append(table)
    comparison = pd.concat(tables, ignore_index=True)
    comparison = comparison[["condition", *[column for column in comparison.columns if column != "condition"]]]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(ANALYSIS_DIR / "comparison.csv", index=False)

    pd.set_option("display.width", 200)
    order = {model: i for i, model in enumerate(MODELS)}
    for column in ["kl_bits", "win_%", "valid_answer_pct", "same_answer_%"]:
        wide = comparison.pivot_table(index=["item", "model"], columns="condition", values=column)
        wide = wide.reindex(columns=[c for c in conditions if c in wide.columns])
        wide = wide.iloc[sorted(range(len(wide)), key=lambda i: (wide.index[i][0], order[wide.index[i][1]]))]
        print(f"\n{column}\n{wide.round(2).to_string(na_rep='-')}")

    table = vs_baseline(conditions)
    (ANALYSIS_DIR / "vs_baseline.tex").write_text(table + "\n")
    print(f"\n{table}")
    table, counts = headline(conditions)
    (ANALYSIS_DIR / "headline.tex").write_text(table + "\n")
    print(f"\n{table}\n\nDecision models vs LLMs at the {LEVEL:.0%} level (DM closer, LLM closer, no difference):")
    for (condition, metric), (dm_wins, llm_wins, ties) in counts.items():
        print(f"  {condition}, {metric.replace(chr(92), '')}: {dm_wins}, {llm_wins}, {ties}")
    plot(comparison)
    print(f"\nSaved comparison.csv, vs_baseline.tex, headline.tex and conditions.png to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
