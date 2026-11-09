"""
Draw analysis_results/mas_pallais/curves.png: for each treatment, the applicants' inattention-corrected fitted curve
and each model's probability of choosing A, with its own. Also writes wtp.csv:
the paper's, the applicants' and each model's fitted WTP.

Usage: python src/experiments/mas_pallais/plotting/curves.py
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np

from common import models
from common.plotting import AXIS_SIZE, GROUPS, INK, MUTED, NAMES, TICK_SIZE, TITLE_SIZE, figure_style, save, style
from experiments.mas_pallais import ANALYSIS_DIR
from experiments.mas_pallais.prompts import TREATMENTS
from experiments.mas_pallais.stat_tests.analysis import compare, curve

AXIS_SIZE, TICK_SIZE, TITLE_SIZE = 1.4 * AXIS_SIZE, 1.4 * TICK_SIZE, 1.4 * TITLE_SIZE  # larger, for this figure only


def plot(curves, path):
    """
    For each treatment, the applicants' inattention-corrected fitted CDF and each model's probability of choosing A
    with its own, a row per treatment, the frontier decision models in the left column and LLMs in the right, each
    column over its legend.
    """
    with figure_style():
        fig = plt.figure(figsize=(16, 5.5 + 3.6 * len(TREATMENTS)), layout="constrained")
        # a margin above the column titles, and a gap between the columns for their end ticks
        fig.get_layout_engine().set(rect=(0, 0, 1, 0.99), wspace=0.06)
        layout = fig.add_gridspec(len(TREATMENTS) + 1, 2, height_ratios=[1] * len(TREATMENTS) + [0.9])
        first = fig.add_subplot(layout[0, 0])
        axes = np.array([[fig.add_subplot(layout[r, c], sharex=first, sharey=first) if r or c else first
                          for c in range(2)] for r in range(len(TREATMENTS))])
        first.set_xlim(-5, 5)
        first.set_xticks([-5, -2.5, 0, 2.5, 5], ["−5", "−2.5", "0", "2.5", "5"])
        first.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        grid = np.linspace(-5, 5, 201)
        for c, (column, members, group) in enumerate([(axes[:, 0], GROUPS["Frontier DMs"], "Frontier DMs"),
                                                      (axes[:, 1], models.LLMS, "LLMs")]):
            for ax, treatment in zip(column, TREATMENTS):
                _, theta = curves[treatment, "humans"]
                ax.plot(grid, curve(theta, 0, grid), color=INK, linewidth=3, label=NAMES["humans"], zorder=3)
                for (t, model), (by_gap, theta) in curves.items():
                    if t == treatment and model in members:
                        color = style(model)[0]
                        ax.scatter(by_gap.index, by_gap, color=color, s=12, alpha=0.5)
                        ax.plot(grid, curve(theta, 0, grid), color=color, linewidth=2, label=NAMES.get(model, model))
                ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle="--")
                ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=TICK_SIZE, labelcolor=INK)
            for ax in column[:-1]:
                ax.tick_params(labelbottom=False)
            column[0].set_title(group, fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
            column[-1].set_xlabel("A's wage minus B's ($/hour)", fontsize=AXIS_SIZE, color=INK)
            # the column's legend underneath it, each model once though not every model is in every treatment
            entries = {label: handle for ax in column for handle, label in zip(*ax.get_legend_handles_labels())}
            key = fig.add_subplot(layout[-1, c])
            key.axis("off")
            legend = key.legend(entries.values(), entries.keys(), frameon=False, fontsize=TICK_SIZE, labelcolor=INK,
                                loc="upper center", ncol=2, handlelength=1.5, columnspacing=1.2)
            legend.set_in_layout(False)  # the key's row holds its space, so the columns stay even
        for ax in axes[:, 1]:
            ax.tick_params(labelleft=False)
        # each row's treatment on its left, and what A is in the empty bottom right corner of its left panel
        for ax, treatment in zip(axes[:, 0], TREATMENTS):
            a = "the M-F 9 am-5 pm job" if treatment == "employer_discretion" else "the alternative"
            ax.set_ylabel(textwrap.fill(treatment.replace("_", " ").capitalize(), 10), fontsize=AXIS_SIZE, color=INK)
            ax.text(0.97, 0.05, f"A = {a}", transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK_SIZE,
                    color=MUTED, fontstyle="italic")
        fig.supylabel("Share choosing A", fontsize=AXIS_SIZE, color=INK)
        save(fig, path)
        plt.close(fig)


if __name__ == "__main__":
    table, curves = compare()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS_DIR / "wtp.csv", index=False)
    print(table.round(2).to_string(index=False, na_rep="-"))
    plot(curves, ANALYSIS_DIR / "curves.png")
    print(f"\nSaved {ANALYSIS_DIR / 'wtp.csv'} and {ANALYSIS_DIR / 'curves.png'}")
