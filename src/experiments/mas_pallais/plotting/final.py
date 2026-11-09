
"""
The paper's Mas & Pallais figure. Left: the work from home treatment's fitted, inattention-corrected
curves from the logit fitted in stat_tests/analysis.py, the decision models' above and the LLMs' below, each panel with
the applicants'. Right: the Wasserstein-1
(earth mover's) distance, in $ per hour, between each model's fitted WTP distribution and the applicants',
per treatment: the average gap between the two distributions' quantiles. It is at least the gap between the
means, and equal to it unless the distributions cross, so whatever exceeds the mean gap comes from a
different spread.

Writes analysis_results/mas_pallais/final.png.

Usage: python src/experiments/mas_pallais/plotting/final.py
"""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.transforms import ScaledTranslation

from common import models
from common.plotting import (AXIS_SIZE, GRID, GROUPS, INK, MUTED, NAMES, TICK_SIZE, TITLE_SIZE, figure_style, save, style,
                             style_axes)
from experiments.mas_pallais import ANALYSIS_DIR
from experiments.mas_pallais.stat_tests.analysis import (compare, curve, distances, fit_logit, human_choices, inattention,
                                                       model_results, wtp)
from experiments.mas_pallais.prompts import TREATMENTS

FIGURE = ANALYSIS_DIR / "mas.png"
AXIS_SIZE, TITLE_SIZE = 1.25 * AXIS_SIZE, 1.25 * TITLE_SIZE  # larger than the shared sizes, for this figure only
CURVE_TREATMENT = "work_from_home"  # the treatment whose fitted curves are shown on the left
GAPS = np.linspace(-5, 5, 201)
FADED_ALPHA = 0.3  # the non-frontier decision models' curves


def fits(treatment):
    """
    The applicants' and each model's fitted (b, c) in the treatment, as in analysis.compare, for the models
    with a WTP (inattention at most analysis.MAX_ALPHA and b > 0), in models.MODELS order.
    """
    humans = human_choices()
    humans = humans[humans["treatment"] == treatment]
    gap, chose_a = humans["gap"].to_numpy(), humans["chose_a"].to_numpy()
    thetas = {"humans": fit_logit(gap, chose_a, inattention(gap, chose_a))}
    results = model_results()
    results = results[results["treatment"] == treatment].dropna(subset=["p_a"])
    for model in [model for model in models.MODELS if model in set(results["model"])]:
        r = results[results["model"] == model]
        gap, p_a = r["gap"].to_numpy(), r["p_a"].to_numpy()
        alpha = inattention(gap, p_a)
        theta = fit_logit(gap, p_a, alpha)
        if pd.notna(wtp(theta, alpha)["mean"]):
            thetas[model] = theta
        else:
            print(f"Left out of the curves, with no WTP: {model}")
    return thetas


def plot_curves(ax, key, thetas, members, group):
    """
    The fitted curves without inattention of the models in members, titled with their group, the applicants' in
    black, the legend in key, the empty panel underneath.
    """
    ax.plot(GAPS, curve(thetas["humans"], 0, GAPS), color=INK, linewidth=3, label="Humans", zorder=3)
    for source in [source for source in thetas if source in members]:
        color, _, linestyle = style(source)
        linestyle = (0, (6, 3)) if linestyle == "--" else linestyle  # LLMs' long dashes, to show on steep curves
        faded = source in models.DECISION_MODELS and source not in GROUPS["Frontier DMs"]  # the frontier ones stand out
        ax.plot(GAPS, curve(thetas[source], 0, GAPS), color=color, linestyle=linestyle, linewidth=2,
                alpha=FADED_ALPHA if faded else 1, label=NAMES.get(source, source))
    ax.axhline(0.5, color=MUTED, linewidth=1.2, alpha=0.6, zorder=1)
    ax.axvline(0, color=MUTED, linewidth=1.2, alpha=0.6, zorder=1)
    ax.set_xlabel("WFH wage minus standard wage ($ per hour)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_title(f"{CURVE_TREATMENT.replace('_', ' ').capitalize()}: {group}", fontsize=TITLE_SIZE, color=INK,
                 fontweight="bold", pad=12)
    style_axes(ax)
    legend_rows(key, *ax.get_legend_handles_labels())


def legend_rows(key, handles, labels, per_row=4):
    """
    The legend in the empty panel key, from its top down, per_row entries a row, each row its own legend so the
    entries in it are evenly spaced, and centred.
    """
    key.axis("off")
    renderer = key.figure.canvas.get_renderer()
    drop = 0  # inches below the top
    for start in range(0, len(handles), per_row):
        anchor = key.transAxes + ScaledTranslation(0, -drop, key.figure.dpi_scale_trans)
        legend = Legend(key, handles[start:start + per_row], labels[start:start + per_row], frameon=False,
                        fontsize=TICK_SIZE, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, 1),
                        bbox_transform=anchor, ncol=per_row, handlelength=1.2, handletextpad=0.5,
                        columnspacing=1.5, borderpad=0.2)
        legend.set_in_layout(False)  # key already holds its space
        key.add_artist(legend)
        drop += legend.get_window_extent(renderer).height / key.figure.dpi


def plot_distances(fig, ax, dist):
    """
    The distance per model and treatment, and averaged, models from closest to furthest on average, leaving out
    the models with no fit in any treatment.
    """
    grid = dist.pivot(index="model", columns="treatment", values="wasserstein")[list(TREATMENTS)].dropna(how="all")
    grid["average"] = grid.mean(axis=1, skipna=False)
    grid = grid.sort_values("average", na_position="last")[["average", *TREATMENTS]]  # the average first
    image = ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0)
    for (i, j), value in np.ndenumerate(grid.to_numpy()):
        text = "no fit" if np.isnan(value) else f"{value:.2f}"
        dark = not np.isnan(value) and value > 0.6 * np.nanmax(grid.to_numpy())
        ax.text(j, i, text, ha="center", va="center", fontsize=1.1 * TICK_SIZE, color="white" if dark else INK)
    ax.axvline(0.5, color="white", linewidth=4)
    # outline the average over the models with one, which sort first
    ax.add_patch(Rectangle((-0.5, -0.5), 1, grid["average"].notna().sum(), fill=False,
                           edgecolor=INK, linewidth=4, clip_on=False, zorder=3))
    ax.set_xticks(range(grid.shape[1]), [textwrap.fill(column.replace("_", " "), 10) for column in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [textwrap.fill(NAMES.get(model, model), 10) for model in grid.index])
    ax.tick_params(length=0, pad=10, labelsize=TICK_SIZE, labelcolor=INK)  # padded clear of the boxes
    ax.get_xticklabels()[0].set_fontweight("bold")  # the average's
    # the decision models' names boxed, to tell them from the LLMs', the frontier ones shaded and bold
    for model, label in zip(grid.index, ax.get_yticklabels()):
        if model in models.DECISION_MODELS:
            frontier = model in GROUPS["Frontier DMs"]
            label.set_bbox(dict(boxstyle="round,pad=0.3", facecolor=GRID if frontier else "none", edgecolor=INK,
                                linewidth=1.5))
            label.set_fontweight("bold" if frontier else "normal")
    ax.set_xlabel("Boxed names are decision models, shaded ones frontier", fontsize=TICK_SIZE, color=INK,
                  fontstyle="italic", labelpad=16)  # the note
    ax.spines[:].set_visible(False)
    ax.set_title("Wasserstein-1 distance", fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.ax.invert_yaxis()  # light at the top, as the closest models are
    #bar.set_label("$ per hour", fontsize=TICK_SIZE, color=INK)
    bar.ax.tick_params(labelsize=TICK_SIZE, colors=MUTED, labelcolor=INK)
    bar.outline.set_visible(False)
    for y, va, text in [(1.02, "bottom", "Better\n↑"), (-0.02, "top", "↓\nWorse")]:  # the bar's ends
        bar.ax.text(0.5, y, text, transform=bar.ax.transAxes, ha="center", va=va, fontsize=TICK_SIZE, color=INK,
                    fontweight="bold", linespacing=1.1)


def plot(thetas, dist, path):
    with figure_style():
        fig = plt.figure(figsize=(20, 15), layout="constrained")  # spaces each row by what it needs
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 0.82])  # the heatmap a little narrower, the curves wider
        # each curve panel over its legend, in a grid of their own so the gap between them fits just that panel
        top, bottom = [grid[row, 0].subgridspec(2, 1, height_ratios=[1, 0.3]) for row in range(2)]
        curves = fig.add_subplot(top[0])
        plot_curves(curves, fig.add_subplot(top[1]), thetas, models.DECISION_MODELS, "decision models")
        plot_curves(fig.add_subplot(bottom[0], sharex=curves, sharey=curves), fig.add_subplot(bottom[1]), thetas,
                    models.LLMS, "LLMs")
        plot_distances(fig, fig.add_subplot(grid[:, 1]), dist)
        fig.get_layout_engine().set(w_pad=0.2, h_pad=0.1, wspace=0.04, hspace=0)
        save(fig, path, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)


if __name__ == "__main__":
    table, _ = compare()
    dist = distances(table)
    print(dist.round(2).to_string(index=False, na_rep="-"))
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    plot(fits(CURVE_TREATMENT), dist, FIGURE)
    print(f"\nSaved {FIGURE}")
