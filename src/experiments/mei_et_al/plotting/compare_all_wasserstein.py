"""
Draw analysis_results/mei_et_al/summary_wasserstein.png: summary_2.png with the Wasserstein-1 (earth mover's)
distance from humans in place of the KL divergence. Turing test win rate against Wasserstein-1 distance, averaged
over the games (left), and mean time against mean cost per call (right). Each game's distance is a percentage of its
range, so games with different ranges can be averaged. Also writes comparison.csv (each game) and
summary_wasserstein.csv (averaged, with calls, valid answers, times and costs).

With --edit, opens the figure drawn from the last summary_wasserstein.csv with labels that can be dragged; pressing S
saves their positions to label_positions_wasserstein.json, which later runs use. Labels without a saved position are
placed automatically; delete the file to place them all automatically again.

Usage: python src/experiments/mei_et_al/plotting/compare_all_wasserstein.py [--edit]
"""

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from common.plotting import (AXIS_SIZE, INK, REFERENCE_STYLES, REFERENCES, TITLE_SIZE, add_legend, edit_labels,
                             figure_style, grouped_points, place_labels, save, scatter_model, style_axes)
from experiments.mei_et_al import ANALYSIS_DIR
from experiments.mei_et_al.plotting.compare_all_2 import collect, draw_divide, plot_time_vs_cost, summarise

OUTPUT_DIR = ANALYSIS_DIR
FIGURE = OUTPUT_DIR / "mei.png"
SUMMARY = OUTPUT_DIR / "summary_wasserstein.csv"
POSITIONS = OUTPUT_DIR / "label_positions_wasserstein.json"  # each panel's label offsets, in points, by group
#EXCLUDED = {"gpt-sol", "gemma"}  # TEMPORARY: left out until their runs finish
EXCLUDED = {}


def plot_turing_vs_wasserstein(ax, summary, right=None):
    """
    Win rate against Wasserstein-1 distance, the x axis up to right if given; returns each group's and reference's
    points, to label.
    """
    for model, win, distance in summary[["model", "win_%", "wasserstein_%"]].itertuples(index=False):
        if model in REFERENCES:
            color, marker = REFERENCE_STYLES[model]
            ax.scatter(win, distance, color=color, marker=marker, s=220, zorder=3)
        else:
            scatter_model(ax, model, win, distance)
    ax.set_xlabel("Turing test: comparisons won (%)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_ylabel("Wasserstein-1 distance (% of range)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_title("Turing test vs Wasserstein-1 distance", fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
    style_axes(ax)
    if right is not None:
        ax.set_xlim(ax.get_xlim()[0], right)
    return grouped_points(summary, "win_%", "wasserstein_%")


def plot(summary, path, edit=False):
    """
    Draw the figure to path or, with edit, show it for its labels to be dragged.
    """
    saved = json.loads(POSITIONS.read_text()) if POSITIONS.exists() else {}
    with figure_style():
        # the same size in inches, so labels sit as they will in the saved figure, but fewer pixels to fit the screen
        fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=70 if edit else 100)
        left_points = plot_turing_vs_wasserstein(left, summary, right=max(55, summary["win_%"].max() + 3))
        families = plot_time_vs_cost(right, summary)
        fig.tight_layout(w_pad=4)
        labels = {"human_likeness": place_labels(left, left_points, saved.get("human_likeness", {}))}  # after the layout,
        draw_divide(right, families)  # which moves the axes
        add_legend(fig, summary)
        if edit:
            edit_labels(fig, labels, POSITIONS, lambda: plot(summary, path))
        else:
            save(fig, path, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit", action="store_true", help="drag the labels of the figure from the last summary_wasserstein.csv")
    if parser.parse_args().edit:
        plot(pd.read_csv(SUMMARY), FIGURE, edit=True)
        return
    comparison, calls = collect()
    if comparison is None:
        print("No results yet")
        return
    comparison = comparison[~comparison["model"].isin(EXCLUDED)].reset_index(drop=True)
    distances = comparison.groupby(["model", "elicitation"], as_index=False)["wasserstein_%"].mean()
    summary = summarise(comparison, calls).merge(distances, how="left")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    summary.round(6).to_csv(SUMMARY, index=False)
    pd.set_option("display.width", 220)
    print(f"\nTuring tests, predictive fit and Wasserstein-1 distance\n{comparison.round(2).to_string(index=False, na_rep='-')}")
    print(f"\nAveraged over the games\n{summary.round(4).to_string(index=False, na_rep='-')}")
    plot(summary, FIGURE)
    print(f"\nSaved comparison.csv, {SUMMARY.name} and {FIGURE.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
