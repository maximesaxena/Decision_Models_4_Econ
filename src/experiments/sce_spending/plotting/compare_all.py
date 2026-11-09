"""
Draw analysis_results/sce_spending/turing_vs_fit.png: Turing test win rate against KL divergence, averaged over the
two questions, in the style of Mei et al.'s summary_2.png, each group in common.plotting.GROUPS getting a shaded
outline around its models' points and a label with an arrow to it. Also writes comparison.csv (each question) and
summary.csv (averaged).

With --edit, opens the figure drawn from the last summary.csv with labels that can be dragged; pressing S saves
their positions to label_positions.json, which later runs use. Labels without a saved position are placed
automatically; delete the file to place them all automatically again.

Usage: python src/experiments/sce_spending/plotting/compare_all.py [--edit]
"""

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from common.plotting import add_legend, edit_labels, figure_style, place_labels, plot_turing_vs_fit, save
from experiments.sce_spending import ANALYSIS_DIR
from experiments.sce_spending.stat_tests.analysis import compare, summarise

FIGURE = ANALYSIS_DIR / "turing_vs_fit.png"
POSITIONS = ANALYSIS_DIR / "label_positions.json"  # the panel's label offsets, in points, by group
#EXCLUDED = {"gpt-sol", "gemma"}  # TEMPORARY: left out until their runs finish
EXCLUDED = {}

def plot(summary, path, edit=False):
    """
    Draw the figure to path or, with edit, show it for its labels to be dragged.
    """
    saved = json.loads(POSITIONS.read_text()) if POSITIONS.exists() else {}
    with figure_style():
        # the same size in inches, so labels sit as they will in the saved figure, but fewer pixels to fit the screen
        fig, ax = plt.subplots(figsize=(9, 7.5), dpi=70 if edit else 100)
        points = plot_turing_vs_fit(ax, summary)
        fig.tight_layout()
        labels = {"human_likeness": place_labels(ax, points, saved.get("human_likeness", {}))}  # after the layout,
        add_legend(fig, summary)  # which moves the axes
        if edit:
            edit_labels(fig, labels, POSITIONS, lambda: plot(summary, path))
        else:
            save(fig, path, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edit", action="store_true", help="drag the labels of the figure from the last summary.csv")
    if parser.parse_args().edit:
        plot(pd.read_csv(ANALYSIS_DIR / "summary.csv"), FIGURE, edit=True)
        return
    comparison = compare()
    comparison = comparison[~comparison["model"].isin(EXCLUDED)].reset_index(drop=True)
    summary = summarise(comparison)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(ANALYSIS_DIR / "comparison.csv", index=False)
    summary.to_csv(ANALYSIS_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Turing tests and predictive fit\n{comparison.round(2).to_string(index=False, na_rep='-')}")
    print(f"\nAveraged over the two questions\n{summary.round(2).to_string(index=False, na_rep='-')}")
    plot(summary, FIGURE)
    print(f"\nSaved comparison.csv, summary.csv and {FIGURE.name} to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
