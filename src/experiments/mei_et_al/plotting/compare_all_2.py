"""
Draw analysis_results/mei_et_al/summary_2.png: Turing test win rate against KL divergence, averaged over
the games (left), and mean time against mean cost per call (right). On the left, each group in
common.plotting.GROUPS gets a shaded outline around its models' points and a label with an arrow to it; on the right,
a dotted curve divides the decision models from the LLMs. Also writes comparison.csv (each game) and summary.csv (averaged, with calls, valid
answers, times and costs).

With --edit, opens the figure drawn from the last summary.csv with labels that can be dragged; pressing S saves
their positions to label_positions_2.json, which later runs use. Labels without a saved position are placed
automatically; delete the file to place them all automatically again.

Usage: python src/experiments/mei_et_al/plotting/compare_all_2.py [--edit]
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from common import models
from common.plotting import (AXIS_SIZE, BAND, INK, LABEL_SIZE, MUTED, REFERENCES, TITLE_SIZE, add_legend, edit_labels,
                             figure_style, place_labels, plot_turing_vs_fit, save, scatter_model, style_axes)
from experiments.mei_et_al import ANALYSIS_DIR
from experiments.mei_et_al.stat_tests.analysis import compare_game, load_calls, load_results
from experiments.mei_et_al.registry import EXPERIMENTS, results_dir

OUTPUT_DIR = ANALYSIS_DIR
FIGURE = OUTPUT_DIR / "summary_2.png"
POSITIONS = OUTPUT_DIR / "label_positions_2.json"  # each panel's label offsets, in points, by group

ELICITATIONS = ["humans", "reference", "choice", "score", "sampled"]
TURING = ["win_%", "tie_%", "lose_%"]
FIT = ["cross_entropy_bits", "kl_bits"]
SIDE_GAP = 18  # points between the dividing curve and the "DMs" and "LLMs" either side of it
SIDE_HEIGHT = 0.9  # dollars per 1,000 calls at which "DMs" and "LLMs" sit


def in_order(table):
    """
    Rows ordered by game, then humans, uniform and models.MODELS, then way of asking.
    """
    model_rank = {reference: i - len(REFERENCES) for i, reference in enumerate(REFERENCES)}
    model_rank |= {model: i for i, model in enumerate(models.MODELS)}
    keys = [table["model"].map(model_rank).fillna(len(model_rank)), table["elicitation"].map(ELICITATIONS.index)]
    if "game" in table:
        keys.insert(0, table["game"].map({game: i for i, game in enumerate(table["game"].unique())}))
    return table.iloc[np.lexsort(keys[::-1])].reset_index(drop=True)


def cost(calls):
    """
    Each call's cost in dollars from its tokens and models.PRICES: 0 if local, blank if unpriced.
    """
    prices = calls["model"].map(models.PRICES)
    input_price = prices.map(lambda price: price[0] if isinstance(price, tuple) else np.nan)
    output_price = prices.map(lambda price: price[1] if isinstance(price, tuple) else np.nan)
    output_tokens = calls["output_tokens"].fillna(0) if "output_tokens" in calls else 0
    dollars = (calls["input_tokens"] * input_price + output_tokens * output_price) / 1e6
    return dollars.mask(calls["device"].isin(models.LOCAL_DEVICES), 0.0)


def collect():
    """
    The comparison with humans for every game, and every call logged.
    """
    rows, calls = [], []
    for experiment in EXPERIMENTS:
        name = experiment.__name__.split(".")[-1]
        results = load_results(results_dir(experiment))
        if results is None:
            print(f"Leaving out {name}: no results yet")
            continue
        not_run = [model for model in models.MODELS if model not in set(results["model"])]
        if not_run:
            print(f"{name}: not yet run on {', '.join(not_run)}")
        humans = experiment.human_choices()
        for game in experiment.GAMES:
            rows += compare_game(game, humans[game.name], results)[0]
        calls.append(load_calls(results_dir(experiment)))
    return (in_order(pd.DataFrame(rows)), pd.concat(calls, ignore_index=True)) if rows else (None, None)


def summarise(comparison, calls):
    """
    The comparison averaged over the games, with calls, valid answers, times and costs. Calls Laya retried
    internally are left out of the times.
    """
    summary = comparison.groupby(["model", "elicitation"], as_index=False).agg(
        games=("game", "size"), **{column: (column, "mean") for column in [*TURING, *FIT]})
    retried = calls["internal_retries"].fillna(0) > 0 if "internal_retries" in calls else pd.Series(False, index=calls.index)
    calls = calls.assign(cost=cost(calls), retried=retried, clean_seconds=calls["seconds"].mask(retried))
    per_call = calls.groupby(["model", "elicitation"], as_index=False).agg(
        calls=("seconds", "size"), internally_retried=("retried", "sum"),
        valid_answer_pct=("answer", lambda answer: 100 * answer.notna().mean()),
        mean_seconds=("clean_seconds", "mean"), median_seconds=("clean_seconds", "median"),
        p90_seconds=("clean_seconds", lambda seconds: seconds.quantile(0.9)), dollars_per_1000_calls=("cost", lambda dollars: 1000 * dollars.mean()),
        device=("device", lambda device: ", ".join(sorted(set(device)))))
    return in_order(summary.merge(per_call, how="left"))


def plain_log_axis(axis):
    """
    Log ticks at 1, 3, 10, 30, ..., written plainly.
    """
    axis.set_major_locator(LogLocator(subs=(1, 3)))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(NullFormatter())


def plot_time_vs_cost(ax, summary):
    """
    Mean seconds against mean dollars per 1,000 calls, local models on their own line; returns the decision models'
    and the LLMs' points, to divide.
    """
    runs = summary[~summary["model"].isin(REFERENCES) & summary["dollars_per_1000_calls"].notna()]
    paid = runs["dollars_per_1000_calls"] > 0
    floor = runs.loc[paid, "dollars_per_1000_calls"].min() / 5 if paid.any() else 1e-6
    runs = runs.assign(y=runs["dollars_per_1000_calls"].where(paid, floor))
    for model, seconds, y in runs[["model", "mean_seconds", "y"]].itertuples(index=False):
        scatter_model(ax, model, seconds, y)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis)
    top = runs["dollars_per_1000_calls"].max() * 2 if paid.any() else floor * 10
    boundary = floor * 2  # below it, a shaded strip for the free runs, which a log axis cannot place
    decades = 10.0 ** np.arange(np.ceil(np.log10(boundary)), np.floor(np.log10(top)) + 1)
    if (~paid).any():
        ax.axhspan(floor / 2, boundary, color=BAND, linewidth=0, zorder=0)
        ax.axhline(boundary, color=MUTED, linewidth=0.8)
        on_spine = dict(transform=ax.get_yaxis_transform(), clip_on=False, zorder=4)
        ax.plot([0, 0], [boundary / 1.25, boundary * 1.25], color="white", linewidth=4, **on_spine)  # a gap in the axis
        for y in (boundary / 1.12, boundary * 1.12):  # drawn as //
            ax.plot(0, y, marker=[(-1, -0.5), (1, 0.5)], markersize=10, markeredgewidth=1.2, color=MUTED, **on_spine)
        ax.set_yticks([floor, *decades], ["run locally", *[f"{decade:g}" for decade in decades]])  # mid-strip
    else:
        ax.set_yticks(decades, [f"{decade:g}" for decade in decades])
    minor = np.asarray(LogLocator(subs=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).tick_values(boundary, top))
    ax.yaxis.set_minor_locator(FixedLocator(minor[minor > boundary * 1.25].tolist()))  # none in the strip or the gap
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(floor / 2, top)
    ax.set_xlabel("Mean seconds per call (log scale)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_ylabel("Mean dollars per 1,000 calls (log scale)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_title("Speed vs cost", fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
    style_axes(ax)
    missing = summary.loc[~summary["model"].isin(REFERENCES) & summary["dollars_per_1000_calls"].isna(), "model"].unique()
    if len(missing):
        print(f"Left out of the speed and cost panel, with no price in models.PRICES: {', '.join(missing)}")
    return {family: runs.loc[runs["model"].isin(members), ["mean_seconds", "y"]].to_numpy()
            for family, members in [("decision models", models.DECISION_MODELS), ("LLMs", models.LLMS)]}


def draw_divide(ax, families):
    """
    A dotted parabola from the top of the axes to the bottom between the decision models' and the LLMs' points,
    x = a + b y + c y^2 in axes fractions: of the tilts b and bends c tried, with the best a for each, the one leaving
    the widest gap on screen, at right angles to the curve, to the nearest point on either side.
    """
    dms, llms = families["decision models"], families["LLMs"]
    if not (len(dms) and len(llms)):
        return
    frame = ax.get_window_extent()
    (dm_x, dm_y), (llm_x, llm_y) = [ax.transAxes.inverted().transform(ax.transData.transform(points)).T
                                    for points in (dms, llms)]
    x, y = np.concatenate([dm_x, llm_x]), np.concatenate([dm_y, llm_y])
    side = np.r_[np.ones(len(dm_x)), -np.ones(len(llm_x))]  # decision models left of the curve, LLMs right
    tilts, bends = [grid.ravel() for grid in np.meshgrid(np.linspace(-1.5, 1.5, 121), np.linspace(-1.5, 1.5, 121))]
    offsets = x - tilts[:, None] * y - bends[:, None] * y ** 2  # each point's a, were the curve to pass through it
    slope = (tilts[:, None] + 2 * bends[:, None] * y) * frame.width / frame.height  # the curve's dx/dy on screen
    scale = frame.width / np.sqrt(1 + slope ** 2)  # pixels at right angles per unit of horizontal fraction
    # the best a for each tilt and bend balances some decision model's gap against some LLM's
    dm, llm = np.meshgrid(np.flatnonzero(side > 0), np.flatnonzero(side < 0))
    middles = ((scale[:, dm] * offsets[:, dm] + scale[:, llm] * offsets[:, llm]) / (scale[:, dm] + scale[:, llm])).reshape(len(tilts), -1)
    gaps = (side * (middles[..., None] - offsets[:, None, :]) * scale[:, None, :]).min(axis=-1)
    heights = np.linspace(0, 1, 50)
    shapes = tilts[:, None] * heights + bends[:, None] * heights ** 2  # each curve less its a
    gaps[(middles + shapes.min(axis=1, keepdims=True) < 0.02) | (middles + shapes.max(axis=1, keepdims=True) > 0.98)] = -np.inf  # stays within the sides
    best = np.unravel_index(gaps.argmax(), gaps.shape)
    middle, tilt, bend = middles[best], tilts[best[0]], bends[best[0]]
    heights = np.linspace(0, 1, 200)
    ax.plot(middle + tilt * heights + bend * heights ** 2, heights, color=MUTED, linewidth=1.8, linestyle="dotted",
            transform=ax.transAxes, zorder=2)
    if gaps[best] <= 0:
        print("No parabola divides the decision models from the LLMs; the dividing curve crosses some points")

    # "DMs" and "LLMs" level with each other either side of the curve, at SIDE_HEIGHT dollars per 1,000 calls
    gap = SIDE_GAP * ax.figure.dpi / 72 / frame.width  # in axes fractions
    height = ax.transAxes.inverted().transform(ax.transData.transform((1, SIDE_HEIGHT)))[1]
    across = middle + tilt * height + bend * height ** 2
    for name, x, ha in [("DMs", across - gap, "right"), ("LLMs", across + gap, "left")]:
        ax.text(x, height, name, ha=ha, va="center", transform=ax.transAxes, fontsize=LABEL_SIZE, color=INK)



def plot(summary, path, edit=False):
    """
    Draw the figure to path or, with edit, show it for its labels to be dragged.
    """
    saved = json.loads(POSITIONS.read_text()) if POSITIONS.exists() else {}
    with figure_style():
        # the same size in inches, so labels sit as they will in the saved figure, but fewer pixels to fit the screen
        fig, (left, right) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=70 if edit else 100)
        left_points = plot_turing_vs_fit(left, summary, right=max(55, summary["win_%"].max() + 3))
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
    parser.add_argument("--edit", action="store_true", help="drag the labels of the figure from the last summary.csv")
    if parser.parse_args().edit:
        plot(pd.read_csv(OUTPUT_DIR / "summary.csv"), FIGURE, edit=True)
        return
    comparison, calls = collect()
    if comparison is None:
        print("No results yet")
        return
    summary = summarise(comparison, calls)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    summary.round(6).to_csv(OUTPUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nTuring tests and predictive fit\n{comparison.round(2).to_string(index=False, na_rep='-')}")
    print(f"\nAveraged over the games\n{summary.round(4).to_string(index=False, na_rep='-')}")
    plot(summary, FIGURE)
    print(f"\nSaved comparison.csv, summary.csv and {FIGURE.name} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
