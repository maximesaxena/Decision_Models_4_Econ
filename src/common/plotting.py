"""
The figures' shared look: each model's color, marker and name, the fonts and sizes, the Turing test against KL
divergence panel, its labels and shaded group outlines, which can be dragged and saved, and the legend.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from common import models

REFERENCES = ["humans", "uniform"]
# each label on the Turing test panel, and the models it points to
GROUPS = {"LLMs": list(models.LLMS), "Frontier DMs": ["jev", "d1-ultra", "pplx", "mammoth"]}

# color, marker, and line style ("-" filled, "--" hollow); variants of a model share its hue family
STYLES = {
    "jev": ("#2a78d6", "o", "-"),
    "jev-3rdparty": ("#104281", "o", "-"),
    "kev": ("#eb6834", "o", "-"),
    "laya": ("#1baf7a", "o", "-"),
    "anyjev": ("#eda100", "o", "-"),
    "luna": ("#7b5cd6", "o", "-"),
    "pplx": ("#e34948", "o", "-"),
    "mammoth": ("#b5179e", "o", "-"),
    "mercury": ("#9a7d0a", "o", "-"),
    "d1-ultra": ("#2e4057", "o", "-"),
    "microsoft": ("#6b7a1f", "o", "-"),
    "clef": ("#a50f15", "o", "-"),
    "qwen": ("#e87ba4", "s", "--"),
    "qwen-nothink": ("#9c2f5f", "s", "--"),
    "gemini": ("#008300", "s", "--"),
    "gemini-lowthink": ("#86b817", "s", "--"),
    "gpt": ("#8b5a2b", "s", "--"),
    "gpt-sol": ("#c4915c", "s", "--"),
    "gemma": ("#00796b", "s", "--"),
    "glm-flash": ("#8fa8e0", "s", "--"),
    "gemini-argon": ("#004d1a", "s", "--"),
    "haiku": ("#d97757", "s", "--"),
}
# names shown on the figures
NAMES = {
    "humans": "Humans",
    "uniform": "Uniform",
    "laya": "Laya",
    "jev": "Jev",
    "jev-3rdparty": "Jev via jevmodel.org",
    "kev": "Kev",
    "anyjev": "AnyJev",
    "luna": "Luna Decisions",
    "pplx": "Perplexity ",
    "mammoth": "Mammoth",
    "mercury": "Mercury",
    "d1-ultra": "D1 Ultra",
    "microsoft": "Microsoft",
    "clef": "Clef Omni",
    "qwen": "Qwen",
    "qwen-nothink": "Qwen (NT)",
    "gemini": "Gemini Flash",
    "gemini-lowthink": "Gemini Flash (LT)",
    "gpt": "Luna LLM",
    "gpt-sol": "Sol",
    "gemma": "Gemma",
    "glm-flash": "GLM Flash",
    "gemini-argon": "Gemini Argon",
    "haiku": "Haiku",
}
INK, MUTED, GRID, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#f3f2ee"
REFERENCE_STYLES = {"humans": (INK, "*"), "uniform": (MUTED, "X")}
FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]  # the first one installed is used
TICK_SIZE, LABEL_SIZE, AXIS_SIZE, TITLE_SIZE = 19, 19, 19, 19
HOLLOW_EDGE = 2.2  # edge width of the LLMs' hollow squares, thick so they read as open boxes beside the solid dots
OUTLINE_PAD = 14  # points between a group's points and the edge of its shaded outline
LEADER_GAP = 20  # labels further than this from their point, in points, get a leader line
LEGEND_COLUMNS = 3  # in each family's legend block


def figure_style():
    """
    The figures' font, with S freed for saving label positions.
    """
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next(font for font in FONTS if font in installed)
    return plt.rc_context({"font.family": font, "keymap.save": ["ctrl+s"]})


def save(fig, path, **kwargs):
    """
    The figure as a 300 dpi PNG at path and as a vector PDF beside it, for LaTeX, its fonts embedded as TrueType so
    the text stays sharp and selectable.
    """
    fig.savefig(path, dpi=300, **kwargs)
    with plt.rc_context({"pdf.fonttype": 42}):
        fig.savefig(path.with_suffix(".pdf"), **kwargs)


def style(model):
    """
    The model's color, marker and line style.
    """
    return STYLES.get(model, (INK, "o", "-"))


def grouped_points(table, x, y):
    """
    Each group's (x, y) points, for the groups with any, then each reference's point under its name.
    """
    groups = {name: members for name, members in GROUPS.items()} | {NAMES[reference]: [reference] for reference in REFERENCES}
    groups = {name: list(zip(*table.loc[table["model"].isin(members), [x, y]].to_numpy().T)) for name, members in groups.items()}
    return {name: points for name, points in groups.items() if points}


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=TICK_SIZE)
    ax.grid(color=GRID, linewidth=0.5)


def rounded_hull(points, pad):
    """
    The convex hull of circles of radius pad around the points, as polygon vertices in the points' units.
    """
    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    cloud = (points[:, None, :] + pad * np.column_stack([np.cos(angles), np.sin(angles)])[None]).reshape(-1, 2)
    cloud = cloud[np.lexsort((cloud[:, 1], cloud[:, 0]))]

    def chain(sequence):  # one half of Andrew's monotone chain
        kept = []
        for point in sequence:
            while len(kept) >= 2 and ((kept[-1][0] - kept[-2][0]) * (point[1] - kept[-2][1])
                                      - (kept[-1][1] - kept[-2][1]) * (point[0] - kept[-2][0])) <= 0:
                kept.pop()
            kept.append(point)
        return kept[:-1]

    return np.array(chain(cloud) + chain(cloud[::-1]))


def outline_pieces(ends, others, pad):
    """
    A group's outlines, in pixels: one rounded hull around its points (ends) unless that would cover another model's
    point (others), in which case the point furthest from the rest gets its own outline, a halo, until none is
    covered. The main outline comes first.
    """
    main, halos = list(ends), []
    while len(main) > 1 and Path(rounded_hull(np.array(main), pad)).contains_points(others).any():
        centre = np.mean(main, axis=0)
        halos.append(main.pop(int(np.argmax([np.hypot(*(point - centre)) for point in main]))))
    return [rounded_hull(np.array(points), pad) for points in [main, *[[halo] for halo in halos]]]


def show_leader(label):
    """
    Show a point's leader line only if its label is far enough from the point to need one; groups always keep their
    arrows.
    """
    if label.arrowprops["arrowstyle"] == "-":
        label.arrow_patch.set_visible(np.hypot(*label.xyann) > LEADER_GAP)


def place_labels(ax, groups, saved):
    """
    Label each entry of groups ({name: [(x, y), ...]}). A name in GROUPS gets a shaded rounded outline around its
    points, or around most of them and a halo around each point that would otherwise bring another model's point
    inside (see outline_pieces), and an arrow from its label to each; any other name labels its single point, with a
    leader line only when the label is far from it. A label sits at its saved (dx, dy, ha, va) offset, in points,
    from the main outline's centre or the point if it has one; otherwise at the nearest spot, inside the axes, that
    clashes least with the points, the outlines and the other labels and arrows. Returns each name's label.
    """
    renderer = ax.figure.canvas.get_renderer()
    pixels = ax.figure.dpi / 72  # pixels per point
    radius = 6 * pixels  # about a marker's half-width
    every = np.concatenate([ax.transData.transform(collection.get_offsets()) for collection in ax.collections])
    markers = [Bbox.from_extents(x - radius, y - radius, x + radius, y + radius) for x, y in every]
    frame = ax.get_window_extent(renderer)
    outlines, shading, mine = {}, {}, {}
    for name, points in groups.items():
        ends = ax.transData.transform(points)
        mine[name] = [np.any(np.all(np.isclose(ends, point), axis=1)) for point in every]
        if name in GROUPS:
            outlines[name] = outline_pieces(ends, every[~np.array(mine[name])], OUTLINE_PAD * pixels)
            shading[name] = [ax.add_patch(Polygon(ax.transData.inverted().transform(outline), facecolor=GRID, alpha=0.6,
                                                  edgecolor=MUTED, linewidth=0.8, zorder=1)) for outline in outlines[name]]
    paths = {name: [Path(outline, closed=True) for outline in pieces] for name, pieces in outlines.items()}
    every_outline = [path for pieces in paths.values() for path in pieces]
    placed, arrows, labels = [], [], {}
    group_arrow = dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=MUTED, linewidth=1, shrinkB=0)

    def annotate(name, dx, dy, ha="center", va="center"):
        """
        The label, its text's box, and its arrows or leader line from the box's centre, in pixels, each with the
        outline it points to (None for a leader line).
        """
        if name in outlines:
            target = outlines[name][0].mean(axis=0)
            arrowprops = group_arrow | dict(patchB=shading[name][0])  # stops at the outline
        else:
            target = ax.transData.transform(groups[name][0])
            arrowprops = dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkB=radius / pixels + 1)
        label = ax.annotate(name, ax.transData.inverted().transform(target), xytext=(dx, dy), textcoords="offset points",
                            fontsize=LABEL_SIZE, color=INK, ha=ha, va=va, arrowprops=arrowprops,
                            bbox=dict(boxstyle="square,pad=0.1", facecolor="none", edgecolor="none"))  # for the halos' arrows
        show_leader(label)
        label.update_positions(renderer)
        box = Text.get_window_extent(label, renderer)
        middle = ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
        ends = [(target, paths[name][0] if name in paths else None)] if label.arrow_patch.get_visible() else []
        ends += [(halo.mean(axis=0), path) for halo, path in zip(outlines.get(name, [])[1:], paths.get(name, [])[1:])]
        return label, box, [(Path([middle, end]), aim) for end, aim in ends]

    def clashes(name, box, own_arrows):
        """
        Text over text counts most, then text over a point, outline or arrow, then an arrow over anything else.
        """
        count = 3 * sum(box.overlaps(other) for other in placed)
        count += 2 * sum(box.overlaps(marker) for marker in markers)
        count += 2 * sum(outline.intersects_bbox(box) for outline in every_outline)
        count += 2 * sum(line.intersects_bbox(box) for line in arrows)
        for arrow, aim in own_arrows:
            # any point or outline on the way counts, including the group's own, but not those it points to
            passed = [marker for marker, own in zip(markers, mine[name])
                      if not (aim.contains_point(marker.get_points().mean(axis=0)) if aim else own)]
            count += 2 * sum(arrow.intersects_bbox(other) for other in placed)
            count += sum(arrow.intersects_bbox(marker) for marker in passed)
            count += sum(arrow.intersects_path(outline) for outline in every_outline if outline is not aim)
            count += sum(arrow.intersects_path(other) for other in arrows)
        inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
        return count + (0 if inside else 100)

    def keep(name, label, box, own_arrows):
        labels[name] = label
        placed.append(box)
        arrows.extend(arrow for arrow, _ in own_arrows)
        for halo in shading.get(name, [])[1:]:  # an arrow from the label to each halo, following it when dragged
            ax.annotate("", halo.get_xy().mean(axis=0), xytext=(0.5, 0.5), textcoords=label.get_bbox_patch(),  # the text alone, not its arrow
                        arrowprops=group_arrow | dict(patchA=label.get_bbox_patch(), patchB=halo))

    def spots(name):
        """
        Where to try the label, nearest first, as (dx, dy, ha, va).
        """
        if name in outlines:
            # just clear of the main outline in each direction, so its arrow is short
            outline = outlines[name][0]
            centre = outline.mean(axis=0)
            label = ax.annotate(name, (0, 0), fontsize=LABEL_SIZE)
            box = Text.get_window_extent(label, renderer)
            label.remove()
            tries = []
            for gap in (10, 18, 28, 42, 60, 85):
                for angle in np.radians(np.arange(0, 360, 22.5)):
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    edge = ((outline - centre) @ direction).max()  # to the outline's furthest point that way
                    half = (abs(direction[0]) * box.width + abs(direction[1]) * box.height) / 2  # the label's half-size that way
                    tries.append((*((edge + half) / pixels + gap) * direction, "center", "center"))
            return tries
        sides = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        return [(gap * x, gap * y, {1: "left", 0: "center", -1: "right"}[x], {1: "bottom", 0: "center", -1: "top"}[y])
                for gap in (14, 25, 40, 60, 85) for x, y in sides]

    for name in [name for name in groups if name in saved] + [name for name in groups if name not in saved]:
        if name in saved:
            keep(name, *annotate(name, *saved[name]))
            continue
        tries, scores = spots(name), []
        for spot in tries:
            label, box, arrow = annotate(name, *spot)
            scores.append(clashes(name, box, arrow))
            label.remove()
            if scores[-1] == 0:
                break
        keep(name, *annotate(name, *tries[int(np.argmin(scores))]))
    return labels


def edit_labels(fig, labels, positions, redraw):
    """
    Show the figure with labels ({panel: {name: label}}) that can be dragged, their arrows following; S saves every
    label's position to positions and calls redraw.
    """
    for panel in labels.values():
        for label in panel.values():
            label.draggable(True)

    def released(_):
        for panel in labels.values():
            for label in panel.values():
                show_leader(label)
        fig.canvas.draw_idle()

    def pressed(event):
        if event.key != "s":
            return
        saved = {panel_name: {name: [*map(float, label.xyann), label.get_horizontalalignment(),
                                     label.get_verticalalignment()] for name, label in panel.items()}
                 for panel_name, panel in labels.items()}
        positions.write_text(json.dumps(saved, indent=1))
        redraw()
        print(f"Saved the label positions to {positions.name} and redrew the figure")

    fig.canvas.mpl_connect("button_release_event", released)
    fig.canvas.mpl_connect("key_press_event", pressed)
    fig.canvas.manager.set_window_title("Drag the labels; press S to save")
    plt.show()


def model_marker(model):
    """
    The model's marker, its face and edge colors, and its edge width: filled for decision models, hollow for LLMs.
    """
    color, marker, linestyle = style(model)
    return (marker, color, "white", 1.2) if linestyle == "-" else (marker, "white", color, HOLLOW_EDGE)


def scatter_model(ax, model, x, y):
    marker, face, edge, width = model_marker(model)
    ax.scatter(x, y, marker=marker, s=110, zorder=3, linewidths=width, facecolors=face, edgecolors=edge)


def legend_handle(model):
    """
    A legend entry: the model's (or reference's) marker and name.
    """
    if model in REFERENCES:
        color, marker = REFERENCE_STYLES[model]
        return Line2D([], [], linestyle="none", marker=marker, markersize=15, color=color, label=NAMES[model])
    marker, face, edge, width = model_marker(model)
    return Line2D([], [], linestyle="none", marker=marker, markersize=12, markeredgewidth=width,
                  markerfacecolor=face, markeredgecolor=edge, label=NAMES.get(model, model))


def add_legend(fig, summary):
    """
    Below the panels, a centred row of the references and, under it, a framed block for each family, decision models
    left and LLMs right (centred if only one has results), its heading on top and its models in models.MODELS order.
    """
    present = set(summary["model"])
    renderer = fig.canvas.get_renderer()
    style = dict(fontsize=AXIS_SIZE, labelcolor=INK, handlelength=1, handletextpad=0.4, columnspacing=1.4,
                 title_fontsize=AXIS_SIZE, alignment="left")
    top = 0
    references = [legend_handle(model) for model in REFERENCES if model in present]
    if references:
        legend = fig.legend(handles=references, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=len(references),
                            frameon=False, borderpad=0.2, **style)
        top = -legend.get_window_extent(renderer).height / fig.bbox.height - 0.01
    families = [(title, handles) for title, members in [("Decision models", models.DECISION_MODELS), ("LLMs", models.LLMS)]
                if (handles := [legend_handle(model) for model in members if model in present])]
    places = [("upper right", 0.49), ("upper left", 0.51)] if len(families) == 2 else [("upper center", 0.5)]
    for (title, handles), (loc, x) in zip(families, places):
        legend = fig.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=(x, top), ncol=LEGEND_COLUMNS,
                            frameon=True, borderpad=0.6, **style)
        legend.get_title().set(fontweight="bold", color=MUTED)
        legend.get_frame().set(edgecolor=GRID, linewidth=1.2, facecolor="white")


def plot_turing_vs_fit(ax, summary, right=None):
    """
    Win rate against KL divergence, the x axis up to right if given; returns each group's and reference's points,
    to label.
    """
    for model, win, kl in summary[["model", "win_%", "kl_bits"]].itertuples(index=False):
        if model in REFERENCES:
            color, marker = REFERENCE_STYLES[model]
            ax.scatter(win, kl, color=color, marker=marker, s=220, zorder=3)
        else:
            scatter_model(ax, model, win, kl)
    ax.set_xlabel("Turing test: comparisons won (%)", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_ylabel("KL divergence from humans", fontsize=AXIS_SIZE, color=INK, fontweight="bold")
    ax.set_title("Turing test vs KL divergence", fontsize=TITLE_SIZE, color=INK, fontweight="bold", pad=12)
    style_axes(ax)
    if right is not None:
        ax.set_xlim(ax.get_xlim()[0], right)
    return grouped_points(summary, "win_%", "kl_bits")
