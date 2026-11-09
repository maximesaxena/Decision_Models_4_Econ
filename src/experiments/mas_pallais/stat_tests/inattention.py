"""
The estimated inattention, alpha, of the applicants and every model (decision models and LLMs), in each of Mas &
Pallais's treatments and averaged over them: the share not choosing A when it pays $5 more, as in analysis.compare.

Writes analysis_results/mas_pallais/inattention.tex.

Usage: python src/experiments/mas_pallais/stat_tests/inattention.py
"""

import numpy as np

from common import models
from common.plotting import NAMES
from experiments.mas_pallais import ANALYSIS_DIR
from experiments.mas_pallais.prompts import TREATMENTS
from experiments.mas_pallais.stat_tests.analysis import MAX_ALPHA, compare

OUTPUT = ANALYSIS_DIR / "inattention.tex"


def name(model):
    return NAMES.get(model, model).strip()


def alphas():
    """
    A row per source, the applicants first and then the models in models.MODELS order, a column per treatment and
    their mean (blank if a treatment is missing).
    """
    table, _ = compare()
    grid = table.pivot(index="source", columns="treatment", values="alpha")[list(TREATMENTS)]
    grid["mean"] = grid.mean(axis=1, skipna=False)
    return grid.loc[["humans", *[model for model in models.MODELS if model in grid.index]]]


def cell(alpha, marked):
    """
    The alpha, daggered if marked and above MAX_ALPHA; a dash if missing.
    """
    if np.isnan(alpha):
        return "--"
    dagger = r"^{\dagger}" if marked and alpha > MAX_ALPHA else ""
    return f"${alpha:.2f}{dagger}$"


def latex(grid):
    """
    The applicants, the decision models and the LLMs, each block under a rule.
    """
    headers = [treatment.replace("_", " ").capitalize() for treatment in TREATMENTS] + ["Mean"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Estimated inattention in each of Mas \& Pallais's treatments}",
             r"\label{tab:mas_pallais_inattention}",
             rf"\begin{{tabular}}{{l{'r' * len(headers)}}}", r"\toprule",
             " & ".join(["", *headers]) + r" \\"]
    blocks = [["humans"], [model for model in grid.index if model in models.DECISION_MODELS],
              [model for model in grid.index if model in models.LLMS]]
    for block in [block for block in blocks if block]:
        lines.append(r"\midrule")
        for source in block:
            row = grid.loc[source]
            cells = [cell(row[treatment], True) for treatment in TREATMENTS] + [cell(row["mean"], False)]
            lines.append(" & ".join([name(source), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              rf"\parbox{{\linewidth}}{{\footnotesize Inattention $\alpha$ in the inattention-corrected logit "
              rf"$P(A \mid \text{{gap}}) = \alpha + (1 - 2\alpha) F(b\,\text{{gap}} + c)$: the share not choosing A "
              rf"when it pays \$5 more, from a linear fit of the choices at wage gaps of \$2 to \$5, clipped to "
              rf"$[0, 0.49]$. Mean is the average over the {len(TREATMENTS)} treatments with equal weights. "
              rf"$^{{\dagger}}$ Above {MAX_ALPHA}, so no WTP is fitted.}}",
              r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    table = latex(alphas())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table + "\n")
    print(table)
    print(f"\nSaved {OUTPUT.name} to {OUTPUT.parent}")
