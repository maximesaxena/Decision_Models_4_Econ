"""
Compare models with Mas & Pallais's (2017) applicants. WTP is estimated for both with the paper's
inattention-corrected logit, P(A | gap) = alpha + (1 - 2 alpha) F(b gap + c), each call counting once, and
each model's fitted WTP distribution is compared with the applicants' by the Wasserstein-1 (earth mover's)
distance, in $ per hour.
"""

import numpy as np
import pandas as pd

from experiments.mas_pallais import HUMAN_DATA, RESULTS_DIR
from experiments.mas_pallais.prompts import TREATMENTS

# Table 5 of the paper: mean, sd, 25th, 50th and 75th percentiles of WTP, in $ per hour
PAPER = {
    "flexible_schedule": (0.48, 2.15, -0.82, 0.48, 1.79),
    "employer_discretion": (3.41, 2.95, 1.63, 3.41, 5.20),
    "work_from_home": (1.33, 1.86, 0.20, 1.33, 2.45),
    "flexible_hours": (-0.22, 2.24, -1.58, -0.22, 1.14),
    "combined_flexible": (1.17, 2.33, -0.25, 1.17, 2.58),
}
STATISTICS = ["mean", "sd", "p25", "median", "p75"]

MAX_ALPHA = 0.4  # above this, choices barely follow the wage and WTP is left blank
SMOOTHING = 0.01  # keeps the cross-entropy finite
QUANTILES = (np.arange(100_000) + 0.5) / 100_000
LOGITS = np.log(QUANTILES / (1 - QUANTILES))  # the standard logistic's quantiles

HUMAN_COLOR = "#898781"
MODEL_COLORS = {"jev": "#2a78d6", "jev-3rdparty": "#173f8a", "kev": "#eb6834", "laya": "#1baf7a", "anyjev": "#eda100", "luna": "#7b5cd6",
                "pplx": "#20808d", "gemini": "#008300"}


def human_choices():
    """
    Every applicant in the five main treatments: treatment, gap (A's wage minus B's) and chose_a.
    """
    data = pd.read_stata(HUMAN_DATA, convert_categoricals=False)
    names = {number: treatment for treatment, (number, _, _) in TREATMENTS.items()}
    data = data[data["treatment_number"].isin(names)]
    # the data's wagegap is B's wage minus A's
    return pd.DataFrame({"treatment": data["treatment_number"].map(names), "gap": -data["wagegap"].astype(float),
                         "chose_a": data["chose_position1"].astype(float)})


def model_results():
    """
    Every model's calls, with p_a (choosing A) from p_1 and p_2 (positions as shown); None if none.
    """
    paths = sorted(RESULTS_DIR.glob("*.csv"))
    if not paths:
        return None
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    results["p_a"] = np.where(results["a_first"], results["p_1"], results["p_2"])
    return results


def inattention(gap, chose_a):
    """
    alpha: the share not choosing A when it pays $5 more, from a linear fit over gaps of $2 to $5.
    """
    high = gap >= 2
    slope, intercept = np.polyfit(gap[high], chose_a[high], 1)
    return float(np.clip(1 - (intercept + 5 * slope), 0, 0.49))


def log_likelihood(theta, gap, chose_a, alpha):
    p = curve(theta, alpha, gap)
    return np.sum(chose_a * np.log(p) + (1 - chose_a) * np.log(1 - p))


def fit_logit(gap, chose_a, alpha):
    """
    Maximum likelihood b and c, by Fisher scoring. chose_a can be a probability.
    """
    X = np.column_stack([gap, np.ones_like(gap)])
    theta = np.array([1.0, 0.0])
    for _ in range(200):
        F = 1 / (1 + np.exp(-np.clip(X @ theta, -30, 30)))
        p = alpha + (1 - 2 * alpha) * F
        slope = (1 - 2 * alpha) * F * (1 - F)  # dp / d(b gap + c)
        weight = slope / (p * (1 - p))
        step = np.linalg.solve((X * (weight * slope)[:, None]).T @ X + 1e-9 * np.eye(2), X.T @ (weight * (chose_a - p)))
        current, size = log_likelihood(theta, gap, chose_a, alpha), 1.0
        while True:
            new = theta + size * step
            new[0] = np.clip(new[0], -100, 100)  # a model that always picks the higher wage has no finite b
            if log_likelihood(new, gap, chose_a, alpha) >= current or size < 1e-8:
                break
            size /= 2
        change, theta = np.abs(new - theta).max(), new
        if change < 1e-10:
            break
    return theta


def curve(theta, alpha, gap):
    """
    The fitted probability of choosing A at each gap.
    """
    return alpha + (1 - 2 * alpha) / (1 + np.exp(-np.clip(theta[0] * gap + theta[1], -30, 30)))


def wtp(theta, alpha):
    """
    Mean, sd and quartiles of WTP for A, as in the paper; blank if alpha > MAX_ALPHA or b <= 0.
    """
    b, c = theta
    if alpha > MAX_ALPHA or b <= 0:
        return dict.fromkeys(STATISTICS, np.nan)
    mean, scale = c / b, 1 / b
    return dict(zip(STATISTICS, [mean, 1 / (0.55 * b), mean - scale * np.log(3), mean, mean + scale * np.log(3)]))


def cross_entropy(chose_a, p):
    p = (1 - SMOOTHING) * p + SMOOTHING / 2
    return -np.mean(chose_a * np.log2(p) + (1 - chose_a) * np.log2(1 - p))


def compare():
    """
    One row per treatment and source (the paper, our fit of the applicants' choices, each model), and
    every curve to plot.
    """
    humans, results = human_choices(), model_results()
    rows, curves = [], {}
    for treatment in TREATMENTS:
        rows.append({"treatment": treatment, "source": "paper (Table 5)", **dict(zip(STATISTICS, PAPER[treatment]))})

        h = humans[humans["treatment"] == treatment]
        gap, chose_a = h["gap"].to_numpy(), h["chose_a"].to_numpy()
        alpha = inattention(gap, chose_a)
        theta = fit_logit(gap, chose_a, alpha)
        rows.append({"treatment": treatment, "source": "humans", "alpha": alpha, **wtp(theta, alpha),
                     "cross_entropy_bits": cross_entropy(chose_a, curve(theta, alpha, gap)), "n": len(h)})
        curves[treatment, "humans"] = (h.groupby("gap")["chose_a"].mean(), theta)

        if results is None:
            continue
        for model, r in results[results["treatment"] == treatment].groupby("model", sort=False):
            valid = 100 * r["p_a"].notna().mean()
            r = r.dropna(subset=["p_a"])
            m_gap, p_a = r["gap"].to_numpy(), r["p_a"].to_numpy()
            m_alpha = inattention(m_gap, p_a)
            m_theta = fit_logit(m_gap, p_a, m_alpha)
            by_gap = r.groupby("gap")["p_a"].mean()  # averaged over maximum wages and orders
            picks_first = np.where(r["a_first"], p_a, 1 - p_a)
            rows.append({"treatment": treatment, "source": model, "alpha": m_alpha, **wtp(m_theta, m_alpha),
                         "cross_entropy_bits": cross_entropy(chose_a, by_gap.reindex(gap).to_numpy()),
                         "position_1_%": 100 * picks_first.mean(), "valid_%": valid, "n": len(r)})
            curves[treatment, model] = (by_gap, m_theta)
    return pd.DataFrame(rows), curves


def quantiles(row):
    """
    The logistic's quantiles, mean + ln(u / (1 - u)) / b, with 1 / b from the quartiles.
    """
    scale = (row["p75"] - row["mean"]) / np.log(3)
    return row["mean"] + scale * LOGITS


def distances(table):
    """
    One row per treatment and model: the Wasserstein-1 distance to the applicants (the average gap between the two
    WTP distributions' quantiles, or the area between their fitted curves without inattention) and the gap in means.
    """
    rows = []
    for treatment in TREATMENTS:
        t = table[table["treatment"] == treatment].set_index("source")
        human = t.loc["humans"]
        for model, row in t.drop(["humans", "paper (Table 5)"]).iterrows():
            rows.append({"treatment": treatment, "model": model, "wasserstein": np.abs(quantiles(row) - quantiles(human)).mean(),
                         "mean_gap": row["mean"] - human["mean"], "sd_ratio": row["sd"] / human["sd"]})
    return pd.DataFrame(rows)
