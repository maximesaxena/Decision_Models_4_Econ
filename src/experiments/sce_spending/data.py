"""
Build data/human/sce_spending/respondents.csv from the two SCE spreadsheets, and draw the personas.

Each spending answer is matched to the same person's core SCE answers from the same month (2020 on).
Demographics, asked only in a person's first month, are carried to their later months.

Usage: python src/experiments/sce_spending/data.py   # rebuilds respondents.csv (a minute or two)
"""

import numpy as np
import pandas as pd

from experiments.sce_spending import CORE_DATA, RESPONDENTS, SPENDING_DATA

N_PERSONAS = 1000
SEED = 0

ANSWERS = ["qsp12n", "qsp12a_1", "qsp12a_2", "qsp12a_3", "qsp13new", "qsp13a_1", "qsp13a_2", "qsp13a_3"]
# what the persona is built from, all of which a persona must have
DEMOGRAPHICS = ["age", "female", "race", "education", "partner", "household_size", "employment",
                "income", "home", "state"]
# the Fed's coarse categories
CATEGORIES = {"_AGE_CAT": "age_cat", "_EDU_CAT": "edu_cat", "_HH_INC_CAT": "income_cat", "_NUM_CAT": "numeracy_cat"}


def read(path):
    """
    A spreadsheet's Data sheet. Row 1 is the Fed's source notice and row 2 the column names.
    """
    from python_calamine import CalamineWorkbook
    rows = CalamineWorkbook.from_path(path).get_sheet_by_name("Data").to_python()
    data = pd.DataFrame(rows[2:], columns=rows[1]).replace("", np.nan)
    return data.apply(lambda column: pd.to_numeric(column, errors="coerce") if column.name not in ["_STATE", *CATEGORIES] else column)


def ticked(data, prefix, codes):
    """
    The codes ticked in a select-all question, joined with ";" (e.g. "1;4"), or NaN if none were.
    """
    picked = pd.DataFrame({code: np.where(data[f"{prefix}_{code}"] == 1, str(code), "") for code in codes}, index=data.index)
    joined = picked.apply(lambda row: ";".join(code for code in row if code), axis=1)
    return joined.replace("", np.nan)


def build():
    spending, core = read(SPENDING_DATA), read(CORE_DATA)
    for data in [spending, core]:
        data["userid"], data["date"] = data["userid"].astype(int), data["date"].astype(int)
    spending = spending[spending["date"] >= core["date"].min()]
    spending = spending[spending[["qsp12n", "qsp13new"]].notna().any(axis=1)]

    # entry-month answers, carried to every month of the same person
    entry = core[core["Q32"].notna()].sort_values("date").groupby("userid").first()
    household = entry[[f"Q45new_{i}" for i in range(1, 9)]]
    demographics = pd.DataFrame({
        "entry_age": entry["Q32"].where(entry["Q32"].between(18, 100)),
        "entry_date": entry.index.map(entry["date"]),
        "female": entry["Q33"].map({1: 1, 2: 0}),
        "race": ticked(entry, "Q35", range(1, 7)),
        "education": entry["Q36"],
        "partner": entry["Q38"].map({1: 1, 2: 0}),
        # you plus everyone else who usually lives with you; NaN if any count is implausible
        "household_size": (1 + household.fillna(0).sum(axis=1)).where((household.fillna(0) <= 15).all(axis=1)),
        "income": entry["Q47"],
        "home": entry["Q43"],
    }, index=entry.index)

    # same-month answers
    month = core[["userid", "date", "weight", "_STATE", *CATEGORIES]].rename(columns={"_STATE": "state", **CATEGORIES})
    month["employment"] = ticked(core, "Q10", range(1, 11))

    data = spending[["userid", "date", *ANSWERS]].merge(month, on=["userid", "date"], how="inner")
    data = data.merge(demographics, left_on="userid", right_index=True, how="left")
    months = (data["date"] // 100 - data["entry_date"] // 100) * 12 + (data["date"] % 100 - data["entry_date"] % 100)
    data["age"] = data["entry_age"] + months.clip(lower=0) // 12
    data = data.drop(columns=["entry_age", "entry_date"])
    assert len(data) == len(spending), "some spending answers have no core answers from the same month"
    return data[["userid", "date", "weight", *ANSWERS, *DEMOGRAPHICS, *CATEGORIES.values()]]


def load():
    return pd.read_csv(RESPONDENTS, dtype={"race": str, "employment": str})


def personas(data=None, n=N_PERSONAS, seed=SEED):
    """
    The respondents the models are asked to be: n people with every demographic and both answers, one
    random month each, in random order.
    """
    data = load() if data is None else data
    complete = data.dropna(subset=["qsp12n", "qsp13new", *DEMOGRAPHICS])
    one_each = complete.sample(frac=1, random_state=seed).drop_duplicates("userid")
    return one_each.sample(n, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    data = build()
    data.to_csv(RESPONDENTS, index=False)
    complete = data.dropna(subset=["qsp12n", "qsp13new", *DEMOGRAPHICS])
    print(f"Saved {len(data)} answers from {data['userid'].nunique()} people to {RESPONDENTS}; "
          f"{complete['userid'].nunique()} people have every demographic and both answers")
