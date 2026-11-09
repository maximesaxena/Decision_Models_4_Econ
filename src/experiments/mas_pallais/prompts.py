"""
The five main treatments of Mas & Pallais (2017), with positions from their Table 2. A is the position
whose value the paper estimates; the wage gap is A's wage minus B's. Each prompt is asked in both orders.

N.B. in the original, the city is specified and there are 68 different metro areas. Here, we simplify things 
by not naming the city explicitly since we do not have the cities on the open-source human data. 
"""

import re

BASELINE = ("The position is 40 hours per week. This is a M-F 9 am-5 pm position. "
            "The work is exclusively on-site in an office downtown. This position pays {wage} dollars per hour.")

ALTERNATIVES = {
    "flexible_schedule": (
        "The position is 40 hours per week. You can make your own schedule. This can be a M-F 9 am-5 pm schedule "
        "or other days and times. The work is exclusively on-site in an office downtown. This position pays {wage} dollars per hour."),
    "flexible_hours": (
        "In this position you can choose the number of hours you work, up to and including 40 hours each week. "
        "The position has a M-F daytime schedule. The work is exclusively on-site in an office downtown. "
        "This position pays {wage} dollars per hour."),
    "work_from_home": (
        "The position is 40 hours per week. This is a M-F 9 am-5 pm position. You have the option of working from "
        "home as well as on-site in an office downtown. This position pays {wage} dollars per hour."),
    "combined_flexible": (
        "You can choose the number of hours you work, up to and including 40 hours each week. You can make your own "
        "schedule. This can be a M-F 9 am-5 pm schedule or other days and times. You have the option of working from "
        "home as well as on-site in an office downtown. This position pays {wage} dollars per hour."),
    "employer_discretion": (
        "The position is 40 hours per week. The hours in this position vary from week to week. You will be given your "
        "work schedule one week in advance. The hours can be morning through evening, weekdays and weekends, but not "
        "nights. The work is exclusively on-site in an office downtown. This position pays {wage} dollars per hour."),
}

# treatment -> (treatment_number in the replication data, position A, position B)
TREATMENTS = {
    "flexible_schedule": (1, ALTERNATIVES["flexible_schedule"], BASELINE),
    "employer_discretion": (2, BASELINE, ALTERNATIVES["employer_discretion"]),
    "work_from_home": (3, ALTERNATIVES["work_from_home"], BASELINE),
    "flexible_hours": (4, ALTERNATIVES["flexible_hours"], BASELINE),
    "combined_flexible": (5, ALTERNATIVES["combined_flexible"], BASELINE),
}

GAPS = [-5, -4, -3, -2.75, -2.5, -2.25, -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0,
        0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4, 5]
MAX_WAGES = [16, 19]

SETUP = ("You are applying for a job as a phone survey interviewer at a call center. As part of the application you are "
         "shown two job positions. The type of work in both positions is the same.")
QUESTION = "If you were selected for both positions, which one would you prefer?"
CRITERIA = {"1": "Position 1", "2": "Position 2"}
ANSWER_FORMAT = "Please give only one choice, and highlight the position you choose with [] (such as [Position 1])."


def prompts():
    """
    Every prompt, with its state and question (decision models) and original (LLMs).
    """
    rows = []
    for treatment, (_, a, b) in TREATMENTS.items():
        for gap in GAPS:
            for max_wage in MAX_WAGES:
                wage_a, wage_b = min(max_wage, max_wage + gap), min(max_wage, max_wage - gap)
                jobs = [a.format(wage=f"{wage_a:.2f}"), b.format(wage=f"{wage_b:.2f}")]
                for a_first in [True, False]:
                    first, second = jobs if a_first else jobs[::-1]
                    state = f"{SETUP}\n\nPosition 1: {first}\n\nPosition 2: {second}"
                    rows.append({"treatment": treatment, "gap": gap, "max_wage": max_wage, "a_first": a_first,
                                 "state": state, "question": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA},
                                 "original": f"{state}\n\n{QUESTION} {ANSWER_FORMAT}"})
    return rows


def parse(response):
    """
    The position ("1" or "2") an LLM chose, read from [] as Mei et al. did; None if there is none.
    """
    answers = {re.sub(r"^position\s*#?\s*", "", match.strip(), flags=re.IGNORECASE)
               for match in re.findall(r"\[(.*?)\]", response)}
    return answers.pop() if len(answers) == 1 and answers <= set(CRITERIA) else None
