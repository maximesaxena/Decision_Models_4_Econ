"""
The two SCE spending questions (QSP12n, QSP13new), worded as in the questionnaire, asked as each persona:
the respondent's demographics only. QSP13new's "Now imagine" becomes "Imagine", as it is asked on its own.
"""

import re

SETUP = "You are taking part in a survey about your household's finances."

QUESTIONS = {
    "qsp12n": ("Suppose next year you were to find your household with 10% more income than you currently expect. "
               "What would you do with the extra income?", {
                   "1": "Save or invest all of it",
                   "2": "Spend or donate all of it",
                   "3": "Use all of it to pay down debts",
                   "4": "Spend some and save some",
                   "5": "Spend some and use part of it to pay down debts",
                   "6": "Save some and use part of it to pay down debts",
                   "7": "Spend some, save some and use some to pay down debts",
               }),
    "qsp13new": ("Imagine that next year you were to find yourself with 10% less household income. What would you do?", {
                   "1": "Cut spending by the whole amount",
                   "2": "Not cut spending at all, but cut my savings by the whole amount",
                   "3": "Not cut spending at all, but increase my debt by borrowing the whole amount",
                   "4": "Cut spending by some and cut savings by some",
                   "5": "Cut spending by some and increase debt by some",
                   "6": "Cut savings by some and increase debt by some",
                   "7": "Cut spending by some, cut savings by some and increase debt some",
               }),
}
ANSWER_FORMAT = "Please give only one choice, and highlight the option you choose with [] (such as [Option 1])."

# the questionnaire's answers, written for a persona; "other" (and anything unlisted) is left out
RACES = {"1": "White", "2": "Black or African American", "3": "American Indian or Alaska Native", "4": "Asian",
         "5": "Native Hawaiian or Other Pacific Islander"}
EDUCATION = {1: "less than high school", 2: "a high school diploma (or equivalent)", 3: "some college but no degree",
             4: "an associate/junior college degree", 5: "a bachelor's degree", 6: "a master's degree",
             7: "a doctoral degree", 8: "a professional degree (such as an MD or JD)"}
EMPLOYMENT = {"1": "working full-time", "2": "working part-time", "3": "not working, but would like to work",
              "4": "temporarily laid off", "5": "on sick or other leave", "6": "permanently disabled or unable to work",
              "7": "retired", "8": "a student", "9": "a homemaker"}
INCOME = {1: "less than $10,000", 2: "$10,000 to $19,999", 3: "$20,000 to $29,999", 4: "$30,000 to $39,999",
          5: "$40,000 to $49,999", 6: "$50,000 to $59,999", 7: "$60,000 to $74,999", 8: "$75,000 to $99,999",
          9: "$100,000 to $149,999", 10: "$150,000 to $199,999", 11: "$200,000 or more"}
HOME = {1: "You own your home.", 2: "You rent your home."}
STATES = {"AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
          "CT": "Connecticut", "DC": "the District of Columbia", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
          "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
          "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
          "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NC": "North Carolina",
          "ND": "North Dakota", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
          "NM": "New Mexico", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
          "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
          "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
          "WI": "Wisconsin", "WY": "Wyoming"}


def persona(person):
    """
    A respondent's demographics as a few sentences in the second person.
    """
    races = [RACES[code] for code in person["race"].split(";") if code in RACES]
    jobs = [EMPLOYMENT[code] for code in person["employment"].split(";") if code in EMPLOYMENT]
    size = int(person["household_size"])
    sentences = [
        f"You are {int(person['age'])} years old and {'female' if person['female'] else 'male'}.",
        f"You consider yourself {' and '.join(races)}." if races else None,
        f"You live in {STATES[person['state']]}." if person["state"] in STATES else None,
        f"The highest level of school you have completed is {EDUCATION[person['education']]}." if person["education"] in EDUCATION else None,
        "You are married or living with a partner." if person["partner"] else "You are not married or living with a partner.",
        "You live alone." if size == 1 else f"There are {size} people in your household, including you.",
        f"You are {' and '.join(jobs)}." if jobs else None,
        f"The total pre-tax income of your household over the past 12 months was {INCOME[person['income']]}." if person["income"] in INCOME else None,
        HOME.get(person["home"]),
    ]
    return " ".join(sentence for sentence in sentences if sentence)


def prompts(people):
    """
    Every prompt, with its state and question (decision models) and original (LLMs).
    """
    rows = []
    for person in people.to_dict("records"):
        state = f"{SETUP} {persona(person)}"
        for item, (question, criteria) in QUESTIONS.items():
            options = "\n".join(f"Option {answer}: {description}" for answer, description in criteria.items())
            rows.append({"userid": person["userid"], "date": person["date"], "item": item, "state": state,
                         "question": {"type": "choice", "instructions": question, "criteria": criteria},
                         "original": f"{state}\n\n{question}\n\n{options}\n\n{ANSWER_FORMAT}"})
    return rows


def parse(response, item):
    """
    The option ("1" to "7") an LLM chose, read from [] as Mei et al. did; None if there is none.
    """
    answers = {re.sub(r"^option\s*#?\s*", "", match.strip(), flags=re.IGNORECASE)
               for match in re.findall(r"\[(.*?)\]", response)}
    return answers.pop() if len(answers) == 1 and answers <= set(QUESTIONS[item][1]) else None
