"""
Ask models the SCE questions as every persona under a robustness variant (variants.py), with the baseline's
run_model, appending each call to results/sce_spending/robustness/<variant>/<model>.csv. A rerun resumes. A pilot
(the first 100 personas) goes to <variant>-pilot/, which the analysis ignores.

Usage: python src/experiments/sce_spending/robustness/run.py human                    # every model in the variant
       python src/experiments/sce_spending/robustness/run.py nosystem gpt-sol --pilot
"""

import argparse
import json

from experiments.mei_et_al.games import MAX_CRITERIA_CHARACTERS, MAX_INSTRUCTIONS_CHARACTERS, MAX_OPTIONS, MAX_STATE_CHARACTERS
from experiments.sce_spending import run as baseline
from experiments.sce_spending.data import personas
from experiments.sce_spending.robustness import RESULTS_DIR
from experiments.sce_spending.robustness.models import load, show_calls
from experiments.sce_spending.robustness.variants import VARIANTS, rows


def check(prompts):
    """
    Every prompt fits Jev's character limits, as check_prompts.py checks the baseline's.
    """
    for row in prompts:
        question = row["question"]
        assert len(row["state"]) <= MAX_STATE_CHARACTERS and len(row["original"]) <= MAX_STATE_CHARACTERS
        assert len(question["instructions"]) <= MAX_INSTRUCTIONS_CHARACTERS
        assert len(question["criteria"]) <= MAX_OPTIONS and len(json.dumps(question["criteria"])) <= MAX_CRITERIA_CHARACTERS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=list(VARIANTS))
    parser.add_argument("models", nargs="*", help="default: every model in the variant")
    parser.add_argument("--pilot", action="store_true", help=f"the first {baseline.PILOT_PERSONAS} personas")
    args = parser.parse_args()

    system, allowed = VARIANTS[args.variant]
    names = args.models or allowed
    assert set(names) <= set(allowed), f"{args.variant} runs only {', '.join(allowed)}"
    people = personas()
    prompts = rows(people.head(baseline.PILOT_PERSONAS) if args.pilot else people, args.variant)
    check(prompts)
    directory = RESULTS_DIR / (f"{args.variant}-pilot" if args.pilot else args.variant)
    directory.mkdir(parents=True, exist_ok=True)
    baseline.RESULTS_DIR = directory  # run_model saves to RESULTS_DIR / <model>.csv
    print(f"{args.variant}: LLM system prompt {system!r}")
    for name in names:
        baseline.run_model(name, show_calls(load(name, system)), prompts, 1)  # prints an LLM's first calls
        print(f"Saved results to {directory / f'{name}.csv'}")
