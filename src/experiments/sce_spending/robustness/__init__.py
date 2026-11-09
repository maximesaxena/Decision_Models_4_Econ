"""
Robustness checks of the SCE spending results: the frontier decision models and three LLMs under small changes to
the prompts (variants.py). Results go in their own folders, apart from the baseline's.
"""

from common.plotting import GROUPS
from experiments.sce_spending import ANALYSIS_DIR as BASELINE_ANALYSIS_DIR, RESULTS_DIR as BASELINE_RESULTS_DIR

RESULTS_DIR = BASELINE_RESULTS_DIR / "robustness"  # one folder per variant
ANALYSIS_DIR = BASELINE_ANALYSIS_DIR / "robustness"
DMS = GROUPS["Frontier DMs"]
LLMS = ["gpt-sol", "gemini-argon", "glm-flash"]
