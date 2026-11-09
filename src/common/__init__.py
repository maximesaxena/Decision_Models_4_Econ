"""Code shared by every experiment: the models, figure style, and where data and results live."""

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
HUMAN_DATA_ROOT = REPO_ROOT / "data" / "human"  
RESULTS_ROOT = REPO_ROOT / "results"  # models' raw answers
ANALYSIS_ROOT = REPO_ROOT / "analysis_results"  # tables and charts
