"""
The New York Fed's SCE Household Spending Survey: what respondents would do with 10% more or less income.
"""

from common import ANALYSIS_ROOT, HUMAN_DATA_ROOT, RESULTS_ROOT

HUMAN_DIR = HUMAN_DATA_ROOT / "sce_spending"
SPENDING_DATA = HUMAN_DIR / "sce-household-spending-microdata.xlsx"
CORE_DATA = HUMAN_DIR / "frbny-sce-public-microdata-20-24.xlsx"
RESPONDENTS = HUMAN_DIR / "respondents.csv"  # built from the two spreadsheets by data.py
RESULTS_DIR = RESULTS_ROOT / "sce_spending"
ANALYSIS_DIR = ANALYSIS_ROOT / "sce_spending"
