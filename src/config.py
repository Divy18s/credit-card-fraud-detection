"""Central configuration: paths, constants, and experiment settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "artifacts"

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
RAW_CSV = DATA_RAW / "creditcard.csv"

RESULTS_CSV = REPORTS / "results_classical.csv"

TARGET_COL = "Class"
RANDOM_STATE = 42

# Time-based split fractions (train on earliest transactions, test on latest)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Remaining 0.15 -> test
