"""Central project settings, so paths and constants are defined in one place."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

UCI_DATASET_ID = 891  # CDC Diabetes Health Indicators, DOI 10.24432/C53919
RAW_FILE = RAW_DIR / "cdc_diabetes_health_indicators.csv"

TARGET = "Diabetes_binary"
# UCI ships a patient ID column. It is an identifier, not a predictor, so we drop it
# to stop models from learning row order.
ID_COLUMN = "ID"

# A fixed seed makes splits and models reproducible.
RANDOM_STATE = 42
