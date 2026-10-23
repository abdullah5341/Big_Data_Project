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

# Split proportions: 70% train, 15% validation (tuning/threshold), 15% test (final, used once).
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# WHO adult BMI bands: <18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, >=30 obese.
BMI_BINS = [0, 18.5, 25, 30, float("inf")]
BMI_LABELS = [0, 1, 2, 3]  # ordinal codes: underweight, normal, overweight, obese

# Chronic conditions counted in the engineered "ComorbidityCount" feature.
COMORBIDITY_COLUMNS = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]

# Hyperparameter search: stratified k-fold CV on the TRAINING set only.
CV_FOLDS = 5
# ROC-AUC is threshold-independent and is the metric most published studies report,
# which keeps our tuning comparable with the literature.
TUNING_METRIC = "roc_auc"
SEARCH_ITERATIONS = {"logistic_regression": 6, "random_forest": 8, "xgboost": 20}

# Models in the order they appear in every table and figure, with display names and fixed colours.
MODEL_NAMES = {
    "dummy": "Dummy (majority)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}

# Bootstrap resamples of the test set, used for 95% confidence intervals.
BOOTSTRAP_ROUNDS = 1000
