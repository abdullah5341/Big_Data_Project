"""
Project settings in one place.

Every other script imports its folders and fixed numbers from here.
If we want to change something (for example the random seed), we change it only here.
"""
from pathlib import Path

# ---------- Folders ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # the main project folder
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------- Dataset ----------
UCI_DATASET_ID = 891   # CDC Diabetes Health Indicators, DOI 10.24432/C53919
RAW_FILE = RAW_DIR / "cdc_diabetes_health_indicators.csv"
TARGET = "Diabetes_binary"   # the column we predict: 0 = no diabetes, 1 = diabetes
ID_COLUMN = "ID"             # only numbers the people, says nothing about health, so we drop it

# Using the same seed everywhere gives the same split and the same models on every run.
RANDOM_STATE = 42

# ---------- Splitting ----------
VAL_SIZE = 0.15    # 15% validation: used to choose each model's decision threshold
TEST_SIZE = 0.15   # 15% test: used only once, for the final results

# ---------- New features ----------
# WHO adult BMI groups: below 18.5 underweight, 18.5-24.9 normal, 25-29.9 overweight, 30+ obese
BMI_BINS = [0, 18.5, 25, 30, float("inf")]
BMI_LABELS = [0, 1, 2, 3]
# These four conditions are added together into the new "ComorbidityCount" feature.
COMORBIDITY_COLUMNS = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]

# ---------- Training ----------
CV_FOLDS = 5
# Most published papers report ROC-AUC, so tuning on it keeps our results comparable.
TUNING_METRIC = "roc_auc"
# How many random combinations of settings to try per model (Random Forest is slow, so fewer).
SEARCH_ITERATIONS = {"logistic_regression": 6, "random_forest": 8, "xgboost": 20}

# Names and colours used in every table and chart.
MODEL_NAMES = {
    "dummy": "Dummy (majority)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}

# ---------- Explanation (SHAP) ----------
DEPLOYED_MODEL = "xgboost"   # best model on validation; it is explained and used in the app
SHAP_SAMPLE_SIZE = 3000      # 3,000 test rows give stable SHAP plots and run in seconds

# ---------- Feature selection ----------
FEATURE_SUBSET_SIZES = [3, 5, 8, 10, 12, 15, 18]   # we try keeping only the top 3, top 5, ...
FEATURE_SELECTION_AUC_TOLERANCE = 0.002            # largest ROC-AUC loss we accept for fewer questions
PERMUTATION_REPEATS = 5

# ---------- Web app ----------
# Scores this close to the threshold are shown as "Borderline": the error analysis showed
# that almost half of these predictions are wrong.
UNCERTAIN_MARGIN = 0.10
BMI_MIN = 12   # smallest BMI in the training data
BMI_MAX = 98   # largest BMI in the training data
