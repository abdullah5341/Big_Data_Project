# Explainable Diabetes Risk Screening (CDC BRFSS 2015)

A coursework project that compares **Logistic Regression, Random Forest and XGBoost** for predicting
diabetes risk from simple health-survey answers. It includes SHAP explanations, error analysis,
feature selection, a scalability test and a **Streamlit** web app.

**Dataset:** [CDC Diabetes Health Indicators, UCI #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(DOI 10.24432/C53919): 253,680 survey answers, 21 features, target `Diabetes_binary`.
It is downloaded automatically, so no manual download is needed.

## Results (test set: 34,422 people, never used for training or tuning)
| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.617 | 0.378 | 0.469 | 0.814 | 0.427 |
| Random Forest | 0.625 | 0.385 | 0.476 | 0.819 | 0.447 |
| **XGBoost** (used in the app) | 0.620 | 0.391 | **0.480** | **0.822** | **0.454** |

- All results and their discussion: [docs/02_results_summary.md](docs/02_results_summary.md)
- How the code works, file by file: [docs/03_code_guide.md](docs/03_code_guide.md)

## Setup
Tested on Python 3.14.8. `requirements.txt` pins the exact library versions used.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web app
```bash
streamlit run app/app.py
```
The app has two pages:
- **Risk check:** questionnaire → risk score → Low / Borderline / High band → explanation of the result.
- **Model performance:** the test results.

The trained model is included in the repo, so the app works straight after cloning.

## Reproduce all results
```bash
python src/run_pipeline.py     # runs steps 1-8 below (about 15-20 minutes)
```
| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `python src/data_loader.py` | Downloads UCI #891 and prints a quality report | `data/raw/` |
| 2 | `python src/preprocess.py` | Removes duplicates, adds 2 features, stratified 70/15/15 split | `data/processed/`, `results/split_summary.csv` |
| 3 | `python src/train.py` | Tunes each model with 5-fold CV on train; chooses the threshold on validation | `models/`, `results/training_summary.csv` |
| 4 | `python src/evaluate.py` | Scores every model once on the test set | `results/test_*.csv`, `figures/eval_*` |
| 5 | `python src/explain.py` | SHAP explanations | `figures/xai_*`, `results/shap_importance.csv` |
| 6 | `python src/feature_selection.py` | Finds how many features are really needed | `figures/fs_01_*`, `results/feature_selection_*.csv` |
| 7 | `python src/scalability.py` | Training time on 25%, 50% and 100% of the data | `figures/scale_01_*`, `results/scalability_benchmark.csv` |
| 8 | `python src/make_architecture_diagram.py` | System architecture figure | `figures/architecture.png` |

The notebooks are opened with `jupyter notebook`:
- `notebooks/01_eda.ipynb` (exploratory data analysis) needs step 1.
- `notebooks/02_error_analysis.ipynb` needs step 4.

All randomness uses `RANDOM_STATE = 42` (in [src/config.py](src/config.py)), so every run gives the same results.

## How data leakage is prevented
- Exact duplicates are removed **before** splitting, and a check proves that no row is in two splits.
- Scaling sits inside an sklearn `Pipeline`, so it only learns from the training folds.
- Class imbalance is handled with class weights, so no synthetic rows are created.
- Settings are tuned on the training data only, the threshold is chosen on validation, and the test set is used once.

## Folder structure
```
├── app/            Streamlit web app
├── data/           raw + processed data (not in git; recreated by the scripts)
├── docs/           project plan, results summary, code guide
├── figures/        all figures for the report and slides
├── models/         saved models (only xgboost.joblib is in git)
├── notebooks/      EDA and error analysis
├── results/        result tables (CSV)
├── src/            the pipeline scripts + settings
└── requirements.txt
```

## Disclaimer
This is a research prototype for screening support only. It is **not** a medical diagnosis.
The risk score ranks people by risk; it is not an exact probability.
