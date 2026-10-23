# Explainable Diabetes Risk Screening (CDC BRFSS 2015)

A leakage-safe comparison of **Logistic Regression, Random Forest and XGBoost** for diabetes risk
screening from self-reported health indicators. It includes SHAP explanations, error analysis, a
scalability benchmark and a **Streamlit** prototype.

**Dataset:** [CDC Diabetes Health Indicators, UCI #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
(DOI 10.24432/C53919), 253,680 survey responses, 21 features, binary target `Diabetes_binary`.
It is downloaded automatically. No manual download is needed.

## Key results (test set, 34,422 people, never used for training or tuning)
| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.617 | 0.378 | 0.469 | 0.814 | 0.427 |
| Random Forest | 0.625 | 0.385 | 0.476 | 0.819 | 0.447 |
| **XGBoost** (deployed) | 0.620 | 0.391 | **0.480** | **0.822** | **0.454** |

The full results and their discussion are in [docs/02_results_summary.md](docs/02_results_summary.md).

## Setup
Tested on Python 3.14.8. `requirements.txt` pins the exact library versions used for the results.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the prototype only (uses the committed model)
```bash
streamlit run app/app.py
```
It has three pages: **Risk check** (questionnaire → risk score, risk band, SHAP explanation),
**Batch scoring** (CSV upload) and **Model performance**. The performance page needs the `results/`
and `figures/` folders, which are included in the repo.

## Reproduce everything
```bash
python src/run_pipeline.py      # ~15-20 min: download → preprocess → train → evaluate → SHAP → benchmark → diagram
jupyter notebook                # then run notebooks/01_eda.ipynb and notebooks/03_error_analysis.ipynb
```
Or run the steps one by one:

| # | Command | What it does | Main outputs |
|---|---|---|---|
| 1 | `python src/data_loader.py` | Downloads UCI #891, prints a quality report | `data/raw/` |
| 2 | `notebooks/01_eda.ipynb` | EDA (5 figures + quality table) | `figures/eda_*`, `results/eda_data_quality.csv` |
| 3 | `python src/preprocess.py` | Deduplication, engineered features, stratified 70/15/15 split | `data/processed/`, `results/split_summary.csv` |
| 4 | `python src/train.py` | 5-fold CV tuning on train, threshold on validation | `models/*.joblib`, `results/training_summary.csv` |
| 5 | `python src/evaluate.py` | One-time test evaluation, bootstrap CIs, curves | `results/test_*.csv`, `figures/eval_*` |
| 6 | `python src/explain.py` | SHAP global and local explanations, importance comparison | `figures/xai_*`, `results/feature_importance_comparison.csv` |
| 6b | `python src/feature_selection.py` | Permutation-importance ranking (train CV), subset size chosen on validation, confirmed on test | `figures/fs_01_*`, `results/feature_selection_*.csv` |
| 7 | `notebooks/03_error_analysis.ipynb` | FN/FP profiles, subgroup fairness, hard cases | `figures/err_*`, `results/error_*.csv` |
| 8 | `python src/scalability.py` | Training time, memory and throughput from 25% to 4× data | `figures/scale_01_*`, `results/scalability_benchmark.csv` |
| 9 | `python src/make_architecture_diagram.py` | System architecture figure | `figures/architecture.png` |

All randomness uses `RANDOM_STATE = 42` ([src/config.py](src/config.py)), so reruns give the same splits and models.
Hyperparameter search times may vary by machine.

## Leakage prevention
- Exact duplicates are removed **before** splitting, and a check confirms that no identical row appears in two splits.
- Scaling is inside an sklearn `Pipeline`, refitted on each CV fold.
- Imbalance is handled with class weights. No synthetic rows are created.
- Tuning uses the training folds only. The threshold is chosen on validation. The test set is used once.

## Folder structure
```
├── app/            Streamlit prototype
├── data/           raw + processed data (git-ignored, recreated by the scripts)
├── docs/           project plan, results summary
├── figures/        all report/slide figures (PNG, 300 dpi)
├── models/         saved pipelines (only xgboost.joblib is committed)
├── notebooks/      EDA and error analysis
├── results/        metrics and tables (CSV)
├── src/            pipeline scripts + config
└── requirements.txt
```

## Disclaimer
This is a research prototype for screening support only. It is **not** a medical diagnosis.
The risk score is a ranking score, not a calibrated probability.
