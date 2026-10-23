# Explainable Diabetes Risk Screening (CDC BRFSS 2015)

Big Data / ML coursework project: a leakage-safe comparison of Logistic Regression,
Random Forest and XGBoost for diabetes risk screening from self-reported health indicators,
with SHAP explanations, error analysis and a Streamlit prototype.

**Dataset:** [CDC Diabetes Health Indicators — UCI #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (DOI 10.24432/C53919), 253,680 records.

> Status: Phase 1 (planning). See [docs/01_project_selection_and_plan.md](docs/01_project_selection_and_plan.md).

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run order (filled in as the project is built)
1. `python src/data_loader.py` — download the dataset
2. `notebooks/01_eda.ipynb` — exploratory analysis
3. `python src/preprocess.py` → `python src/train.py` → `python src/evaluate.py` → `python src/explain.py`
4. `streamlit run app/app.py` — web prototype

## Folder structure
See section 22 of the planning document.

## Disclaimer
This is a research prototype for screening support only. It is **not** a medical diagnosis.
