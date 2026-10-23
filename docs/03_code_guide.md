# Code Guide: How the Project Works

Read the files in this order. Each step saves a file that the next step reads.

| Step | File | What it does | Main output |
|---|---|---|---|
| 0 | `src/config.py` | All settings in one place: folders, random seed, split sizes, model names | — |
| 1 | `src/data_loader.py` | Downloads the dataset from UCI and prints a quality report | `data/raw/` |
| – | `notebooks/01_eda.ipynb` | Charts that describe the data (EDA) | `figures/eda_*` |
| 2 | `src/preprocess.py` | Removes duplicates, adds 2 features, splits 70/15/15 | `data/processed/` |
| 3 | `src/train.py` | Tunes and trains LR, RF and XGBoost, and chooses each model's threshold | `models/` |
| 4 | `src/evaluate.py` | Scores every model once on the test set | `results/test_metrics.csv` |
| – | `notebooks/02_error_analysis.ipynb` | Who the model gets wrong, and why | `figures/err_*` |
| 5 | `src/explain.py` | SHAP explanations for XGBoost | `figures/xai_*` |
| 6 | `src/feature_selection.py` | How many features the model really needs | `results/feature_selection_*` |
| 7 | `src/scalability.py` | Training time on 25%, 50% and 100% of the data | `figures/scale_01_*` |
| 8 | `src/make_architecture_diagram.py` | Draws the system architecture figure | `figures/architecture.png` |
| – | `app/app.py` | The web app | — |
| – | `src/run_pipeline.py` | Runs steps 1–8 in order with one command | — |

---

## Step 0: `config.py`
- **Why it exists:** every fixed number lives here, so there are no unexplained numbers hidden in the code.
- **Most important line:** `RANDOM_STATE = 42`. The same seed means the same split and the same models on every run. This is what makes the results reproducible.

## Step 1: `data_loader.py`
- `download_dataset()` downloads UCI dataset 891, joins the features and the target into one table, drops the `ID` column, and saves a CSV.
- `load_dataset()` reads the saved CSV, so later steps don't download the data again.
- `print_quality_report()` prints the facts for the report: 253,680 rows, 0 missing values, 24,206 duplicates, and 13.9% with diabetes.
- **Viva:** *"Where does your data come from?"* From the CDC's BRFSS 2015 telephone health survey, published on the UCI repository as dataset 891.

## Step 2: `preprocess.py`
- `remove_duplicates()` drops exact copies **before** splitting. Otherwise the same row could be in both train and test, and the test score would look better than it really is.
- `add_engineered_features()` adds `BMICategory` (the WHO BMI groups) and `ComorbidityCount` (how many of high BP, high cholesterol, stroke and heart disease a person has). The app calls the **same function**, so the app and the experiments always prepare data the same way.
- `split_data()` makes two cuts: first 15% for test, then validation out of what is left. `stratify` keeps the diabetes share at about 15% in every part.
- `check_no_overlap()` proves that no row is in two splits. It stops with an error if one is.
- **Viva:** *"How did you prevent data leakage?"* Duplicates are removed before the split, scaling sits inside the pipeline, settings are tuned on train only, the threshold is chosen on validation, and the test set is used once.

## Step 3: `train.py`
- `load_split()` splits a file into **X** (the answers) and **y** (diabetes yes or no).
- `build_models()` creates the three models, each with a range of settings to try:
  - **Logistic Regression** has a `StandardScaler` first. LR is sensitive to the scale of the numbers; tree models are not.
  - **Class imbalance:** `class_weight="balanced"` for LR and RF, and `scale_pos_weight` (about 5.5) for XGBoost. Mistakes on people with diabetes count more, so the model doesn't just ignore them.
- `tune()` uses `RandomizedSearchCV`. It tries N random combinations of settings and scores each one with 5-fold cross-validation **on the training data only**.
- `choose_threshold()` finds the score cut-off with the best F1 **on the validation set**. It isn't 0.5, because class weights push all scores upwards.
- `main()` saves the dummy model, then for each model: tune → choose the threshold → save the model **together with its threshold** in `models/`.
- **Viva:** *"Why class weights instead of SMOTE?"* Class weights create no fake rows, so there's no risk of synthetic data leaking into validation or test, and they're simpler.

## Step 4: `evaluate.py`
- `compute_metrics()` calculates all the numbers. The meaning of each metric is written in its docstring.
- `main()` loads each saved model, applies its **saved** threshold, and writes `test_metrics.csv`. It also saves `test_predictions.csv`, which the error-analysis notebook uses.
- **Viva:** *"Why does the Dummy model have the highest accuracy?"* It says "no diabetes" for everyone, and 85% of people really don't have diabetes. That's why we judge the models by recall, F1, ROC-AUC and PR-AUC.

## Notebook `02_error_analysis.ipynb`
- It groups every test person into TP, FP, TN or FN (true or false positive or negative).
- It compares the diabetics who were missed with the ones who were caught. The missed ones look healthier: better self-rated health, lower BMI, less high blood pressure.
- It checks recall by sex, age and income group, which is the fairness check.
- It shows that 84% of XGBoost's misses are also missed by both other models. So the limit is the data (no blood-test results), not the algorithm.

## Step 5: `explain.py`
- `shap.TreeExplainer` calculates exact SHAP values for XGBoost, using a sample of 3,000 test people.
- It produces a **bar chart** (overall importance), a **beeswarm** (whether high or low values push risk up or down) and **3 waterfall charts** (individual people).
- `pick_example_people()` picks the most confident true positive, the worst miss (false negative) and the most confident false alarm (false positive).
- **Viva:** *"Does SHAP prove that BMI causes diabetes?"* No. It shows what the model learned from the data, which are associations, not causes.

## Step 6: `feature_selection.py`
- **Rank** the features on the train data, using permutation importance (shuffle one feature and see how much the score drops).
- **Choose** how many to keep on validation: the smallest number within 0.002 ROC-AUC of using all features.
- **Check** the choice once on test.
- **Result:** 10 features give almost the same ROC-AUC as all 23 (0.820 vs 0.822). That means a shorter questionnaire is possible.

## Step 7: `scalability.py`
- `clone()` copies the tuned settings without the training. Each model is then trained on 25%, 50% and 100% of the data and timed.
- This is the evidence for the Big Data discussion: the data is about 31 MB in memory, and training takes seconds, so one laptop is enough today.

## App: `app/app.py`
- `load_model()` loads the saved XGBoost once, using `@st.cache_resource`.
- `show_questionnaire()` shows the form, converts the answers into the dataset's codes, calculates BMI from height and weight, and rejects BMIs outside 12–98.
- `show_result()` follows the same steps as training: `add_engineered_features` → saved model → risk score → risk band (Low / Borderline / High) → SHAP explanation.
- `show_model_performance()` shows the test results table and the main figures.

---

## Key ideas in one sentence each
| Term | Meaning |
|---|---|
| Data leakage | Information from the test data sneaks into training, so results look better than they really are. |
| Stratified split | Every part (train, validation, test) keeps the same share of people with diabetes. |
| Pipeline | Steps such as scaler → model, trained as one unit so the scaler only learns from training data. |
| 5-fold cross-validation | Split the training data into 5 parts; train on 4 and score on 1, five times, then average. |
| Class weights | Mistakes on the rare class (diabetes) cost more, so the model doesn't ignore it. |
| Threshold | The score above which we flag a person; chosen on validation (0.64 for XGBoost). |
| Recall | Of the people with diabetes, the share we catch. |
| Precision | Of the people we flag, the share who really have diabetes. |
| ROC-AUC | The chance that a random person with diabetes gets a higher score than a random person without it (0.5 = guessing, 1 = perfect). |
| PR-AUC | Area under the precision–recall curve. A random model scores the prevalence (0.153). |
| SHAP value | How much one answer pushed one person's score up or down. |
| Permutation importance | How much the score drops when one feature's values are shuffled. |
