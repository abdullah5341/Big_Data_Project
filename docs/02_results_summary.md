# Results Summary: Working Notes for the Report

> These are **notes and numbers**, not report text. Write the report in your own words.
> Every number below comes from our own runs (files in `results/`). Literature values are
> marked with the verification status from `docs/01_project_selection_and_plan.md`.

## 1. Data (from `data_loader.py`, `preprocess.py`)
| Item | Value |
|---|---|
| Raw rows / features | 253,680 / 21 |
| Missing values | 0 |
| Exact duplicates removed | 24,206 (9.5%), 98.7% of them class 0 |
| Rows after deduplication | 229,474 |
| Positive share (before → after deduplication) | 13.93% → 15.29% |
| Train / Val / Test | 160,631 / 34,421 / 34,422 (stratified, 15.3% positive in each) |
| Engineered features | `BMICategory` (WHO bands), `ComorbidityCount` (HighBP + HighChol + Stroke + HeartDisease) |

## 2. Model results on the test set (`results/test_metrics.csv`)
| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | Train time |
|---|---|---|---|---|---|---|---|---|
| Dummy (majority) | 0.847 | 0.000 | 0.000 | 1.000 | 0.000 | 0.500 | 0.153 | – |
| Logistic Regression | 0.786 | 0.378 | 0.617 | 0.817 | 0.469 | 0.814 | 0.427 | ~0.3 s |
| Random Forest | 0.790 | 0.385 | 0.625 | 0.819 | 0.476 | 0.819 | 0.447 | ~9–10 s |
| **XGBoost** | 0.794 | 0.391 | 0.620 | 0.826 | **0.480** | **0.822** | **0.454** | ~2 s |

- **XGBoost vs Logistic Regression:** ROC-AUC 0.822 vs 0.814 (+0.008). XGBoost is the best model, but only slightly better than the simple model (answers RQ2).
- **Thresholds (chosen on validation, max F1):** LR 0.62 · RF 0.60 · XGB 0.64
- **CV ROC-AUC on train (5-fold):** LR 0.807 ± 0.003 · RF 0.811 ± 0.003 · XGB 0.815 ± 0.003. Validation and test agree, so there is no sign of overfitting.

## 3. Comparison with published studies (same dataset)
This is the final comparison used in the report (Table VI). It covers three studies; the requirement asks for at least three.

| Study | Year | Best model | Acc | Prec | Rec | F1 | ROC-AUC | Our XGBoost | Difference (AUC) | Imbalance handling / protocol |
|---|---|---|---|---|---|---|---|---|---|---|
| Shechter & Shomron (JSIM, peer-reviewed) | 2026 | CatBoost | 0.86 | – | – | – | 0.83 | 0.822 | −0.008 | Resampling variants explored; evaluated on imbalanced data |
| Majcherek et al. (PLOS ONE, peer-reviewed) | 2025 | Extra Trees | >0.90 | – | – | – | 0.99 | 0.822 | −0.168 | SMOTE / ADASYN / random oversampling |
| Islam et al. (TCCE 2025 / arXiv) | 2025 | Stacking (XGB+KNN) | 0.948 | 0.929 | 0.970 | 0.949 | 0.990 | 0.822 | −0.168 | SMOTE + Tomek links; timing vs. split not stated; test-set size does not match a 20% split |

– = not reported in the source.

**Verification:**
- **Shechter & Shomron:** checked against the article page.
- **Islam et al.:** checked against the full arXiv text.
- **Majcherek et al.:** the numbers come from the abstract. Its use of SMOTE, ADASYN and random oversampling was confirmed from the PLOS ONE article listing. The exact split ratio was not found.

**Discussed in the text but not in the table:**
- **Aich et al. (CopulaSMOTE):** oversampling only on the training folds; gains mainly in F1 and recall. Exact numbers were not verified, so they are left out.
- **Preprint 202411.1265:** dropped, because its authors could not be verified.

**How to argue it:** our AUC sits within about 0.01 of the three studies that evaluate on real, imbalanced data. The studies reporting about 0.99 use balanced or resampled evaluation, or don't report when resampling happened. Those numbers are not comparable, so we do **not** claim to beat or lose to them. Our Accuracy (0.79) is lower than P1's 0.86 because we choose the threshold for F1/recall, not for accuracy. Even the Dummy model scores 0.85 accuracy.

## 4. Explainability (`results/shap_importance.csv`)
- **Top 5 by mean |SHAP| (XGBoost):** GenHlth, ComorbidityCount, Age, BMI, HighBP
- **Discussion points:**
  - ComorbidityCount absorbed the credit of HighChol, Stroke and HeartDisease. This is the correlated-features limitation of SHAP.
  - CholCheck = 0 lowers predicted risk. This is plausibly *diagnosis/access bias*: people outside the healthcare system are less often diagnosed.
  - Heavy alcohol use lowers predicted risk. This is counter-intuitive and plausibly reverse causation (diagnosed people are told to stop drinking). It shows SHAP gives association, not causation.

## 4b. Feature selection (`src/feature_selection.py`)
**Method:** rank features by permutation importance in 5-fold CV on **train**. Choose the smallest top-k
subset on **validation** whose ROC-AUC is within 0.002 of all features. Confirm once on **test**.
XGBoost keeps its tuned hyperparameters.

| k | 3 | 5 | 8 | **10** | 12 | 15 | 18 | 23 (all) |
|---|---|---|---|---|---|---|---|---|
| Val ROC-AUC | 0.790 | 0.811 | 0.815 | **0.816** | 0.817 | 0.817 | 0.817 | 0.818 |

- **Selected (10):** GenHlth, Age, BMI, ComorbidityCount, CholCheck, HvyAlcoholConsump, Sex, Income, HighBP, MentHlth
- **Test, all 23 vs selected 10:** ROC-AUC 0.822 vs 0.820 · PR-AUC 0.454 vs 0.450 · F1 0.480 vs 0.478 · Recall 0.620 vs 0.613
- **Interpretation:** more than half the features add almost nothing. A shorter questionnaire is possible, which is useful for real screening.
  Note that ComorbidityCount is built from HighBP, HighChol, Stroke and HeartDisease, and BMI needs height and weight. So
  the 10 model features correspond to **12 survey questions** (vs 21).
- **Decision:** the deployed app keeps all 23 features (best test AUC; the difference is small). A shortened questionnaire is future work.
- The permutation ranking agrees with SHAP on the top 4 (GenHlth, Age, BMI, ComorbidityCount).

## 5. Error analysis (`notebooks/02_error_analysis.ipynb`)
| Finding | Number |
|---|---|
| FN / FP / TP / TN | 1,999 / 5,086 / 3,266 / 24,071 |
| Missed vs caught diabetics, average GenHlth | 2.71 vs 3.70 (missed people rate their health better) |
| Missed vs caught, average BMI | 28.6 vs 34.2 |
| Missed vs caught, HighBP share | 51% vs 90% |
| Missed vs caught, ComorbidityCount | 1.12 vs 2.14 |
| False negatives with BMI < 30 | 64.6% |
| Test rows within ±0.10 of the threshold | 20.1%, with an error rate of 46.7% (vs 14.0% further away) |
| Share of all errors near the threshold | 45.7% |
| Diabetics missed by **all three** models | 1,685 (32.0%) |
| XGBoost misses also missed by LR and RF | **84.3%** → the limit is the data (no glucose/HbA1c), not the algorithm |

**False positives** look clinically similar to true positives (average BMI 32.6, HighBP 87%, GenHlth 3.48). Many are plausibly undiagnosed or high-risk people, which is acceptable for screening.

### Subgroup check (`results/error_subgroup_metrics.csv`)
| Group | Prevalence | Recall | FPR |
|---|---|---|---|
| Female / Male | 14.2% / 16.7% | 0.616 / 0.625 | 0.155 / 0.200 |
| Age 18–44 / 45–64 / 65+ | 4.3% / 15.4% / 21.6% | **0.281** / 0.597 / 0.679 | 0.025 / 0.158 / 0.303 |
| Income low / middle / high | 24.7% / 17.0% / 10.9% | 0.764 / 0.643 / **0.485** | 0.336 / 0.212 / 0.103 |
| Education ≤ high school / some college+ | 19.6% / 13.2% | 0.684 / 0.575 | 0.254 / 0.139 |

→ Sex is balanced. **Young adults (recall 0.28) and high-income people (0.49) are missed most.** The model follows prevalence. Report this as a fairness limitation. A possible mitigation (future work) is group-specific thresholds.

## 6. Big Data platform evaluation, CLO3 (`results/scalability_benchmark.csv`)
| Training rows | LR fit | RF fit | XGB fit | XGB predictions/s |
|---|---|---|---|---|
| 40k (25%) | 0.06 s | 1.6 s | 0.7 s | ~0.89 M |
| 80k (50%) | 0.11 s | 3.4 s | 1.1 s | ~0.89 M |
| 160k (100%) | 0.25 s | 9.1 s | 1.8 s | ~0.84 M |

- The training data is **30.8 MB** in memory, so a single laptop is enough. Spark would add cluster start-up and data-transfer overhead with no benefit at this size.
- **XGBoost's training time grows more slowly than the data** (4× the rows → about 2.4× the time). Random Forest grows faster (4× the rows → about 6× the time) and its saved model is 64 MB. This is a practical reason to deploy XGBoost.
- **Future, about 100× (all BRFSS years ≈ 25M+ rows):** Parquet storage, then PySpark / Spark MLlib or distributed XGBoost (e.g. on Databricks). Same pipeline logic.
- **Streaming:** questionnaire events → Kafka → scoring service with the saved pipeline (XGBoost already makes about 0.85 million predictions per second on this laptop) → drift monitoring → scheduled retraining.

## 7. Limitations to discuss
1. Self-reported survey data. Labels miss undiagnosed diabetes. There are no lab features.
2. Single year (2015), US only. Temporal and geographic generalisation is untested.
3. The prediabetes coding is ambiguous (UCI documentation vs observed counts). Verify this.
4. Risk scores are not calibrated probabilities (because of class weighting). The app says so.
5. SHAP: correlated features share credit; it gives associations, not causes.
6. There is a performance gap across age and income groups.
7. Hyperparameter search was limited (randomised, 6–20 settings) to keep it laptop-friendly.
