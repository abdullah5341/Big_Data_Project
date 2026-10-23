# Project Selection, Methodology & Roadmap

> **Status:** Planning document (Phase 1). Written with AI help to brainstorm and plan.
> It is **not** report text. Write the final report in your own words and verify every
> citation yourself before using it.
> Status labels used below: **✅ Verified** (checked against the source on 23-Oct-2026) ·
> **⚠️ Partly verified** · **❌ Not verified yet**.

---

## 1. Candidate Project Ideas (6)

Dataset facts for ideas 2–6 come from general knowledge and are **❌ Not verified yet**. Only idea 1 (the recommended one) and the credit-card dataset size/licence were checked.

### Idea 1 — Diabetes Risk Screening from Lifestyle & Health Survey Indicators (Healthcare)
| # | Item | Detail |
|---|---|---|
| 1 | Title | Explainable, leakage-safe ML for diabetes risk screening using CDC BRFSS 2015 health indicators |
| 2 | Problem | Many adults with diabetes or prediabetes are undiagnosed. A cheap questionnaire-based screen could flag high-risk people for a proper blood test. |
| 3 | Why real | Diabetes is a major chronic disease. A survey-only screen needs no lab work and could run in a pharmacy, a clinic or online. |
| 4 | ML task | Binary classification (diabetes/prediabetes vs. no diabetes), imbalanced |
| 5 | Dataset | CDC Diabetes Health Indicators (cleaned BRFSS 2015) |
| 6 | Source | UCI ML Repository id 891 (DOI 10.24432/C53919); Kaggle mirror by A. Teboul |
| 7 | Size | 253,680 rows × 21 features + target |
| 8 | Features | HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age (bucketed), Education, Income |
| 9 | Target | `Diabetes_binary` |
| 10 | Models | Logistic Regression, Random Forest, XGBoost |
| 11 | XAI | SHAP (TreeExplainer) + LR coefficients as a cross-check |
| 12 | Prototype | Streamlit questionnaire → risk probability → SHAP explanation |
| 13 | Difficulty | **Easy–Moderate** |
| 14 | ≥3 recent studies | **Yes ✅** (see §5) |
| 15 | Gap | Published results vary widely (AUC ≈ 0.82–0.83 vs. ≈ 0.99). Several papers do not say whether resampling happened before or after the split. Error analysis is weak. Few papers offer a reproducible prototype. |
| 16 | Contribution | A leakage-safe, imbalance-aware comparison of 3 models + SHAP + error analysis + working screening prototype |
| 17 | Viva Qs | "Why is your accuracy lower than paper X?", "Is self-reported data reliable?", "How did you avoid leakage?" |
| 18 | Risks | Self-reported data; the realistic ceiling is about AUC 0.82–0.83, so our results will look "lower" than inflated papers (we turn this into a strength) |
| 19 | Impl. difficulty | Low. Tabular data, no missing values, runs on a laptop |
| 20 | Why suitable | Large, clean and public. Many recent papers. Features are easy to explain. Ideal for SHAP. A simple prototype is enough. |

### Idea 2 — Credit Card Default Prediction (Finance)
| Item | Detail |
|---|---|
| Problem / why real | Banks must estimate which customers will miss next month's payment. Bad estimates cost money and affect credit access. |
| Task | Binary classification (default next month), imbalanced (~22% default ❌) |
| Dataset / source | *Default of Credit Card Clients* (Taiwan), UCI id 350. **30,000 rows, CC BY 4.0 ✅** |
| Features | Credit limit, sex, education, marriage, age, 6 months of repayment status, bill amounts and payments (23 features ❌) |
| Models / XAI | LR, RF, XGBoost / SHAP |
| Prototype | Streamlit customer form → default probability |
| Difficulty | Easy–Moderate |
| Recent studies | Likely yes (a 2025 SHAP-stability arXiv paper and a 2024 preprint were found). Not fully verified. |
| Gap / novelty | Fairness for sex/age and SHAP stability. A smaller gap, because this dataset is heavily studied. |
| Risks | The data is from 2005 (old). The `PAY_x` coding is confusing. Thousands of papers make novelty hard. |

### Idea 3 — Telecom Customer Churn Prediction (Business)
| Item | Detail |
|---|---|
| Dataset | IBM Telco Customer Churn (Kaggle), ~7,043 rows, ~20 features ❌ |
| Task / models | Binary churn; LR, RF, XGBoost; SHAP |
| Difficulty | Easy |
| Weakness | Small and synthetic-like. "Big data" is hard to justify. It is a very common student project, so many papers are low quality. |

### Idea 4 — Bank Term-Deposit Marketing Response (Business)
| Item | Detail |
|---|---|
| Dataset | UCI Bank Marketing, ~45,211 rows, 16 features ❌ |
| Task / models | Binary (subscribes yes/no); LR, RF, XGBoost; SHAP |
| Difficulty | Easy |
| Weakness | The `duration` feature leaks the target (known issue). Data is from 2008–2010. Real-world value is less compelling. |

### Idea 5 — Fake Job Posting Detection (NLP)
| Item | Detail |
|---|---|
| Dataset | EMSCAD "Real or Fake Job Posting", ~17,880 postings ❌ |
| Task / models | Text classification; TF-IDF + LR, Linear SVM, RF/XGBoost; LIME |
| Difficulty | Moderate |
| Weakness | Heavy imbalance (~5% fake ❌). Text preprocessing adds work. Finding comparison papers with the same split is harder. |

### Idea 6 — Stroke Risk Prediction (Healthcare)
| Item | Detail |
|---|---|
| Dataset | Kaggle Stroke Prediction, ~5,110 rows ❌ |
| Task / models | Binary; LR, RF, XGBoost; SHAP |
| Difficulty | Easy |
| Weakness | Very small, with very few positives (~5% ❌). The data's origin is unclear, so lecturers may question its validity. |

---

## 2. Ranking (best → worst for one Master's student)

| Rank | Idea | Marks potential | Dataset quality | Literature | Ease | Viva defensibility | Overall |
|---|---|---|---|---|---|---|---|
| 1 | **Diabetes BRFSS** | High | High (253k, no missing values, official CDC source) | Strong, recent (2024–2026) | Easy | High | ⭐ **BEST** |
| 2 | Credit card default | Med–High | Good, but 2005 data | Very strong, but saturated | Easy | Med–High | **SECOND** |
| 3 | Telco churn | Medium | Small | Many but weak | Very easy | Medium | **BACKUP** |
| 4 | Bank marketing | Medium | Leakage trap | Moderate | Easy | Medium | |
| 5 | Fake jobs (NLP) | Med–High | OK | Moderate | Moderate | Medium | |
| 6 | Stroke | Low–Med | Small, unclear origin | Many but weak | Easy | Low | |

- **BEST CHOICE:** Idea 1 — Diabetes risk screening (CDC BRFSS 2015)
- **SECOND BEST:** Idea 2 — Credit card default
- **BACKUP CHOICE:** Idea 3 — Telco churn

### Why the best choice wins
1. **Real "big-ish" data.** 253,680 records is a defensible size for a Big Data module (big enough to discuss scaling, small enough for a laptop).
2. **Clean.** The UCI page says there are no missing values, so there is no complicated imputation.
3. **Features a non-expert can explain.** These include BMI, blood pressure, age and general health. No medical jargon is needed.
4. **A real, defensible gap.** Recent papers on the *same* dataset report results ranging from AUC ≈ 0.83 to ≈ 0.99. That large spread is a genuine, checkable observation (see §5) and gives us a clear story: *"we measure honestly what this data can actually deliver."*
5. **Strong fit for SHAP and a Streamlit prototype.**

---

## 3. Dataset Verification (CDC Diabetes Health Indicators)

| Check | Finding | Status |
|---|---|---|
| Name | CDC Diabetes Health Indicators | ✅ |
| URL | https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators | ✅ |
| DOI | 10.24432/C53919 | ✅ |
| Kaggle mirror | https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset | ✅ exists |
| Original source | CDC BRFSS telephone survey (UCI links the BRFSS annual data page) | ✅ |
| Size | 253,680 instances | ✅ |
| Features | UCI says "21 features". Its description text also says "35 features" (an inconsistency on the page). | ⚠️ confirm when loading |
| Feature list | 10 names confirmed on the UCI page (HighBP … PhysActivity). The rest are from the Kaggle listing. | ⚠️ confirm when loading |
| Target | `Diabetes_binary`. UCI: "0 = no diabetes, 1 = prediabetes or diabetes" | ✅ (but see the note below) |
| Classes | 218,334 negative (86.1%) / 35,346 positive (13.9%), from our `src/data_loader.py` run | ✅ |
| Missing values | 0 (our run) | ✅ |
| Duplicates | 24,206 exact duplicate rows (9.5%) (our run) | ✅ |
| Value ranges | All match the codebook. MentHlth/PhysHlth go 0–30 (UCI text says "1-30"). | ✅ |
| Sensitive attributes | Sex, Income, Education (flagged by UCI) | ✅ |
| Release | Linked on UCI on 25-Sep-2023. Survey data is from 2015 (UCI points to the BRFSS 2014 page, so the survey year is ambiguous). The Kaggle version was published around 2021 ❌ | ⚠️ |
| Licence | UCI says "See linked dataset for licensing". BRFSS is US-government public data. Kaggle licence not checked. | ❌ verify |
| Used in research | Yes, widely | ✅ |
| ≥3 studies since 2020 | Yes, at least 5 found (§5) | ✅ |

> **Important label note.** UCI says label 1 = "prediabetes **or** diabetes", but at least one
> GitHub project claims prediabetes is coded 0 in the Kaggle binary file. **Before training, you
> must check** which file you use and how prediabetes is coded. Write the answer in the report.
> **Recommendation:** load the UCI version with `ucimlrepo` (`fetch_ucirepo(id=891)`) so the data
> source is citable and reproducible.

---

## 4. Literature (initial set)

Do **not** cite anything below until you have opened the paper yourself.

| # | Paper | Venue / Year | Data | Models | Reported result | Limitation / note | Status |
|---|---|---|---|---|---|---|---|
| P1 | Shechter, S. & Shomron, N. "Machine Learning for Diabetes Screening: Insights from the BRFSS and Pima Indians Diabetes Databases." DOI 10.29024/jsim.283 | J. Scientific Innovation in Medicine, 2026 (peer-reviewed) | BRFSS 2015 (253,680) + PIDD | LR, KNN, RF, GB, CatBoost, XGBoost | CatBoost **AUC 0.83, Acc 0.86**. XGBoost Acc 0.87. Recall-tuned LR: recall 0.97, acc 0.43, AUC 0.82. Uses SHAP. | Self-reported data. The BRFSS split method is not clearly stated. | ✅ from article page |
| P2 | Majcherek, D., Ciesielski, A., Sobczak, P. "AI-driven analysis of diabetes risk determinants in U.S. adults…" DOI 10.1371/journal.pone.0328655 | PLOS ONE, 2025 | BRFSS 2015 (253,680) | 18 models | Extra Trees **Acc > 90%, AUC 0.99** (abstract) | Very high compared with P1, P4 and P5. Check whether balancing happened before the split. | ⚠️ abstract only; full text not read |
| P3 | Islam, M. N. et al. "An Improved Ensemble-Based ML Model with Feature Optimization for Early Diabetes Prediction." arXiv:2512.02023 | TCCE-2025 conference / arXiv, 2025 | BRFSS 2015 | LR, RF, GB, XGB, LGBM, CatBoost, KNN, NB, SVC, stacking | Stacking: **Acc 0.9482, P 0.9293, R 0.9702, F1 0.9493, AUC 0.9895**. Plain LR AUC 0.8135. | SMOTE+Tomek used, but the paper does not say whether before or after the split. Its confusion matrix totals 77,724, which does not match a 20% test set of 253,680 (~50,736). | ✅ from arXiv HTML |
| P4 | Aich, A., Murshed, M. M., Wade, B., Hewage, S. "CopulaSMOTE…" arXiv:2506.17326 | arXiv preprint, 2025 | BRFSS 2015 Diabetes Health Indicators (253,680, 13.9% positive) | RF, GB, XGB, LR, MLP + oversamplers | Best AUC ≈ **0.828**. Best F1 ≈ **0.446** | Oversampling applied **only to the training fold** (leakage-safe) | ⚠️ setup verified; numbers need checking against the paper's table |
| P5 | (Authors ❌) "A Comparative Study of ML Approaches for Diabetes Risk Prediction: Insights from SHAP and Feature Importance." Preprints.org 202411.1265 | Preprint, 2024 (not peer-reviewed) | BRFSS | LR, XGBoost, RF | XGBoost **AUC 0.83**. SHAP: GenHlth, HighBP, BMI | Not peer-reviewed | ⚠️ |
| P6 | Allani, U. "Interactive Diabetes Risk Prediction Using Explainable ML: A Dash-Based Approach with SHAP, LIME…" arXiv:2505.05683 | arXiv, 2025 | BRFSS 2015 | LR, RF, XGB, LGBM, KNN, NN | LightGBM + undersampling had the best recall (no numbers in the abstract) | Includes a prototype (related work for our app) | ⚠️ abstract only |
| P7 | Horestani, F. J. & Owrang O, M. M. "Predicting Diabetes with ML Analysis of Income and Health Factors." arXiv:2404.13260 | arXiv, 2024 | BRFSS 2015 | not read | not read | | ⚠️ abstract only |
| P8 | Xie, Z., Nikolayeva, O., Luo, J., Li, D. "Building Risk Prediction Models for Type 2 Diabetes Using ML Techniques." DOI 10.5888/pcd16.190109 | Preventing Chronic Disease, 2019 | **BRFSS 2014** (138,146), a different subset | SVM, DT, LR, RF, NN, NB | AUC 0.718–0.795 | Background reference only, not a same-dataset comparison | ✅ |
| P9 | Burrows, N. R. et al. "Incidence of End-Stage Renal Disease Attributed to Diabetes…" MMWR | 2017 | — | — | — | The dataset's introductory paper on UCI | ✅ |
| P10 | Journal of Intelligent Decision Making & Information Science (2025): "Explainable ML-Based Decision Support for Diabetes Risk Screening Using Public Health Indicator Data" | 2025 | Kaggle BRFSS indicators | — | SHAP + LIME; top factors GenHlth, age, BMI, HighBP | PDF could not be read | ⚠️ |

**For the 15+ references in the final report**, also add method references (verify each yourself): Lundberg & Lee 2017 (SHAP, NeurIPS); Chen & Guestrin 2016 (XGBoost, KDD); Breiman 2001 (Random Forest); Pedregosa et al. 2011 (scikit-learn); Chawla et al. 2002 (SMOTE); Saito & Rehmsmeier 2015 (PR curves for imbalanced data); Kaufman et al. 2012 (leakage in data mining); the CDC BRFSS documentation; and the IDF Diabetes Atlas or CDC National Diabetes Statistics Report for motivation figures.

---

## 5. Research Gap (realistic, not exaggerated)

What we observed in the papers above (all checkable):
- On the **same** 253,680-row dataset, reported AUC ranges from **≈ 0.82–0.83** (P1, P4, P5) to **≈ 0.99** (P2, P3).
- P4 explicitly oversamples only the training data and gets ≈ 0.83. P3 does not say when SMOTE was applied, and its test-set size does not match a normal 20% split.
- *Our own arithmetic, still a hypothesis:* 77,724 is close to 20% of a deduplicated majority class doubled by SMOTE. That would fit oversampling **before** splitting. We **must not** claim this as fact. We only say "the evaluation protocol is not fully reported".
- Error analysis (who gets misclassified, and why) is rarely reported.
- Few papers release code or a working tool.

**Gap statement (draft):** *"Published results on the BRFSS 2015 Diabetes Health Indicators dataset differ substantially, and evaluation protocols (especially when class balancing is applied relative to the train/test split) are often not reported. There is a need for a transparent, leakage-safe benchmark of interpretable and ensemble models, with explanation and error analysis, delivered as a reproducible screening prototype."*

**Contribution (honest):** This is *not* a new algorithm. We provide (1) a leakage-safe, reproducible comparison of LR, RF and XGBoost; (2) imbalance handling and threshold selection on a validation set; (3) SHAP global and local explanations; (4) a structured error analysis; (5) a working Streamlit screening prototype; and (6) a critical comparison with 3+ published studies.

---

## 6. Final Project Definition

### 6.1 Title options
1. *Explainable and Leakage-Safe Machine Learning for Diabetes Risk Screening Using CDC BRFSS Health Indicators*
2. *A Reproducible Comparison of Interpretable and Ensemble Models for Survey-Based Diabetes Risk Prediction*
3. *From Survey to Screening: An Explainable Diabetes Risk Prediction Prototype on 250,000 BRFSS Records*

### 6.2 Problem statement
A large share of adults with diabetes or prediabetes are unaware of their condition (find and cite a CDC/IDF figure). Laboratory tests are accurate but need a clinic visit. Self-reported lifestyle and health indicators are cheap to collect, but it is unclear how well ML can screen with them alone. Published results on the same benchmark dataset are inconsistent, and evaluation protocols are not always transparent. This project builds and evaluates a transparent, leakage-safe screening model and prototype.

### 6.3 Motivation
- Early detection allows lifestyle changes and treatment before complications.
- Questionnaire screening is scalable and low-cost.
- Clinicians and users need to see *why* someone is flagged (explainability).
- Inflated accuracy claims can mislead practitioners, so honest evaluation matters.

### 6.4 Aim
To develop and evaluate an explainable, leakage-safe ML pipeline and web prototype that estimates diabetes risk from self-reported health indicators, and to compare it fairly against recent published studies.

### 6.5 Objectives (measurable)
1. Carry out EDA on the 253,680-record dataset and document class balance, duplicates and key feature relationships.
2. Train and tune Logistic Regression, Random Forest and XGBoost on a stratified train/validation/test split, fitting all preprocessing on training data only.
3. Evaluate the models on the untouched test set with ROC-AUC, PR-AUC, recall, precision, F1 and a confusion matrix, and compare them with ≥3 published studies.
4. Explain the best model with SHAP (global and per-person) and analyse false negatives and false positives.
5. Deliver a Streamlit prototype that returns risk probability, a risk band and an explanation, with a README for full reproducibility.

### 6.6 Research questions
- **RQ1:** Which of LR, RF and XGBoost gives the best discrimination (ROC-AUC, PR-AUC) and minority-class recall on this dataset under a leakage-safe protocol?
- **RQ2:** How much does an ensemble model actually improve on an interpretable baseline (LR)?
- **RQ3:** Which indicators drive predictions according to SHAP, and do they agree with known clinical risk factors?
- **RQ4:** Which groups or feature patterns account for most misclassifications, and how do our results compare with published studies?

### 6.7 Scope
**Included:** the binary target, 3 models, class weighting and threshold tuning, SHAP, error analysis, a Streamlit app, and a scalability benchmark (training time on 25/50/100% of the data).
**Excluded:** deep learning, the 3-class target (mentioned only as future work), clinical validation, deployment to the cloud, real patient data, and LLMs.

---

## 7. Machine Learning Approach

| Role | Model | Why |
|---|---|---|
| **Baseline + interpretable** | Logistic Regression (`class_weight='balanced'`) | Simple and fast. Coefficients show direction of risk. A strong baseline in medicine. |
| **Bagging ensemble** | Random Forest | Captures non-linear patterns. Robust. Easy to explain as "many trees vote". |
| **Stronger boosting model** | XGBoost (`scale_pos_weight`) | Usually the best on tabular data. Works directly with SHAP TreeExplainer. |
| (Reference only) | Dummy "always negative" classifier | Shows why accuracy alone is misleading (≈86% accuracy with 0 recall) |

No deep learning: on 21 mostly binary tabular features, tree ensembles usually match or beat neural networks, are cheaper, and are easier to explain.

**Imbalance strategy (simple and leakage-safe):**
1. Main approach: class weights, which create no synthetic data and have no leakage risk.
2. Optional ablation: SMOTE inside an `imblearn.Pipeline`, so it runs only on the training folds.
3. Choose the decision threshold on the **validation** set (e.g. to reach recall ≥ 0.75, or maximum F1). Never choose it on the test set.

---

## 8. Data Processing Pipeline

```
Raw data (UCI 891, via ucimlrepo)
 → Load & schema check (dtypes, value ranges, e.g. BMI 12–98, Age 1–13)
 → Quality check (missing values, duplicates, impossible values)
 → EDA (on the full data, descriptive only; no decisions that use the test labels)
 → Duplicate handling (decision documented below)
 → Outliers: keep BMI extremes (real people); optionally cap BMI at the 99.5th percentile on TRAIN only
 → Encoding: none needed (already numeric). Treat GenHlth/Age/Education/Income as ordinal.
 → Feature engineering (light): e.g. BMI category (WHO bands), "number of chronic conditions" (HighBP+HighChol+Stroke+HeartDisease)
 → STRATIFIED SPLIT 70 / 15 / 15  (train / validation / test), random_state fixed
 → Fit scaler (LR only) on TRAIN → apply to val/test (sklearn Pipeline)
 → Feature selection: compare all 21 vs. a reduced set chosen by permutation importance on VALIDATION
 → Training + tuning: RandomizedSearchCV with stratified 5-fold CV on TRAIN only
 → Threshold selection on VALIDATION
 → Final evaluation on TEST, once
 → SHAP on the test set
 → Save pipeline (joblib) → Streamlit app
```

**Leakage rules (memorise these for the viva):**
- Split **first**. Scaling, SMOTE, feature selection and threshold tuning see training/validation data only.
- Put preprocessing inside an sklearn `Pipeline`, so cross-validation refits it on each fold.
- **Duplicates:** identical rows can fall into both train and test, which inflates scores. Recommended: drop exact duplicates *before* splitting and report how many there were. As a check, also report test results with duplicates kept, to show the effect.
- Open the test set once, at the end.

---

## 9. EDA Plan

**Essential for the report (6):**
1. Target class distribution (bar chart with %), which shows the imbalance
2. Data quality table (missing values, duplicates, value ranges)
3. Diabetes rate by GenHlth, Age, and HighBP (grouped bar charts)
4. BMI distribution by class (box or violin plot)
5. Correlation heatmap (Spearman, since the features are ordinal/binary)
6. Diabetes rate by Income/Education (links to the fairness discussion)

**Optional / appendix:** histograms of all features, a before/after deduplication comparison, PhysHlth/MentHlth distributions.

---

## 10. Evaluation Plan

- **Primary metrics** (because of imbalance): **ROC-AUC, PR-AUC, recall (sensitivity) of the positive class, F1**
- **Secondary:** precision, specificity, accuracy, confusion matrix, training time
- **Why accuracy is not enough:** a model that predicts "no diabetes" for everyone already gets ≈86% accuracy.
- **Plots:** ROC curves (3 models on one plot), PR curves, 3 confusion matrices, a calibration curve for the best model (optional)
- **Statistical check (optional, simple):** 95% bootstrap confidence intervals for test AUC

### Final comparison table (fill in only after experiments)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time (s) |
|---|---|---|---|---|---|---|---|
| Dummy | [RESULT AFTER TRAINING] | | | | | | |
| Logistic Regression | [RESULT AFTER TRAINING] | | | | | | |
| Random Forest | [RESULT AFTER TRAINING] | | | | | | |
| XGBoost | [RESULT AFTER TRAINING] | | | | | | |

---

## 11. XAI Plan (one primary method: SHAP)

- **Why:** screening decisions affect people. We must show *which answers* raised a person's risk, and check that the model relies on sensible factors.
- **What SHAP does (simply):** it splits each prediction into contributions from each feature. The contributions add up from the average risk to this person's risk.
- **How:** `shap.TreeExplainer` on XGBoost, using a sample of ~2,000 test rows (fast).
- **Visuals:** (1) global bar plot of mean |SHAP|; (2) beeswarm summary plot; (3) waterfall plots for one true positive, one false negative and one false positive; (4) in the app, a per-user waterfall or bar chart.
- **Cross-check:** compare with LR coefficients and RF impurity importance. Agreement increases trust.
- **Limitations:** SHAP explains the *model*, not biology. It shows association, not causation. Correlated features (e.g. GenHlth/PhysHlth/DiffWalk) share credit unpredictably.

---

## 12. Comparison With Published Studies (template)

| Study | Year | Dataset | Model | Acc | Prec | Rec | F1 | ROC-AUC | Our result | Difference |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 Shechter & Shomron | 2026 | BRFSS 2015 (253,680) | CatBoost | 0.86 | n/s | n/s | n/s | 0.83 | [RESULT] | [ ] |
| P3 Islam et al. | 2025 | BRFSS 2015 | Stacking (XGB+KNN) | 0.9482 | 0.9293 | 0.9702 | 0.9493 | 0.9895 | [RESULT] | [ ] |
| P4 Aich et al. | 2025 | BRFSS 2015 (253,680) | GB + oversampling | n/s | n/s | n/s | ≈0.446* | ≈0.828* | [RESULT] | [ ] |
| P2 Majcherek et al. | 2025 | BRFSS 2015 (253,680) | Extra Trees | >0.90 | n/s | n/s | n/s | 0.99 | [RESULT] | [ ] |
| P5 Preprint | 2024 | BRFSS | XGBoost | n/s | n/s | n/s | n/s | 0.83 | [RESULT] | [ ] |

n/s = not stated in the part we read. \* = confirm against the paper's table.

**Explain differences by:** split ratio, deduplication, *when* resampling was done (before or after the split), balanced vs. imbalanced test set, threshold choice, feature selection, and hyperparameters. **Never claim "better" unless the numbers show it under comparable conditions.** It is fine, and stronger, to say: *"our AUC matches the leakage-safe studies (P1, P4); the much higher reported values may come from evaluating on a resampled test set, which is not comparable."*

---

## 13. System Architecture

```
            ┌──────────────────── OFFLINE (training) ────────────────────┐
 UCI/CDC ──►│ Load → Quality check → EDA → Split 70/15/15 → Pipeline     │
 BRFSS 2015 │ (scale/weights) → Train LR/RF/XGB → Tune (CV on train)      │
            │ → Threshold (val) → Test eval → SHAP → save model.joblib    │
            └───────────────────────────────┬─────────────────────────────┘
                                            ▼
 User ─► Streamlit form ─► Input validation ─► Same saved pipeline ─► XGBoost
                                                                   │
            Result dashboard ◄─ SHAP explanation ◄─ Probability + risk band
```
- **Web interface:** questionnaire with friendly labels (e.g. an "Age group" dropdown instead of 1–13).
- **Input validation:** ranges, required fields, BMI calculated from height/weight.
- **Preprocessing pipeline:** the *same* fitted sklearn pipeline used in training, so there is no training/serving mismatch.
- **Model → probability → risk band** (Low / Moderate / High, using the validation-chosen threshold).
- **SHAP explanation:** the top factors raising or lowering this person's risk.
- **Dashboard:** probability, band, explanation, and a clear "not a diagnosis" disclaimer.

---

## 14. Web Prototype (Streamlit)

Page 1, **Risk check**: form → Predict → probability gauge/bar, risk band, SHAP chart, plain-English summary (e.g. "high blood pressure and BMI 34 increased your estimated risk").
Page 2, **Model info**: comparison table, ROC curve, global SHAP plot (loaded from `results/`).
Optional: CSV upload for batch prediction.
Not included: login, database, Docker, cloud.

---

## 15. Big Data Platform (CLO3)

**Current implementation:** about 253k rows × 22 columns of small integers, which is **tens of MB in memory**. pandas + scikit-learn + XGBoost on one laptop is the right tool. Spark would add cluster overhead with no benefit at this size. *We show this with evidence:* record training time at 25/50/100% of the data and peak memory use.

**Advantages of the current stack:** free, simple, reproducible, rich ML/XAI ecosystem. **Disadvantages:** limited to one machine's RAM, no distributed training, no streaming.

**Future scalability:**
- **100× data (~25M rows, e.g. all BRFSS years):** Parquet storage, then Polars/Dask or **Spark (PySpark MLlib)** for preprocessing and training, or XGBoost's distributed mode. Managed options: Databricks or cloud ML services.
- **Streaming (e.g. online questionnaires):** Kafka → Spark Structured Streaming or a lightweight API → the saved model scores each record. Monitor data drift. Retrain on a schedule.
- **The 5 Vs:** volume (moderate), velocity (batch annual survey), variety (structured), veracity (self-reported), value (screening).

---

## 16. Error Analysis Plan

1. **False negatives** (missed diabetics, the most harmful error): profile them by BMI, Age, GenHlth and HighBP. Expected: young, normal-BMI people who report good health.
2. **False positives:** older people with high BP and BMI but no diabetes. These are plausibly high-risk/prediabetic, a useful discussion point.
3. **Errors by subgroup:** recall and FPR by Sex, Age group and Income band (a fairness check).
4. **Low-confidence predictions:** probability within ±0.1 of the threshold. Measure their share and their error rate.
5. **SHAP on misclassified cases:** use waterfall plots to show which features misled the model.
**Conclusion type:** what the data *cannot* capture (no glucose or HbA1c) limits recall for "hidden" cases.

---

## 17. Ethical Considerations (relevant only)

- **Privacy:** data is public and anonymised. The app stores no user inputs.
- **Bias/fairness:** Sex, Income and Education are sensitive. Report subgroup performance, and consider training without Income/Education as a check.
- **Misuse:** this is a screening aid, **not a diagnosis**. The app shows a disclaimer and recommends a blood test.
- **Self-report bias:** some people with diabetes may not know it, so some labels are wrong.
- **Generalisability:** US adults, 2015 only.
- **Transparency:** SHAP and open code.
- **Licence:** BRFSS is public US-government data. Cite UCI and CDC.

---

## 18. Report Outline (IEEE two-column, ~12–13 pages)

| Section | Pages | Notes |
|---|---|---|
| Title + Abstract | 0.3 | 150–200 words |
| 1 Introduction (background, problem, motivation, objectives, RQs, contribution) | 1.2 | |
| 2 Literature Review (approaches, comparison table, gap) | 2.0 | Table of P1–P7 |
| 3 Methodology (dataset, EDA, preprocessing, features, split, models, tuning, SHAP, metrics) | 2.5 | Pipeline figure |
| 4 System Design (architecture, data flow, prototype) | 1.0 | Architecture diagram |
| 5 Implementation (tech, code structure, saved model, UI) | 0.8 | App screenshot |
| 6 Results (EDA findings, model table, confusion matrices, ROC/PR, SHAP, literature comparison, error analysis) | 3.0 | Most figures go here |
| 7 Discussion (findings, why the best model won, limitations, platform/CLO3, scalability, ethics) | 1.5 | |
| 8 Conclusion & Future Work | 0.4 | |
| References (15+), Appendices | not counted | |

---

## 19. Presentation Plan (7 minutes)

| Slide | Show | Say (simple) | Time | Likely question |
|---|---|---|---|---|
| 1 Title | Title, name, module | "I built an explainable diabetes risk screening tool from CDC survey data." | 0:10 | — |
| 2 Problem & Motivation | Undiagnosed-diabetes statistic (cited), screening idea | "Many people don't know they have diabetes. Can a 21-question survey flag them?" | 0:50 | "Why survey data and not blood tests?" |
| 3 Objectives & Gap | Spread of published AUCs (0.83 vs 0.99) | "Same dataset, very different results. I wanted an honest, leakage-safe benchmark." | 0:40 | "Are you saying those papers are wrong?" |
| 4 Dataset & Method | 253,680 rows, 13.9% positive, pipeline arrow | "Split first, then fit everything on training data only." | 0:30 | "How did you handle duplicates?" |
| 5 ML Models | LR / RF / XGB roles | "A baseline, a bagging model and a boosting model." | 0:20 | "Why not deep learning?" |
| 6 Architecture | Diagram | "Offline training, saved pipeline, Streamlit app." | 0:10 | — |
| 7 Live Demo | App: 2 profiles (low and high risk) + SHAP | Walk through inputs and the explanation | 2:30 | "What if I enter BMI 200?" |
| 8 Results | Metrics table, ROC curves, literature comparison | "XGBoost best AUC [X]. In line with leakage-safe studies." | 1:15 | "Why is your accuracy lower than 94%?" |
| 9 XAI & Errors | SHAP beeswarm, FN profile | "GenHlth, BMI, HighBP, Age matter most. Missed cases look healthy." | 0:45 | "Does SHAP prove causation?" |
| 10 Conclusion | 3 takeaways + limitations | "Honest evaluation, explainable, working tool. Not a diagnosis." | 0:20 | "What next?" |

Keep a recorded demo video as a backup in case the live demo fails.

---

## 20. Viva Questions (30)

Format: **Short** · **Detailed** · ❌ *Don't say*

**A. Problem/domain**
1. *Why this problem?* · **Short:** Undiagnosed diabetes is common, and survey screening is cheap. · **Detailed:** Early detection prevents complications. A questionnaire can be used at scale before a lab test. · ❌ "It is an easy dataset."
2. *Who would use it?* · **Short:** Public-health screening and pharmacies. · **Detailed:** It flags people to refer for an HbA1c/glucose test. It does not replace a doctor. · ❌ "Doctors can use it to diagnose."
3. *Why include prediabetes in the positive class?* · **Short:** That is how the dataset defines label 1 (to be confirmed). Both groups benefit from early action. · ❌ "I didn't check."

**B. Dataset**
4. *Why this dataset?* · **Short:** Large, official CDC source, clean, and used by many recent papers. · **Detailed:** 253,680 records, no missing values, ≥5 papers from 2024–26 to compare with. · ❌ "It was the first Kaggle result."
5. *Is self-reported data reliable?* · **Short:** It has bias, which we discuss as a limitation. · **Detailed:** Recall bias and undiagnosed cases create noisy labels. This limits the achievable AUC to ~0.83. · ❌ "It is perfectly accurate."
6. *Data from 2015 is old. Is that a problem?* · **Short:** Risk factors are stable, but prevalence has changed. We list it as a limitation. · ❌ "Age doesn't matter."
7. *How imbalanced is it?* · **Short:** About 14% positive (from our EDA: [value]). · ❌ "It's balanced."

**C. Preprocessing**
8. *Duplicates?* · **Short:** Dropped before the split, with the count reported, to prevent train/test overlap. · ❌ "There were none" (unless the check confirms it).
9. *Why no one-hot encoding?* · **Short:** Features are binary or ordinal integers already. Trees handle ordinal values natively. · ❌ "I forgot."
10. *Why scale only for LR?* · **Short:** LR is sensitive to feature scale. Trees are not. · ❌ "Scaling improves every model."
11. *Feature engineering?* · **Short:** BMI category and a comorbidity count. Kept only if they help on validation. · ❌ "Engineered features always help."

**D. Models**
12. *Why these three?* · **Short:** An interpretable baseline, bagging, and boosting give a fair spectrum. · ❌ "They are popular."
13. *Why not deep learning?* · **Short:** Small tabular feature set. Trees match or beat NNs here and are explainable. · ❌ "DL is too hard for me."
14. *RF vs XGBoost, simply?* · **Short:** RF builds independent trees and averages them. XGBoost builds trees one after another, each fixing earlier errors. · ❌ Long equations.
15. *How was tuning done?* · **Short:** RandomizedSearchCV, 5-fold stratified CV, training set only. · ❌ "I tuned on the test set."
16. *Which model would you deploy?* · **Short:** The best on validation AUC/recall with acceptable speed, likely XGBoost. LR if the difference is tiny, for transparency. · ❌ "The one with highest accuracy."

**E. Evaluation**
17. *Why not accuracy?* · **Short:** Always predicting "no" gives ~86%. · ❌ "Accuracy is the best metric."
18. *ROC-AUC vs PR-AUC?* · **Short:** PR-AUC focuses on the minority class, so it is more informative under imbalance. · ❌ "They are the same."
19. *How was the threshold chosen?* · **Short:** On the validation set, to meet the recall target. · ❌ "0.5 by default" or "on the test set".
20. *Can results be trusted?* · **Short:** Leakage-safe protocol, one-time test evaluation, CIs, code released. · ❌ "100% trustworthy."

**F. XAI**
21. *Why SHAP?* · **Short:** Consistent per-person and global explanations. Fast for trees. · ❌ "It's trendy."
22. *Does SHAP show causation?* · **Short:** No. It shows how the model uses features. · ❌ "BMI causes diabetes according to SHAP."
23. *What if SHAP disagrees with the LR coefficients?* · **Short:** That can come from non-linearity or correlated features. We discuss it. · ❌ "One of them is wrong."

**G. Literature**
24. *Why are your results lower than 94–99%?* · **Short:** Our test set is real and imbalanced, with no synthetic rows. Our result matches the leakage-safe studies. · ❌ "Those authors cheated."
25. *What is your research gap?* · **Short:** Inconsistent, under-reported evaluation on this dataset, plus little error analysis and few prototypes. · ❌ "Nobody has ever done this."
26. *What is your novelty?* · **Short:** A transparent benchmark + XAI + error analysis + prototype, not a new algorithm. · ❌ "A new model."

**H. Big Data**
27. *Why is this a Big Data project?* · **Short:** It has a quarter-million records, and we evaluate the platform choice with a measured scaling test. We also design for 100× data and streaming. · ❌ "Because it's big."
28. *Why not Spark?* · **Short:** It fits in memory. Spark adds overhead. Our benchmark shows a laptop is enough today. · ❌ "Spark is useless."
29. *Streaming?* · **Short:** Kafka → model service → drift monitoring → periodic retraining. · ❌ "Just retrain every second."

**I. Prototype**
30. *What if a user enters invalid data?* · **Short:** Validation limits ranges and dropdowns limit choices. BMI is calculated from height and weight. · ❌ "It will crash."
31. *Does the app use the same preprocessing as training?* · **Short:** Yes, the same saved sklearn pipeline. · ❌ "I re-coded it in the app."

**J. Limitations/Ethics**
32. *Biggest weakness?* · **Short:** No lab features, so recall is limited for "silent" cases. Self-reported US data from 2015. · ❌ "No weaknesses."
33. *What happens when the model is wrong?* · **Short:** A false negative may delay a test, so the app tells everyone with risk factors to get tested. A false positive costs one blood test. · ❌ "It's never wrong."
34. *Fairness?* · **Short:** We report recall by sex, age and income group, and discuss any gaps. · ❌ "ML is objective."

### Key defence answers (the 20 must-prepare questions)
1–3 (problem, dataset, models): see Q1, Q4, Q12.
4 Deep learning: Q13.
5 Big Data: Q27 + the scaling benchmark figure.
6–7 Gap/novelty: Q25–26.
8 *"Is your model better?"* We don't claim it is. We claim *comparable to leakage-safe work and more transparent*.
9 Leakage: split first, Pipeline, deduplicate, threshold on validation, test used once.
10–11 Metrics: Q17–18.
12 SHAP: Q21.
13, 16 Limitations/weakness: Q32.
14 100× data: Parquet + Spark MLlib / distributed XGBoost.
15 Streaming: Q29.
17 Another semester: add the 3-class target, add more BRFSS years (temporal validation), calibration, fairness mitigation.
18 Deploy: Q16.
19 Trust: Q20.
20 Wrong prediction: Q33.

---

## 21. Implementation Roadmap

| Phase | Tasks | Output / files | You must understand | Common mistakes |
|---|---|---|---|---|
| 1 Topic & Dataset (Wk 1) | Confirm topic. Download via `ucimlrepo`. Check the label coding. | `data/raw/`, `src/data_loader.py` | What each feature means | Using the 50/50 balanced Kaggle file without saying so |
| 2 Literature (Wk 1–3) | Read P1–P7. Fill the comparison table. Collect 15+ refs. | `docs/literature_table.md` | Each paper's split and balancing method | Citing unread papers |
| 3 EDA (Wk 2–3) | The 6 essential plots + quality table | `notebooks/01_eda.ipynb`, `figures/eda_*` | Imbalance, duplicates | Too many plots with no interpretation |
| 4 Preprocessing (Wk 3–4) | Deduplicate, split, pipeline, features | `src/preprocess.py`, `data/processed/` | Why split first | Scaling before splitting |
| 5 Models (Wk 4–6) | LR/RF/XGB + tuning + thresholds | `src/train.py`, `models/*.joblib` | Hyperparameters you tuned | Tuning on test |
| 6 Evaluation (Wk 6–7) | Metrics, curves, bootstrap CI, scaling benchmark | `src/evaluate.py`, `results/metrics.csv` | Every metric | Reporting only accuracy |
| 7 XAI + errors (Wk 7–8) | SHAP plots, FN/FP analysis, subgroup checks | `src/explain.py`, `notebooks/03_error_analysis.ipynb` | How to read a beeswarm plot | Causal claims |
| 8 Prototype (Wk 8–9) | Streamlit app | `app/app.py` | How the app loads the pipeline | Re-implementing preprocessing in the app |
| 9 Report (Wk 9–12) | IEEE LaTeX/Word template | `report/` | Your own story | AI-written prose over 20% |
| 10 Presentation (Wk 12–13) | 10 slides + demo rehearsal + backup video | `presentation/` | Timings | Overrunning the demo |

---

## 22. Folder Structure

```
Big_Data_Project/
├── data/
│   ├── raw/            # original download (git-ignored, re-downloadable)
│   └── processed/      # train/val/test splits (git-ignored)
├── notebooks/          # 01_eda, 02_modelling, 03_error_analysis (exploration + figures)
├── src/                # reusable code: data_loader, preprocess, train, evaluate, explain, config
├── models/             # saved pipelines (.joblib)
├── app/                # Streamlit app
├── figures/            # PNGs used in the report/slides
├── results/            # metrics CSV/JSON, comparison tables
├── docs/               # planning notes, literature table
├── report/             # IEEE report source
├── presentation/       # slides
├── requirements.txt
└── README.md
```

## 23. Coding Plan

| Library | Used for |
|---|---|
| pandas, numpy | Data handling |
| ucimlrepo | Downloading UCI dataset 891 reproducibly |
| matplotlib, seaborn | EDA and result plots |
| scikit-learn | Split, Pipeline, LR, RF, CV, metrics |
| xgboost | Boosting model |
| imbalanced-learn | Optional SMOTE ablation (inside the pipeline) |
| shap | Explanations |
| joblib | Saving the model |
| streamlit | Web prototype |
| jupyter | Notebooks |

**Run order:** `src/data_loader.py` → `notebooks/01_eda.ipynb` → `src/preprocess.py` → `src/train.py` → `src/evaluate.py` → `src/explain.py` → `notebooks/03_error_analysis.ipynb` → `streamlit run app/app.py`.

We will build these **one script at a time**, each explained before moving on.

---

## 24. What YOU Must Verify Yourself
- [ ] The label coding of prediabetes in the file you use
- [ ] Full texts of P1–P7 (splits, resampling timing, exact numbers)
- [ ] The authors of P5 (Preprints 202411.1265)
- [ ] The dataset licence wording (UCI/Kaggle/CDC)
- [ ] All motivation statistics (CDC/IDF)
- [ ] All results come from your own runs
