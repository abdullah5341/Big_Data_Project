"""Streamlit prototype: diabetes risk screening with SHAP explanations.

Run from the project root:  streamlit run app/app.py
Needs models/xgboost.joblib (created by src/train.py, also committed to the repo).

The app reuses the training code's feature engineering and the saved sklearn pipeline,
so a user's answers go through exactly the same steps as the training data.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import (BMI_MAX, BMI_MIN, DEPLOYED_MODEL, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
                    UNCERTAIN_MARGIN)
from preprocess import add_engineered_features

# ---------- Questionnaire options (codes match the BRFSS codebook on UCI) ----------
AGE_GROUPS = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6, "50-54": 7,
              "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80 or older": 13}
GENERAL_HEALTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
EDUCATION = {"Never attended school / kindergarten": 1, "Grades 1-8": 2, "Grades 9-11": 3,
             "High school graduate / GED": 4, "Some college or technical school": 5, "College graduate": 6}
INCOME = {"Less than $10,000": 1, "$10,000-$14,999": 2, "$15,000-$19,999": 3, "$20,000-$24,999": 4,
          "$25,000-$34,999": 5, "$35,000-$49,999": 6, "$50,000-$74,999": 7, "$75,000 or more": 8}
YES_NO_QUESTIONS = {
    "HighBP": "Have you been told you have high blood pressure?",
    "HighChol": "Have you been told you have high cholesterol?",
    "CholCheck": "Have you had a cholesterol check in the past 5 years?",
    "Smoker": "Have you smoked at least 100 cigarettes in your life?",
    "Stroke": "Have you ever had a stroke?",
    "HeartDiseaseorAttack": "Have you had coronary heart disease or a heart attack?",
    "PhysActivity": "Any physical activity in the past 30 days (not counting your job)?",
    "Fruits": "Do you eat fruit at least once a day?",
    "Veggies": "Do you eat vegetables at least once a day?",
    "HvyAlcoholConsump": "Heavy drinking (men >14 drinks/week, women >7 drinks/week)?",
    "AnyHealthcare": "Do you have any health care coverage?",
    "NoDocbcCost": "In the past year, did cost stop you from seeing a doctor?",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
}
FRIENDLY_NAMES = {
    "GenHlth": "general health rating", "Age": "age group", "BMI": "BMI", "HighBP": "high blood pressure",
    "HighChol": "high cholesterol", "ComorbidityCount": "number of chronic conditions",
    "BMICategory": "BMI category", "DiffWalk": "difficulty walking", "Sex": "sex", "Income": "income",
    "Education": "education", "MentHlth": "poor mental health days", "PhysHlth": "poor physical health days",
    "CholCheck": "recent cholesterol check", "HvyAlcoholConsump": "heavy alcohol use",
    "PhysActivity": "physical activity", "Smoker": "smoking history", "Stroke": "stroke history",
    "HeartDiseaseorAttack": "heart disease history", "Fruits": "daily fruit", "Veggies": "daily vegetables",
    "AnyHealthcare": "health care coverage", "NoDocbcCost": "cost barrier to seeing a doctor",
}


@st.cache_resource
def load_model():
    bundle = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")
    explainer = shap.TreeExplainer(bundle["pipeline"].named_steps["model"])
    return bundle, explainer


def risk_band(score: float, threshold: float) -> tuple[str, str]:
    """Three bands instead of a hard yes/no, because scores near the threshold are often wrong."""
    if score >= threshold + UNCERTAIN_MARGIN:
        return "High risk", "error"
    if score <= threshold - UNCERTAIN_MARGIN:
        return "Low risk", "success"
    return "Borderline / uncertain", "warning"


def collect_answers() -> dict | None:
    """Render the questionnaire. Returns raw answers in dataset format, or None if not submitted."""
    with st.form("questionnaire"):
        col1, col2 = st.columns(2)
        with col1:
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            age = st.selectbox("Age group", list(AGE_GROUPS), index=8)
            gen_health = st.select_slider("In general, your health is…", list(GENERAL_HEALTH), value="Good")
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=1.0)
            ment_hlth = st.slider("Days of poor MENTAL health in the past 30 days", 0, 30, 0)
            phys_hlth = st.slider("Days of poor PHYSICAL health in the past 30 days", 0, 30, 0)
            education = st.selectbox("Education", list(EDUCATION), index=4)
            income = st.selectbox("Household income", list(INCOME), index=5)
        with col2:
            yes_no = {col: int(st.checkbox(question, value=col in {"CholCheck", "AnyHealthcare",
                                                                   "PhysActivity", "Fruits", "Veggies"}))
                      for col, question in YES_NO_QUESTIONS.items()}
        submitted = st.form_submit_button("Estimate my risk", type="primary")

    if not submitted:
        return None
    bmi = weight_kg / (height_cm / 100) ** 2
    if not BMI_MIN <= bmi <= BMI_MAX:
        st.error(f"Calculated BMI is {bmi:.1f}, outside the range the model was trained on "
                 f"({BMI_MIN}-{BMI_MAX}). Please check height and weight.")
        return None
    return {
        **yes_no, "BMI": round(bmi), "GenHlth": GENERAL_HEALTH[gen_health], "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth, "Sex": int(sex == "Male"), "Age": AGE_GROUPS[age],
        "Education": EDUCATION[education], "Income": INCOME[income],
    }


def explain_in_words(explanation: shap.Explanation, top_n: int = 3) -> tuple[list[str], list[str]]:
    contributions = pd.Series(explanation.values, index=explanation.feature_names).sort_values()
    increasing = [FRIENDLY_NAMES.get(f, f) for f in contributions[contributions > 0].index[::-1][:top_n]]
    decreasing = [FRIENDLY_NAMES.get(f, f) for f in contributions[contributions < 0].index[:top_n]]
    return increasing, decreasing


def show_result(answers: dict, bundle: dict, explainer: shap.TreeExplainer) -> None:
    features = add_engineered_features(pd.DataFrame([answers]))[bundle["features"]]
    score = float(bundle["pipeline"].predict_proba(features)[0, 1])
    threshold = bundle["threshold"]
    band, style = risk_band(score, threshold)

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk score", f"{score:.2f}")
    m2.metric("Decision threshold", f"{threshold:.2f}")
    m3.metric("BMI (calculated)", answers["BMI"])
    getattr(st, style)(f"**{band}**")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption("The risk score ranks people by risk; it is not a calibrated probability. The model was trained "
               "with class weights, which shifts scores upward. The threshold was chosen on validation data.")

    explanation = explainer(features)[0]
    increasing, decreasing = explain_in_words(explanation)
    st.subheader("Why this result?")
    if increasing:
        st.markdown(f"**Raised your score most:** {', '.join(increasing)}")
    if decreasing:
        st.markdown(f"**Lowered your score most:** {', '.join(decreasing)}")
    fig = plt.figure()
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close(fig)
    st.caption("Bars show how much each answer pushed the score up (red) or down (blue) from the average, "
               "in log-odds. These are associations learned from survey data, not causes.")


def batch_page(bundle: dict) -> None:
    st.write("Upload a CSV with the 21 original BRFSS indicator columns (same names and codes as the UCI dataset).")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is None:
        return
    data = pd.read_csv(uploaded)
    raw_features = [f for f in bundle["features"] if f not in {"BMICategory", "ComorbidityCount"}]
    missing = [c for c in raw_features if c not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    features = add_engineered_features(data[raw_features])[bundle["features"]]
    data["risk_score"] = bundle["pipeline"].predict_proba(features)[:, 1].round(4)
    data["risk_band"] = [risk_band(s, bundle["threshold"])[0] for s in data["risk_score"]]
    st.dataframe(data, use_container_width=True)
    st.download_button("Download results", data.to_csv(index=False), "risk_scores.csv", "text/csv")


def model_info_page() -> None:
    st.write("Results on the held-out test set (34,422 people, never used for training or tuning).")
    metrics_file = RESULTS_DIR / "test_metrics.csv"
    if metrics_file.exists():
        st.dataframe(pd.read_csv(metrics_file).drop(columns=["tp", "fp", "tn", "fn"]), hide_index=True)
    for title, image in [("ROC curves", "eval_01_roc_curves.png"),
                         ("What drives the model (SHAP)", "xai_02_shap_beeswarm.png"),
                         ("Confusion matrices", "eval_03_confusion_matrices.png")]:
        if (FIGURES_DIR / image).exists():
            st.markdown(f"**{title}**")
            st.image(str(FIGURES_DIR / image))


def main() -> None:
    st.set_page_config(page_title="Diabetes Risk Screening", layout="wide")
    st.title("Diabetes Risk Screening")
    st.warning("Research prototype for screening support only. This is **not a diagnosis**. "
               "Anyone concerned about diabetes should take a blood test (HbA1c or fasting glucose).")

    bundle, explainer = load_model()
    page = st.sidebar.radio("Page", ["Risk check", "Batch scoring (CSV)", "Model performance"])
    st.sidebar.caption("Model: XGBoost trained on CDC BRFSS 2015 (UCI dataset 891). No answers are stored.")

    if page == "Risk check":
        answers = collect_answers()
        if answers is not None:
            show_result(answers, bundle, explainer)
    elif page == "Batch scoring (CSV)":
        batch_page(bundle)
    else:
        model_info_page()


if __name__ == "__main__":
    main()
