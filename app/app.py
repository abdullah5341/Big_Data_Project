"""
Web app (Streamlit): diabetes risk screening with an explanation of the result.

Run from the project folder:   streamlit run app/app.py
It uses models/xgboost.joblib (made by src/train.py and included in the repo).

The user's answers go through the SAME feature function and the SAME saved model as the
training data, so the app always behaves exactly like the model we evaluated.
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# Let the app import our own code from the src/ folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import (BMI_MAX, BMI_MIN, DEPLOYED_MODEL, FIGURES_DIR, MODELS_DIR, RESULTS_DIR,
                    UNCERTAIN_MARGIN)
from preprocess import add_engineered_features

# ---------- Answer options. The numbers are the codes used in the dataset (UCI codebook). ----------
AGE_GROUPS = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6, "50-54": 7,
              "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80 or older": 13}
GENERAL_HEALTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
EDUCATION = {"Never attended school / kindergarten": 1, "Grades 1-8": 2, "Grades 9-11": 3,
             "High school graduate / GED": 4, "Some college or technical school": 5, "College graduate": 6}
INCOME = {"Less than $10,000": 1, "$10,000-$14,999": 2, "$15,000-$19,999": 3, "$20,000-$24,999": 4,
          "$25,000-$34,999": 5, "$35,000-$49,999": 6, "$50,000-$74,999": 7, "$75,000 or more": 8}

# Yes/no questions: dataset column -> question shown to the user.
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
# These start ticked, because most people answer "yes" to them.
TICKED_BY_DEFAULT = ["CholCheck", "AnyHealthcare", "PhysActivity", "Fruits", "Veggies"]

# Plain-English names for the explanation text.
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


@st.cache_resource   # load the model only once, not on every click
def load_model():
    saved = joblib.load(MODELS_DIR / f"{DEPLOYED_MODEL}.joblib")
    explainer = shap.TreeExplainer(saved["pipeline"].named_steps["model"])
    return saved, explainer


def get_risk_band(score, threshold):
    """Three bands instead of a hard yes/no, because scores near the threshold are often wrong."""
    if score >= threshold + UNCERTAIN_MARGIN:
        return "High risk"
    if score <= threshold - UNCERTAIN_MARGIN:
        return "Low risk"
    return "Borderline / uncertain"


def show_questionnaire():
    """Show the form. Returns the answers as dataset codes, or None until the button is pressed."""
    with st.form("questionnaire"):
        left, right = st.columns(2)
        with left:
            sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
            age = st.selectbox("Age group", list(AGE_GROUPS), index=8)   # starts at "60-64"
            general_health = st.select_slider("In general, your health is…", list(GENERAL_HEALTH), value="Good")
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=1.0)
            mental_days = st.slider("Days of poor MENTAL health in the past 30 days", 0, 30, 0)
            physical_days = st.slider("Days of poor PHYSICAL health in the past 30 days", 0, 30, 0)
            education = st.selectbox("Education", list(EDUCATION), index=4)
            income = st.selectbox("Household income", list(INCOME), index=5)
        with right:
            answers = {}
            for column, question in YES_NO_QUESTIONS.items():
                ticked = st.checkbox(question, value=column in TICKED_BY_DEFAULT)
                answers[column] = 1 if ticked else 0
        submitted = st.form_submit_button("Estimate my risk", type="primary")

    if not submitted:
        return None

    # Input validation: the model only knows BMIs between 12 and 98.
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < BMI_MIN or bmi > BMI_MAX:
        st.error(f"Calculated BMI is {bmi:.1f}, outside the range the model was trained on "
                 f"({BMI_MIN}-{BMI_MAX}). Please check height and weight.")
        return None

    # Convert every answer to the code the dataset uses.
    answers["BMI"] = round(bmi)   # the dataset stores BMI as a whole number
    answers["GenHlth"] = GENERAL_HEALTH[general_health]
    answers["MentHlth"] = mental_days
    answers["PhysHlth"] = physical_days
    answers["Sex"] = 1 if sex == "Male" else 0
    answers["Age"] = AGE_GROUPS[age]
    answers["Education"] = EDUCATION[education]
    answers["Income"] = INCOME[income]
    return answers


def show_result(answers, saved, explainer):
    # Same steps as in training: add the two new features, then use the training column order.
    person = add_engineered_features(pd.DataFrame([answers]))
    person = person[saved["features"]]

    score = float(saved["pipeline"].predict_proba(person)[0, 1])
    threshold = saved["threshold"]
    band = get_risk_band(score, threshold)

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{score:.2f}")
    col2.metric("Decision threshold", f"{threshold:.2f}")
    col3.metric("BMI (calculated)", answers["BMI"])

    if band == "High risk":
        st.error(f"**{band}**")
    elif band == "Low risk":
        st.success(f"**{band}**")
    else:
        st.warning(f"**{band}**")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption("The risk score ranks people by risk; it is not an exact probability. The model was "
               "trained with class weights, which pushes scores upwards. The threshold was chosen on "
               "validation data.")

    # Explanation: how much each answer pushed this person's score up or down.
    shap_values = explainer(person)[0]
    pushes = pd.Series(shap_values.values, index=person.columns)
    raised = pushes[pushes > 0].sort_values(ascending=False).head(3)
    lowered = pushes[pushes < 0].sort_values().head(3)

    st.subheader("Why this result?")
    if len(raised) > 0:
        names = [FRIENDLY_NAMES[feature] for feature in raised.index]
        st.markdown(f"**Raised your score most:** {', '.join(names)}")
    if len(lowered) > 0:
        names = [FRIENDLY_NAMES[feature] for feature in lowered.index]
        st.markdown(f"**Lowered your score most:** {', '.join(names)}")

    shap.plots.waterfall(shap_values, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")
    st.caption("Red bars pushed the score up and blue bars pushed it down, compared with the average "
               "person (in log-odds). These are patterns learned from survey data, not causes.")


def show_model_performance():
    st.write("Results on the held-out test set (34,422 people, never used for training or tuning).")
    metrics_file = RESULTS_DIR / "test_metrics.csv"
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file).drop(columns=["tp", "fp", "tn", "fn"])
        st.dataframe(metrics, hide_index=True)

    figures = [
        ("ROC curves", "eval_01_roc_curves.png"),
        ("What drives the model (SHAP)", "xai_02_shap_beeswarm.png"),
        ("Confusion matrices", "eval_03_confusion_matrices.png"),
    ]
    for title, file_name in figures:
        image_path = FIGURES_DIR / file_name
        if image_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(image_path))


def main():
    st.set_page_config(page_title="Diabetes Risk Screening", layout="wide")
    st.title("Diabetes Risk Screening")
    st.warning("Research prototype for screening support only. This is **not a diagnosis**. "
               "Anyone concerned about diabetes should take a blood test (HbA1c or fasting glucose).")

    saved, explainer = load_model()
    page = st.sidebar.radio("Page", ["Risk check", "Model performance"])
    st.sidebar.caption("Model: XGBoost trained on CDC BRFSS 2015 (UCI dataset 891). No answers are stored.")

    if page == "Risk check":
        answers = show_questionnaire()
        if answers is not None:
            show_result(answers, saved, explainer)
    else:
        show_model_performance()


if __name__ == "__main__":
    main()
