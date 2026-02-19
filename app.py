import streamlit as st
import pandas as pd
from predict import predict

st.set_page_config(page_title="Thyroid Risk Predictor")

st.title("🩺 Thyroid Risk Prediction System")
st.write("Enter patient details below:")

age = st.number_input("Age", min_value=15, max_value=90)
gender = st.selectbox("Gender", ["Male", "Female"])
familyhistory = st.selectbox("Family history of thyroid disorders", ["Yes", "No"])
pregnant = st.selectbox("Pregnant", ["Yes", "No"])

st.write("---")
st.subheader("🏥 Thyroid Evaluation Form")
st.write("Tick all the symptoms you are currently facing:")

Goiter = st.checkbox("Goiter")
Fatigue = st.checkbox("Fatigue")
Weight_Change = st.checkbox("Weight Change")
Hair_Loss = st.checkbox("Hair Loss")
Heart_Rate_Changes = st.checkbox("Heart Rate Changes")
Sensitivity_to_Cold_or_Heat = st.checkbox("Sensitivity to Cold or Heat")
Increased_Sweating = st.checkbox("Increased Sweating")
Muscle_Weakness = st.checkbox("Muscle Weakness")
Constipation_or_More_Bowel_Movements = st.checkbox(
    "Constipation or More Bowel Movements"
)
Depression_or_Anxiety = st.checkbox("Depression or Anxiety")
Difficulty_Concentrating_or_Memory_Problems = st.checkbox(
    "Difficulty Concentrating or Memory Problems"
)
Dry_or_Itchy_Skin = st.checkbox("Dry or Itchy Skin")

# Map inputs to numeric format
user_input = {
    "Age": (age - 15) / (90 - 15),  # scaled 0-1
    "Gender": 1 if gender == "Male" else 0,
    "Pregnancy": 1 if pregnant == "Yes" else 0,
    "Family_History_of_Thyroid": 1 if familyhistory == "Yes" else 0,
    "Goiter": int(Goiter),
    "Fatigue": int(Fatigue),
    "Weight_Change": int(Weight_Change),
    "Hair_Loss": int(Hair_Loss),
    "Heart_Rate_Changes": int(Heart_Rate_Changes),
    "Sensitivity_to_Cold_or_Heat": int(Sensitivity_to_Cold_or_Heat),
    "Increased_Sweating": int(Increased_Sweating),
    "Muscle_Weakness": int(Muscle_Weakness),
    "Constipation_or_More_Bowel_Movements": int(Constipation_or_More_Bowel_Movements),
    "Depression_or_Anxiety": int(Depression_or_Anxiety),
    "Difficulty_Concentrating_or_Memory_Problems": int(
        Difficulty_Concentrating_or_Memory_Problems
    ),
    "Dry_or_Itchy_Skin": int(Dry_or_Itchy_Skin),
}

if st.button("Predict Thyroid Risk"):
    result = predict(user_input)

    st.subheader("Class Probabilities")
    st.dataframe(
        pd.DataFrame(
            {
                "Risk Level": list(result["class_probabilities"].keys()),
                "Probability": list(result["class_probabilities"].values()),
            }
        ),
        hide_index=True,
    )

    st.subheader("Prediction Results")
    # st.write(f"**Predicted Risk Class:** {result['predicted_class']}")
    st.write(f"ML Risk Score: {round(result['ml_risk_score'],3)}")
    st.write(f"Importance Score: {round(result['importance_score'],3)}")
    st.write(f"Final Weighted Reality Score: {round(result['final_weighted_score'],3)}")
    st.write(
        f"**:yellow[⚠️Interpreted Risk Level for this case: {result['interpreted_risk']}]**"
    )

    st.subheader("Top Feature Contributions")
    st.dataframe(result["feature_contributions"].head(10))
