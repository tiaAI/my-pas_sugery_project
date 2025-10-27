import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import xgboost
# Load the model
model = joblib.load('xgb_model.pkl')

# Define feature names
feature_names = [
'Maternal_age','Gestational_age','Gravidity','Parturition','Prior_C-sections_number',
'Uterine_surgery_number','MRI_Risk_Score'
]
# print(model.predict_proba(input)[0][1])

# Streamlit user interface
st.title("PAS Hysterectomy Risk Model")

# age: numerical input
Maternal_age = st.number_input("Maternal Age (day):",  min_value=0, max_value=500, value=261)

# sex: categorical selection
Gestational_age = st.number_input("Gestational Age (year):", min_value=0, max_value=100, value=26)

# cp: categorical selection
Gravidity = st.number_input("Gravidity:", min_value=0, max_value=50, value=6)

# trestbps: numerical input
Parturition = st.number_input("Parturition:", min_value=0, max_value=50, value=2)

# trestbps: numerical input
Prior_C_sections_number = st.number_input("Prior C-Sections Number:", min_value=0, max_value=50, value=2)

# chol: numerical input
Uterine_surgery_number = st.number_input("Uterine Surgery Number:", min_value=0, max_value=50, value=0)

# fbs: categorical selection
MRI_Risk_Score = st.number_input("MRI_Risk_Score:", min_value=0, max_value=50, value=13)


# Process inputs and make predictions
feature_values = [Maternal_age,Gestational_age,Gravidity,Parturition,Prior_C_sections_number,Uterine_surgery_number,MRI_Risk_Score]
features = np.array([feature_values])

if st.button("Predict"):
    # Predict class and probabilities
    predicted_proba = model.predict_proba(features)[0][1]

    # Display prediction results
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    if predicted_proba >= 0.2894:
        advice = (
        f"According to our model prediction, you are at high risk of requiring hysterectomy during cesarean delivery, "
        f"with an estimated probability of {predicted_proba:.1%} based on the PAS Risk Model. "
        "Although this is only a predictive result, it indicates a potentially elevated risk of intraoperative complications. "
        "We strongly recommend that you consult with an obstetric specialist as soon as possible for further evaluation "
        "and to develop an individualized prenatal management plan to ensure appropriate medical support and safe delivery."
        )
    else:
        advice = (
        f"According to our model prediction, your risk of requiring hysterectomy during cesarean delivery is low. "
        f"The PAS Risk Model estimates your probability of avoiding this complication at {predicted_proba:.1f}%. "
        "While the risk is low, we still recommend continuing regular prenatal check-ups and close monitoring "
        "to ensure a safe and smooth delivery process."
            )
    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model, base_score=0.5)
    shap_values = explainer.shap_values(pd.DataFrame(features, columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], 
                   pd.DataFrame(features, columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
