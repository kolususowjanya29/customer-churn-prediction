import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction")
st.write("Provide customer information to assess churn risk.")


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/feature_names.pkl")
    return model, scaler, features


model, scaler, feature_names = load_artifacts()

# User Inputs
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
total_charges = st.number_input(
    "Total Charges ($)", 18.0, 9000.0, float(tenure * monthly_charges)
)
contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

if st.button("Predict Churn"):
    # Construct zeroed dictionary matching trained features
    input_dict = {feat: 0 for feat in feature_names}

    if "tenure" in input_dict:
        input_dict["tenure"] = tenure
    if "MonthlyCharges" in input_dict:
        input_dict["MonthlyCharges"] = monthly_charges
    if "TotalCharges" in input_dict:
        input_dict["TotalCharges"] = total_charges
    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Prediction
    prob = model.predict_proba(input_scaled)[0][1]
    churn_percentage = int(prob * 100)

    st.markdown("---")
    st.metric("Churn Probability", f"{churn_percentage}%")

    if churn_percentage >= 70:
        st.error("Prediction: High Churn Risk")
    elif churn_percentage >= 40:
        st.warning("Prediction: Medium Churn Risk")
    else:
        st.success("Prediction: Low Churn Risk")

    st.write("**Top Factors:** Contract, Tenure, Monthly Charges")
