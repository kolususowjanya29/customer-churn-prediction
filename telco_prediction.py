import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train_and_save_artifacts():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors="coerce")
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X_raw = df.drop(columns=['Churn'])
    
    categorical_cols = X_raw.select_dtypes(include=['object', 'str']).columns
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    X = X.astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_scaled, y)
    
    joblib.dump(model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    
    if not os.path.exists(model_path):
        with st.spinner("Training initial model..."):
            train_and_save_artifacts()
            
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    
    # Initialize TreeExplainer for XGBoost model
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

model, scaler, feature_names, explainer = load_artifacts()

# UI Layout
st.title("📉 Customer Churn Prediction & SHAP Analysis")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(tenure * monthly_charges))
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

if st.button("Calculate Churn Risk", type="primary"):
    # Build zero-filled baseline matching trained features
    input_dict = {feat: 0.0 for feat in feature_names}

    if "tenure" in input_dict: input_dict["tenure"] = float(tenure)
    if "MonthlyCharges" in input_dict: input_dict["MonthlyCharges"] = float(monthly_charges)
    if "TotalCharges" in input_dict: input_dict["TotalCharges"] = float(total_charges)

    if f"Contract_{contract}" in input_dict:
        input_dict[f"Contract_{contract}"] = 1.0
    if f"InternetService_{internet_service}" in input_dict:
        input_dict[f"InternetService_{internet_service}"] = 1.0
    if f"PaymentMethod_{payment}" in input_dict:
        input_dict[f"PaymentMethod_{payment}"] = 1.0
    if f"PaperlessBilling_{paperless}" in input_dict:
        input_dict[f"PaperlessBilling_{paperless}"] = 1.0
    if f"TechSupport_{tech_support}" in input_dict:
        input_dict[f"TechSupport_{tech_support}"] = 1.0

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    churn_percentage = int(prob * 100)

    st.markdown("---")
    st.metric(label="Estimated Churn Probability", value=f"{churn_percentage}%")

    if churn_percentage >= 70:
        st.error("🚨 High Churn Risk")
    elif churn_percentage >= 40:
        st.warning("⚠️ Medium Churn Risk")
    else:
        st.success("✅ Low Churn Risk")

    # ---------------------------------------------------------
    # SHAP Explanation Section
    # ---------------------------------------------------------
    st.subheader("💡 Prediction Explanation (SHAP)")
    
    # Calculate SHAP values for the single scaled input row
    shap_values = explainer(input_scaled)
    
    # Override feature names in the Explanation object for clear labeling
    shap_values.feature_names = feature_names

    # Render Waterfall Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], max_display=7, show=False)
    st.pyplot(fig)
    plt.close()
