import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train_and_save_artifacts():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load and clean data
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors="coerce")
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    
    # 2. Fit Scaler and Model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_scaled, y)
    
    # 3. Save Artifacts
    joblib.dump(model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    
    # If artifacts do not exist, run training step once
    if not os.path.exists(model_path):
        with st.spinner("Training initial model... Please wait."):
            train_and_save_artifacts()
            
    model = joblib.load(model_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, features

model, scaler, feature_names = load_artifacts()
