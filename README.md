# 📉 Telco Customer Churn Prediction & Explainability System

A end-to-end Machine Learning solution for predicting customer churn in the telecommunications sector. This system combines data engineering, high-performance XGBoost modeling, Explainable AI (SHAP), and an interactive Streamlit web dashboard.

---

## 📌 Executive Summary

Acquiring new customers in telecommunications can cost up to 5 times more than retaining existing ones. This repository provides an end-to-end prediction pipeline designed to identify high-risk accounts before they cancel service. By integrating SHAP (SHapley Additive exPlanations), retention teams receive both a churn probability score and clear feature-level drivers to craft targeted retention offers.

---

## 🏗️ Architecture & Project Structure

```text
customer-churn-prediction/
├── models/                     # Saved model artifacts (.pkl)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── app.py                      # Main Streamlit application
├── generate_deck.py            # Automated PowerPoint presentation generator
├── requirements.txt            # Environment dependencies
└── README.md                   # Project documentation
