import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully! Shape:", df.shape)

# ---------------------------------------------------------
# 2. Preprocessing & Data Cleaning
# ---------------------------------------------------------
# Clean Missing Values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors="coerce")
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customerID column if present
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Convert Target to Binary (1 / 0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Optional EDA Plot
plt.figure(figsize=(6, 4))
sns.countplot(x="Contract", hue="Churn", data=df, palette="viridis")
plt.title("Churn Rate by Contract Type")
plt.close()  # Close plot to prevent blocking non-interactive environments

# One-Hot Encode Categorical Features
categorical_cols = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate Features (X) and Target (y)
X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# ---------------------------------------------------------
# 3. Train-Test Split & Scaling
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Model Training & Benchmarking
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1 Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
        }
    )

comparison_df = pd.DataFrame(results)
print("\n=== MODEL BENCHMARK RESULTS ===")
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 5. Hyperparameter Tuning on XGBoost
# ---------------------------------------------------------
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss", random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
)
grid.fit(X_train_scaled, y_train)
best_model = grid.best_estimator_

# ---------------------------------------------------------
# 6. SHAP Explainability
# ---------------------------------------------------------
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 6))
plt.title("SHAP Feature Importance Summary")
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.close()

# ---------------------------------------------------------
# 7. Save Artifacts
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("\nModel, Scaler, and Feature Names saved successfully to models/ directory!")
