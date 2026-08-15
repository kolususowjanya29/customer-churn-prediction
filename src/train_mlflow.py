import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from data_pipeline import clean_and_preprocess_data

def run_experiment():
    mlflow.set_experiment("Telecom_Churn_Prediction")

    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(raw_path)
    df_processed, scaler = clean_and_preprocess_data(df, is_training=True)
    
    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1. SMOTE Class Imbalance Handling
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 2. MLflow Tracking
    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
        mlflow.log_params(params)
        mlflow.log_param("sampling_strategy", "SMOTE")

        model = XGBClassifier(**params, eval_metric='logloss')
        model.fit(X_train_resampled, y_train_resampled)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Log Metrics
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, probs))

        # Log Model to MLflow
        mlflow.sklearn.log_model(model, "xgboost_churn_model")

        # Save local artifacts for FastAPI/Streamlit
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/best_churn_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(X_train.columns.tolist(), "models/model_columns.pkl")
        print("Model trained with SMOTE and tracked in MLflow successfully.")

if __name__ == "__main__":
    run_experiment()
