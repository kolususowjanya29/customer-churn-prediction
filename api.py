from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Telecom Churn Prediction API", version="1.0")

model = joblib.load("models/best_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
model_columns = joblib.load("models/model_columns.pkl")

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    TechSupport: str

@app.get("/")
def home():
    return {"status": "Online", "service": "Churn Prediction API"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        raw_dict = data.dict()
        raw_dict['TotalCharges'] = raw_dict['tenure'] * raw_dict['MonthlyCharges']
        
        raw_df = pd.DataFrame([raw_dict])
        encoded_df = pd.get_dummies(raw_df)
        
        final_input = pd.DataFrame(0, index=[0], columns=model_columns)
        for col in encoded_df.columns:
            if col in final_input.columns:
                final_input[col] = encoded_df[col]
                
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        final_input[num_cols] = scaler.transform(final_input[num_cols])
        
        prob = float(model.predict_proba(final_input)[0][1])
        risk_level = "High Churn Risk" if prob >= 0.5 else "Low Churn Risk"
        
        return {
            "churn_probability": f"{round(prob * 100, 2)}%",
            "prediction": risk_level,
            "raw_score": prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
