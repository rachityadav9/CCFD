from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI(title="Credit Risk Prediction API")

# -------- Request Schema --------
class Applicant(BaseModel):
    AMT_CREDIT: float
    AMT_GOODS_PRICE: float
    DAYS_EMPLOYED: float
    EXT_SOURCE_3: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str

# -------- Prediction Endpoint -----------
@app.post("/predict")
def predict(applicant: Applicant):

    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    # Numeric fields
    input_df.at[0, "AMT_CREDIT"] = applicant.AMT_CREDIT
    input_df.at[0, "AMT_GOODS_PRICE"] = applicant.AMT_GOODS_PRICE
    input_df.at[0, "DAYS_EMPLOYED"] = applicant.DAYS_EMPLOYED
    input_df.at[0, "EXT_SOURCE_3"] = applicant.EXT_SOURCE_3

    # Income Encoding
    income_col = f"NAME_INCOME_TYPE_{applicant.NAME_INCOME_TYPE}"
    if income_col in input_df.columns:
        input_df.at[0, income_col] = 1

    # Education Encoding
    edu_col = f"NAME_EDUCATION_TYPE_{applicant.NAME_EDUCATION_TYPE}"
    if edu_col in input_df.columns:
        input_df.at[0, edu_col] = 1

    # Scale
    input_scaled = scaler.transform(input_df)

    probability = model.predict_proba(input_scaled)[0][1]

    # Risk logic
    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return {
        "default_probability": round(float(probability), 4),
        "risk_level": risk
    }