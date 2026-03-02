# CCFD
🧠 AI Credit Risk Intelligence Platform

An end-to-end Machine Learning web application that predicts loan default probability using financial and behavioral indicators.

This system combines data preprocessing, feature engineering, model training, API deployment, and a modern AI dashboard interface into a production-ready fintech solution.

🚀 Live Deployment

🌐 Deployed on Render
🔗 Public API + Web Interface
⚡ Built with FastAPI + Uvicorn

📌 Project Overview

This project predicts the probability that a loan applicant will default using a trained Logistic Regression model.

It includes:

Data preprocessing pipeline

Feature scaling (StandardScaler)

Class imbalance handling (SMOTE)

Model training & evaluation

REST API deployment

Advanced Dark-Themed AI Dashboard UI

📊 Dataset

The model was trained using the Home Credit Default Risk dataset.

Source:
Kaggle – Home Credit Default Risk

The dataset contains applicant financial, employment, and credit behavior data.

🛠 Machine Learning Pipeline
1️⃣ Data Cleaning

Removed high-missing-value columns (>60%)

Handled null values

Encoded categorical variables

Feature selection

2️⃣ Feature Engineering

One-hot encoding

Feature scaling using StandardScaler

Balanced classes using SMOTE

3️⃣ Model Training

Logistic Regression classifier

Trained on scaled features

Evaluated using:

ROC-AUC

Precision

Recall

F1-score

4️⃣ Model Export

credit_risk_model.pkl

scaler.pkl

model_columns.pkl

🧠 Model Performance

ROC-AUC: ~0.74

Improved recall for minority class

Balanced classification strategy

The system prioritizes risk detection while maintaining overall stability.

🌐 Backend Architecture

Built using:

FastAPI (REST API)

Uvicorn (ASGI server)

Pydantic (Data validation)

Joblib (Model loading)

Pandas (Data transformation)

API Endpoint

POST /predict

Request Schema:

{
  "AMT_CREDIT": float,
  "AMT_GOODS_PRICE": float,
  "DAYS_EMPLOYED": float,
  "EXT_SOURCE_3": float,
  "NAME_INCOME_TYPE": string,
  "NAME_EDUCATION_TYPE": string
}

Response:

{
  "default_probability": float,
  "risk_level": "Low Risk" | "Medium Risk" | "High Risk"
}
🎨 Frontend

Custom-built Dark AI Control Room UI featuring:

Two-panel dashboard layout

Large probability visualization

Dynamic risk badges

Responsive design

Modern fintech aesthetics

Technologies:

HTML

CSS (Glassmorphism + Gradient theme)

Vanilla JavaScript (Fetch API)

🧩 Project Structure
├── api.py
├── templates/
│   └── index.html
├── credit_risk_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── render.yaml
└── README.md
⚙ Deployment

Deployed using:

GitHub repository

Render Web Service

Uvicorn production server

Start Command:

uvicorn api:app --host 0.0.0.0 --port 10000
🔬 Key Learnings

Handling highly imbalanced datasets

Building full ML pipelines

Converting ML models into REST APIs

Deploying AI systems publicly

Designing production-level dashboards

Debugging real-world deployment issues

📈 Future Improvements

SHAP-based explainability dashboard

Confidence score visualization

Model monitoring & logging

Admin analytics panel

Model retraining pipeline

Authentication & access control

👨‍💻 Author

Rachit Yadav
Assistant Security Analyst
AI & ML Enthusiast
Engineering Background
