import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk AI", layout="centered")

st.title("💳 Credit Risk Prediction System")
st.markdown("AI-powered Loan Default Probability Estimator")

st.divider()

st.subheader("📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    amt_credit = st.number_input("Credit Amount", value=200000.0)
    days_employed = st.number_input("Days Employed", value=1000.0)

with col2:
    amt_goods = st.number_input("Goods Price", value=180000.0)
    ext_source_3 = st.slider(
        "External Risk Score (0 = Risky, 1 = Safe)",
        0.0, 1.0, 0.5
    )

income_type = st.selectbox(
    "Income Type",
    ["Working", "Commercial associate", "Pensioner", "State servant"]
)

education_type = st.selectbox(
    "Education Level",
    ["Secondary / secondary special", "Higher education", "Incomplete higher"]
)

st.divider()

if st.button("🚀 Analyze Risk"):

    api_url = "https://ccfd-i7ks.onrender.com/predict"

    payload = {
        "AMT_CREDIT": amt_credit,
        "AMT_GOODS_PRICE": amt_goods,
        "DAYS_EMPLOYED": days_employed,
        "EXT_SOURCE_3": ext_source_3,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education_type
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()

        probability = result["default_probability"]
        risk = result["risk_level"]

        st.subheader("📊 Risk Analysis Result")
        st.progress(probability)
        st.write(f"### Default Probability: {probability}")

        if risk == "Low Risk":
            st.success("🟢 Low Risk")
        elif risk == "Medium Risk":
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

    else:
        st.error("API Error. Please try again.")