# Bank Customer Churn Risk Scoring App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("churn_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bank Churn Risk Scoring", layout="wide")

st.title("🏦 Bank Customer Churn Risk Predictor")

st.write(
    "This application predicts **customer churn probability** and assigns a **risk level** "
    "based on behavioral and engagement patterns."
)

# -----------------------------
# Sidebar – Customer Input
# -----------------------------
st.sidebar.header("Customer Information")

CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
Age = st.sidebar.slider("Age", 18, 100, 40)
Tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
Balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0)
EstimatedSalary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 60000.0)
NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 2)
HasCrCard = st.sidebar.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.sidebar.selectbox("Is Active Member", [0, 1])

Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

# -----------------------------
# Feature Engineering
# -----------------------------
Balance_to_Salary = Balance / (EstimatedSalary + 1)
Product_Density = NumOfProducts / (Tenure + 1)
Engagement_Product = IsActiveMember * NumOfProducts
Age_Tenure = Age * Tenure

# -----------------------------
# Build input dataframe
# -----------------------------
input_data = {
    "CreditScore": CreditScore,
    "Age": Age,
    "Tenure": Tenure,
    "Balance": Balance,
    "NumOfProducts": NumOfProducts,
    "HasCrCard": HasCrCard,
    "IsActiveMember": IsActiveMember,
    "EstimatedSalary": EstimatedSalary,
    "Balance_to_Salary": Balance_to_Salary,
    "Product_Density": Product_Density,
    "Engagement_Product": Engagement_Product,
    "Age_Tenure": Age_Tenure,
    "Geography_Germany": 1 if Geography == "Germany" else 0,
    "Geography_Spain": 1 if Geography == "Spain" else 0,
    "Gender_Male": 1 if Gender == "Male" else 0,
}

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]

# -----------------------------
# Prediction
# -----------------------------
churn_prob = model.predict_proba(input_df)[0][1]

if churn_prob >= 0.75:
    risk_level = "🔴 High Risk"
elif churn_prob >= 0.50:
    risk_level = "🟠 Medium Risk"
else:
    risk_level = "🟢 Low Risk"

# -----------------------------
# Output Section
# -----------------------------
st.subheader("Prediction Result")

st.metric("Churn Probability", f"{churn_prob:.2%}")
st.metric("Risk Level", risk_level)

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("Why is this customer at risk?")

# -----------------------------
# SHAP – Local explanation (single customer)
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

shap_values_churn = shap_values.values[0, :, 1]
base_value = shap_values.base_values[0, 1]

# Create a matplotlib Figure explicitly
fig, ax = plt.subplots(figsize=(10, 6))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_churn,
        base_values=base_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    ),
    show=False
)

st.pyplot(fig)
plt.close(fig)




# -----------------------------
# What-if Analysis
# -----------------------------
st.subheader("What-if Scenario")

st.write(
    "Adjust **engagement or product count** to observe how churn risk changes."
)

