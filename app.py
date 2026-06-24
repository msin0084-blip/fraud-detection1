import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from groq import Groq

rf = joblib.load('notebooks/rf_model.pkl')
scaler_amount = joblib.load('notebooks/scaler_amount.pkl')
scaler_time = joblib.load('notebooks/scaler_time.pkl')

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("Fraud Detection System")
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount ($)", min_value=0.0, value=100.0)
    time = st.number_input("Time (seconds from first transaction)", min_value=0.0, value=0.0)

st.subheader("V Features")
v_features = {}
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i-1) % 4]:
        v_features[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

if st.button("Analyse Transaction"):
    input_dict = {'Time': time, 'Amount': amount}
    for i in range(1, 29):
        input_dict[f'V{i}'] = v_features.get(f'V{i}', 0.0)

    input_df = pd.DataFrame([input_dict])
    input_df['Amount_scaled'] = scaler_amount.transform(input_df[['Amount']])[0][0]
    input_df['Time_scaled'] = scaler_time.transform(input_df[['Time']])[0][0]
    input_df = input_df.drop(columns=['Amount', 'Time'])
    input_df = input_df.drop(columns=['Time_scaled'])

    prob = rf.predict_proba(input_df)[0][1]
    label = "🔴 FRAUD" if prob > 0.5 else "🟢 LEGITIMATE"

    st.markdown(f"## {label}")
    st.metric("Fraud Probability", f"{prob:.2%}")

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1],
        shap_values[0][:,1],
        feature_names=input_df.columns.tolist(),
        show=False
    )
    st.pyplot(fig)

    top_features = sorted(
        zip(input_df.columns, shap_values[0][:,1]),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    feature_summary = ", ".join([f"{f}: {v:.3f}" for f, v in top_features])

    prompt = f"""A transaction was flagged by a fraud detection model.
Fraud probability: {prob:.2%}
Top contributing features (SHAP values): {feature_summary}

Write a concise risk report (3-4 sentences) explaining why this transaction was flagged, the risk level, and a recommendation."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    st.subheader("AI Risk Report")
    st.write(response.choices[0].message.content)