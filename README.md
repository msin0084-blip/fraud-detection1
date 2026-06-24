# Fraud Detection System
Live Demo: https://fraud-detection1-uel6z2ycrqzndvmfavpfb8.streamlit.app/
An AI-powered fraud detection web app built with machine learning and LLM explainability.

## What it does
- Detects fraudulent credit card transactions using a Random Forest model (84% F1, 0.98 ROC-AUC)
- Explains predictions using SHAP values (feature importance per transaction)
- Generates plain-English risk reports using Groq LLaMA via LLM API
- Interactive Streamlit dashboard for live transaction analysis

## Tech Stack
- Python, scikit-learn, imbalanced-learn (SMOTE)
- SHAP for explainability
- Groq API (LLaMA 3.1) for AI risk reports
- Streamlit for the web dashboard

## Dataset
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 0.17% fraud rate

## Setup
```bash
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
streamlit run app.py
```

## Results
| Metric | Score |
|--------|-------|
| F1 Score (Fraud) | 0.84 |
| ROC-AUC | 0.98 |
| Precision | 0.86 |
| Recall | 0.82 |
