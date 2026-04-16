# 🩺 Chronic Kidney Disease Predictor

A clean, interactive Streamlit web app for predicting Chronic Kidney Disease (CKD) using machine learning.

## Features

- **3 ML models** — Decision Tree, Logistic Regression, Random Forest
- **Live prediction** — adjust patient vitals from the sidebar and get instant results
- **Visual analytics** — accuracy comparison, feature distributions, confusion matrix
- Clean, minimal UI with Plotly charts

## Dataset

UCI ML Repository — Chronic Kidney Disease dataset (400 records, 25 features).  
5 key features used: `sod`, `hemo`, `bp`, `wbcc`, `sc`

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy**
