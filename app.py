import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}

/* Background */
.stApp {
    background: #f5f4f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8e6df !important;
}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label {
    color: #a09f99 !important;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Hero title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    font-weight: 400;
    color: #1a1a2e;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b6b7b;
    font-weight: 300;
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}

/* Cards */
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border: 1px solid #e4e2da;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9a99a3;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #1a1a2e;
    line-height: 1;
}
.metric-unit {
    font-size: 0.8rem;
    color: #9a99a3;
    margin-top: 0.2rem;
}

/* Section headers */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9a99a3;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e4e2da;
}

/* Result pill */
.result-ckd {
    background: #fef0f0;
    border: 1.5px solid #f5c6c6;
    color: #c0392b;
    border-radius: 50px;
    padding: 0.7rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 500;
    display: inline-block;
    margin-top: 0.5rem;
}
.result-notckd {
    background: #f0faf4;
    border: 1.5px solid #b7e4c7;
    color: #1e8449;
    border-radius: 50px;
    padding: 0.7rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 500;
    display: inline-block;
    margin-top: 0.5rem;
}

/* Accuracy badge */
.acc-badge {
    background: #1a1a2e;
    color: #e8e6df;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-block;
}
.acc-best {
    background: #c0392b;
    color: white;
}

/* Divider */
.thin-divider {
    border: none;
    border-top: 1px solid #e4e2da;
    margin: 1.5rem 0;
}

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data & model loading ───────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    data = pd.read_csv("chronic_kidney_disease.csv")
    data.replace('?', np.nan, inplace=True)
    data.columns = data.columns.str.strip().str.replace("'", "")
    for col in data.columns:
        if col != 'class' and pd.api.types.is_string_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce')

    features = ['sod', 'hemo', 'bp', 'wbcc', 'sc']
    X = data[features]
    y = data['class'].astype(str).str.strip().map({'ckd': 1, 'notckd': 0})

    mask = y.notna()
    X, y = X[mask], y[mask]
    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, preds),
            "preds": preds,
            "y_test": y_test,
        }

    return data, X, y, results, features


data, X, y, results, FEATURES = load_and_train()
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_model_name]["model"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Patient Inputs")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    sod  = st.slider("Sodium (sod)", 100.0, 160.0, float(X['sod'].mean()), 0.5,
                     help="Serum Sodium in mEq/L")
    hemo = st.slider("Haemoglobin (hemo)", 3.0, 18.0, float(X['hemo'].mean()), 0.1,
                     help="Haemoglobin in gms")
    bp   = st.slider("Blood Pressure (bp)", 50, 180, int(X['bp'].mean()), 2,
                     help="Diastolic BP in mm/Hg")
    wbcc = st.slider("WBC Count (wbcc)", 3000, 20000, int(X['wbcc'].mean()), 100,
                     help="White Blood Cell Count in cells/cumm")
    sc   = st.slider("Serum Creatinine (sc)", 0.4, 15.0, float(X['sc'].mean()), 0.1,
                     help="Serum Creatinine in mgs/dl")

    st.markdown("<hr style='border-color:#2d2d4e; margin: 1.2rem 0'>", unsafe_allow_html=True)
    selected_model = st.selectbox("Model", list(results.keys()),
                                  index=list(results.keys()).index(best_model_name))
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=True, type="primary")


# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Chronic Kidney<br>Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Machine learning classification on clinical markers</div>', unsafe_allow_html=True)

# ── Row 1 — dataset stats ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
ckd_count    = int(y.sum())
notckd_count = len(y) - ckd_count

for col, label, value, unit in [
    (c1, "Total Records",      len(data),    "patients"),
    (c2, "CKD Cases",          ckd_count,    f"{ckd_count/len(y)*100:.0f}% of total"),
    (c3, "Non-CKD Cases",      notckd_count, f"{notckd_count/len(y)*100:.0f}% of total"),
    (c4, "Features Used",      len(FEATURES), "clinical markers"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# ── Row 2 — model accuracy + prediction result ────────────────────────────────
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section-label">Model Accuracy Comparison</div>', unsafe_allow_html=True)

    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100 for n in names]
    colors = ["#c0392b" if n == best_model_name else "#d4d0c8" for n in names]

    fig = go.Figure(go.Bar(
        x=accs, y=names, orientation='h',
        marker_color=colors,
        text=[f"{a:.1f}%" for a in accs],
        textposition='outside',
        textfont=dict(size=13, color='#1a1a2e'),
        hovertemplate='%{y}: %{x:.2f}%<extra></extra>',
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=60, t=10, b=10),
        height=200,
        xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color='#1a1a2e')),
        showlegend=False,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    for name in names:
        badge_cls = "acc-best" if name == best_model_name else "acc-badge"
        st.markdown(
            f'<span class="{badge_cls}">{name} — {results[name]["accuracy"]*100:.2f}%</span>&nbsp;&nbsp;',
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        input_data = np.array([[sod, hemo, bp, wbcc, sc]])
        chosen = results[selected_model]["model"]
        pred   = chosen.predict(input_data)[0]
        proba  = chosen.predict_proba(input_data)[0] if hasattr(chosen, "predict_proba") else None

        if pred == 1:
            st.markdown('<div class="result-ckd">⚠️ CKD Detected</div>', unsafe_allow_html=True)
            st.markdown("<br>The model predicts the presence of **Chronic Kidney Disease** based on the entered values.", unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-notckd">✅ No CKD Detected</div>', unsafe_allow_html=True)
            st.markdown("<br>The model predicts **no Chronic Kidney Disease** based on the entered values.", unsafe_allow_html=True)

        if proba is not None:
            st.markdown(f"<br><span style='font-size:0.85rem;color:#9a99a3'>Confidence — CKD: {proba[1]*100:.1f}% | Not CKD: {proba[0]*100:.1f}%</span>", unsafe_allow_html=True)

        st.markdown(f"<br><span style='font-size:0.8rem;color:#9a99a3'>Model used: {selected_model}</span>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#9a99a3; font-size:0.95rem; padding: 1rem 0; line-height:1.7">
            Adjust the patient values in the<br>sidebar and click <strong style='color:#1a1a2e'>Run Prediction</strong>.
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# ── Row 3 — feature distributions ─────────────────────────────────────────────
st.markdown('<div class="section-label">Feature Distributions by Class</div>', unsafe_allow_html=True)

feat_labels = {
    "sod": "Sodium", "hemo": "Haemoglobin",
    "bp": "Blood Pressure", "wbcc": "WBC Count", "sc": "Serum Creatinine"
}

full = X.copy()
full['class'] = y.values

cols = st.columns(5)
for i, feat in enumerate(FEATURES):
    with cols[i]:
        ckd_vals    = full[full['class'] == 1][feat].dropna()
        notckd_vals = full[full['class'] == 0][feat].dropna()

        fig2 = go.Figure()
        fig2.add_trace(go.Box(
            y=ckd_vals, name="CKD",
            marker_color="#c0392b", line_width=1.5,
            boxpoints=False,
        ))
        fig2.add_trace(go.Box(
            y=notckd_vals, name="No CKD",
            marker_color="#1e8449", line_width=1.5,
            boxpoints=False,
        ))
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title=dict(text=feat_labels[feat], font=dict(size=12, color='#1a1a2e'), x=0.5),
            margin=dict(l=5, r=5, t=30, b=5),
            height=220,
            showlegend=False,
            yaxis=dict(showgrid=True, gridcolor='#e4e2da', tickfont=dict(size=10)),
            xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Row 4 — confusion matrix ───────────────────────────────────────────────────
st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Confusion Matrix — ' + selected_model + '</div>', unsafe_allow_html=True)

cm_col, info_col = st.columns([2, 3], gap="large")

with cm_col:
    cm = confusion_matrix(results[selected_model]["y_test"], results[selected_model]["preds"])
    fig3 = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Not CKD", "CKD"], y=["Not CKD", "CKD"],
        text_auto=True,
        color_continuous_scale=["#f5f4f0", "#c0392b"],
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        coloraxis_showscale=False,
        font=dict(size=13, color='#1a1a2e'),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with info_col:
    report = classification_report(
        results[selected_model]["y_test"],
        results[selected_model]["preds"],
        target_names=["Not CKD", "CKD"],
        output_dict=True,
    )
    rep_df = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
    rep_df = rep_df[["precision", "recall", "f1-score", "support"]].round(3)
    rep_df.columns = ["Precision", "Recall", "F1-Score", "Support"]
    rep_df.index.name = "Class"
    st.dataframe(rep_df, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#b0afb9; font-size:0.78rem; letter-spacing:0.05em'>
    CHRONIC KIDNEY DISEASE PREDICTOR &nbsp;·&nbsp; UCI ML REPOSITORY DATASET
</div>
""", unsafe_allow_html=True)
