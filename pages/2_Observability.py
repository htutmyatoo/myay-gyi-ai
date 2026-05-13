"""
pages/3_📊_Observability.py — Model Observability Page
Displays model performance metrics, ROC curve, confusion matrix,
and global SHAP feature importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from pathlib import Path

############################
# PAGE CONFIG
############################
st.set_page_config(
    page_title="Myay Gyi AI | Observability",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Report a bug': "https://matrix.to/#/@htutmyatoo:matrix.org",
    }
)

############################
# CSS
############################
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="block-container"] { padding: 1rem 2rem 0rem 2rem; }
[data-testid="stMetric"] {
    background-color: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 8px; text-align: center; padding: 15px 0;
}
.alert-box { padding: 10px 14px; border-radius: 6px; margin: 5px 0;
             font-size: 0.82rem; font-weight: 500; line-height: 1.5; }
.alert-orange { background:#2d1e12; border-left:3px solid #e67e22; color:#f5c97a; }
.alert-blue   { background:#12202d; border-left:3px solid #3498db; color:#7ac4f5; }
.page-header  { font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                color:#555; letter-spacing:0.06em; margin-bottom:0.25rem; }
.metric-row   { display:flex; justify-content:space-between; font-size:0.82rem;
                padding:5px 0; border-bottom:1px solid #1e1e1e; }
</style>
""", unsafe_allow_html=True)

############################
# LOAD REAL MODEL METRICS
# Works reliably from pages/2_Observability.py
############################
# Root project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR   = BASE_DIR / "data"

############################
# HELPERS
############################
def load_json(path, default):
    """
    Safely load JSON file.
    Returns default if file missing / invalid.
    """
    try:
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def load_csv(path):
    """
    Safely load CSV file.
    Returns empty DataFrame if missing / invalid.
    """
    try:
        path = Path(path)
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


############################
# 1. MODEL METRICS
# data/model_metrics.json
############################
PLACEHOLDER_METRICS = load_json(
    DATA_DIR / "model_metrics.json",
    {
        "Accuracy": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1 Score": 0.0,
        "AUC-ROC": 0.0,
        "Cohen Kappa": 0.0,
        "Log Loss": 0.0
    }
)

############################
# 2. ROC CURVE
# data/roc_curve.csv
############################
df_roc = load_csv(DATA_DIR / "roc_curve.csv")

if not df_roc.empty and {"fpr", "tpr"}.issubset(df_roc.columns):
    fpr_vals = df_roc["fpr"].values
    tpr_vals = df_roc["tpr"].values
else:
    fpr_vals = np.array([0, 1])
    tpr_vals = np.array([0, 1])

############################
# 3. GLOBAL SHAP
# data/shap_importance.csv
############################
df_shap = load_csv(DATA_DIR / "shap_importance.csv")

if not df_shap.empty and {"feature", "importance"}.issubset(df_shap.columns):
    PLACEHOLDER_SHAP = dict(
        zip(df_shap["feature"], df_shap["importance"])
    )
else:
    PLACEHOLDER_SHAP = {}

############################
# 4. CONFUSION MATRIX
# data/confusion_matrix.csv
############################
df_cm = load_csv(DATA_DIR / "confusion_matrix.csv")

if not df_cm.empty:
    CONF_MATRIX = df_cm.values
else:
    CONF_MATRIX = np.array([[0, 0], [0, 0]])

############################
# CHART BUILDERS
############################
def make_roc_curve(fpr, tpr, auc_score: float) -> go.Figure:
    """
    ROC curve with diagonal reference line.
    AUC score displayed as annotation.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"Model (AUC = {auc_score:.4f})",
        line=dict(color="#3498db", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random classifier",
        line=dict(color="#555", width=1, dash="dash")
    ))
    fig.update_layout(
        height=320, template="plotly_dark",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.1),
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="#0e1117"
    )
    return fig


def make_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Annotated heatmap confusion matrix.
    Rows = actual class, columns = predicted class.
    """
    labels = ["No aftershock", "Aftershock"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"Actual: {l}" for l in labels],
        colorscale="Blues",
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18, family="IBM Plex Mono")
    ))
    fig.update_layout(
        height=280, template="plotly_dark",
        margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="#0e1117"
    )
    return fig


def make_global_shap(shap_dict: dict) -> go.Figure:
    """
    Global SHAP feature importance bar chart.
    Shows mean absolute SHAP value per feature across all predictions.
    """
    sorted_items = sorted(shap_dict.items(), key=lambda x: x[1])
    feats  = [i[0] for i in sorted_items]
    values = [i[1] for i in sorted_items]

    fig = go.Figure(go.Bar(
        x=values, y=feats,
        orientation="h",
        marker_color="#3498db",
        marker_opacity=0.8,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="IBM Plex Mono")
    ))
    fig.update_layout(
        height=280, template="plotly_dark",
        xaxis_title="Mean |SHAP value|",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=50, t=10, b=20),
        paper_bgcolor="#0e1117"
    )
    return fig


def make_feedback_chart() -> go.Figure | None:
    """
    Visualise collected feedback ratings from data/feedback.json.
    Returns None if no feedback data exists yet.
    """
    fb_path = "/data/feedback.json"
    if not os.path.exists(fb_path):
        return None

    with open(fb_path) as f:
        data = json.load(f)
    entries = [e for e in data.get("feedback", []) if e.get("rating") is not None]
    if not entries:
        return None

    df = pd.DataFrame(entries)
    rating_counts = df.groupby(["page", "rating"]).size().reset_index(name="count")

    fig = px.bar(
        rating_counts, x="rating", y="count", color="page",
        barmode="group",
        labels={"rating": "Star Rating", "count": "Count", "page": "Page"},
        template="plotly_dark",
        color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71"]
    )
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=10, b=20),
        paper_bgcolor="#0e1117"
    )
    return fig


############################
# SIDEBAR
############################
with st.sidebar:
    st.markdown("## Myay Gyi AI 📡")
    st.markdown("*Real-Time Seismic Monitor*")
    tab_select = st.radio(
        "View",
        ["Model Performance", "Feature Importance", "User Feedback"],
        label_visibility="collapsed"
    )
    st.markdown(
        """<div style="font-size:0.72rem;color:#555;line-height:1.8;margin-top:12px;">
        Target: M4.0+ aftershock within 72h<br>
        Method: XGBoost classifier<br>
        Training data: USGS historical (2000–2024)<br>
        Developer: Htut Myat Oo<br>
        Version: 2.0.0
        </div>""",
        unsafe_allow_html=True
    )        

############################
# PAGE HEADER
############################
st.markdown('<div class="page-header">MYAY GYI AI / MODEL OBSERVABILITY</div>', unsafe_allow_html=True)
st.markdown("### 📊 Model Observability")
st.caption("Monitors the aftershock prediction model's performance metrics,interpretability, and user satisfaction over time.")
############################
# TAB: MODEL PERFORMANCE
############################
if tab_select == "Model Performance":

    col_metrics, col_roc, col_cm = st.columns((1, 1.8, 1.4), gap="medium")

    with col_metrics:
        st.markdown("#### Performance Metrics")
        st.caption("XGBoost Classifier · Test set evaluation")

        for name, val in PLACEHOLDER_METRICS.items():
            st.markdown(
                f'<div class="metric-row">'
                f'<span style="color:#aaa;">{name}</span>'
                f'<span style="font-family:IBM Plex Mono;font-weight:600;">{val:.4f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("AUC-ROC", f"{PLACEHOLDER_METRICS['AUC-ROC']:.4f}", " vs baseline 0.5")
        st.metric("F1 Score", f"{PLACEHOLDER_METRICS['F1 Score']:.4f}")

    with col_roc:
        st.markdown("#### ROC Curve")
        st.caption(f"AUC = {PLACEHOLDER_METRICS['AUC-ROC']:.4f} — higher is better (max 1.0)")
        st.plotly_chart(
            make_roc_curve(fpr_vals, tpr_vals, PLACEHOLDER_METRICS["AUC-ROC"]),
            use_container_width=True
        )
        st.markdown(
            '<div class="alert-box alert-blue">'
            'AUC > 0.88 indicates strong discriminative ability — '
            'the model correctly ranks a random aftershock event above a non-event '
            '88% of the time.</div>',
            unsafe_allow_html=True
        )

    with col_cm:
        st.markdown("#### Confusion Matrix")
        st.caption("Rows = actual · Columns = predicted")
        st.plotly_chart(make_confusion_matrix(CONF_MATRIX), use_container_width=True)

        tn, fp, fn, tp = CONF_MATRIX.ravel()
        total = tn + fp + fn + tp
        st.markdown(
            f'<div class="metric-row"><span style="color:#aaa;">True Negatives</span>'
            f'<span style="font-family:IBM Plex Mono;">{tn} ({tn/total*100:.1f}%)</span></div>'
            f'<div class="metric-row"><span style="color:#aaa;">False Positives</span>'
            f'<span style="font-family:IBM Plex Mono;color:#f5c97a;">{fp} ({fp/total*100:.1f}%)</span></div>'
            f'<div class="metric-row"><span style="color:#aaa;">False Negatives</span>'
            f'<span style="font-family:IBM Plex Mono;color:#f5a5a5;">{fn} ({fn/total*100:.1f}%)</span></div>'
            f'<div class="metric-row"><span style="color:#aaa;">True Positives</span>'
            f'<span style="font-family:IBM Plex Mono;color:#7af5a5;">{tp} ({tp/total*100:.1f}%)</span></div>',
            unsafe_allow_html=True
        )

############################
# TAB: FEATURE IMPORTANCE
############################
elif tab_select == "Feature Importance":

    col_shap, col_detail = st.columns((1.5, 1), gap="medium")

    with col_shap:
        st.markdown("#### Global Feature Importance (SHAP)")
        st.caption(
            "Mean absolute SHAP value across all test predictions. "
            "Higher value = stronger influence on the model output."
        )
        st.plotly_chart(make_global_shap(PLACEHOLDER_SHAP), use_container_width=True)

    with col_detail:
        st.markdown("#### Feature Descriptions")

        descriptions = {
            "Magnitude":         ("Main shock size — the strongest predictor of aftershock activity.",    "🔴 High"),
            "Depth (km)":        ("Shallow events (<70 km) dramatically increase aftershock probability.", "🔴 High"),
            "Seismicity rate":   ("Background M3+ events in 30 days within 200 km radius.",               "🟠 Medium"),
            "Fault zone":        ("Whether event occurred near a known active fault (Sagaing etc.).",      "🟠 Medium"),
            "Hours since event": ("Time elapsed — risk decays following Omori's Law.",                    "🟡 Medium"),
            "Region":            ("Geographic region encoding.",                                           "🟢 Low"),
            "Season":            ("Month of year — minor seasonal correlations in some regions.",          "🟢 Low"),
        }

        for feat, (desc, importance) in descriptions.items():
            with st.popover(f"{feat} — {importance}"):
                st.caption(desc)

        st.divider()
        st.markdown(
            '<div class="alert-box alert-blue">'
            '<strong>SHAP (SHapley Additive exPlanations)</strong> fairly distributes '
            'each feature\'s contribution to a prediction using game theory. '
            'It is model-agnostic and supports both global and per-prediction explanations.</div>',
            unsafe_allow_html=True
        )

############################
# TAB: USER FEEDBACK
############################
elif tab_select == "User Feedback":

    st.markdown("#### User Feedback Summary")

    fb_path = "data/feedback.json"
    if not os.path.exists(fb_path):
        st.info("No feedback collected yet. Feedback forms are available on each page.")
    else:
        with open(fb_path) as f:
            data = json.load(f)
        entries = data.get("feedback", [])

        if not entries:
            st.info("No feedback entries found.")
        else:
            df_fb = pd.DataFrame(entries)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Responses", len(df_fb))

            valid_ratings = df_fb[df_fb["rating"].notna()]
            if not valid_ratings.empty:
                col_b.metric("Avg Rating", f"{valid_ratings['rating'].mean():.1f} / 5")

            col_c.metric("Pages Covered", df_fb["page"].nunique())

            fig_fb = make_feedback_chart()
            if fig_fb:
                st.plotly_chart(fig_fb, use_container_width=True)

            st.markdown("#### Recent Comments")
            comments = df_fb[df_fb["comment"].notna() & (df_fb["comment"] != "")]
            for _, row in comments.tail(5).iterrows():
                st.markdown(
                    f'<div class="alert-box alert-blue">'
                    f'<strong>{row["page"]}</strong> · ⭐ {row.get("rating","?")}<br>'
                    f'{row["comment"]}</div>',
                    unsafe_allow_html=True
                )

############################
# FEEDBACK FORM
############################
st.divider()
with st.popover("💬 Give Feedback for Observability", width="stretch"):
        rating  = st.feedback("stars", key="obs_rating")
        fb_text = st.text_area("Any comments about the model performance display?",
                               height=80, key="obs_fb_text")
        if st.button("Submit Feedback", key="obs_fb_submit", type="primary"):
            from datetime import datetime
            os.makedirs("data", exist_ok=True)
            existing = []
            if os.path.exists(fb_path):
                with open(fb_path) as f:
                    existing = json.load(f).get("feedback", [])
            existing.append({
                "page":    "Observability",
                "rating":  rating,
                "comment": fb_text,
                "time":    datetime.now().isoformat()
            })
            with open(fb_path, "w") as f:
                json.dump({"feedback": existing}, f, indent=2)
            st.success("Thanks for your feedback!")
