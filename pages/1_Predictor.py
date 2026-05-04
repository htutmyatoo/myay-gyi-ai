"""
pages/2_🔮_Predictor.py — Aftershock Risk Predictor

Features:
  - Input mode toggle in sidebar: Recent Events (searchable list) or Manual Entry
  - Live USGS feed with searchable selectbox
  - Fault selector with proximity auto-suggestion
  - Real XGBoost model + SHAP — loads from models/ when available
  - Graceful fallback to heuristic placeholder if model files are missing
  - Feedback form saved to data/feedback.json
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests
import pickle
import json
import os
from datetime import datetime

############################
# PAGE CONFIG
############################
st.set_page_config(
    page_title="Myay Gyi AI | Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Report a bug': "https://matrix.to/#/@htutmyatoo:matrix.org",
    }
)

############################
# SHARED CSS
############################
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
[data-testid="block-container"] { padding: 1rem 2rem 0 2rem; }
[data-testid="stMetric"] {
    background:#1a1a1a; border:1px solid #2a2a2a;
    border-radius:8px; text-align:center; padding:15px 0;
}
.alert-box {
    padding:10px 14px; border-radius:6px; margin:5px 0;
    font-size:0.82rem; font-weight:500; line-height:1.5;
}
.alert-red    { background:#2d1212; border-left:3px solid #e74c3c; color:#f5a5a5; }
.alert-orange { background:#2d1e12; border-left:3px solid #e67e22; color:#f5c97a; }
.alert-blue   { background:#12202d; border-left:3px solid #3498db; color:#7ac4f5; }
.alert-green  { background:#12291a; border-left:3px solid #2ecc71; color:#7af5a5; }
.alert-purple { background:#1e1229; border-left:3px solid #9b59b6; color:#d7a5f5; }
.risk-chip {
    display:inline-flex; align-items:center; gap:4px;
    font-size:0.75rem; padding:3px 10px; border-radius:4px;
    margin:2px; font-weight:500;
}
.chip-red    { background:#2d1212; color:#f5a5a5; border:1px solid #e74c3c44; }
.chip-orange { background:#2d1e12; color:#f5c97a; border:1px solid #e67e2244; }
.chip-green  { background:#12291a; color:#7af5a5; border:1px solid #2ecc7144; }
.page-header {
    font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
    color:#555; letter-spacing:0.06em; margin-bottom:0.25rem;
}
.event-card {
    background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px;
    padding:10px 14px; margin:4px 0; font-size:0.82rem; line-height:1.7;
}
.fault-badge {
    display:inline-block; font-size:0.7rem; font-weight:600;
    padding:2px 7px; border-radius:3px; margin-left:6px;
    vertical-align:middle; letter-spacing:0.04em;
}
.fault-mm { background:#3d1a1a; color:#f5a5a5; }
.fault-gl { background:#12202d; color:#7ac4f5; }
.disclaimer {
    font-size:0.72rem; color:#666; line-height:1.6;
    border:1px solid #2a2a2a; border-radius:6px;
    padding:10px 12px; margin-top:8px;
}
.section-label {
    font-size:0.7rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; color:#666; margin:14px 0 6px;
    border-bottom:1px solid #2a2a2a; padding-bottom:4px;
}
.model-badge {
    display:inline-block; font-size:0.72rem; font-weight:600;
    padding:3px 10px; border-radius:4px; margin-left:8px;
}
.badge-real { background:#12291a; color:#7af5a5; border:1px solid #2ecc7144; }
.badge-demo { background:#2d1e12; color:#f5c97a; border:1px solid #e67e2244; }
</style>
""", unsafe_allow_html=True)


############################
# FAULT DATABASE
############################
FAULTS = {
    "Sagaing Fault (Myanmar)": {
        "lat": 21.5, "lon": 96.0, "region": "myanmar",
        "type": "Strike-slip", "length_km": 1200,
        "note": "Most seismically active fault in Myanmar — runs N–S through the country"
    },
    "Shan Scarp Fault (Myanmar)": {
        "lat": 21.0, "lon": 98.5, "region": "myanmar",
        "type": "Thrust", "length_km": 400,
        "note": "Eastern Myanmar — linked to several destructive historical earthquakes"
    },
    "Kyaukkyan Fault (Myanmar)": {
        "lat": 24.5, "lon": 97.5, "region": "myanmar",
        "type": "Strike-slip", "length_km": 250,
        "note": "Northern Myanmar — intersects the Sagaing Fault system"
    },
    "Dawei Fault (Myanmar)": {
        "lat": 14.5, "lon": 98.5, "region": "myanmar",
        "type": "Thrust", "length_km": 300,
        "note": "Southern Myanmar / Tanintharyi region"
    },
    "Three Pagodas Fault (Thailand-Myanmar)": {
        "lat": 15.5, "lon": 99.0, "region": "myanmar",
        "type": "Strike-slip", "length_km": 500,
        "note": "Cross-border fault — 1983 Kanchanaburi earthquake"
    },
    "San Andreas Fault (USA)": {
        "lat": 36.0, "lon": -120.5, "region": "global",
        "type": "Strike-slip", "length_km": 1300,
        "note": "California — Pacific / North American plate boundary"
    },
    "North Anatolian Fault (Turkey)": {
        "lat": 40.8, "lon": 33.0, "region": "global",
        "type": "Strike-slip", "length_km": 1500,
        "note": "One of the world's most active faults — 1999 Izmit M7.6"
    },
    "Sumatra-Andaman Subduction Zone": {
        "lat": 4.0, "lon": 95.0, "region": "global",
        "type": "Subduction", "length_km": 5500,
        "note": "Generated the 2004 Indian Ocean tsunami (M9.1)"
    },
    "Japan Trench Subduction Zone": {
        "lat": 38.0, "lon": 143.5, "region": "global",
        "type": "Subduction", "length_km": 800,
        "note": "Generated the 2011 Tohoku earthquake and tsunami (M9.0)"
    },
    "Alpine Fault (New Zealand)": {
        "lat": -43.5, "lon": 170.5, "region": "global",
        "type": "Strike-slip", "length_km": 850,
        "note": "M8+ event considered overdue on the Pacific-Australian boundary"
    },
    "Dead Sea Transform Fault": {
        "lat": 31.5, "lon": 35.5, "region": "global",
        "type": "Strike-slip", "length_km": 1000,
        "note": "Middle East — runs from Red Sea to Turkey"
    },
    "Himalayan Frontal Thrust (India-Nepal)": {
        "lat": 28.0, "lon": 85.0, "region": "global",
        "type": "Thrust", "length_km": 2400,
        "note": "Generated the 2015 Nepal M7.8 earthquake"
    },
    "Cascadia Subduction Zone (USA/Canada)": {
        "lat": 45.5, "lon": -124.0, "region": "global",
        "type": "Subduction", "length_km": 1000,
        "note": "Pacific Northwest — capable of M9+ megathrust events"
    },
    "No specific fault / Unknown": {
        "lat": None, "lon": None, "region": "global",
        "type": "Unknown", "length_km": None,
        "note": "Select if the event is not near a known major fault"
    },
}

############################
# MODEL LOADING
############################
MODEL_PATH     = "models/xgb_aftershock_model.pkl"
EXPLAINER_PATH = "models/shap_explainer.pkl"
FEATURES_PATH  = "models/feature_cols.json"
METADATA_PATH  = "models/model_metadata.json"

@st.cache_resource
def load_model():
    """
    Load trained XGBoost model and SHAP explainer from disk.
    Returns (model, explainer, feature_cols, metadata) or None tuple if files missing.
    """
    if not all(os.path.exists(p) for p in [MODEL_PATH, EXPLAINER_PATH, FEATURES_PATH]):
        return None, None, None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(EXPLAINER_PATH, "rb") as f:
            explainer = pickle.load(f)
        with open(FEATURES_PATH) as f:
            feature_cols = json.load(f)
        metadata = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
        return model, explainer, feature_cols, metadata
    except Exception as e:
        st.warning(f"Model files found but failed to load: {e}")
        return None, None, None, None

model, explainer, feature_cols, model_metadata = load_model()
MODEL_LOADED = model is not None

############################
# LIVE USGS DATA
############################
@st.cache_data(ttl=600)
def load_recent_earthquakes(min_mag: float, days: int) -> pd.DataFrame:
    """
    Fetch recent M>=min_mag earthquakes from USGS (last N days).
    Cached 10 minutes. Returns empty DataFrame on failure.
    """
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    rows = []
    for eq in data["features"]:
        prop = eq["properties"]
        geo  = eq["geometry"]
        mag  = prop.get("mag")
        if mag is None or mag < min_mag:
            continue
        t = pd.to_datetime(prop["time"], unit="ms", utc=True)
        if t < cutoff:
            continue
        rows.append({
            "place":     prop.get("place", "Unknown"),
            "time":      t,
            "magnitude": mag,
            "depth":     geo["coordinates"][2],
            "longitude": geo["coordinates"][0],
            "latitude":  geo["coordinates"][1],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time", ascending=False).reset_index(drop=True)


############################
# HELPERS
############################
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2)**2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fault_proximity_km(eq_lat, eq_lon, fault_name) -> float:
    """Approximate distance in km from epicentre to a named fault centroid."""
    f = FAULTS.get(fault_name, {})
    if f.get("lat") is None:
        return 9999.0
    return haversine_km(eq_lat, eq_lon, f["lat"], f["lon"])


def time_ago(ts: pd.Timestamp) -> str:
    """UTC timestamp → human-readable relative time."""
    secs = int((pd.Timestamp.now(tz="UTC") - ts).total_seconds())
    if secs < 60:      return "Just now"
    elif secs < 3600:  return f"{secs//60}m ago"
    elif secs < 86400: return f"{secs//3600}h {(secs%3600)//60}m ago"
    return f"{secs//86400}d ago"


def mag_icon(m: float) -> str:
    if m >= 6:   return "🔴"
    elif m >= 5: return "🟠"
    elif m >= 4: return "🟡"
    return "🟢"


def depth_label(d: float):
    if d < 70:    return "Shallow (<70 km)",         "alert-red"
    elif d < 300: return "Intermediate (70–300 km)", "alert-orange"
    return         "Deep (>300 km)",                  "alert-blue"


def closest_fault(lat, lon, fault_names) -> str:
    """Return name of closest fault from a list, by centroid distance."""
    dists = {fn: fault_proximity_km(lat, lon, fn) for fn in fault_names}
    return min(dists, key=dists.get)


############################
# PREDICTION ENGINE
############################
def run_prediction(magnitude, depth, seismicity_rate,
                   hours_since, fault_prox, fault_type,
                   eq_lat=None, eq_lon=None):
    """
    Run aftershock prediction using real model if loaded, else heuristic fallback.
    Returns standardised result dict consumed by all UI components.
    """
    if MODEL_LOADED:
        return _predict_real(magnitude, depth, seismicity_rate,
                             hours_since, fault_prox, fault_type,
                             eq_lat, eq_lon)
    return _predict_heuristic(magnitude, depth, seismicity_rate,
                              hours_since, fault_prox, fault_type)


def _predict_real(magnitude, depth, seismicity_rate, hours_since,
                  fault_prox, fault_type, eq_lat, eq_lon):
    """
    Predict using loaded XGBoost model and SHAP explainer.
    Constructs feature vector matching training schema.
    """
    import shap as shap_lib

    depth_cat = 0 if depth < 70 else (1 if depth < 300 else 2)
    lat  = eq_lat  if eq_lat  is not None else 19.0
    lon  = eq_lon  if eq_lon  is not None else 96.0
    now  = pd.Timestamp.now()
    
    type_weight = {
        "Subduction": 1.4,
        "Thrust": 1.2,
        "Strike-slip": 1.0,
        "Unknown": 0.8
    }

    fault_pressure = (
        max(0, (150 - fault_prox) / 150)
        * type_weight.get(fault_type, 1.0)
        * magnitude
    )

    row = {
        "magnitude":           magnitude,
        "depth":               depth,
        "depth_category":      depth_cat,
        "seismicity_rate_30d": seismicity_rate,
        "seismicity_rate_7d":  max(0, seismicity_rate // 4),
        "hour_of_day":         now.hour,
        "month":               now.month,
        "lat_bin":             round(lat / 5) * 5,
        "lon_bin":             round(lon / 5) * 5,
        "is_myanmar_region":   int(5 <= lat <= 35 and 85 <= lon <= 110),
        "nearest_fault_km": fault_prox,
        "is_near_fault": int(fault_prox < 50),
        "fault_pressure": fault_pressure
    }
    X = pd.DataFrame([row])[feature_cols]

    prob     = float(model.predict_proba(X)[0, 1])
    sv       = explainer.shap_values(X)[0]
    shap_d   = {f: round(float(v), 4) for f, v in zip(feature_cols, sv)}

    risk = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")
    p24  = float(np.clip(prob * 1.10, 0, 0.99))
    p48  = float(prob * 0.72)
    p72  = float(prob * 0.48)

    drivers = _build_drivers(magnitude, depth, fault_prox, fault_type,
                              seismicity_rate, hours_since)

    return {"prob_72h": prob, "prob_24h": p24, "prob_48h": p48,
            "prob_72h_only": p72, "risk_level": risk,
            "shap_values": shap_d, "drivers": drivers, "real_model": True}


def _predict_heuristic(magnitude, depth, seismicity_rate,
                        hours_since, fault_prox, fault_type):
    """
    Heuristic fallback (Omori-Law inspired) when model files are not present.
    Used during development before the Colab notebook has been run.
    """
    base  = 0.10
    base += min((magnitude - 4.5) / 10.0 * 2.5, 0.45)
    base += max(0, (70 - depth) / 70) * 0.20
    base += min(seismicity_rate / 60, 0.15)
    fb    = {"Subduction": 0.06, "Thrust": 0.04, "Strike-slip": 0.03}.get(fault_type, 0)
    if fault_prox < 50:    base += 0.10
    elif fault_prox < 150: base += 0.06
    elif fault_prox < 300: base += 0.03
    base += fb
    base *= max(0.75, 1.0 - hours_since / 500)
    base  = float(np.clip(base, 0.02, 0.97))

    risk = "High" if base >= 0.65 else ("Moderate" if base >= 0.40 else "Low")
    shap = {
        "Magnitude":         round(min((magnitude-4.5)/10.0*2.5, 0.45), 4),
        "Depth (km)":        round(max(0, (70-depth)/70)*0.20, 4),
        "Seismicity rate":   round(min(seismicity_rate/60, 0.15), 4),
        "Fault proximity":   round(0.10 if fault_prox<50 else 0.06 if fault_prox<150
                                   else 0.03 if fault_prox<300 else 0.0, 4),
        "Fault type":        round(fb, 4),
        "Hours since event": round(-(hours_since/500)*0.08, 4),
    }
    drivers = _build_drivers(magnitude, depth, fault_prox, fault_type,
                              seismicity_rate, hours_since)
    return {"prob_72h": base, "prob_24h": float(np.clip(base*1.10,0,0.99)),
            "prob_48h": base*0.72, "prob_72h_only": base*0.48,
            "risk_level": risk, "shap_values": shap,
            "drivers": drivers, "real_model": False}


def _build_drivers(magnitude, depth, fault_prox, fault_type,
                    seismicity_rate, hours_since):
    drivers = []
    if magnitude >= 6.5:        drivers.append(("M6.5+ main shock",       "high"))
    elif magnitude >= 5.5:      drivers.append(("M5.5+ main shock",       "high"))
    elif magnitude >= 5.0:      drivers.append(("M5.0+ main shock",       "medium"))
    if depth < 30:              drivers.append(("Very shallow (<30 km)",   "high"))
    elif depth < 70:            drivers.append(("Shallow depth (<70 km)",  "medium"))
    if fault_prox < 50:         drivers.append(("Within 50 km of fault",  "high"))
    elif fault_prox < 150:      drivers.append(("Within 150 km of fault", "medium"))
    if seismicity_rate > 20:    drivers.append(("Active seismic zone",     "medium"))
    if fault_type == "Subduction": drivers.append(("Subduction zone",      "medium"))
    if hours_since < 6:         drivers.append(("Very recent (<6h)",       "high"))
    elif hours_since < 24:      drivers.append(("Recent event (<24h)",     "medium"))
    return drivers

############################
# CHARTS
############################
def make_gauge(probability: float, risk_level: str) -> go.Figure:
    color = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}[risk_level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 40, "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0,100], "tickwidth":1, "tickcolor":"#444",
                     "tickfont": {"size":10}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a1a", "borderwidth": 0,
            "steps": [
                {"range":[0,  40], "color":"#12291a"},
                {"range":[40, 65], "color":"#2d1e12"},
                {"range":[65,100], "color":"#2d1212"},
            ],
            "threshold": {"line":{"color":color,"width":3},
                          "thickness":0.8, "value":round(probability*100,1)}
        }
    ))
    fig.update_layout(height=215, margin=dict(l=20,r=20,t=20,b=0),
                      paper_bgcolor="#0e1117", font_color="#ccc")
    return fig


def make_time_window_chart(p24, p48, p72) -> go.Figure:
    probs   = [p24*100, p48*100, p72*100]
    windows = ["0 – 24 h", "24 – 48 h", "48 – 72 h"]
    colors  = ["#e74c3c" if p>=65 else "#f39c12" if p>=40 else "#2ecc71" for p in probs]
    fig = go.Figure(go.Bar(
        x=probs, y=windows, orientation="h", marker_color=colors,
        text=[f"{p:.0f}%" for p in probs], textposition="outside",
        textfont=dict(size=13, family="IBM Plex Mono")
    ))
    fig.update_layout(height=155, margin=dict(l=10,r=50,t=8,b=8),
                      template="plotly_dark",
                      xaxis=dict(range=[0,118], showgrid=False, showticklabels=False),
                      yaxis=dict(showgrid=False),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    return fig


def make_shap_chart(shap_values: dict) -> go.Figure:
    items  = sorted(shap_values.items(), key=lambda x: abs(x[1]))
    feats  = [i[0] for i in items]
    vals   = [i[1] for i in items]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in vals], textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono")
    ))
    fig.add_vline(x=0, line_width=1, line_color="#444")
    fig.update_layout(height=max(220, len(feats)*35),
                      margin=dict(l=10,r=60,t=10,b=10),
                      template="plotly_dark",
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False),
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    return fig


############################
# SIDEBAR
############################
with st.sidebar:
    st.markdown("## Myay Gyi AI📡")
    st.markdown("*Real-Time Seismic Monitor*")
    # Model status badge
    if MODEL_LOADED:
        ver = model_metadata.get("training_date", "")[:10] if model_metadata else ""
        st.markdown(
            f'<div class="alert-box alert-green" style="text-align:center;">'
            f'✅ <strong>Model is active</strong><br>'
            f'<span style="font-size:0.72rem;">XGBoost · trained {ver}</span></div><br>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="alert-box alert-red" style="text-align:center;">'
            '⚠️ <strong>Model is unavailable</strong><br>'
            '<span style="font-size:0.72rem;">We\'re currently experiencing issues</span>'
            '</div><br>',
            unsafe_allow_html=True
        )
    st.markdown('<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#666;margin-bottom:6px;">'
                'Event List Filters</div>', unsafe_allow_html=True)
    recent_days    = st.slider("Show last N days", 1, 30, 7)
    recent_min_mag = st.slider("Min magnitude", 4.0, 7.5, 4.5, 0.1)
    if st.button("🔄 Refresh list", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
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
# LOAD LIVE DATA
############################
df_recent = load_recent_earthquakes(recent_min_mag, recent_days)

############################
# PAGE HEADER
############################
st.markdown('<div class="page-header">MYAY GYI AI / AFTERSHOCK PREDICTOR</div>',
            unsafe_allow_html=True)
st.markdown("### 🔮 Aftershock Risk Predictor")
st.caption("Estimates the probability of a M4.0+ aftershock within 72 hours of a triggering event.")
############################
# MAIN LAYOUT
############################
col_input, col_output, col_explain = st.columns((1.3, 1.7, 1.4), gap="medium")

# ══════════════════════════════════════════════════════
# LEFT — INPUT PANEL
# ══════════════════════════════════════════════════════
with col_input:
    with st.container(height=655, border=False):
        st.markdown("#### Event Details")

        # ── Initialise session state ───────────────────────
        if "selected_eq" not in st.session_state:
            st.session_state.selected_eq = None

        eq = None  # resolved below based on mode

        st.markdown('<div class="section-label">Recent Shocks</div>',
                    unsafe_allow_html=True)

        if df_recent.empty:
            st.markdown(
                f'<div class="alert-box alert-blue">No M{recent_min_mag:.1f}+ events '
                f'in the last {recent_days} days. Adjust sidebar filters or switch to '
                f'Manual Entry.</div>',
                unsafe_allow_html=True
            )
        else:
            # Searchable selectbox — Streamlit's selectbox supports typing to filter
            def eq_label(row):
                loc = row["place"][:45] if len(row["place"]) > 45 else row["place"]
                return (f"{mag_icon(row['magnitude'])} "
                        f"M{row['magnitude']:.1f}  ·  {loc}  ·  {time_ago(row['time'])}")

            placeholder = "— Type to search or scroll to select —"
            options = [placeholder] + [eq_label(r) for _, r in df_recent.iterrows()]

            chosen = st.selectbox(
                f"M{recent_min_mag:.1f}+ events · last {recent_days}d · "
                f"{len(df_recent)} found",
                options,
                key="eq_select"
            )

            if chosen != placeholder:
                idx = options.index(chosen) - 1
                st.session_state.selected_eq = df_recent.iloc[idx].to_dict()
                eq = st.session_state.selected_eq
            else:
                eq = None

        if eq:
            dlbl, _ = depth_label(eq["depth"])
            st.markdown(
                f'<div class="event-card">'
                f'<strong>{mag_icon(eq["magnitude"])} M{eq["magnitude"]:.1f}</strong>'
                f'&nbsp;—&nbsp;{eq["place"]}<br>'
                f'⏱ {time_ago(eq["time"])}<br>'
                f'📍 {eq["latitude"]:.3f}°N, {eq["longitude"]:.3f}°E<br>'
                f'⬇ Depth: <strong>{eq["depth"]:.0f} km</strong> '
                f'<span style="color:#f5a5a5;">({dlbl})</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # ── Fault selector (shared between both modes) ─────
        st.markdown('<div class="section-label">Nearest Major Fault</div>',
                    unsafe_allow_html=True)

        fault_region = st.radio("", ["Myanmar / SE Asia", "Global"],
                                horizontal=True, label_visibility="collapsed",
                                key="fault_region")
        fault_pool   = {k: v for k, v in FAULTS.items()
                        if v["region"] == ("myanmar" if fault_region.startswith("Myanmar") else "global")
                        or k == "No specific fault / Unknown"}
        fault_names  = list(fault_pool.keys())

        # Auto-suggest closest fault
        default_idx = 0
        if eq:
            closest = closest_fault(eq["latitude"], eq["longitude"], fault_names)
            if closest in fault_names:
                default_idx = fault_names.index(closest)

        selected_fault = st.selectbox("", fault_names,
                                    index=default_idx, label_visibility="collapsed",
                                    key="fault_select")
        fault_info = FAULTS.get(selected_fault, {})

        if fault_info.get("lat"):
            ref_lat = eq["latitude"]  if eq else fault_info["lat"]
            ref_lon = eq["longitude"] if eq else fault_info["lon"]
            prox_km = fault_proximity_km(ref_lat, ref_lon, selected_fault)
            badge_css = "fault-mm" if fault_info.get("region") == "myanmar" else "fault-gl"
            badge_txt = "Myanmar" if fault_info.get("region") == "myanmar" else "Global"
            prox_str  = (f'<br>📏 ~<strong>{prox_km:.0f} km</strong> from event epicentre'
                        if eq else "")
            st.markdown(
                f'<div class="event-card">'
                f'<strong>{selected_fault}</strong>'
                f'<span class="fault-badge {badge_css}">{badge_txt}</span><br>'
                f'{fault_info["type"]} &nbsp;·&nbsp; ~{fault_info["length_km"]:,} km<br>'
                f'<span style="color:#777;font-size:0.77rem;">{fault_info["note"]}</span>'
                f'{prox_str}</div>',
                unsafe_allow_html=True
            )

        # ── Parameter form ─────────────────────────────────
        st.markdown(
            f'<div class="section-label">'
            f'{"Auto-filled parameters (editable)" if eq else "Event parameters"}'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.form("predictor_form"):
            default_mag   = float(eq["magnitude"]) if eq else 5.5
            default_depth = float(eq["depth"])     if eq else 20.0
            default_hours = float(
                (pd.Timestamp.now(tz="UTC") - eq["time"]).total_seconds() / 3600
            ) if eq else 1.0

            magnitude = st.slider("Magnitude", 3.0, 9.5,
                                value=round(default_mag, 1), step=0.1,
                                help="Main shock magnitude — auto-filled from selected event")
            depth = st.number_input("Depth (km)", 0.0, 700.0,
                                    value=round(default_depth, 1), step=1.0,
                                    help="Hypocentre depth in km")
            seismicity_rate = st.slider(
                "Background seismicity (M3+ events, 30 days, 200 km radius)",
                0, 100, 14,
                help="Count from USGS catalog search, or leave at default"
            )
            hours_since = st.number_input(
                "Hours since main shock", 0.0, 720.0,
                value=round(min(default_hours, 720.0), 1), step=0.5,
                help="Auto-filled from selected event — adjust if needed"
            )
            submitted = st.form_submit_button(
                "🔮 Predict Aftershock Risk",
                use_container_width=True, type="primary"
            )

        # Quick depth + proximity indicators
        dlbl, dcss = depth_label(depth)
        st.markdown(
            f'<div class="alert-box {dcss}" style="margin-top:6px;">'
            f'Depth class: <strong>{dlbl}</strong></div>',
            unsafe_allow_html=True
        )
        if fault_info.get("lat") and eq:
            p   = fault_proximity_km(eq["latitude"], eq["longitude"], selected_fault)
            css = "alert-red" if p < 50 else "alert-orange" if p < 150 else "alert-blue"
            st.markdown(
                f'<div class="alert-box {css}">'
                f'Fault distance: <strong>~{p:.0f} km</strong></div>',
                unsafe_allow_html=True
            )
    
# ══════════════════════════════════════════════════════
# CENTER — OUTPUT
# ══════════════════════════════════════════════════════
with col_output:
    with st.container(height=655, border=False):
        st.markdown("#### Prediction Output")

        if not submitted:
            st.markdown(
                '<div class="alert-box alert-blue" style="text-align:center;padding:30px 14px;">'
                '← Configure the event details on the left<br>'
                'then click <strong>Predict</strong></div>',
                unsafe_allow_html=True
            )
            st.plotly_chart(make_gauge(0.0, "Low"), use_container_width=True)
        else:
            fault_prox = (
                fault_proximity_km(eq["latitude"], eq["longitude"], selected_fault)
                if (eq and fault_info.get("lat")) else
                (200.0 if fault_info.get("lat") else 9999.0)
            )
            fault_type = fault_info.get("type", "Unknown")
            eq_lat = eq["latitude"]  if eq else None
            eq_lon = eq["longitude"] if eq else None

            result = run_prediction(magnitude, depth, seismicity_rate,
                                    hours_since, fault_prox, fault_type,
                                    eq_lat, eq_lon)

            prob = result["prob_72h"]
            risk = result["risk_level"]

            st.plotly_chart(make_gauge(prob, risk), use_container_width=True)

            verdict_map = {
                "High":     ("alert-red",    "⚠️ High risk — maintain precautionary measures"),
                "Moderate": ("alert-orange", "⚡ Moderate risk — monitor situation closely"),
                "Low":      ("alert-green",  "✅ Lower risk — continue standard monitoring"),
            }
            vcss, vtxt = verdict_map[risk]
            st.markdown(
                f'<div class="alert-box {vcss}" style="text-align:center;">'
                f'<strong>{vtxt}</strong></div>',
                unsafe_allow_html=True
            )

            if eq:
                st.markdown(
                    f'<div class="alert-box alert-purple">'
                    f'📌 <strong>M{eq["magnitude"]:.1f}</strong> — {eq["place"]}<br>'
                    f'Fault: <strong>{selected_fault}</strong>'
                    f'{f" (~{fault_prox:.0f} km)" if fault_info.get("lat") else ""}'
                    f'</div><br>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="alert-box alert-purple">'
                    f'📌 Manual input · Fault: <strong>{selected_fault}</strong>'
                    f'</div><br>', unsafe_allow_html=True
                )

            st.markdown("**Probability by time window**")
            st.plotly_chart(
                make_time_window_chart(
                    result["prob_24h"], result["prob_48h"], result["prob_72h_only"]
                ),
                use_container_width=True
            )

            # m1, m2, m3 = st.columns(3)
            # m1.metric("0–24 h",  f"{result['prob_24h']*100:.0f}%")
            # m2.metric("24–48 h", f"{result['prob_48h']*100:.0f}%")
            # m3.metric("48–72 h", f"{result['prob_72h_only']*100:.0f}%")

            st.markdown("**Key risk factors**")
            chip_map = {"high": "chip-red", "medium": "chip-orange", "low": "chip-green"}
            st.markdown(
                "".join(f'<span class="risk-chip {chip_map[l]}">{n}</span>'
                        for n, l in result["drivers"]),
                unsafe_allow_html=True
            )

            if not result.get("real_model"):
                st.markdown(
                    '<div class="alert-box alert-orange" style="margin-top:10px;">'
                    '⚠️ Demo model output — run the Colab notebook and place model files '
                    'in <code>models/</code> for real predictions.</div>',
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════════════════════
# RIGHT — SHAP EXPLANATION
# ══════════════════════════════════════════════════════
with col_explain:
    with st.container(height=655, border=False):
        st.markdown("#### Why this prediction?")

        if not submitted:
            st.caption("Feature importance will appear after prediction.")
            st.markdown(
                '<div class="alert-box alert-blue" style="margin-top:10px;">'
                '🔍 SHAP values quantify each feature\'s contribution to the probability, '
                'making the model transparent and trustworthy for decision-makers.</div>',
                unsafe_allow_html=True
            )
        else:
            label = "SHAP values (real model)" if result.get("real_model") else "Feature contributions (demo)"
            st.caption(f"{label} · Red = higher risk · Blue = lower risk")
            st.plotly_chart(make_shap_chart(result["shap_values"]), use_container_width=True)

            for feat, val in sorted(result["shap_values"].items(),
                                    key=lambda x: abs(x[1]), reverse=True):
                direction = "▲" if val > 0 else "▼"
                color     = "#f5a5a5" if val > 0 else "#7ac4f5"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.78rem;padding:3px 0;border-bottom:1px solid #1e1e1e;">'
                    f'<span style="color:#aaa;">{feat}</span>'
                    f'<span style="color:{color};font-family:IBM Plex Mono;font-weight:600;">'
                    f'{direction} {abs(val):.4f}</span></div>',
                    unsafe_allow_html=True
                )

        st.divider()
        if fault_info.get("lat"):
            st.markdown("**Selected fault**")
            st.markdown(
                f'<div class="alert-box alert-purple">'
                f'<strong>{selected_fault}</strong><br>'
                f'{fault_info.get("type","—")} · ~{fault_info.get("length_km","?")} km<br>'
                f'<span style="font-size:0.75rem;color:#aaa;">{fault_info.get("note","")}</span>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown(
            '<div class="disclaimer">⚠️ <strong>Important:</strong> Probabilistic estimate '
            'only — not a guarantee of future seismic events. Fault proximity uses centroid '
            'approximations, not full fault trace geometry. Always follow guidance from '
            'official local emergency management authorities.</div>',
            unsafe_allow_html=True
        )

############################
# FEEDBACK
############################
st.divider()
with st.popover("💬 Give Feedback for Prediction", width="stretch"):
    rating  = st.feedback("stars", key="pred_rating")
    fb_text = st.text_area("Was the prediction useful? Any comments?",
                               height=80, key="pred_fb_text")

    if st.button("Submit Feedback", key="pred_fb_btn", type="primary"):
        fb_path = "data/feedback.json"
        os.makedirs("data", exist_ok=True)
        existing = []
        if os.path.exists(fb_path):
            with open(fb_path) as f:
                existing = json.load(f).get("feedback", [])
        existing.append({
            "page":        "Predictor",
            "rating":      rating,
            "comment":     fb_text,
            "time":        datetime.now().isoformat(),
            "model_type":  "real" if MODEL_LOADED else "demo",
            "inputs": {
            "magnitude":      magnitude if submitted else None,
            "depth":          depth if submitted else None,
            "fault":          selected_fault,
            "selected_event": eq["place"] if (submitted and eq) else None,
            }
        })
        with open(fb_path, "w") as f:
            json.dump({"feedback": existing}, f, indent=2)
        st.success("Thanks! Your feedback helps improve the model.")