"""
Home.py — Live Earthquake Monitor Dashboard
Displays real-time seismic activity with map, timeline, and depth analysis.
"""

############################
# IMPORT LIBRARIES
############################
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from numpy.random import default_rng as rng

############################
# PAGE CONFIGURATION
############################
st.set_page_config(
    page_title="Myay Gyi AI | Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state= "auto",
    menu_items={
        'Report a bug': "https://matrix.to/#/@htutmyatoo:matrix.org",
    }
)

############################
# SHARED CSS (imported by all pages)
############################
def load_css():
    """Inject shared CSS styles across all pages."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
    }     
    [data-testid="block-container"] {
        padding: 1rem 2rem 0rem 2rem;
        max-width:95% !important;
    }
    [data-testid="stMetric"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        text-align: center;
        padding: 15px 0;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .alert-box {
        padding: 10px 14px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 0.82rem;
        font-weight: 500;
        line-height: 1.5;
    }
    .alert-red    { background:#2d1212; border-left:3px solid #e74c3c; color:#f5a5a5; }
    .alert-orange { background:#2d1e12; border-left:3px solid #e67e22; color:#f5c97a; }
    .alert-blue   { background:#12202d; border-left:3px solid #3498db; color:#7ac4f5; }
    .alert-green  { background:#12291a; border-left:3px solid #2ecc71; color:#7af5a5; }
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #666;
        margin: 16px 0 6px;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 4px;
    }
    .page-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #555;
        letter-spacing: 0.06em;
        margin-bottom: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

############################
# DATA LOADING
############################
@st.cache_data(ttl=600)
def load_earthquake_data():
    """
    Fetch last 30 days of earthquake data from USGS GeoJSON feed.
    Returns cleaned DataFrame with engineered depth and magnitude categories.
    """
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for eq in data["features"]:
        prop = eq["properties"]
        geo  = eq["geometry"]
        mag  = prop["mag"]
        if mag is None:
            continue
        rows.append({
            "place":     prop["place"],
            "time":      pd.to_datetime(prop["time"], unit="ms"),
            "magnitude": mag,
            "depth":     geo["coordinates"][2],
            "longitude": geo["coordinates"][0],
            "latitude":  geo["coordinates"][1],
            "tsunami":   prop.get("tsunami", 0),
        })

    df = pd.DataFrame(rows)

    df["depth_category"] = pd.cut(
        df["depth"],
        bins=[0, 70, 300, 10000],
        labels=["Shallow", "Intermediate", "Deep"]
    )
    df["mag_category"] = pd.cut(
        df["magnitude"],
        bins=[0, 2, 4, 6, 8, 12],
        labels=["Minor (<2)", "Light (2–4)", "Moderate (4–6)", "Strong (6–8)", "Major (8+)"]
    )
    return df


############################
# LOAD & FILTER DATA
############################
try:
    df = load_earthquake_data()
except Exception as e:
    st.error(f"⚠️ Failed to load USGS data: {e}")
    st.stop()

############################
# SIDEBAR
############################
with st.sidebar:
    st.markdown("# Myay Gyi AI 📡")
    st.markdown("*Real-Time Seismic Monitor*")

    st.markdown('<div class="section-label">Region</div>', unsafe_allow_html=True)
    region = st.radio("", ["Myanmar 🇲🇲", "Global 🌍"], horizontal=True, label_visibility="collapsed")

    st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
    min_mag   = st.slider("Min Magnitude", 0.0, 8.0, 4.5, 0.1)
    days_back = st.slider("Time Window (days)", 1, 30, 30)
    depth_filter = st.multiselect(
        "Depth Category",
        ["Shallow", "Intermediate", "Deep"],
        default=["Shallow", "Intermediate", "Deep"]
    )

    st.markdown('<div class="section-label">Visualisation</div>', unsafe_allow_html=True)
    view_mode    = st.radio("", ["Epicenter Map", "Activity Timeline", "Depth Analysis"], label_visibility="collapsed")
    color_theme  = st.selectbox("Map Color Scale", ["inferno", "plasma", "magma", "viridis", "cividis"])

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """<div style="font-size:0.72rem;color:#555;line-height:1.8;margin-top:12px;">
        Data: USGS GeoJSON API<br>
        Auto-refresh: 10 mins<br>
        Developer: Htut Myat Oo<br>
        Version: 2.0.0
        </div>""",
        unsafe_allow_html=True
    )

############################
# FILTER DATA
############################
is_myanmar = region.startswith("Myanmar")
cutoff     = pd.Timestamp.now() - pd.Timedelta(days=days_back)

df_f = df.copy()
if is_myanmar:
    df_f = df_f[df_f["latitude"].between(5, 35) & df_f["longitude"].between(85, 110)]
df_f = df_f[df_f["time"] >= cutoff]
df_f = df_f[df_f["magnitude"] >= min_mag]
if depth_filter:
    df_f = df_f[df_f["depth_category"].isin(depth_filter)]

############################
# HELPER FUNCTIONS
############################
def time_ago(ts):
    """Convert timestamp to human-readable relative time string."""
    secs = int((pd.Timestamp.now() - ts).total_seconds())
    if secs < 60:    return "Just now"
    elif secs < 3600: return f"{secs//60}m ago"
    elif secs < 86400:
        return f"{secs//3600}h {(secs%3600)//60}m ago"
    return f"{secs//86400}d ago"


def last_quake_status(ts):
    """Return Streamlit status element type and label based on event recency."""
    hours = (pd.Timestamp.now() - ts).total_seconds() / 3600
    label = time_ago(ts)
    if hours < 1:    return "error", label
    elif hours < 6:  return "warning", label
    return "success", label


def get_trend(df):
    """
    Compare event count in last 24h vs previous 24h.
    Returns arrow symbol, descriptive text, and delta color string.
    """
    now  = pd.Timestamp.now()
    t0   = df[df["time"] >= now - pd.Timedelta(hours=24)]
    t1   = df[(df["time"] < now - pd.Timedelta(hours=24)) &
              (df["time"] >= now - pd.Timedelta(hours=48))]
    if t1.empty:
        return "→", "No comparison data", "off"
    diff = len(t0) - len(t1)
    if diff > len(t1) * 0.1:
        return "↑", f"+{diff} vs yesterday", "inverse"
    elif diff < -len(t1) * 0.1:
        return "↓", f"{diff} vs yesterday", "normal"
    return "→", "Stable vs yesterday", "off"


def mag_icon(m):
    """Return coloured circle emoji indicating magnitude severity."""
    if m >= 6:   return "🔴"
    elif m >= 5: return "🟠"
    elif m >= 4: return "🟡"
    return "🟢"


############################
# CHART FUNCTIONS
############################
def make_map(df, theme, myanmar):
    """
    Build interactive scatter mapbox coloured by magnitude.
    Auto-zooms to Myanmar when myanmar=True.
    """
    center = {"lat": 19.0, "lon": 96.0} if myanmar else {"lat": 20, "lon": 0}
    zoom   = 4 if myanmar else 1
    fig = px.scatter_mapbox(
        df, lat="latitude", lon="longitude",
        color="magnitude", size=df["magnitude"].clip(lower=1), size_max=18,
        hover_name="place",
        hover_data={"magnitude": ":.1f", "depth": ":.0f", "time": True,
                    "latitude": False, "longitude": False},
        color_continuous_scale=theme, zoom=zoom, center=center,
        mapbox_style="carto-darkmatter", template="plotly_dark"
    )
    fig.update_layout(
        height=450, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Mag", thickness=10, len=0.5)
    )
    return fig


def make_timeline(df):
    """
    Dual-axis chart: daily earthquake count (bars) + average magnitude (line).
    Reveals both frequency and intensity trends simultaneously.
    """
    df = df.copy()
    df["date"] = df["time"].dt.date
    daily = df.groupby("date").agg(
        count=("magnitude","count"),
        avg_mag=("magnitude","mean")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["count"],
                         name="Count", marker_color="#e67e22", opacity=0.75))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["avg_mag"],
                             name="Avg Magnitude", yaxis="y2",
                             line=dict(color="#3498db", width=2), mode="lines+markers"))
    fig.update_layout(
        height=380, template="plotly_dark",
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis=dict(title="Event Count"),
        yaxis2=dict(title="Avg Magnitude", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified"
    )
    return fig


def make_depth_scatter(df):
    """
    Scatter plot of depth vs magnitude, coloured by depth category.
    Used to identify shallow high-magnitude events (highest damage risk).
    """
    fig = px.scatter(
        df, x="depth", y="magnitude", color="depth_category",
        size="magnitude", size_max=14, hover_name="place",
        labels={"depth":"Depth (km)","magnitude":"Magnitude","depth_category":"Depth Type"},
        color_discrete_map={
            "Shallow":         "#e74c3c",
            "Intermediate": "#f39c12",
            "Deep":           "#3498db"
        },
        template="plotly_dark"
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=10, b=20),
                      legend=dict(orientation="h", y=1.08))
    return fig


def make_mag_histogram(df):
    """Stacked magnitude histogram coloured by depth category."""
    fig = px.histogram(
        df, x="magnitude", nbins=30, color="depth_category",
        color_discrete_map={
            "Shallow":         "#e74c3c",
            "Intermediate": "#f39c12",
            "Deep":           "#3498db"
        },
        template="plotly_dark", barmode="stack"
    )
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=10,b=20),
                      showlegend=False,
                      xaxis_title="Magnitude", yaxis_title="Count")
    return fig


############################
# LAYOUT
############################
title_region = "Myanmar 🇲🇲" if is_myanmar else "Global 🌍"
st.markdown(f'<div class="page-header">MYAY GYI AI/ LIVE DASHBOARD / {title_region.upper()}</div>', unsafe_allow_html=True)
st.markdown("### 📡 Real-Time Seismic Monitor")

col_l, col_c, col_r = st.columns((1.5, 4.5, 2), gap="medium")

# ── LEFT: Metrics ──────────────────────────────────────────────
with col_l:
    with st.container(height=655, border=False):
        st.markdown("#### Summary")

        if df_f.empty:
            st.warning("No events match current filters.")
        else:
            latest   = df_f.sort_values("time", ascending=False).iloc[0]
            strongest = df_f.loc[df_f["magnitude"].idxmax()]
            status, label = last_quake_status(latest["time"])
            arrow, trend_txt, delta_col = get_trend(df_f)

            st.metric("Total Events",      len(df_f))
            st.metric("Max Magnitude",     f"{df_f['magnitude'].max():.1f}")
            st.metric("Avg Depth (km)",    f"{df_f['depth'].mean():.1f}")
            st.metric("vs Yesterday",      arrow, trend_txt, delta_color=delta_col)

            st.markdown("**⏱ Last Event**")
            getattr(st, status)(label)

            st.markdown("**🌋 Strongest Event**")
            st.markdown(
                f'<div class="alert-box alert-red">'
                f'M{strongest["magnitude"]:.1f} — {strongest["place"][:30]}<br>'
                f'<span style="font-size:0.72rem;opacity:0.8;">{time_ago(strongest["time"])}</span>'
                f'</div>', unsafe_allow_html=True
            )

            tsunami_ct = int(df_f["tsunami"].sum())
            if tsunami_ct:
                st.markdown(
                    f'<div class="alert-box alert-orange">⚠️ {tsunami_ct} tsunami-associated event(s)</div>',
                    unsafe_allow_html=True
                )

            st.markdown("**Magnitude Breakdown**")
            for cat, cnt in df_f["mag_category"].value_counts().sort_index().items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.78rem;padding:2px 0;">'
                    f'<span style="color:#999;">{cat}</span>'
                    f'<span style="font-weight:600;">{cnt}</span></div>',
                    unsafe_allow_html=True
                )

# ── CENTER: Charts + Table ──────────────────────────────────────
with col_c:
    with st.container(height=655, border=False):
        df = rng(0).standard_normal((10, 1))

        tab1, tab2, tab3 = st.tabs([f"{view_mode}", "Magnitude Distribution", "Recent Events"])

        if df_f.empty:
            st.info("No data to display for the current filters.")
        else:
            if view_mode == "Epicenter Map":
                tab1.header(f"Epicenter Map — {title_region}")
                tab1.plotly_chart(make_map(df_f, color_theme, is_myanmar), use_container_width=True)

            elif view_mode == "Activity Timeline":
                tab1.header(f"Daily Activity — {title_region}")
                tab1.plotly_chart(make_timeline(df_f), use_container_width=True)

            else:
                tab1.header(f"Depth vs Magnitude — {title_region}")
                tab1.plotly_chart(make_depth_scatter(df_f), use_container_width=True)

            tab2.header("Magnitude Distribution")
            tab2.plotly_chart(make_mag_histogram(df_f), use_container_width=True)

            tab3.header("Recent Events")
            table = df_f.sort_values("time", ascending=False).head(15).copy()
            table["Time"]       = table["time"].dt.strftime("%d %b %H:%M")
            table["Location"]   = table["place"].str.split(",").str[0]
            table["Mag"]        = table["magnitude"].round(1)
            table["Depth (km)"] = table["depth"].round(0).astype(int)
            table["Type"]       = table["depth_category"].astype(str)
            tab3.dataframe(table[["Time","Location","Mag","Depth (km)","Type"]],
                        hide_index=True, use_container_width=True)
            
            ############################
            # FEEDBACK (bottom of every page)
            ############################
            with st.popover("💬 Give Feedback for Dashboard", width="stretch"):
                rating   = st.feedback("stars", key="home_rating")
                fb_text  = st.text_area("Any comments or suggestions?", height=80, key="home_fb_text")
                if st.button("Submit Feedback", key="home_fb_submit", type="primary"):
                    import json, os
                    from datetime import datetime
                    fb_path = "data/feedback.json"
                    os.makedirs("data", exist_ok=True)
                    existing = []
                    if os.path.exists(fb_path):
                        with open(fb_path) as f:
                            existing = json.load(f).get("feedback", [])
                    existing.append({
                        "page":    "Dashboard",
                        "rating":  rating,
                        "comment": fb_text,
                        "time":    datetime.now().isoformat()
                    })
                    with open(fb_path, "w") as f:
                        json.dump({"feedback": existing}, f, indent=2)
                    st.success("Thanks for your feedback!")                        

# ── RIGHT: Insights ─────────────────────────────────────────────
with col_r:
    with st.container(height=655, border=False):
        st.markdown("#### Insights")

        if df_f.empty:
            st.info("No data available.")
        else:
            recent_24h = df_f[df_f["time"] >= pd.Timestamp.now() - pd.Timedelta(hours=24)]
            if not recent_24h.empty:
                top_region = (
                    recent_24h["place"].str.split(",").str[0]
                    .value_counts().idxmax()
                )
                st.markdown("**📍 Most Active Region (24h)**")
                st.info(top_region)

            st.markdown("**🪨 Depth Distribution**")
            depth_counts = df_f["depth_category"].value_counts()
            total = depth_counts.sum()
            css_map = {
                "Shallow":         "alert-red",
                "Intermediate": "alert-orange",
                "Deep":           "alert-blue"
            }
            for cat, cnt in depth_counts.items():
                pct = cnt / total * 100
                css = css_map.get(str(cat), "alert-green")
                st.markdown(
                    f'<div class="alert-box {css}">{cat}<br>'
                    f'<strong>{cnt} events ({pct:.0f}%)</strong></div>',
                    unsafe_allow_html=True
                )

            shallow_major = df_f[
                (df_f["depth_category"] == "Shallow") &
                (df_f["magnitude"] >= 5.5)
            ]
            if not shallow_major.empty:
                st.markdown(
                    f'<div class="alert-box alert-red">⚠️ {len(shallow_major)} shallow M5.5+ '
                    f'event(s) — elevated surface damage risk</div>',
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🔴 Recent M5.0+ Events (72h)**")
            strong_recent = df_f[
                (df_f["magnitude"] >= 5.0) &
                (df_f["time"] >= pd.Timestamp.now() - pd.Timedelta(hours=72))
            ].sort_values("time", ascending=False).head(5)

            if strong_recent.empty:
                st.markdown(
                    '<div class="alert-box alert-green">✅ No M5.0+ events in last 72h</div>',
                    unsafe_allow_html=True
                )
            else:
                for _, row in strong_recent.iterrows():
                    st.markdown(
                        f'<div class="alert-box alert-orange">'
                        f'{mag_icon(row["magnitude"])} <strong>M{row["magnitude"]:.1f}</strong>'
                        f' — {row["place"][:28]}<br>'
                        f'<span style="font-size:0.72rem;opacity:0.8;">'
                        f'{time_ago(row["time"])} · {row["depth"]:.0f} km depth</span>'
                        f'</div>', unsafe_allow_html=True
                    )

            st.caption(
                f"{len(df_f):,} events · Last {days_back}d · M≥{min_mag} · {region}"
            )

            with st.popover("Help", icon="💡", width="stretch"):
                st.markdown("""        
                **Depth classification:**
                - 🔴 Shallow <70 km — higher surface impact
                - 🟠 Intermediate 70–300 km
                - 🔵 Deep >300 km — rarely felt
                
                **Chart Classification:**
                - **Epicenter Map**: geographic distribution of earthquake locations by magnitude
                - **Activity Timeline**: daily frequency of recorded seismic events
                - **Depth Analysis**: distribution of earthquake focal depths across recorded seismic events
                - **Magnitude Distribution**: frequency spread of earthquake magnitudes showing intensity patterns
                """)
