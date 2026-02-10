# 📡 Live Earthquake Monitor

[![Spotify](https://img.shields.io/badge/Dev%20Soundtrack-Kids%20(The%20Midnight)-1DB954?logo=spotify&logoColor=white)](https://open.spotify.com/track/5DDDOmwf5A73gx139EfoOU?si=ec769bf36364473f)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Shawarma-Support-yellow?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/htutmyatoo)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
[![Data Source](https://img.shields.io/badge/Data%20Source-USGS%20GeoJSON-orange)](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson)
![Visualization](https://img.shields.io/badge/Visualization-Plotly-informational)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=htutmyatoo.live-earthquake-monitor)
![Last Commit](https://img.shields.io/github/last-commit/htutmyatoo/live-earthquake-monitor)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/github/license/htutmyatoo/live-earthquake-monitor)

An interactive, real-time **earthquake monitoring dashboard** built with **Streamlit** to visualize seismic activity both in **Myanmar** and globally using live data from the **USGS Earthquake Hazards Program**.

## 🖥️ Demo App
<div align="center">
  <a href="https://live-earthquake-monitor.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
  </a>
</div>

![Myanmar Demo](assets/myanmar_demo.png)
![Global Demo](assets/global_demo.png)

## 🎯 Purpose & Design Philosophy

Earthquake data is abundant, but **raw data alone does not create understanding**.

This dashboard aims to:
- Convert live seismic data into **human-readable insights**
- Support **regional awareness** (Myanmar-focused view)
- Provide **trend-based context**, not just points on a map

## 🚀 Key Features

### 🌍 Geographic Scope
- **Myanmar Mode 🇲🇲**
  - Filters earthquakes within Myanmar’s geographic bounds
- **Global Mode 🌍**
  - Displays worldwide seismic activity for comparison

### 📊 Visualization Modes
- **Epicenter Map**
  - Interactive geographic distribution of earthquakes
  - Size and color represent magnitude
- **Activity Timeline**
  - Daily earthquake frequency
  - Useful for spotting trends and anomalies

### 📈 Real-Time Metrics
- Total earthquakes detected
- Maximum recorded magnitude
- Average depth (km)
- Time since the last earthquake (human-readable)

### 🧠 Insight Layer
- Most active region in the last 24 hours
- Activity trend:
  - Increasing ↑
  - Decreasing ↓
  - Stable →

### 🎛️ User Controls
- Minimum magnitude filter
- Map color scale selector
- Manual data refresh
- Cached live data (auto-refresh every 10 minutes)

## 🧠 Data Source

[**USGS Earthquake Hazards Program (GeoJSON API)**](https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson)

### Data Fields Used
- `place` — textual location description
- `time` — UTC timestamp
- `mag` — earthquake magnitude
- `depth` — depth in kilometers
- `latitude`, `longitude` — epicenter coordinates

## ⚙️ Installation & Local Setup

### Project Structure
```
live-earthquake-monitor/
│
├── streamlit_app.py      # Main dashboard application
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── LICENSE               # MIT License
├── .gitignore
└── assets/               # Screenshots & media
```

### 1. Install Prerequisites
- Git
- Python 3.8+
- A code editor (VS Code works well)
- Internet connection (live API)

### 2. Clone the Repository
```sh
git clone https://github.com/htutmyatoo/live-earthquake-monitor.git
cd live-earthquake-monitor
```

### 3. Create a Python Virtual Environment
#### macOS / Linux
```sh
python3 -m venv venv
source venv/bin/activate
```
#### Windows (PowerShell)
```sh
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the App
```sh
streamlit run streamlit_app.py
```

Open in your browser:
```sh
http://localhost:8501
```

## 🧭 References

- [**Streamlit Community Tutorial**](https://discuss.streamlit.io/t/building-a-dashboard-in-python-using-streamlit/60621)
- [**DataProfessor — Population Dashboard**](https://github.com/dataprofessor/population-dashboard)
