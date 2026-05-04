# 📡 Myay Gyi AI: Live Earthquake Monitor & Aftershock Predictor

[![Spotify](https://img.shields.io/badge/Dev%20Soundtrack-Kids%20(The%20Midnight)-1DB954?logo=spotify&logoColor=white)](https://open.spotify.com/track/5DDDOmwf5A73gx139EfoOU?si=ec769bf36364473f)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Model-XGBoost-success)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
[![Data Source](https://img.shields.io/badge/Data%20Source-USGS%20GeoJSON-orange)](https://earthquake.usgs.gov/)
![Visualization](https://img.shields.io/badge/Visualization-Plotly-informational)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=htutmyatoo.myay-gyi-ai)
![Last Commit](https://img.shields.io/github/last-commit/htutmyatoo/live-earthquake-monitor)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/github/license/htutmyatoo/live-earthquake-monitor)

Real-time **seismic intelligence platform** built with **Streamlit** to monitor live earthquakes, explore trends, and estimate aftershock probability both in **Myanmar** and globally using live data from the **USGS Earthquake Hazards Program** and machine learning techniques.

## 🖥️ Demo App
<div align="center">
  <a href="https://myay-gyi-ai-v1.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Streamlit App">
  </a>
</div>

![Global Demo](assets/global_demo.png)

> [!CAUTION]
> Disclaimer: This tool is for educational and analytical purposes only. It is not an official emergency warning system. Always follow local authorities and seismic agencies during emergencies.

## 🧠 What is Myay Gyi AI?

> **Myay Gyi (မြေကြီး)** means **Earth** in Burmese. 

This project transforms raw earthquake feeds into an easy-to-understand intelligence dashboard with an integrated **Aftershock Risk Predictor**.

Designed for:
- Public awareness  
- Students & researchers  
- Journalists  
- Emergency planners  

## 🚀 Core Features

### 📡 1. Live Earthquake Dashboard

Track real-time earthquakes from the **USGS GeoJSON API**.

[![Myanmar Demo](assets/myanmar_demo.png)](https://myay-gyi-ai-v1.streamlit.app/)

### Includes:

- 🌍 **Global Mode**
- 🇲🇲 **Myanmar Focus Mode**
- 📍 Interactive epicenter map
- 📈 Daily activity timeline
- 🌊 Depth analysis
- 📊 Magnitude insights
- 🔄 Auto-refresh every 10 minutes

### 🔮 2. Aftershock Risk Predictor

Estimate the probability of a **M4.0+ aftershock within 72 hours** after a triggering earthquake.

[![Myanmar Demo](assets/myanmar_demo.png)](https://myay-gyi-ai-v1.streamlit.app/)

### Powered by:

- **XGBoost Classifier**
- Historical USGS seismic data
- Engineered geospatial features
- Fault proximity signals
- SHAP explainability

### Input Modes:

- Live recent earthquakes
- Manual custom scenario testing

## 🛠 Tech Stack

| Layer | Tools |
|------|------|
| Frontend | Streamlit |
| Charts | Plotly |
| Data | Pandas / NumPy |
| ML | XGBoost / Scikit-learn / Colab|
| Explainability | SHAP |
| API Source | USGS |
| Deployment | Streamlit Cloud |

## 📂 Project Structure
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

## ⚙️ Installation & Local Setup

### 1. Clone Repo
```sh
git clone https://github.com/htutmyatoo/myay-gyi-ai.git
cd myay-gyi-ai
```

### 2. Create Environment
#### Windows (PowerShell)
```sh
python -m venv venv
venv\Scripts\activate
```
#### macOS / Linux
```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Launch App
```sh
streamlit run Home.py
```

Open in your browser:
```sh
http://localhost:8501
```

## 🎯 Why This Product Matters

Most earthquake dashboards only display dots on a map.

Myay Gyi AI goes further:
- Live monitoring 
- Human-readable alerts 
- Trend detection 
- Regional focus on Myanmar 
- Predictive risk modelling 
- Explainable AI 

## 📌 Future Roadmap
- SMS / Telegram alerts
- Fault-line live overlays
- Mobile app version
- Government disaster dashboard mode
- Deep learning seismic forecasting
- Crowd-sourced felt reports

## 🧭 References

- [**Streamlit Community Tutorial**](https://discuss.streamlit.io/t/building-a-dashboard-in-python-using-streamlit/60621)
- [**DataProfessor — Population Dashboard**](https://github.com/dataprofessor/population-dashboard)
- [**XGBoost Documentation**](https://xgboost.readthedocs.io/en/release_3.2.0)
- [**Streamlit Documentation**](https://docs.streamlit.io/)
- [**SHAP Explainability Docs**](https://shap.readthedocs.io/en/latest/)

## 🍥 Support
<a href="https://ko-fi.com/J3J21UINNT" target="_blank">
  <img src="https://storage.ko-fi.com/cdn/brandasset/v2/support_me_on_kofi_dark.png?_gl=1*mz6i7q*_gcl_au*MTE3MDY3MDM4NC4xNzcxNDUyMzcx*_ga*MTY2NTkxNjMxNy4xNzcxNDUyMzcy*_ga_M13FZ7VQ2C*czE3NzI0NTgwOTQkbzEyJGcxJHQxNzcyNDU4NDc4JGo1MSRsMCRoMA.." width = 200 alt="Ko-fi.com"/>
</a>
