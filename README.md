# 🌡️ Predictive Thermal-Aware Resource Allocation Framework
## Multi-Core Computing Systems & AI Digital Twin Telemetry

> **MCA Research Project** | Python · scikit-learn · psutil · Streamlit · Plotly

---

## 🎯 Overview

An advanced scientific framework providing **Predictive Thermal-Aware Resource Allocation** for multi-core computing architectures. By combining **real-time thermodynamic hardware telemetry**, **AI Digital Twin replication**, and **machine learning regressors**, the system dynamically forecasts future core utilization and thermal bottlenecks to dispatch compute tasks onto optimal cores—mitigating thermal throttling and reducing workload rejection rates.

---

## 🎨 Scientific Design System & Palette

Designed for scientific research dashboards with sophisticated typography (serif headings and monospace technical telemetry) and an exact palette:

| Color Token | Hex Code | Purpose |
|---|---|---|
| **Space Cadet** | `#25344F` | Primary Background |
| **Slate Gray** | `#617891` | Metric Panels & Cards |
| **Tan** | `#D5B893` | Primary Accents & Highlights |
| **Coffee** | `#6F4D38` | Secondary Accents |
| **Caput Mortuum** | `#632024` | Thermal Alerts & Critical States |

---

## 🏗️ Architecture

```
mca/
├── app.py                      # Main Streamlit application entry point & scientific styling
├── requirements.txt            # Project dependencies
├── .streamlit/config.toml      # Space Cadet dark theme configuration
│
├── core/
│   ├── monitor.py              # Thermodynamic & system resource telemetry
│   ├── digital_twin.py         # AI Digital Twin simulation engine
│   ├── predictor.py            # ML forecasting (Random Forest, GBM, Ridge, Linear)
│   ├── allocator.py            # Predictive thermal-aware resource allocator
│   └── scheduler.py            # Multi-core scheduling algorithms & comparative benchmark
│
├── data/
│   └── collector.py            # Telemetry buffering & feature engineering
│
├── ui/
│   ├── dashboard.py            # Live System & Thermal Monitor
│   ├── predictor_ui.py         # ML Prediction & Digital Twin Telemetry
│   ├── allocator_ui.py         # Thermal-Aware Resource Allocator
│   └── comparison_ui.py        # Scheduling Algorithm Benchmark & Analytics
│
└── models/                     # Serialized scikit-learn model artifacts
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app.py
```

The scientific dashboard will initialize at **http://localhost:8501**.

---

## 📱 Core Dashboard Capabilities

### 🌡️ Live System & Thermal Monitor
- **Thermodynamic Telemetry**: Per-core and package temperature monitoring (°C) with thermal headroom tracking.
- **Scientific Gauges**: CPU load, RAM allocation, package thermal status, and disk utilization.
- **Multi-Core Heatmap**: Core-by-core load distribution mapped to the scientific color gradient.
- **Digital Twin Coherence**: Real-time correlation tracking between digital twin model and physical silicon.

### 🤖 ML Prediction & Digital Twin Telemetry
- **Model Regressors**: Random Forest, Gradient Boosting, Ridge, and Linear Regression.
- **Multi-Step Forecasting**: Horizon forecasting with 95% confidence intervals.
- **Thermodynamic What-If Simulator**: Virtual workload injection testing saturation and thermal throttling risk without host degradation.

### ⚙️ Thermal-Aware Resource Allocator
- **Dual Policy Evaluation**: Compare classical static threshold gating vs. ML-guided thermal headroom allocation.
- **Core Load Balancing**: Automatic dispatch to coolest available cores.
- **Audit Logging**: Comprehensive decision log capturing throughput, wait time, and rejection reductions.

### 📊 Scheduling Algorithm Benchmark & Analytics
- **5 Scheduling Algorithms**: FCFS, Shortest Job First (SJF), Round Robin, Priority, and ML-Guided Predictive.
- **Multi-Core Gantt Telemetry**: Multi-channel Gantt visualization of task execution.
- **Multivariate Radar**: Five-dimensional performance trade-off analysis.

---

## 🛠️ Tech Stack

- **UI Framework**: Streamlit ≥ 1.32
- **Data Visualization**: Plotly Interactive Engine ≥ 5.20
- **Machine Learning**: scikit-learn ≥ 1.4, joblib ≥ 1.3
- **Hardware Telemetry**: psutil ≥ 5.9
- **Data Engineering**: pandas ≥ 2.2, numpy ≥ 1.26
