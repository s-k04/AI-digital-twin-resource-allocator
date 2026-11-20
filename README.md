# 🧠 AI-Driven Digital Twin Framework
## Predictive Resource Allocation in Multi-Core Computing Systems

> **MCA Research Project** | Python · scikit-learn · psutil · Streamlit · Plotly

---

## 🎯 Overview

An intelligent framework that creates a **Digital Twin** of your multi-core computing system, uses **Machine Learning** to predict future CPU and RAM usage, applies **smart resource allocation**, and benchmarks it against traditional scheduling algorithms.

---

## 🎨 Scientific Design System & Palette

Designed with an elegant, premium scientific dashboard aesthetic using sophisticated typography (serif headings and clean sans-serif/monospace technical data) and the exact palette:

| Color Token | Hex Code | Purpose |
|---|---|---|
| **Space Cadet** | `#25344F` | Primary Background |
| **Slate Gray** | `#617891` | Metric Panels & Cards |
| **Tan** | `#D5B893` | Primary Accents & Highlights |
| **Coffee** | `#6F4D38` | Secondary Accents |
| **Caput Mortuum** | `#632024` | System Alerts & Critical States |

---

## 🏗️ Architecture

```
mca/
├── app.py                      # Main Streamlit dashboard (entry point)
├── requirements.txt            # Project dependencies
├── .streamlit/config.toml      # Space Cadet dark theme configuration
│
├── core/
│   ├── monitor.py              # Real-time psutil monitoring
│   ├── digital_twin.py         # Digital Twin engine
│   ├── predictor.py            # ML prediction (Random Forest, GBM, Ridge, Linear)
│   ├── allocator.py            # Intelligent resource allocator
│   └── scheduler.py            # Scheduling algorithms + benchmarks
│
├── data/
│   └── collector.py            # Data collection + feature engineering
│
├── ui/
│   ├── dashboard.py            # Live Monitor page
│   ├── predictor_ui.py         # ML Prediction & Digital Twin page
│   ├── allocator_ui.py         # Resource Allocation page
│   └── comparison_ui.py        # Scheduler Benchmark page
│
└── models/                     # Saved ML models (auto-created)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

---

## 📱 Features

### 🖥️ Live Monitor
- **Real-Time Telemetry**: Real-time CPU (per core), RAM, and Disk gauges
- **CPU Core Load Heatmap**: Core-by-core load distribution mapped to the scientific color gradient
- **Time-Series Telemetry**: Multi-variable animated curves tracking CPU, RAM, and Digital Twin state
- **Digital Twin Deviation Tracking**: Coherence and deviation metrics between digital replica and host
- **Top Process Table**: Detailed process table with resource attribution

### 🤖 ML Prediction & Digital Twin
- **Models**: Random Forest, Gradient Boosting, Ridge, Linear Regression
- **Horizon Forecasting**: Predict CPU and RAM N steps into the future with 95% confidence intervals
- **Feature Importance**: Time-series feature attribution and contribution analysis
- **What-If Simulator**: Inject synthetic workloads into the digital twin to evaluate system saturation without touching the real machine

### ⚙️ Resource Allocator
- **Dual Policy Evaluation**: Compare classical static threshold gating vs. ML-guided headroom-aware allocation
- **Task Queue**: Priority-based task scheduling
- **Core Assignment**: Multi-core load balancing with visual core distribution
- **Side-by-Side Statistics**: Benchmark wait times, rejection mitigation, and throughput

### 📊 Scheduler Benchmark
- **5 Scheduling Algorithms**: FCFS, SJF, Round Robin, Priority, and Predictive (ML-guided)
- **Gantt Charts**: Animated multi-channel Gantt visualization of task execution
- **Metrics**: Avg Wait Time, Turnaround, Throughput, CPU Utilization, Makespan
- **Performance Radar**: Multivariate radar chart visualizing comparative algorithm efficiency
- **Improvement Analysis**: Quantitative improvement evaluation (Predictive vs Traditional)

---

## 🔬 Research Topics Covered

| Topic | Implementation |
|---|---|
| CPU/RAM Prediction | scikit-learn Random Forest + GBM models |
| Real-Time Monitoring | psutil system calls |
| Digital Twin | Virtual system replica with synthetic workload injection |
| Intelligent Allocation | ML-predicted headroom-based resource reservation |
| Scheduler Comparison | FCFS vs SJF vs RR vs Priority vs Predictive |

---

## 🛠️ Technologies

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit ≥ 1.32 | Dashboard UI |
| psutil ≥ 5.9 | System monitoring |
| scikit-learn ≥ 1.4 | ML models |
| pandas ≥ 2.2 | Data manipulation |
| plotly ≥ 5.20 | Interactive charts |
| numpy ≥ 1.26 | Numerical computing |
| joblib ≥ 1.3 | Model persistence |
