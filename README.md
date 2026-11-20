# 🧠 AI-Driven Digital Twin Framework
## Predictive Resource Allocation in Multi-Core Computing Systems

> **MCA Research Project** | Python · scikit-learn · psutil · Streamlit · Plotly

---

## 🎯 Overview

An intelligent framework that creates a **Digital Twin** of your multi-core computing system, uses **Machine Learning** to predict future CPU and RAM usage, applies **smart resource allocation**, and benchmarks it against traditional scheduling algorithms.

---

## 🏗️ Architecture

```
mca/
├── app.py                      # Streamlit dashboard (entry point)
├── requirements.txt
├── .streamlit/config.toml      # Dark theme config
│
├── core/
│   ├── monitor.py              # Real-time psutil monitoring
│   ├── digital_twin.py         # Digital Twin engine
│   ├── predictor.py            # ML prediction (Random Forest, GBM, Linear)
│   ├── allocator.py            # Intelligent resource allocator
│   └── scheduler.py            # Scheduling algorithms + benchmarks
│
├── data/
│   └── collector.py            # Data collection + feature engineering
│
├── ui/
│   ├── dashboard.py            # Live Monitor page
│   ├── predictor_ui.py         # Prediction & Digital Twin page
│   ├── allocator_ui.py         # Resource Allocation page
│   └── comparison_ui.py        # Scheduler Benchmark page
│
└── models/                     # Saved ML models (auto-created)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd mca
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**

---

## 📱 Features

### 🖥️ Live Monitor
- Real-time CPU (per core) and RAM gauges
- CPU core load heatmap
- Animated time-series charts
- Digital Twin deviation tracking
- Top process table

### 🤖 ML Prediction & Digital Twin
- **Models**: Random Forest, Gradient Boosting, Ridge, Linear Regression
- Train on live-collected data
- Predict CPU/RAM N steps into the future
- Confidence intervals
- Feature importance visualization
- **What-If Simulator**: Inject synthetic workloads into the digital twin

### ⚙️ Resource Allocator
- Task queue with priority scheduling
- **Traditional**: Static CPU/RAM threshold-based allocation
- **ML-Guided**: Predictive headroom-aware allocation
- Core assignment visualization
- Side-by-side performance comparison

### 📊 Scheduler Benchmark
- **5 algorithms**: FCFS, SJF, Round Robin, Priority, Predictive (ML-guided)
- Animated Gantt charts
- Metrics: Avg Wait Time, Turnaround, Throughput, CPU Utilization, Makespan
- Performance radar chart
- Improvement analysis (Predictive vs Traditional)

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

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥1.32 | Dashboard UI |
| psutil | ≥5.9 | System monitoring |
| scikit-learn | ≥1.4 | ML models |
| pandas | ≥2.2 | Data manipulation |
| plotly | ≥5.20 | Interactive charts |
| numpy | ≥1.26 | Numerical computing |
| joblib | ≥1.3 | Model persistence |
