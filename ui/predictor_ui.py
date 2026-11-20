"""
ML Prediction & Digital Twin UI page.
Train machine learning models, inspect forecast horizons, and execute what-if simulations.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.monitor import get_full_snapshot, get_system_info
from core.predictor import ResourcePredictor, AVAILABLE_MODELS
from core.digital_twin import DigitalTwin


def _init_state():
    if "predictor" not in st.session_state or not hasattr(st.session_state.predictor, "sample_count"):
        st.session_state.predictor = ResourcePredictor(buffer_size=500, horizon=5)
    if "pred_twin" not in st.session_state:
        info = get_system_info()
        st.session_state.pred_twin = DigitalTwin(
            num_cores=info["cpu_logical_cores"],
            total_ram_gb=info["total_ram_gb"],
        )
    if "pred_history" not in st.session_state:
        st.session_state.pred_history = {
            "ts": [], "real_cpu": [], "pred_cpu": [],
            "real_ram": [], "pred_ram": []
        }


def _make_pred_chart(history: dict, target: str = "cpu"):
    """Render prediction vs actual chart in scientific palette."""
    color_real = "#D5B893" if target == "cpu" else "#617891"
    color_pred = "#6F4D38"
    fig = go.Figure()
    ts = history["ts"]
    real = history[f"real_{target}"]
    pred = history[f"pred_{target}"]

    fig.add_trace(go.Scatter(
        x=ts, y=real, name=f"Real {target.upper()} (%)",
        line=dict(color=color_real, width=2),
        fill="tozeroy",
        fillcolor="rgba(213, 184, 147, 0.08)" if target == "cpu" else "rgba(97, 120, 145, 0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=pred, name=f"Predicted {target.upper()} (%)",
        line=dict(color=color_pred, width=2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(111, 77, 56, 0.08)"
    ))

    # Upper headroom reference line
    fig.add_hline(
        y=90, line_dash="dash", line_color="#632024", opacity=0.75,
        annotation_text="Warning Threshold 90%",
        annotation_font_color="#E8B4B8",
        annotation_font_family="JetBrains Mono"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(color="#D5B893", family="Inter", size=11),
            bgcolor="rgba(29, 42, 64, 0.6)"
        ),
        title=dict(
            text=f"{target.upper()} — Actual vs Predicted",
            font=dict(color="#D5B893", size=13, family="Playfair Display, serif")
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono"),
            ticksuffix="%", range=[0, 105]
        ),
    )
    return fig


def _make_feature_importance_chart(importance: dict, title: str):
    """Render feature importance bar chart using the scientific palette."""
    if not importance:
        return None
    names = list(importance.keys())[:10]
    vals = [importance[n] for n in names]

    palette_cycle = ["#D5B893", "#617891", "#6F4D38", "#A3B8CC", "#B89973"]
    bar_colors = [palette_cycle[i % len(palette_cycle)] for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="rgba(213, 184, 147, 0.3)", width=1)),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(color="#D5B893", family="JetBrains Mono", size=10),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=30, t=35, b=10),
        title=dict(text=title, font=dict(color="#D5B893", size=13, family="Playfair Display, serif")),
        yaxis=dict(
            tickfont=dict(color="#F4EFEB", family="JetBrains Mono", size=10),
            autorange="reversed",
            showgrid=False
        ),
        xaxis=dict(
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono"),
            showgrid=True,
            gridcolor="rgba(97, 120, 145, 0.2)"
        ),
    )
    return fig


def render():
    _init_state()
    predictor: ResourcePredictor = st.session_state.predictor
    twin: DigitalTwin = st.session_state.pred_twin

    st.markdown("## 🤖 ML Prediction & Digital Twin")
    st.markdown(
        "<p style='color:#9FB0C2;font-size:0.92rem;margin-top:-6px;margin-bottom:18px;'>"
        "Train machine learning models to predict future CPU/RAM usage and run what-if simulations."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Telemetry Collection ──────────────────────────────────────────────────
    snap = get_full_snapshot()
    cpu = snap["cpu"]["overall_percent"]
    ram = snap["ram"]["percent"]
    num_procs = len(snap["processes"])
    predictor.add_sample(cpu, ram, num_procs)
    twin.update(snap["cpu"]["per_core_percent"], ram)

    samples = getattr(predictor, "sample_count", len(getattr(predictor, "data_buffer", [])))
    if samples < 50:
        st.info(f"📊 Buffer: **{samples}** samples collected  |  Minimum to train: **50** (Click **⚡ Pre-Fill 120 Samples** below to test immediately!)")
    else:
        st.success(f"📊 Buffer: **{samples}** samples collected  |  Ready to train!")

    # ── Model Configuration Form ──────────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        cpu_model = st.selectbox("CPU Model", list(AVAILABLE_MODELS.keys()), index=0, key="cpu_model_sel")
    with tc2:
        ram_model = st.selectbox("RAM Model", list(AVAILABLE_MODELS.keys()), index=0, key="ram_model_sel")
    with tc3:
        horizon = st.slider("Prediction Horizon (steps)", 1, 30, 5, key="pred_horizon")
        predictor.horizon = horizon

    col_train, col_fill, col_reset = st.columns([2, 2, 1])
    with col_train:
        if st.button("🚀 Train Models", key="train_btn", type="primary", disabled=(samples < 50)):
            with st.spinner("Training ML models..."):
                metrics = predictor.train(cpu_model, ram_model)
            if "error" in metrics:
                st.error(metrics["error"])
            else:
                st.success("✅ Models trained successfully!")
                st.session_state.train_metrics = metrics
    with col_fill:
        if st.button("⚡ Pre-Fill 120 Samples", key="fill_buf_btn", help="Quickly populates buffer with realistic history"):
            predictor.generate_synthetic_samples(120, base_cpu=cpu, base_ram=ram)
            st.rerun()
    with col_reset:
        if st.button("🗑️ Reset Buffer", key="reset_buf_btn"):
            predictor.data_buffer.clear()
            predictor.trained = False
            st.session_state.pred_history = {
                "ts": [], "real_cpu": [], "pred_cpu": [],
                "real_ram": [], "pred_ram": []
            }
            st.rerun()

    # ── Model Validation Metrics ──────────────────────────────────────────────
    if "train_metrics" in st.session_state and st.session_state.train_metrics:
        m = st.session_state.train_metrics
        st.markdown("### 📊 Model Performance Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f"<div style='border-bottom:1px solid rgba(213,184,147,0.3);padding-bottom:4px;margin-bottom:8px;'>"
                f"<span style='font-family:\"Playfair Display\",serif;color:#D5B893;font-weight:600;'>CPU Model ({cpu_model})</span>"
                f"</div>", unsafe_allow_html=True
            )
            cm = m.get("cpu", {})
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE", f"{cm.get('MAE', 0):.3f}%")
            mc2.metric("RMSE", f"{cm.get('RMSE', 0):.3f}%")
            mc3.metric("R²", f"{cm.get('R2', 0):.3f}")
        with m2:
            st.markdown(
                f"<div style='border-bottom:1px solid rgba(97,120,145,0.4);padding-bottom:4px;margin-bottom:8px;'>"
                f"<span style='font-family:\"Playfair Display\",serif;color:#617891;font-weight:600;'>RAM Model ({ram_model})</span>"
                f"</div>", unsafe_allow_html=True
            )
            rm = m.get("ram", {})
            mr1, mr2, mr3 = st.columns(3)
            mr1.metric("MAE", f"{rm.get('MAE', 0):.3f}%")
            mr2.metric("RMSE", f"{rm.get('RMSE', 0):.3f}%")
            mr3.metric("R²", f"{rm.get('R2', 0):.3f}")

    # ── Live Inference & Predictive Curves ────────────────────────────────────
    if predictor.trained:
        pred = predictor.predict()
        if pred:
            st.markdown("### 🔮 Live Prediction")
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(
                "Predicted CPU", f"{pred['cpu_predicted']}%",
                delta=f"{pred['cpu_predicted']-cpu:+.1f}% from now"
            )
            p2.metric(
                "CPU CI",
                f"[{pred['cpu_ci_low']}% — {pred['cpu_ci_high']}%]"
            )
            p3.metric(
                "Predicted RAM", f"{pred['ram_predicted']}%",
                delta=f"{pred['ram_predicted']-ram:+.1f}% from now"
            )
            p4.metric(
                "RAM CI",
                f"[{pred['ram_ci_low']}% — {pred['ram_ci_high']}%]"
            )

            # Append to history
            h = st.session_state.pred_history
            h["ts"].append(datetime.now())
            h["real_cpu"].append(cpu)
            h["pred_cpu"].append(pred["cpu_predicted"])
            h["real_ram"].append(ram)
            h["pred_ram"].append(pred["ram_predicted"])
            for k in h:
                if isinstance(h[k], list) and len(h[k]) > 60:
                    h[k] = h[k][-60:]

        # Charts
        if len(st.session_state.pred_history["ts"]) > 2:
            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(
                    _make_pred_chart(st.session_state.pred_history, "cpu"),
                    use_container_width=True, key="pred_cpu_chart"
                )
            with pc2:
                st.plotly_chart(
                    _make_pred_chart(st.session_state.pred_history, "ram"),
                    use_container_width=True, key="pred_ram_chart"
                )

        # Feature Importance
        st.markdown("### 🧠 Feature Importance")
        fi1, fi2 = st.columns(2)
        with fi1:
            imp_cpu = predictor.get_feature_importance("cpu")
            if imp_cpu:
                fig_imp = _make_feature_importance_chart(imp_cpu, "CPU Model — Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True, key="fi_cpu")
        with fi2:
            imp_ram = predictor.get_feature_importance("ram")
            if imp_ram:
                fig_imp = _make_feature_importance_chart(imp_ram, "RAM Model — Feature Importance")
                st.plotly_chart(fig_imp, use_container_width=True, key="fi_ram")

    # ── Digital Twin What-If Simulator ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Digital Twin — What-If Simulator")
    st.markdown(
        "<p style='color:#9FB0C2;font-size:0.92rem;margin-top:-6px;'>"
        "Inject a synthetic workload into the <b>Digital Twin</b> to see how the system would respond <i>without</i> touching the real machine."
        "</p>",
        unsafe_allow_html=True
    )

    ws1, ws2, ws3 = st.columns(3)
    with ws1:
        wi_cpu = st.slider("Extra CPU Load (%)", 0, 80, 20, key="wi_cpu")
    with ws2:
        wi_ram = st.slider("Extra RAM (GB)", 0.0, 8.0, 1.0, step=0.5, key="wi_ram")
    with ws3:
        wi_dur = st.slider("Duration (s)", 5, 120, 30, key="wi_dur")

    if st.button("▶️ Run Simulation", key="whatif_btn", type="primary"):
        scenario = {"extra_cpu_pct": wi_cpu, "extra_ram_gb": wi_ram, "duration_s": wi_dur}
        result = twin.run_whatif_simulation(scenario)
        twin.inject_workload("What-If Load", wi_cpu, wi_ram, wi_dur)

        sat = result["saturation_risk"]
        card_border = "#632024" if sat else "#D5B893"
        card_bg = "rgba(99, 32, 36, 0.28)" if sat else "rgba(97, 120, 145, 0.22)"
        status_title = "Saturation Risk Detected" if sat else "Simulation Result — Nominal"

        st.markdown(
            f'<div style="background:{card_bg};border:1px solid {card_border};border-left:4px solid {card_border};'
            f'padding:16px;border-radius:8px;margin-top:10px;">'
            f'<h4 style="font-family:\'Playfair Display\',serif;color:{card_border};margin:0 0 6px 0;">{status_title}</h4>'
            f'<p style="color:#F4EFEB;font-family:\'JetBrains Mono\',monospace;margin:4px 0;">'
            f'Predicted CPU: <b style="color:#D5B893">{result["predicted_cpu_avg_pct"]}%</b> &nbsp;|&nbsp; '
            f'Predicted RAM: <b style="color:#617891">{result["predicted_ram_pct"]}%</b></p>'
            f'<p style="color:#F4EFEB;margin:6px 0 0;font-size:0.9rem;">{result["recommendation"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Active synthetic workloads table
    if twin.synthetic_workloads:
        st.markdown("#### ⚡ Active Synthetic Workloads")
        df_wl = pd.DataFrame([{
            "Name": w["name"],
            "CPU Load": f"{w['cpu_load']}%",
            "RAM": f"{w['ram_load_gb']} GB",
            "Progress": f"{min(100, w['elapsed_s']/w['duration_s']*100):.0f}%",
            "Remaining (s)": max(0, round(w['duration_s'] - w['elapsed_s'], 1))
        } for w in twin.synthetic_workloads])
        st.dataframe(df_wl, use_container_width=True, hide_index=True)

    if st.button("🔄 Refresh Predictions", key="refresh_pred"):
        st.rerun()
