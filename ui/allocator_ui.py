"""
Thermal-Aware Resource Allocation UI page.
Submit compute tasks, compare traditional static allocation vs ML-guided predictive allocation,
and observe core load and thermal distribution in real-time.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.allocator import ResourceAllocator, generate_random_tasks
from core.monitor import get_system_info, get_full_snapshot
from core.predictor import ResourcePredictor


def _init_state():
    if "alloc_ml" not in st.session_state or st.session_state.get("alloc_reset"):
        info = get_system_info()
        st.session_state.alloc_ml = ResourceAllocator(
            num_cores=info["cpu_logical_cores"],
            total_ram_gb=info["total_ram_gb"]
        )
        st.session_state.alloc_trad = ResourceAllocator(
            num_cores=info["cpu_logical_cores"],
            total_ram_gb=info["total_ram_gb"]
        )
        st.session_state.alloc_log = []
        st.session_state.alloc_reset = False
    if "alloc_predictor" not in st.session_state or not hasattr(st.session_state.alloc_predictor, "sample_count"):
        st.session_state.alloc_predictor = ResourcePredictor(buffer_size=200, horizon=5)


def _core_diagram(core_load: list, title: str, default_color: str = "#D5B893"):
    """Render multi-core load bars styled with the scientific palette."""
    fig = go.Figure()
    for i, load in enumerate(core_load):
        # Color coding: Nominal (Tan/Slate), Elevated (Coffee), Critical (Caput Mortuum)
        if load >= 85:
            bar_color = "#632024"  # Caput Mortuum (thermal throttling / saturation hazard)
        elif load >= 70:
            bar_color = "#6F4D38"  # Coffee (elevated core stress)
        else:
            bar_color = default_color  # Tan or Slate Gray

        fig.add_trace(go.Bar(
            x=[f"C{i}"], y=[load],
            marker=dict(
                color=bar_color,
                line=dict(color="rgba(213, 184, 147, 0.3)", width=1)
            ),
            showlegend=False,
            text=[f"{load:.0f}%"], textposition="outside",
            textfont=dict(color="#F4EFEB", family="JetBrains Mono", size=10),
        ))

    # Reference threshold lines
    fig.add_hline(
        y=85, line_dash="dash", line_color="#632024", opacity=0.7,
        annotation_text="85% Throttling Cap", annotation_font_color="#E8B4B8",
        annotation_font_family="JetBrains Mono", annotation_font_size=9
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210, margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text=title, font=dict(color="#D5B893", size=12, family="Playfair Display, serif")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#9FB0C2", family="JetBrains Mono")),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono"),
            range=[0, 115], ticksuffix="%"
        ),
        barmode="group",
    )
    return fig


def _stats_card(title: str, stats: dict, accent_color: str):
    """Render an elegant scientific statistics panel."""
    rows = ""
    for k, v in stats.items():
        rows += (
            f'<tr style="border-bottom:1px solid rgba(97,120,145,0.2);">'
            f'<td style="color:#9FB0C2;padding:5px 8px;font-size:0.84rem;font-family:\'Inter\',sans-serif;">{k}</td>'
            f'<td style="color:#F4EFEB;padding:5px 8px;font-family:\'JetBrains Mono\',monospace;font-size:0.9rem;text-align:right;">'
            f'<b>{v}</b></td></tr>'
        )
    return (
        f'<div style="background:rgba(97, 120, 145, 0.18);border:1px solid rgba(97, 120, 145, 0.4);'
        f'border-left:4px solid {accent_color};padding:14px 16px;border-radius:8px;margin:6px 0;">'
        f'<h4 style="font-family:\'Playfair Display\',serif;color:{accent_color};margin:0 0 10px 0;font-size:1.05rem;">{title}</h4>'
        f'<table style="width:100%;border-collapse:collapse;">{rows}</table></div>'
    )


def render():
    _init_state()
    alloc_ml: ResourceAllocator = st.session_state.alloc_ml
    alloc_trad: ResourceAllocator = st.session_state.alloc_trad
    predictor: ResourcePredictor = st.session_state.alloc_predictor

    # Telemetry sampling
    snap = get_full_snapshot()
    cpu_now = snap["cpu"]["overall_percent"]
    ram_now = snap["ram"]["percent"]
    thermal_data = snap.get("thermal", {})
    predictor.add_sample(cpu_now, ram_now, len(snap["processes"]))

    pred = predictor.predict() if predictor.trained else None
    pred_cpu = pred["cpu_predicted"] if pred else cpu_now
    pred_ram = pred["ram_predicted"] if pred else ram_now

    st.markdown("## ⚙️ Thermal-Aware Predictive Allocator")
    st.markdown(
        "<p style='color:#9FB0C2;font-size:0.92rem;margin-top:-6px;margin-bottom:18px;'>"
        "Benchmark <b>traditional static-threshold</b> provisioning against <b>ML-guided thermal & headroom-aware</b> allocation. "
        "Intelligently dispatches compute tasks to cooler cores, minimizing thermal throttling and task rejections."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Live Telemetry Cards ──────────────────────────────────────────────────
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Host CPU Load", f"{cpu_now:.1f}%")
    lc2.metric("Package Thermal", f"{thermal_data.get('package_temp_c', 42.0):.1f}°C")
    lc3.metric("RAM Allocation", f"{ram_now:.1f}%")
    lc4.metric("Predicted Next CPU", f"{pred_cpu:.1f}%" if pred else "Model Untrained")

    st.markdown("<hr style='border-color:rgba(97,120,145,0.3);margin:14px 0;'>", unsafe_allow_html=True)

    # ── Task Submission Console ───────────────────────────────────────────────
    st.markdown("### 📥 Compute Task Dispatch Console")
    with st.form("task_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        t_name = fc1.text_input("Task Identifier", value="ML Inference Worker", key="task_name_inp")
        t_cpu = fc2.slider("Required CPU Capacity (%)", 1, 80, 20, key="task_cpu")
        t_ram = fc3.slider("Memory Allocation (GB)", 0.1, 8.0, 1.0, step=0.1, key="task_ram")
        t_dur = fc4.slider("Execution Duration (s)", 1, 60, 10, key="task_dur")
        t_pri = st.slider("Dispatch Priority Level (1=Highest, 10=Lowest)", 1, 10, 5, key="task_prio")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            submitted = st.form_submit_button("🚀 Dispatch Task", type="primary", use_container_width=True)
        with col_b2:
            auto_submit = st.form_submit_button("🎲 Inject Batch of 10 Random Tasks", use_container_width=True)

    if submitted:
        t_ml = alloc_ml.submit_task(t_name, t_cpu, t_ram, t_dur, t_pri)
        t_tr = alloc_trad.submit_task(t_name, t_cpu, t_ram, t_dur, t_pri)
        alloc_ml.allocate_next_task("ml", pred_cpu, pred_ram)
        alloc_trad.allocate_next_task("traditional")
        alloc_ml.release_finished_tasks()
        alloc_trad.release_finished_tasks()
        st.session_state.alloc_log.append({
            "Timestamp": datetime.now().strftime("%H:%M:%S"),
            "Task": t_name, "CPU %": t_cpu, "RAM GB": t_ram, "Duration": f"{t_dur}s",
            "ML Policy Decision": t_ml.status, "Traditional Decision": t_tr.status,
            "Target Cores": str(t_ml.assigned_cores) if t_ml.assigned_cores else "None",
        })
        st.success(f"Task '{t_name}' dispatched! ML Decision: {t_ml.status} | Traditional Policy: {t_tr.status}")

    if auto_submit:
        tasks = generate_random_tasks(10, seed=int(time.time()) % 1000)
        for td in tasks:
            alloc_ml.submit_task(td["name"], td["cpu_pct"], td["ram_gb"], td["duration_s"], td["priority"])
            alloc_trad.submit_task(td["name"], td["cpu_pct"], td["ram_gb"], td["duration_s"], td["priority"])
        while alloc_ml.task_queue:
            alloc_ml.allocate_next_task("ml", pred_cpu, pred_ram)
        while alloc_trad.task_queue:
            alloc_trad.allocate_next_task("traditional")
        alloc_ml.release_finished_tasks()
        alloc_trad.release_finished_tasks()
        st.success("Batch of 10 tasks processed across both allocation engines!")
        st.rerun()

    # ── Multi-Core Distribution Diagrams ──────────────────────────────────────
    st.markdown("### 🔧 Multi-Core Load & Thermal Balance")
    cd1, cd2 = st.columns(2)
    with cd1:
        st.plotly_chart(
            _core_diagram(alloc_trad.core_load, "Traditional Allocator — Core Stress", "#617891"),
            use_container_width=True, key="core_trad"
        )
    with cd2:
        st.plotly_chart(
            _core_diagram(alloc_ml.core_load, "ML-Guided Predictive Allocator — Core Stress", "#D5B893"),
            use_container_width=True, key="core_ml"
        )

    # ── Comparative Statistics Panels ─────────────────────────────────────────
    st.markdown("### 📊 Policy Performance Comparative Metrics")
    stats_trad = alloc_trad.get_stats()
    stats_ml = alloc_ml.get_stats()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(
            _stats_card("🔵 Traditional Static Policy", {
                "Total Tasks Ingested": stats_trad["total_submitted"],
                "Successfully Completed": stats_trad["completed"],
                "Rejected / Throttled": stats_trad["rejected"],
                "Mean Queue Wait Time": f"{stats_trad['avg_wait_time_s']}s",
                "Cluster CPU Utilization": f"{stats_trad['cpu_utilization_pct']}%",
                "Committed Memory": f"{stats_trad['ram_used_gb']} GB",
            }, "#617891"),
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            _stats_card("🟢 ML Predictive Provisioning", {
                "Total Tasks Ingested": stats_ml["total_submitted"],
                "Successfully Completed": stats_ml["completed"],
                "Rejected / Throttled": stats_ml["rejected"],
                "Mean Queue Wait Time": f"{stats_ml['avg_wait_time_s']}s",
                "Cluster CPU Utilization": f"{stats_ml['cpu_utilization_pct']}%",
                "Committed Memory": f"{stats_ml['ram_used_gb']} GB",
            }, "#D5B893"),
            unsafe_allow_html=True,
        )

    # Rejection improvement callout
    trad_rej = stats_trad["rejected"]
    ml_rej = stats_ml["rejected"]
    if trad_rej > 0:
        impr = round((trad_rej - ml_rej) / trad_rej * 100, 1)
        st.markdown(
            f'<div style="background:rgba(213, 184, 147, 0.15);border:1px solid rgba(213, 184, 147, 0.4);'
            f'border-left:4px solid #D5B893;padding:12px 16px;border-radius:6px;margin:12px 0;">'
            f'<span style="font-family:\'Playfair Display\',serif;font-weight:700;color:#D5B893;">'
            f'Headroom Optimization:</span> '
            f'<span style="color:#F4EFEB;">ML-Guided predictive allocation mitigated task rejections by '
            f'<b style="color:#D5B893;">{impr}%</b> versus traditional threshold gating '
            f'({trad_rej} rejected vs {ml_rej} under predictive).</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Decision Telemetry Audit Log ──────────────────────────────────────────
    if st.session_state.alloc_log:
        st.markdown("### 📋 Dispatch Decision Audit Log")
        df_log = pd.DataFrame(st.session_state.alloc_log)
        st.dataframe(df_log, use_container_width=True, hide_index=True)

    # Reset Allocators
    col_rst, _ = st.columns([1, 3])
    with col_rst:
        if st.button("🔄 Reset Allocation State", key="reset_alloc"):
            alloc_ml.reset()
            alloc_trad.reset()
            st.session_state.alloc_log = []
            st.rerun()

    # Completed tasks table
    if alloc_ml.completed_tasks:
        st.markdown("### ✅ Task Execution Telemetry (ML Allocator)")
        rows = [t.to_dict() for t in alloc_ml.completed_tasks[-20:]]
        df_done = pd.DataFrame(rows)
        st.dataframe(df_done, use_container_width=True, hide_index=True)
