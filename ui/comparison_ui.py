"""
Scheduler Comparison UI page.
Animated Gantt chart, metrics comparison table, improvement analysis.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.scheduler import (
    generate_sim_tasks, run_all_schedulers,
    schedule_fcfs, schedule_sjf, schedule_round_robin,
    schedule_priority, schedule_predictive,
)

# ── Scientific Palette Tokens for Schedulers ──
# Exact palette: Space Cadet #25344F, Slate Gray #617891, Tan #D5B893, Coffee #6F4D38, Caput Mortuum #632024
ALGO_COLORS = {
    "FCFS": "#617891",          # Slate Gray
    "SJF": "#8FA5BD",           # Lightened Slate
    "Round Robin": "#6F4D38",   # Coffee
    "Priority": "#632024",      # Caput Mortuum
    "Predictive": "#D5B893",    # Tan (Hero ML Policy)
}


def _make_gantt(gantt_data: list, algo_name: str, color: str, num_cores: int = 1):
    """Render Gantt chart styled in the scientific palette."""
    if not gantt_data:
        return go.Figure()
    df = pd.DataFrame(gantt_data)
    fig = go.Figure()

    for _, row in df.iterrows():
        core_label = f"Core {row['core']}" if num_cores > 1 else "CPU"
        fig.add_trace(go.Bar(
            x=[row["finish"] - row["start"]],
            y=[core_label],
            base=[row["start"]],
            orientation="h",
            name=row["name"],
            marker=dict(
                color=color,
                opacity=0.85,
                line=dict(color="rgba(213, 184, 147, 0.4)", width=0.8)
            ),
            text=row["name"],
            textposition="inside",
            textfont=dict(color="#1D2A40" if color == "#D5B893" else "#F4EFEB", family="JetBrains Mono", size=9),
            hovertemplate=(
                f"<b>{row['name']}</b><br>"
                f"Start: {row['start']:.2f}s<br>"
                f"Finish: {row['finish']:.2f}s<br>"
                f"Duration: {row['finish']-row['start']:.2f}s<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(140, 50 + num_cores * 40),
        margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text=f"Gantt — {algo_name}", font=dict(color=color, size=13, family="Playfair Display, serif")),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono"),
            title_font=dict(color="#D5B893", family="Inter", size=11)
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color="#F4EFEB", family="JetBrains Mono")),
        barmode="overlay",
    )
    return fig


def _make_comparison_bar(metrics_df: pd.DataFrame, metric: str, title: str, lower_is_better=True):
    """Render scientific comparative bar chart."""
    fig = go.Figure()
    for algo in metrics_df.index:
        val = metrics_df.loc[algo, metric]
        color = ALGO_COLORS.get(algo, "#617891")
        fig.add_trace(go.Bar(
            x=[algo], y=[val],
            name=algo,
            marker=dict(
                color=color,
                line=dict(color="rgba(213, 184, 147, 0.35)", width=1)
            ),
            text=[f"{val:.3f}"],
            textposition="outside",
            textfont=dict(color="#F4EFEB", family="JetBrains Mono", size=10),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text=title, font=dict(color="#D5B893", size=12, family="Playfair Display, serif")),
        showlegend=False,
        xaxis=dict(tickfont=dict(color="#F4EFEB", family="Inter"), showgrid=False),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono")
        ),
    )
    return fig


def _make_radar_chart(metrics_df: pd.DataFrame):
    """Render spider radar comparison across normalized dimensions."""
    categories = ["Throughput", "CPU Util", "Avg Wait (inv)", "Avg Turnaround (inv)", "Makespan (inv)"]
    fig = go.Figure()

    normed = metrics_df.copy()
    for col in ["avg_wait_time", "avg_turnaround", "makespan"]:
        if col in normed.columns:
            max_val = normed[col].max()
            if max_val > 0:
                normed[col] = 1 - normed[col] / max_val
    if "throughput" in normed.columns:
        max_t = normed["throughput"].max()
        if max_t > 0:
            normed["throughput"] = normed["throughput"] / max_t
    if "cpu_utilization" in normed.columns:
        max_cu = normed["cpu_utilization"].max()
        if max_cu > 0:
            normed["cpu_utilization"] = normed["cpu_utilization"] / max_cu

    for algo in normed.index:
        vals = [
            normed.loc[algo, "throughput"] if "throughput" in normed.columns else 0,
            normed.loc[algo, "cpu_utilization"] if "cpu_utilization" in normed.columns else 0,
            normed.loc[algo, "avg_wait_time"] if "avg_wait_time" in normed.columns else 0,
            normed.loc[algo, "avg_turnaround"] if "avg_turnaround" in normed.columns else 0,
            normed.loc[algo, "makespan"] if "makespan" in normed.columns else 0,
        ]
        vals = [max(0, min(1, v)) for v in vals]
        algo_color = ALGO_COLORS.get(algo, "#617891")
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=algo,
            line=dict(color=algo_color, width=2),
            fillcolor=algo_color + "28",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="rgba(97, 120, 145, 0.3)",
                tickfont=dict(color="#9FB0C2", family="JetBrains Mono", size=9)
            ),
            angularaxis=dict(
                gridcolor="rgba(97, 120, 145, 0.3)",
                tickfont=dict(color="#F4EFEB", family="Inter", size=10)
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=30, r=30, t=35, b=30),
        legend=dict(
            font=dict(color="#D5B893", family="Inter"),
            bgcolor="rgba(29, 42, 64, 0.6)"
        ),
        title=dict(
            text="Algorithm Performance Radar",
            font=dict(color="#D5B893", size=13, family="Playfair Display, serif")
        ),
    )
    return fig


def render():
    st.markdown("## 📊 Scheduling Algorithm Benchmark")
    st.markdown(
        "<p style='color:#9FB0C2;font-size:0.92rem;margin-top:-6px;margin-bottom:18px;'>"
        "Simulate task workloads and compare <b>5 scheduling algorithms</b> — "
        "FCFS, SJF, Round Robin, Priority, and <b>ML-Guided Predictive</b> — "
        "with animated Gantt charts and statistical metrics."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Simulation Configuration ──────────────────────────────────────────────
    st.markdown("### ⚙️ Simulation Settings")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        n_tasks = st.slider("Number of Tasks", 5, 30, 15, key="sched_n_tasks")
    with sc2:
        n_cores = st.slider("Number of CPU Cores", 1, 8, 4, key="sched_cores")
    with sc3:
        rr_quantum = st.slider("Round Robin Quantum (s)", 0.5, 5.0, 2.0, step=0.5, key="sched_quantum")
    with sc4:
        pred_load = st.slider("Predicted System Load (%)", 0, 100, 30, key="sched_pred_load")

    seed = st.number_input("Task Generation Seed", value=42, min_value=1, max_value=9999, key="sched_seed")

    run_btn = st.button("▶️ Run Simulation", type="primary", key="run_sched")

    if run_btn or "sched_results" not in st.session_state:
        tasks = generate_sim_tasks(n_tasks, seed=int(seed))
        with st.spinner("Running all scheduling algorithms..."):
            results = run_all_schedulers(tasks, rr_quantum, n_cores, pred_load)
        st.session_state.sched_results = results
        st.session_state.sched_n_cores = n_cores
        st.success(f"✅ Simulated {n_tasks} tasks across 5 scheduling algorithms.")

    if "sched_results" not in st.session_state:
        return

    results = st.session_state.sched_results
    n_cores_used = st.session_state.get("sched_n_cores", 4)

    # ── Metrics Table ─────────────────────────────────────────────────────────
    st.markdown("### 📋 Performance Metrics Comparison")
    metrics_rows = {}
    for algo, res in results.items():
        if "metrics" in res and res["metrics"]:
            metrics_rows[algo] = res["metrics"]

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).T
        display_df = metrics_df.copy()
        col_rename = {
            "avg_wait_time": "Avg Wait (s)",
            "avg_turnaround": "Avg Turnaround (s)",
            "avg_response_time": "Avg Response (s)",
            "throughput": "Throughput (t/s)",
            "cpu_utilization": "CPU Util (%)",
            "total_tasks": "Tasks",
            "makespan": "Makespan (s)",
        }
        display_df.rename(columns=col_rename, inplace=True)

        st.dataframe(
            display_df.round(3),
            use_container_width=True
        )

    # ── Improvement Analysis ──────────────────────────────────────────────────
    if "FCFS" in metrics_rows and "Predictive" in metrics_rows:
        st.markdown("### 🏆 Predictive vs Traditional — Improvement Analysis")
        fcfs_m = metrics_rows["FCFS"]
        pred_m = metrics_rows["Predictive"]
        improvements = []
        for metric, label, lower_better in [
            ("avg_wait_time", "Avg Wait Time", True),
            ("avg_turnaround", "Avg Turnaround", True),
            ("cpu_utilization", "CPU Utilization", False),
            ("throughput", "Throughput", False),
        ]:
            if metric in fcfs_m and metric in pred_m:
                base = fcfs_m[metric]
                new = pred_m[metric]
                if base != 0:
                    pct = ((base - new) / base * 100) if lower_better else ((new - base) / base * 100)
                    better = pct > 0
                    improvements.append({
                        "Metric": label,
                        "FCFS": round(base, 3),
                        "Predictive": round(new, 3),
                        "Improvement %": f"{'↑' if better else '↓'} {abs(pct):.1f}%",
                        "Result": "✅ Better" if better else "⚠️ Worse",
                    })
        if improvements:
            df_impr = pd.DataFrame(improvements)
            st.dataframe(df_impr, use_container_width=True, hide_index=True)

    # ── Comparative Bar Charts ────────────────────────────────────────────────
    if metrics_rows:
        st.markdown("### 📈 Metric Charts")
        bc1, bc2 = st.columns(2)
        with bc1:
            st.plotly_chart(
                _make_comparison_bar(metrics_df, "avg_wait_time", "Avg Wait Time (s) — Lower is Better"),
                use_container_width=True, key="bar_wait"
            )
        with bc2:
            st.plotly_chart(
                _make_comparison_bar(metrics_df, "cpu_utilization", "CPU Utilization (%) — Higher is Better", False),
                use_container_width=True, key="bar_cpu"
            )
        bc3, bc4 = st.columns(2)
        with bc3:
            st.plotly_chart(
                _make_comparison_bar(metrics_df, "throughput", "Throughput (tasks/s) — Higher is Better", False),
                use_container_width=True, key="bar_throughput"
            )
        with bc4:
            st.plotly_chart(
                _make_comparison_bar(metrics_df, "makespan", "Makespan (s) — Lower is Better"),
                use_container_width=True, key="bar_makespan"
            )

        # Radar chart
        st.markdown("### 🕸️ Performance Radar")
        st.plotly_chart(_make_radar_chart(metrics_df), use_container_width=True, key="radar")

    # ── Gantt Charts ──────────────────────────────────────────────────────────
    st.markdown("### 🗓️ Gantt Charts")
    for algo, res in results.items():
        gantt = res.get("gantt", [])
        if gantt:
            color = ALGO_COLORS.get(algo, "#617891")
            is_multi = algo == "Predictive"
            st.plotly_chart(
                _make_gantt(gantt, algo, color, n_cores_used if is_multi else 1),
                use_container_width=True,
                key=f"gantt_{algo}"
            )

    # ── Task Details ──────────────────────────────────────────────────────────
    with st.expander("📋 Task Schedule Details (Predictive)", expanded=False):
        pred_res = results.get("Predictive", {})
        pred_tasks = pred_res.get("tasks", [])
        if pred_tasks:
            rows = [{
                "ID": t.task_id, "Name": t.name, "Priority": t.priority,
                "Burst (s)": t.burst_time, "Arrival (s)": t.arrival_time,
                "Start (s)": round(t.start_time, 2) if t.start_time else None,
                "Finish (s)": round(t.finish_time, 2) if t.finish_time else None,
                "Wait (s)": round(t.waiting_time, 2) if t.waiting_time else None,
                "Turnaround (s)": round(t.turnaround_time, 2) if t.turnaround_time else None,
            } for t in pred_tasks]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
