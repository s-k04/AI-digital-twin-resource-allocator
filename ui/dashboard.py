"""
Live System Monitoring Dashboard page.
Real-time CPU, RAM, Disk, and process tracking with Digital Twin synchronization.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.monitor import get_full_snapshot, get_system_info
from core.digital_twin import DigitalTwin


HISTORY_LEN = 60   # seconds of telemetry history


def _init_state():
    if "cpu_history" not in st.session_state:
        st.session_state.cpu_history = []
    if "ram_history" not in st.session_state:
        st.session_state.ram_history = []
    if "time_history" not in st.session_state:
        st.session_state.time_history = []
    if "twin" not in st.session_state:
        info = get_system_info()
        st.session_state.twin = DigitalTwin(
            num_cores=info["cpu_logical_cores"],
            total_ram_gb=info["total_ram_gb"]
        )
    if "sys_info" not in st.session_state:
        st.session_state.sys_info = get_system_info()


def _make_scientific_gauge(value, title, color, suffix="%", max_val=100, threshold=85):
    """Render a sophisticated gauge styled in the scientific palette."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "#D5B893", "family": "Playfair Display, serif"}},
        number={"suffix": suffix, "font": {"size": 22, "color": "#F4EFEB", "family": "JetBrains Mono, monospace"}},
        delta={"reference": 50, "increasing": {"color": "#632024"}, "decreasing": {"color": "#D5B893"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#9FB0C2", "tickfont": {"color": "#9FB0C2", "size": 10, "family": "JetBrains Mono"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1D2A40",
            "bordercolor": "rgba(97, 120, 145, 0.4)",
            "steps": [
                {"range": [0, max_val * 0.5], "color": "#182335"},
                {"range": [max_val * 0.5, max_val * 0.75], "color": "#212F45"},
                {"range": [max_val * 0.75, max_val], "color": "rgba(99, 32, 36, 0.3)"},
            ],
            "threshold": {"line": {"color": "#632024", "width": 2}, "thickness": 0.75, "value": threshold},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=190,
        margin=dict(l=15, r=15, t=35, b=10),
    )
    return fig


def _make_core_heatmap(per_core: list):
    """Render CPU core load distribution heatmap."""
    n = len(per_core)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    z = []
    text_grid = []
    row_z = []
    row_t = []

    for i, v in enumerate(per_core):
        row_z.append(round(v, 1))
        row_t.append(f"C{i}<br>{v:.0f}%")
        if len(row_z) == cols:
            z.append(row_z)
            text_grid.append(row_t)
            row_z, row_t = [], []
    if row_z:
        while len(row_z) < cols:
            row_z.append(0)
            row_t.append("")
        z.append(row_z)
        text_grid.append(row_t)

    scientific_scale = [
        [0.0, "#192436"],   # Deep Space Cadet
        [0.3, "#2C3C56"],   # Slate Navy
        [0.55, "#617891"],  # Slate Gray
        [0.75, "#D5B893"],  # Tan
        [0.88, "#6F4D38"],  # Coffee
        [1.0, "#632024"],   # Caput Mortuum (Peak load)
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text_grid,
        texttemplate="%{text}",
        textfont={"family": "JetBrains Mono, monospace", "size": 11, "color": "#F4EFEB"},
        colorscale=scientific_scale,
        zmin=0, zmax=100,
        showscale=True,
        colorbar={
            "title": {"text": "Load %", "font": {"color": "#D5B893", "family": "Inter", "size": 11}},
            "tickfont": {"color": "#9FB0C2", "family": "JetBrains Mono", "size": 10},
            "outlinecolor": "rgba(97, 120, 145, 0.4)",
            "len": 0.9,
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=170,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def _make_time_chart(time_hist, cpu_hist, ram_hist, twin_hist):
    """Render animated multi-metric time series chart."""
    fig = go.Figure()
    ts = time_hist[-HISTORY_LEN:]
    cpu = cpu_hist[-HISTORY_LEN:]
    ram = ram_hist[-HISTORY_LEN:]
    twin = twin_hist[-HISTORY_LEN:] if twin_hist else []

    # CPU load: Tan accent
    fig.add_trace(go.Scatter(
        x=ts, y=cpu, name="Real CPU (%)",
        line=dict(color="#D5B893", width=2),
        fill="tozeroy", fillcolor="rgba(213, 184, 147, 0.08)"
    ))
    # RAM allocation: Slate Gray accent
    fig.add_trace(go.Scatter(
        x=ts, y=ram, name="RAM (%)",
        line=dict(color="#617891", width=2),
        fill="tozeroy", fillcolor="rgba(97, 120, 145, 0.12)"
    ))
    # Digital Twin: Coffee dashed
    if twin:
        fig.add_trace(go.Scatter(
            x=ts, y=twin, name="Twin CPU (%)",
            line=dict(color="#6F4D38", width=1.8, dash="dot"),
        ))

    # Critical threshold line in Caput Mortuum
    fig.add_hline(
        y=85, line_dash="dash", line_color="#632024", opacity=0.8,
        annotation_text="Warning 85%", annotation_font_color="#E8B4B8",
        annotation_font_family="JetBrains Mono"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=10, r=10, t=15, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(color="#D5B893", family="Inter", size=11),
            bgcolor="rgba(29, 42, 64, 0.6)"
        ),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            tickfont=dict(color="#9FB0C2", family="JetBrains Mono")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(97, 120, 145, 0.2)",
            range=[0, 105], tickfont=dict(color="#9FB0C2", family="JetBrains Mono"),
            ticksuffix="%"
        ),
    )
    return fig


def render():
    _init_state()
    st.markdown("## 🖥️ Live System Monitor")
    st.markdown(
        "<p style='color:#9FB0C2;font-size:0.92rem;margin-top:-6px;margin-bottom:18px;'>"
        "Real-time CPU, RAM, Disk and process tracking with Digital Twin synchronization."
        "</p>",
        unsafe_allow_html=True
    )

    info = st.session_state.sys_info

    # ── System Info Expander ──────────────────────────────────────────────────
    with st.expander("ℹ️ System Architecture Info", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Physical Cores", info["cpu_physical_cores"])
        c2.metric("Logical Cores", info["cpu_logical_cores"])
        c3.metric("Total RAM", f"{info['total_ram_gb']} GB")
        c4.metric("Max CPU Freq", f"{info['cpu_max_freq_mhz']} MHz" if info['cpu_max_freq_mhz'] else "Dynamic")
        st.caption(f"OS: {info['system']} · Host: {info['node']} · Architecture: {info['processor']} · Python: {info['python_version']}")

    # ── Real-Time Snapshot ────────────────────────────────────────────────────
    snap = get_full_snapshot()
    cpu_data = snap["cpu"]
    ram_data = snap["ram"]
    disk_data = snap["disk"]
    net_data = snap["network"]
    procs = snap["processes"]

    # Update state histories
    st.session_state.cpu_history.append(cpu_data["overall_percent"])
    st.session_state.ram_history.append(ram_data["percent"])
    st.session_state.time_history.append(datetime.now())

    # Update digital twin
    twin: DigitalTwin = st.session_state.twin
    twin.update(cpu_data["per_core_percent"], ram_data["percent"])
    twin_hist = [float(np.mean(c)) for c in twin.twin_cpu_history]

    # ── Scientific Telemetry Gauges ───────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            _make_scientific_gauge(cpu_data["overall_percent"], "CPU Usage", "#D5B893"),
            use_container_width=True, key="gauge_cpu"
        )
    with g2:
        st.plotly_chart(
            _make_scientific_gauge(ram_data["percent"], "RAM Usage", "#617891"),
            use_container_width=True, key="gauge_ram"
        )
    with g3:
        st.plotly_chart(
            _make_scientific_gauge(disk_data["percent"], "Disk Usage", "#6F4D38"),
            use_container_width=True, key="gauge_disk"
        )

    # ── Multi-Core Heatmap ────────────────────────────────────────────────────
    st.markdown("#### 🔥 CPU Core Load Heatmap")
    st.plotly_chart(
        _make_core_heatmap(cpu_data["per_core_percent"]),
        use_container_width=True, key="core_heatmap"
    )

    # ── Time Series Telemetry ─────────────────────────────────────────────────
    st.markdown("#### 📈 CPU & RAM Over Time")
    if len(st.session_state.time_history) > 1:
        st.plotly_chart(
            _make_time_chart(
                st.session_state.time_history,
                st.session_state.cpu_history,
                st.session_state.ram_history,
                twin_hist,
            ),
            use_container_width=True,
            key="time_chart"
        )
    else:
        st.info("Collecting data... refresh to see the chart.")

    # ── Secondary Telemetry Metrics ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RAM Used", f"{ram_data['used_gb']} GB", f"{ram_data['percent']:.1f}%")
    col2.metric("RAM Free", f"{ram_data['available_gb']} GB")
    col3.metric("Net Sent", f"{net_data['bytes_sent_mb']} MB")
    col4.metric("Net Recv", f"{net_data['bytes_recv_mb']} MB")

    # ── Digital Twin Status Banner ───────────────────────────────────────────
    dev = twin.get_deviation()
    st.markdown("#### 🌐 Digital Twin Status")
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Twin Health Score", f"{dev['health_score']}%",
               help="How closely the twin tracks the real system")
    tc2.metric("CPU Deviation", f"{dev['cpu_deviation']:.2f}%")
    tc3.metric("RAM Deviation", f"{dev['ram_deviation']:.2f}%")

    health = dev["health_score"]
    if health > 80:
        badge_bg = "rgba(213, 184, 147, 0.15)"
        badge_border = "#D5B893"
        badge_color = "#D5B893"
        desc = "Excellent synchronization with real system."
    elif health > 60:
        badge_bg = "rgba(111, 77, 56, 0.22)"
        badge_border = "#6F4D38"
        badge_color = "#F2DDD0"
        desc = "Minor drift detected."
    else:
        badge_bg = "rgba(99, 32, 36, 0.28)"
        badge_border = "#632024"
        badge_color = "#F8D3D6"
        desc = "High deviation — twin may need recalibration."

    st.markdown(
        f'<div style="background:{badge_bg};border:1px solid {badge_border};border-left:4px solid {badge_border};'
        f'padding:12px 16px;border-radius:6px;margin:10px 0 16px;">'
        f'<span style="font-family:\'Playfair Display\',serif;font-weight:700;color:{badge_color};margin-right:8px;">'
        f'Twin Health: {health}%</span> · '
        f'<span style="color:#F4EFEB;font-size:0.88rem;">{desc}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Top Process Table ─────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Top Processes")
    if procs:
        df_procs = pd.DataFrame(procs)[["pid", "name", "cpu_percent", "memory_percent", "status", "num_threads"]]
        df_procs.columns = ["PID", "Process", "CPU %", "RAM %", "Status", "Threads"]
        df_procs = df_procs.fillna(0).round(2)
        st.dataframe(
            df_procs,
            use_container_width=True,
            hide_index=True,
        )

    # Swap memory notice
    if ram_data.get("swap_total_gb", 0) > 0:
        st.caption(f"💾 Swap: {ram_data['swap_used_gb']} / {ram_data['swap_total_gb']} GB ({ram_data['swap_percent']}%)")

    # Footer controls
    st.caption(f"🕐 Last updated: {snap['timestamp']}")
    if st.button("🔄 Refresh Now", key="refresh_monitor"):
        st.rerun()
