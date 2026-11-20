"""
Predictive Thermal-Aware Resource Allocation Framework
Multi-Core Computing Systems & AI-Driven Digital Twin
Main Streamlit Application Entry Point
"""
import streamlit as st
import sys
import os

# ─── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Thermal-Aware Resource Allocation",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Predictive Thermal-Aware Resource Allocation in Multi-Core Computing Systems — AI Digital Twin Framework",
    },
)

# ─── Global Scientific CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography: Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

/* ── Scientific Palette Tokens ── */
:root {
    --space-cadet:     #25344F; /* Primary background */
    --slate-gray:      #617891; /* Cards and panels */
    --tan:             #D5B893; /* Accents and highlights */
    --coffee:          #6F4D38; /* Secondary accents */
    --caput-mortuum:   #632024; /* Thermal alerts & critical states */
    
    --bg-deep:         #1D2A40;
    --text-primary:    #F4EFEB;
    --text-secondary:  #D5B893;
    --text-muted:      #9FB0C2;
    --border-subtle:   rgba(97, 120, 145, 0.35);
    --border-card:     rgba(97, 120, 145, 0.45);
    --border-accent:   rgba(213, 184, 147, 0.35);
}

/* ── App Shell & Background ── */
.stApp {
    background-color: var(--space-cadet) !important;
    background-image: radial-gradient(circle at 18% 12%, #2c3c59 0%, #25344F 60%, #1c273c 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1D2A40 !important;
    border-right: 1px solid var(--border-subtle) !important;
    box-shadow: 2px 0 12px rgba(10, 16, 26, 0.35) !important;
}
[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-muted) !important;
}

/* ── Main Container Spacing ── */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px !important;
}

/* ── Sophisticated Serif Headings ── */
h1, h2, h3, h4, .serif-heading {
    font-family: 'Playfair Display', 'Cinzel', Georgia, serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.015em !important;
}
h1 {
    font-size: 2.1rem !important;
    color: var(--tan) !important;
    border-bottom: 1px solid rgba(213, 184, 147, 0.25);
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem !important;
}
h2 {
    font-size: 1.6rem !important;
    color: var(--tan) !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}
h3 {
    font-size: 1.25rem !important;
    color: #E2CEB9 !important;
    margin-top: 1.2rem !important;
    margin-bottom: 0.5rem !important;
}
h4 {
    font-size: 1.05rem !important;
    color: var(--tan) !important;
    font-weight: 500 !important;
}

/* ── Technical Data Typography ── */
code, pre, .technical-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: rgba(97, 120, 145, 0.2) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 16px rgba(15, 23, 38, 0.25) !important;
    transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(213, 184, 147, 0.55) !important;
    background: rgba(97, 120, 145, 0.28) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--tan) !important;
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: rgba(97, 120, 145, 0.24) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1.1rem !important;
}
.stButton > button:hover {
    background: rgba(213, 184, 147, 0.18) !important;
    border-color: var(--tan) !important;
    color: var(--tan) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(10, 16, 26, 0.2) !important;
}
.stButton > button[kind="primary"] {
    background: var(--tan) !important;
    color: #1c273c !important;
    border: 1px solid var(--tan) !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(213, 184, 147, 0.2) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #E2CEB9 !important;
    border-color: #E2CEB9 !important;
    color: #121a28 !important;
    box-shadow: 0 4px 16px rgba(213, 184, 147, 0.35) !important;
}

/* ── Controls: Inputs, Selectboxes, Sliders ── */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div > input {
    background: #1D2A40 !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stTextInput > div > div > input:focus {
    border-color: var(--tan) !important;
    box-shadow: 0 0 0 1px var(--tan) !important;
}
.stSlider > div > div > div > div {
    background: var(--tan) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--tan) !important;
    border: 2px solid var(--space-cadet) !important;
    box-shadow: 0 0 6px rgba(0, 0, 0, 0.4) !important;
}

/* ── Scientific Alert & Status Boxes ── */
.stInfo {
    background: rgba(97, 120, 145, 0.22) !important;
    border: 1px solid var(--border-card) !important;
    border-left: 4px solid var(--slate-gray) !important;
    border-radius: 6px !important;
    color: #E5ECF2 !important;
}
.stSuccess {
    background: rgba(213, 184, 147, 0.16) !important;
    border: 1px solid rgba(213, 184, 147, 0.45) !important;
    border-left: 4px solid var(--tan) !important;
    border-radius: 6px !important;
    color: #F8F5F2 !important;
}
.stWarning {
    background: rgba(111, 77, 56, 0.25) !important;
    border: 1px solid rgba(111, 77, 56, 0.6) !important;
    border-left: 4px solid var(--coffee) !important;
    border-radius: 6px !important;
    color: #F2DDD0 !important;
}
.stError {
    background: rgba(99, 32, 36, 0.3) !important;
    border: 1px solid rgba(99, 32, 36, 0.75) !important;
    border-left: 4px solid var(--caput-mortuum) !important;
    border-radius: 6px !important;
    color: #F8D3D6 !important;
}

/* ── DataFrames & Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #1D2A40 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(97, 120, 145, 0.14) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    overflow: hidden;
}
.streamlit-expanderHeader {
    background: rgba(97, 120, 145, 0.2) !important;
    color: var(--tan) !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 0.95rem !important;
    border-radius: 7px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(97, 120, 145, 0.3) !important;
    margin: 1.4rem 0 !important;
}

/* ── Captions ── */
.stCaption, caption, small {
    color: var(--text-muted) !important;
    font-size: 0.76rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1D2A40 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 6px !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(97, 120, 145, 0.35) !important;
    color: var(--tan) !important;
    border-bottom: 2px solid var(--tan) !important;
}

/* ── Forms ── */
[data-testid="stForm"] {
    background: rgba(97, 120, 145, 0.16) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 10px !important;
    padding: 1.2rem !important;
    box-shadow: 0 4px 16px rgba(15, 23, 38, 0.2) !important;
}

/* ── Plotly Chart Containers ── */
.js-plotly-plot, .stPlotlyChart {
    border-radius: 8px !important;
    border: 1px solid rgba(97, 120, 145, 0.25) !important;
    background: rgba(29, 42, 64, 0.5) !important;
    padding: 4px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: var(--tan) !important;
}

/* ── Custom Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1D2A40; }
::-webkit-scrollbar-thumb { background: rgba(97, 120, 145, 0.6); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--tan); }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Research Header / Monogram
        st.markdown("""
        <div style="text-align:center;padding:18px 8px 12px;border-bottom:1px solid rgba(97,120,145,0.3);">
            <div style="
                display:inline-block;
                width:44px;
                height:44px;
                line-height:44px;
                border-radius:50%;
                background:rgba(213,184,147,0.15);
                border:1px solid rgba(213,184,147,0.4);
                font-size:1.4rem;
                margin-bottom:8px;
            ">🌡️</div>
            <h1 style="
                font-family:'Playfair Display', serif !important;
                font-size:1.15rem !important;
                font-weight:700 !important;
                color:#D5B893 !important;
                margin:0 !important;
                line-height:1.3;
                border:none !important;
                padding:0 !important;
            ">Thermal-Aware<br>Resource Allocation</h1>
            <p style="color:#9FB0C2;font-size:0.72rem;margin:6px 0 0;letter-spacing:0.04em;text-transform:uppercase;">
                AI Digital Twin & Telemetry
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        # Navigation
        pages = {
            "🌡️  Live Thermal Monitor": "monitor",
            "🤖  ML Prediction & Twin": "predictor",
            "⚙️  Resource Allocator": "allocator",
            "📊  Scheduler Benchmark": "benchmark",
        }

        if "page" not in st.session_state:
            st.session_state.page = "monitor"

        st.markdown("<p style='color:#9FB0C2;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;padding:0 4px;margin-bottom:6px;'>Navigation</p>", unsafe_allow_html=True)
        for label, key in pages.items():
            is_active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown('<hr style="margin:14px 0;border-color:rgba(97,120,145,0.3);">', unsafe_allow_html=True)

        # Refresh Controls
        st.markdown("<p style='color:#9FB0C2;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;padding:0 4px;margin-bottom:6px;'>Telemetry Stream</p>", unsafe_allow_html=True)
        auto_refresh = st.toggle("🔄 Continuous Polling (3s)", value=False, key="auto_refresh")

        # System Telemetry Mini
        st.markdown('<hr style="margin:14px 0;border-color:rgba(97,120,145,0.3);">', unsafe_allow_html=True)
        st.markdown("<p style='color:#9FB0C2;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.12em;padding:0 4px;margin-bottom:8px;'>Core Telemetry</p>", unsafe_allow_html=True)
        try:
            from core.monitor import get_full_snapshot
            snap = get_full_snapshot()
            cpu = snap["cpu"]["overall_percent"]
            ram = snap["ram"]["percent"]
            disk = snap["disk"]["percent"]
            thermal = snap.get("thermal", {})
            temp = thermal.get("package_temp_c", 42.0)
            status_color = thermal.get("status_color", "#D5B893")

            def mini_bar(label, val, val_str, bar_color):
                pct = min(100.0, max(0.0, val))
                return (
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:0.75rem;color:#9FB0C2;margin-bottom:3px;font-family:\'Inter\',sans-serif;">'
                    f'<span>{label}</span><span style="color:{bar_color};font-family:\'JetBrains Mono\',monospace;font-weight:600;">{val_str}</span></div>'
                    f'<div style="background:#162132;border-radius:3px;height:5px;overflow:hidden;border:1px solid rgba(97,120,145,0.3);">'
                    f'<div style="background:{bar_color};width:{pct}%;height:5px;border-radius:2px;'
                    f'transition:width 0.3s ease;"></div></div></div>'
                )

            st.markdown(
                mini_bar("CPU Load", cpu, f"{cpu:.1f}%", "#D5B893") +
                mini_bar("Thermal (Pkg)", temp, f"{temp:.1f}°C", status_color) +
                mini_bar("RAM Alloc", ram, f"{ram:.1f}%", "#617891") +
                mini_bar("Disk Storage", disk, f"{disk:.1f}%", "#6F4D38"),
                unsafe_allow_html=True
            )
        except Exception:
            st.caption("Telemetry unavailable")

        st.markdown('<hr style="margin:14px 0;border-color:rgba(97,120,145,0.3);">', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#788A9E;font-size:0.67rem;text-align:center;line-height:1.5;">'
            '<span style="color:#D5B893;font-family:\'Playfair Display\',serif;font-size:0.75rem;letter-spacing:0.02em;">'
            'Thermal-Aware Digital Twin</span><br>'
            'Predictive Multi-Core Research System<br>'
            '<span style="color:rgba(97,120,145,0.5);">──────────────</span><br>'
            '<span style="font-family:\'JetBrains Mono\',monospace;color:#9FB0C2;">'
            '#25344F · #617891 · #D5B893 · #6F4D38 · #632024</span></div>',
            unsafe_allow_html=True
        )

        return auto_refresh


# ─── Main Dispatcher ──────────────────────────────────────────────────────────
def main():
    auto_refresh = render_sidebar()

    page = st.session_state.get("page", "monitor")

    if page == "monitor":
        from ui.dashboard import render
        render()
    elif page == "predictor":
        from ui.predictor_ui import render
        render()
    elif page == "allocator":
        from ui.allocator_ui import render
        render()
    elif page == "benchmark":
        from ui.comparison_ui import render
        render()

    # Auto-refresh loop
    if auto_refresh:
        import time
        time.sleep(3)
        st.rerun()


if __name__ == "__main__":
    main()
