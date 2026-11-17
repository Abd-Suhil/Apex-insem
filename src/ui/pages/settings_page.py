"""
Settings & Workspace Configuration Page
Theme toggle, model configuration audit, session cache management, and application metadata.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
from src.ui.components import render_header, render_kpi_card, render_info_banner


def render_settings_page():
    render_header(
        title="Settings & Workspace Preferences",
        subtitle="Manage display themes, environment configuration, and pipeline parameters.",
        badge="CONFIGURATION"
    )

    # 1. Interface Theme Toggle
    st.markdown("<div class='kpi-title'>Interface & Appearance</div>", unsafe_allow_html=True)
    current_theme = st.session_state.get("theme", "dark")
    theme_choice = st.radio(
        "Display Mode:",
        options=["Dark Modern (Deep Navy / Charcoal)", "Light Clean (SaaS Light)"],
        index=0 if current_theme == "dark" else 1,
        horizontal=True,
        key="theme_radio_select"
    )
    new_theme = "dark" if "Dark" in theme_choice else "light"
    if new_theme != current_theme:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Pipeline Defaults & Session State Audit
    st.markdown("<div class='kpi-title'>Active Pipeline Parameters</div>", unsafe_allow_html=True)
    df = st.session_state.get("df")
    model = st.session_state.get("model")
    features = st.session_state.get("selected_features", [])
    k_val = st.session_state.get("k_clusters", 5)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown(f"""
        <b>Dataset Identity:</b> <code>{st.session_state.get('dataset_name', 'Mall Customers Benchmark')}</code><br/>
        <b>Sample Cardinality:</b> <code>{len(df) if df is not None else 0:,} records</code><br/>
        <b>Active Feature Space:</b> <code>{features}</code>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"""
        <b>Cluster Cardinality (K):</b> <code>{k_val}</code><br/>
        <b>Model Status:</b> <code>{'Trained & Persisted' if model is not None else 'Uninitialized'}</code><br/>
        <b>Initialization Strategy:</b> <code>k-means++</code>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Workspace Reset Action
    st.markdown("<div class='kpi-title'>Workspace Management</div>", unsafe_allow_html=True)
    st.markdown("Reset pipeline state and re-load canonical benchmark dataset from disk.")

    if st.button("Reset Workspace & Clear Cache", type="secondary", key="reset_cache_btn"):
        for key in ["df", "model", "preprocessor", "metrics", "personas", "model_trained"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Session state cleared. Reloading benchmark...")
        st.rerun()

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. System Metadata
    st.markdown("<div class='kpi-title'>Platform Specifications</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="iq-card">
        <b>Platform:</b> CustomerIQ — Intelligent Customer Segmentation & Business Analytics<br/>
        <b>Engine:</b> Unsupervised Scikit-Learn K-Means++ Engine with Dynamic Silhouette Optimization<br/>
        <b>Visualization Layer:</b> Plotly Express + Seaborn & Matplotlib Core<br/>
        <b>Reporting Engine:</b> ReportLab Document Object Model Architecture<br/>
        <b>Architecture:</b> Modular Decoupled Streamlit Multipage SaaS Pipeline
    </div>
    """, unsafe_allow_html=True)
