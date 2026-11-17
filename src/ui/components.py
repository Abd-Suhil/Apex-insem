"""
Reusable UI Components for CustomerIQ
"""

import streamlit as st


def render_header(title: str, subtitle: str, badge: str = "ENTERPRISE ML PLATFORM"):
    """Render a standardized page title and description block."""
    st.markdown(f"""
    <div class="iq-header">
        <div class="iq-badge">{badge}</div>
        <div class="iq-title">{title}</div>
        <div class="iq-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, subtitle: str = "", delta: str = ""):
    """Render a KPI metric card."""
    delta_html = f'<div style="color: #10B981; font-size: 0.8rem; font-weight: 600; margin-top: 4px;">{delta}</div>' if delta else ""
    sub_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_info_banner(message: str):
    """Render an informative styled callout banner."""
    st.markdown(f"""
    <div class="info-callout">
        {message}
    </div>
    """, unsafe_allow_html=True)
