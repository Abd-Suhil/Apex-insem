"""
Overview Dashboard Page
Executive summary, high-level KPIs, segment distribution, and quick action routing.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.ui.theme import get_plotly_theme


def render_overview_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Executive Overview",
        subtitle="Real-time macro view of customer segments, key behavioral indicators, and cluster distribution.",
        badge="EXECUTIVE DASHBOARD"
    )

    df = st.session_state.get("df")
    model = st.session_state.get("model")
    prep = st.session_state.get("preprocessor")
    metrics = st.session_state.get("metrics", {})
    personas = st.session_state.get("personas", {})
    k_clusters = st.session_state.get("k_clusters", 5)

    if df is None:
        st.warning("No dataset loaded. Please navigate to Data Explorer.")
        return

    # Safety: Ensure cluster column is present
    if "cluster" not in df.columns:
        if model is not None and prep is not None:
            try:
                X_s, _ = prep.transform(df)
                df["cluster"] = model.predict(X_s)
                st.session_state["df"] = df
            except Exception:
                pass

    has_clusters = "cluster" in df.columns

    # 1. Responsive KPI Cards Row (Using 5 columns with fluid typography)
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    with kpi_col1:
        render_kpi_card(
            title="Total Customers",
            value=f"{len(df):,}",
            subtitle="Analyzed profiles"
        )
    with kpi_col2:
        render_kpi_card(
            title="Active Segments",
            value=f"{k_clusters if has_clusters else '-'}",
            subtitle="Discovered clusters"
        )
    with kpi_col3:
        avg_income = df["annual_income"].mean() if "annual_income" in df else 0
        render_kpi_card(
            title="Avg Annual Income",
            value=f"${avg_income:.1f}k",
            subtitle="Benchmark mean"
        )
    with kpi_col4:
        avg_spend = df["spending_score"].mean() if "spending_score" in df else 0
        render_kpi_card(
            title="Avg Spending Score",
            value=f"{avg_spend:.1f}",
            subtitle="Scale of 1–100"
        )
    with kpi_col5:
        sil = metrics.get("silhouette_score", 0.0) if has_clusters else 0.0
        render_kpi_card(
            title="Silhouette Score",
            value=f"{sil:.4f}" if has_clusters else "N/A",
            subtitle="Separation quality"
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Main Row: Cluster Scatter Overview & Population Share
    if has_clusters:
        col_main_left, col_main_right = st.columns([1.5, 1])

        with col_main_left:
            st.markdown("<div class='kpi-title' style='margin-bottom: 6px;'>Customer Segmentation Map</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df,
                x="annual_income",
                y="spending_score",
                color=df["cluster"].astype(str),
                hover_data=["customer_id", "age"] if "age" in df.columns else ["customer_id"],
                labels={
                    "annual_income": "Annual Income ($k)",
                    "spending_score": "Spending Score (1–100)",
                    "color": "Segment"
                },
                template="plotly_dark" if theme == "dark" else "plotly_white",
                title="Income vs. Spending Score by Cluster",
                height=380
            )
            fig_scatter.update_layout(**pl_theme)
            fig_scatter.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.8, color="rgba(255,255,255,0.2)")))
            st.plotly_chart(fig_scatter, use_container_width=True, config={"responsive": True})

        with col_main_right:
            st.markdown("<div class='kpi-title' style='margin-bottom: 6px;'>Segment Share Breakdown</div>", unsafe_allow_html=True)
            cluster_counts = df["cluster"].value_counts().reset_index()
            cluster_counts.columns = ["cluster", "count"]
            cluster_counts["persona_name"] = cluster_counts["cluster"].apply(
                lambda c: (personas.get(c) or personas.get(str(c)) or {}).get("persona_name", f"Cluster {c}")
            )
            fig_donut = px.pie(
                cluster_counts,
                values="count",
                names="persona_name",
                hole=0.55,
                template="plotly_dark" if theme == "dark" else "plotly_white",
                title="Customer Portfolio Composition",
                height=380
            )
            fig_donut.update_layout(**pl_theme)
            fig_donut.update_traces(textinfo="percent+label", showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True, config={"responsive": True})
    else:
        render_info_banner("No clusters computed yet. Please visit <b>Clustering Studio</b> to train the K-Means model on your data.")

    # 3. Macro Feature Distributions (Income & Spending)
    st.markdown("<div class='kpi-title' style='margin-top: 12px; margin-bottom: 6px;'>Macro Feature Distributions</div>", unsafe_allow_html=True)
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        fig_inc = px.histogram(
            df,
            x="annual_income",
            nbins=18,
            marginal="box",
            template="plotly_dark" if theme == "dark" else "plotly_white",
            title="Annual Income Distribution ($k)",
            labels={"annual_income": "Annual Income ($k)"},
            color_discrete_sequence=["#6366F1"],
            height=320
        )
        fig_inc.update_layout(**pl_theme)
        st.plotly_chart(fig_inc, use_container_width=True, config={"responsive": True})

    with dist_col2:
        fig_spd = px.histogram(
            df,
            x="spending_score",
            nbins=18,
            marginal="box",
            template="plotly_dark" if theme == "dark" else "plotly_white",
            title="Spending Score Distribution (1–100)",
            labels={"spending_score": "Spending Score (1–100)"},
            color_discrete_sequence=["#EC4899"],
            height=320
        )
        fig_spd.update_layout(**pl_theme)
        st.plotly_chart(fig_spd, use_container_width=True, config={"responsive": True})

    # 4. Quick Action Routing Bar
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Quick Navigation Actions</div>", unsafe_allow_html=True)
    act_col1, act_col2, act_col3, act_col4 = st.columns(4)

    with act_col1:
        if st.button("Explore Data Studio", use_container_width=True, key="quick_data_btn"):
            st.session_state["nav_selection"] = "Data Explorer"
            st.rerun()
    with act_col2:
        if st.button("Retrain Clustering Model", use_container_width=True, key="quick_train_btn"):
            st.session_state["nav_selection"] = "Clustering Studio"
            st.rerun()
    with act_col3:
        if st.button("Score New Customer", use_container_width=True, key="quick_score_btn"):
            st.session_state["nav_selection"] = "Customer Prediction"
            st.rerun()
    with act_col4:
        if st.button("Export PDF Executive Report", use_container_width=True, key="quick_report_btn"):
            st.session_state["nav_selection"] = "Reports & Exports"
            st.rerun()
