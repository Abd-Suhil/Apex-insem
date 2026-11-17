"""
Exploratory Data Analytics Page
Interactive Plotly visualizations for statistical distributions, correlations, and pairwise dynamics.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header
from src.ui.theme import get_plotly_theme


def render_analytics_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Exploratory Analytics",
        subtitle="Univariate distributions, bivariate relationships, and multidimensional correlation heatmaps.",
        badge="STATISTICAL DISCOVERY"
    )

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded.")
        return

    # Tabs for structured exploration
    tab_dist, tab_bivar, tab_corr = st.tabs([
        "Distribution Analysis",
        "Bivariate Relationships",
        "Correlation & Pairwise"
    ])

    # 1. Distributions Tab
    with tab_dist:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            fig_inc = px.histogram(
                df, x="annual_income", nbins=18, marginal="violin",
                color_discrete_sequence=["#6366F1"],
                title="Annual Income Distribution ($k)",
                labels={"annual_income": "Annual Income ($k)"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=320
            )
            fig_inc.update_layout(**pl_theme)
            st.plotly_chart(fig_inc, use_container_width=True, config={"responsive": True})

        with d_col2:
            fig_spd = px.histogram(
                df, x="spending_score", nbins=18, marginal="violin",
                color_discrete_sequence=["#EC4899"],
                title="Spending Score Distribution (1–100)",
                labels={"spending_score": "Spending Score (1–100)"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=320
            )
            fig_spd.update_layout(**pl_theme)
            st.plotly_chart(fig_spd, use_container_width=True, config={"responsive": True})

        d_col3, d_col4 = st.columns(2)
        with d_col3:
            if "age" in df.columns:
                fig_age = px.histogram(
                    df, x="age", nbins=16, marginal="box",
                    color_discrete_sequence=["#10B981"],
                    title="Age Distribution (Years)",
                    labels={"age": "Age (Years)"},
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=320
                )
                fig_age.update_layout(**pl_theme)
                st.plotly_chart(fig_age, use_container_width=True, config={"responsive": True})
            else:
                st.info("Age column not found in dataset.")

        with d_col4:
            if "gender" in df.columns:
                gender_counts = df["gender"].value_counts().reset_index()
                gender_counts.columns = ["Gender", "Count"]
                fig_gen = px.bar(
                    gender_counts, x="Gender", y="Count",
                    color="Gender",
                    color_discrete_sequence=["#8B5CF6", "#3B82F6"],
                    title="Gender Demographics",
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=320
                )
                fig_gen.update_layout(**pl_theme)
                st.plotly_chart(fig_gen, use_container_width=True, config={"responsive": True})
            else:
                st.info("Gender column not found in dataset.")

    # 2. Bivariate Relationships Tab
    with tab_bivar:
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_inc_spd = px.scatter(
                df, x="annual_income", y="spending_score",
                color="gender" if "gender" in df.columns else None,
                hover_data=["customer_id"] if "customer_id" in df.columns else None,
                trendline="ols",
                title="Annual Income vs. Spending Score",
                labels={"annual_income": "Annual Income ($k)", "spending_score": "Spending Score (1–100)"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=380
            )
            fig_inc_spd.update_layout(**pl_theme)
            fig_inc_spd.update_traces(marker=dict(size=7, opacity=0.8))
            st.plotly_chart(fig_inc_spd, use_container_width=True, config={"responsive": True})

        with b_col2:
            if "age" in df.columns:
                fig_age_spd = px.scatter(
                    df, x="age", y="spending_score",
                    color="gender" if "gender" in df.columns else None,
                    hover_data=["customer_id"] if "customer_id" in df.columns else None,
                    trendline="ols",
                    title="Age vs. Spending Score",
                    labels={"age": "Age (Years)", "spending_score": "Spending Score (1–100)"},
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=380
                )
                fig_age_spd.update_layout(**pl_theme)
                fig_age_spd.update_traces(marker=dict(size=7, opacity=0.8))
                st.plotly_chart(fig_age_spd, use_container_width=True, config={"responsive": True})
            else:
                st.info("Age column not present.")

    # 3. Correlation & Pairwise Tab
    with tab_corr:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["customer_id", "cluster"]]
        if len(numeric_cols) >= 2:
            c_col1, c_col2 = st.columns([1, 1.3])
            with c_col1:
                corr_matrix = df[numeric_cols].corr().round(3)
                fig_heat = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Purples",
                    title="Pearson Correlation Heatmap",
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=380
                )
                fig_heat.update_layout(**pl_theme)
                st.plotly_chart(fig_heat, use_container_width=True, config={"responsive": True})

            with c_col2:
                fig_pair = px.scatter_matrix(
                    df,
                    dimensions=numeric_cols,
                    color="cluster" if "cluster" in df.columns else None,
                    title="Pairwise Feature Interaction Matrix",
                    template="plotly_dark" if theme == "dark" else "plotly_white",
                    height=380
                )
                fig_pair.update_layout(**pl_theme)
                st.plotly_chart(fig_pair, use_container_width=True, config={"responsive": True})
        else:
            st.info("At least 2 numeric features required for correlation analysis.")
