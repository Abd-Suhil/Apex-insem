"""
Customer Segments Page
In-depth cluster profiling, demographic benchmarks, centroid geometry, and persona distribution.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.ui.theme import get_plotly_theme
from src.profiling import generate_cluster_profiles


def render_segments_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Customer Segments & Personas",
        subtitle="Behavioral clustering profiles, centroid coordinates, and comparative persona benchmarks.",
        badge="SEGMENTATION PROFILES"
    )

    df = st.session_state.get("df")
    model = st.session_state.get("model")
    prep = st.session_state.get("preprocessor")
    personas = st.session_state.get("personas", {})
    features = st.session_state.get("selected_features", ["annual_income", "spending_score"])

    if df is None or model is None or "cluster" not in df.columns:
        st.warning("Please train or load a clustering model first.")
        return

    # 1. Cluster Scatter Plot with Centroids
    st.markdown("<div class='kpi-title'>Segment Spatial Distribution</div>", unsafe_allow_html=True)
    f1, f2 = features[0], features[1]

    # Retrieve unscaled centroids
    centroids_scaled = model.cluster_centers_
    centroids_orig = prep.inverse_transform(centroids_scaled)

    fig_seg = px.scatter(
        df,
        x=f1,
        y=f2,
        color=df["cluster"].astype(str),
        hover_data=["customer_id", "age"] if "age" in df.columns else ["customer_id"],
        labels={
            f1: f1.replace('_', ' ').title(),
            f2: f2.replace('_', ' ').title(),
            "color": "Segment"
        },
        title=f"Cluster Separation: {f1.replace('_', ' ').title()} vs. {f2.replace('_', ' ').title()}",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=400
    )
    fig_seg.update_layout(**pl_theme)
    fig_seg.update_traces(marker=dict(size=8, opacity=0.85))

    # Add Centroid Star Markers
    f1_idx = prep.features.index(f1) if f1 in prep.features else 0
    f2_idx = prep.features.index(f2) if f2 in prep.features else 1

    fig_seg.add_trace(go.Scatter(
        x=centroids_orig[:, f1_idx],
        y=centroids_orig[:, f2_idx],
        mode="markers+text",
        marker=dict(
            symbol="star",
            size=18,
            color="#F59E0B",
            line=dict(color="#FFFFFF" if theme == "dark" else "#000000", width=1.5)
        ),
        text=[f"Centroid {i}" for i in range(len(centroids_orig))],
        textposition="top center",
        name="Cluster Centroids"
    ))
    st.plotly_chart(fig_seg, use_container_width=True, config={"responsive": True})

    # 2. Detailed Cluster Demographics Table & Bar Comparisons
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Cluster Demographic Benchmarks</div>", unsafe_allow_html=True)

    profile_df = generate_cluster_profiles(df, cluster_col="cluster")

    # Attach persona names with safe dictionary lookup
    persona_names = [
        (personas.get(c) or personas.get(str(c)) or {}).get("persona_name", f"Cluster {c}")
        for c in profile_df["cluster"]
    ]
    profile_df.insert(1, "persona_name", persona_names)

    st.dataframe(profile_df, use_container_width=True)

    # 3. Comparative Metric Bar Charts
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        if "mean_annual_income" in profile_df.columns:
            fig_bar_inc = px.bar(
                profile_df,
                x="persona_name",
                y="mean_annual_income",
                color="persona_name",
                title="Mean Annual Income by Segment ($k)",
                labels={"persona_name": "Segment", "mean_annual_income": "Mean Income ($k)"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=320
            )
            fig_bar_inc.update_layout(**pl_theme, showlegend=False)
            st.plotly_chart(fig_bar_inc, use_container_width=True, config={"responsive": True})

    with b_col2:
        if "mean_spending_score" in profile_df.columns:
            fig_bar_spd = px.bar(
                profile_df,
                x="persona_name",
                y="mean_spending_score",
                color="persona_name",
                title="Mean Spending Score by Segment (1–100)",
                labels={"persona_name": "Segment", "mean_spending_score": "Mean Spending Score"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=320
            )
            fig_bar_spd.update_layout(**pl_theme, showlegend=False)
            st.plotly_chart(fig_bar_spd, use_container_width=True, config={"responsive": True})

    # 4. Individual Persona Deep-Dive Cards
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Segment Personas & Tactical Focus</div>", unsafe_allow_html=True)

    sorted_keys = sorted(personas.keys(), key=lambda x: int(x))
    for c_id in sorted_keys:
        p = personas[c_id]
        with st.expander(f"Cluster {c_id}: **{p['persona_name']}** — {p.get('percentage', 0):.1f}% of base", expanded=False):
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown(f"**Customer Volume:** {p.get('customer_count', 0)} ({p.get('percentage', 0):.1f}%)")
                st.markdown(f"**Average Annual Income:** ${p.get('avg_income', 0):.1f}k")
                st.markdown(f"**Average Spending Score:** {p.get('avg_spending', 0):.1f}/100")
                if p.get("avg_age") is not None:
                    st.markdown(f"**Average Age:** {p.get('avg_age'):.1f} years")
            with p_col2:
                st.markdown(f"**Behavioral Profile:**<br/>{p.get('description', '')}", unsafe_allow_html=True)
                st.markdown(f"**Strategic Marketing Focus:**<br/>{p.get('marketing_strategy', '')}", unsafe_allow_html=True)
                st.markdown(f"**Retention Priority:** `{p.get('retention_priority', '')}`")
