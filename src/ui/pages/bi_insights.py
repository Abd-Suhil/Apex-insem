"""
Business Intelligence Page
Strategic decision frameworks, cohort spending dynamics, and revenue optimization roadmaps.
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


def render_bi_insights_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Business Intelligence & Strategic Playbooks",
        subtitle="Data-driven revenue opportunities, churn risk assessment, and empirical marketing playbooks.",
        badge="COMMERCIAL INTELLIGENCE"
    )

    df = st.session_state.get("df")
    personas = st.session_state.get("personas", {})
    if df is None or "cluster" not in df.columns:
        st.warning("Please run clustering first to generate business intelligence.")
        return

    profile_df = generate_cluster_profiles(df, cluster_col="cluster")

    # 1. Macro Summary Cards
    st.markdown("<div class='kpi-title'>Cohort Portfolio Value Analysis</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        high_income_custs = len(df[df["annual_income"] >= df["annual_income"].median()])
        render_kpi_card("Affluent Base (> Median)", f"{high_income_custs} customers", f"{(high_income_custs/len(df)*100):.1f}% of total portfolio")

    with c2:
        high_spend_custs = len(df[df["spending_score"] >= 60])
        render_kpi_card("High Velocity Spenders (Score >= 60)", f"{high_spend_custs} customers", f"{(high_spend_custs/len(df)*100):.1f}% active shoppers")

    with c3:
        avg_basket_index = (df["annual_income"] * (df["spending_score"] / 100)).mean()
        render_kpi_card("Revenue Potential Index", f"{avg_basket_index:.1f}", "Synthetic purchasing power index")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Quadrant Opportunity Matrix (Plotly Bubble Chart)
    st.markdown("<div class='kpi-title'>Strategic Segmentation Quadrant Matrix</div>", unsafe_allow_html=True)
    
    # Prepare bubble data
    bubble_data = []
    for _, row in profile_df.iterrows():
        c = int(row["cluster"])
        p = personas.get(c) or personas.get(str(c)) or {}
        bubble_data.append({
            "Cluster": f"Cluster {c}",
            "Persona": p.get("persona_name", f"Cluster {c}"),
            "Mean Income ($k)": row.get("mean_annual_income", 50),
            "Mean Spending Score": row.get("mean_spending_score", 50),
            "Customer Count": int(row.get("customer_count", 0)),
            "Portfolio Share (%)": float(row.get("percentage", 0))
        })
    bubble_df = pd.DataFrame(bubble_data)

    fig_matrix = px.scatter(
        bubble_df,
        x="Mean Income ($k)",
        y="Mean Spending Score",
        size="Customer Count",
        color="Persona",
        hover_name="Persona",
        text="Cluster",
        size_max=40,
        title="Segment Value Matrix (Bubble Size = Customer Volume)",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=380
    )
    fig_matrix.update_layout(**pl_theme)
    fig_matrix.add_hline(y=50, line_dash="dot", line_color="rgba(150,150,150,0.4)")
    fig_matrix.add_vline(x=df["annual_income"].mean(), line_dash="dot", line_color="rgba(150,150,150,0.4)")
    fig_matrix.update_traces(textposition="top center")
    st.plotly_chart(fig_matrix, use_container_width=True, config={"responsive": True})

    # 3. Data-Backed Playbooks vs. Empirical Observations
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Strategic Recommendations by Segment</div>", unsafe_allow_html=True)

    for _, row in bubble_df.iterrows():
        c_label = row["Cluster"]
        c_num = int(c_label.replace("Cluster ", ""))
        p_info = personas.get(c_num) or personas.get(str(c_num)) or {}

        with st.expander(f"📌 {row['Persona']} ({row['Portfolio Share (%)']}% Share)", expanded=True):
            o_col1, o_col2 = st.columns(2)
            with o_col1:
                st.markdown(f"""
                <b>📊 Empirical Statistical Observations:</b><br/>
                • Mean Annual Income: <b>${row['Mean Income ($k)']:.1f}k</b><br/>
                • Mean Spending Score: <b>{row['Mean Spending Score']:.1f} / 100</b><br/>
                • Segment Size: <b>{row['Customer Count']} customers</b> ({row['Portfolio Share (%)']}%)
                """, unsafe_allow_html=True)
            with o_col2:
                st.markdown(f"""
                <b>🎯 Prescriptive Commercial Strategy:</b><br/>
                • <b>Action:</b> {p_info.get('marketing_strategy', 'Standard engagement')}<br/>
                • <b>Retention Classification:</b> <code>{p_info.get('retention_priority', 'Normal')}</code>
                """, unsafe_allow_html=True)
