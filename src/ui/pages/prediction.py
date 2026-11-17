"""
Customer Prediction Page
Real-time individual inference and batch scoring engine with interactive spatial positioning.
Engineered for multi-resolution screen adaptability.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.ui.theme import get_plotly_theme
from src.data_loader import load_customer_data


def render_prediction_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Customer Scoring & Inference",
        subtitle="Classify single customer prospects or batch customer cohorts into active behavioral segments.",
        badge="REAL-TIME INFERENCE"
    )

    df = st.session_state.get("df")
    model = st.session_state.get("model")
    prep = st.session_state.get("preprocessor")
    personas = st.session_state.get("personas", {})
    features = st.session_state.get("selected_features", ["annual_income", "spending_score"])

    if model is None or prep is None or df is None:
        st.warning("Model and Preprocessor not loaded. Please train or load a model from Clustering Studio.")
        return

    mode = st.radio("Inference Mode:", ["Single Prospect Scoring", "Batch Cohort CSV Scoring"], horizontal=True, key="pred_mode_radio")

    if mode == "Single Prospect Scoring":
        st.markdown("<div class='kpi-title'>Prospect Behavioral Inputs</div>", unsafe_allow_html=True)

        with st.form("single_predict_form"):
            col_in1, col_in2, col_in3 = st.columns(3)
            with col_in1:
                input_income = st.slider("Annual Income ($k):", min_value=10, max_value=150, value=75, step=1, key="pred_income_slider")
            with col_in2:
                input_spending = st.slider("Spending Score (1–100):", min_value=1, max_value=100, value=80, step=1, key="pred_spending_slider")
            with col_in3:
                input_age = st.slider("Customer Age:", min_value=18, max_value=85, value=32, step=1, key="pred_age_slider")

            submit_pred = st.form_submit_button("Predict Customer Segment", use_container_width=True)

        if submit_pred:
            # Build sample dataframe matching preprocessor requirements
            sample_df = pd.DataFrame({
                "annual_income": [input_income],
                "spending_score": [input_spending],
                "age": [input_age],
                "customer_id": [9999]
            })

            # Scale and Predict
            X_s, _ = prep.transform(sample_df)
            pred_cluster = int(model.predict(X_s)[0])
            p_data = personas.get(pred_cluster) or personas.get(str(pred_cluster)) or {}

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            res_box1, res_box2 = st.columns([1, 2])

            with res_box1:
                render_kpi_card(
                    title="Assigned Segment",
                    value=f"Cluster {pred_cluster}",
                    subtitle=p_data.get("persona_name", "Standard")
                )
                st.markdown(f"**Retention Priority:** `{p_data.get('retention_priority', 'Standard')}`")

            with res_box2:
                st.markdown(f"""
                <div class="iq-card">
                    <div class="kpi-title">Segment Profile & Marketing Action Plan</div>
                    <div style="font-size: 0.95rem; margin-bottom: 6px;"><b>Persona Narrative:</b> {p_data.get('description', '')}</div>
                    <div style="font-size: 0.95rem; color: #6366F1;"><b>Targeted Strategy:</b> {p_data.get('marketing_strategy', '')}</div>
                </div>
                """, unsafe_allow_html=True)

            # Spatial Visualization: New Point vs. Existing Population
            st.markdown("<div class='kpi-title' style='margin-top: 15px;'>Spatial Position Relative to Customer Base</div>", unsafe_allow_html=True)
            f1, f2 = features[0], features[1]

            fig_pred = px.scatter(
                df,
                x=f1,
                y=f2,
                color=df["cluster"].astype(str),
                opacity=0.35,
                title="Prospect Spatial Placement on Customer Map",
                labels={f1: f1.replace('_', ' ').title(), f2: f2.replace('_', ' ').title(), "color": "Segment"},
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=380
            )
            fig_pred.update_layout(**pl_theme)

            # Exact coordinate mapping from sample_df
            new_x = float(sample_df[f1].iloc[0])
            new_y = float(sample_df[f2].iloc[0])

            fig_pred.add_trace(go.Scatter(
                x=[new_x],
                y=[new_y],
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=20,
                    color="#EC4899",
                    line=dict(color="#FFFFFF", width=2)
                ),
                text=["★ New Customer Prospect"],
                textposition="top center",
                name="New Prospect"
            ))
            st.plotly_chart(fig_pred, use_container_width=True, config={"responsive": True})

    else:
        st.markdown("<div class='kpi-title'>Batch Customer Cohort Scoring</div>", unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload CSV for Cohort Inference", type=["csv"], key="batch_scoring_uploader")

        if batch_file is not None:
            try:
                batch_df = load_customer_data(batch_file)
                X_batch_scaled, _ = prep.transform(batch_df)
                batch_preds = model.predict(X_batch_scaled)

                scored_df = batch_df.copy()
                scored_df["cluster"] = batch_preds
                scored_df["persona_name"] = [
                    (personas.get(c) or personas.get(str(c)) or {}).get("persona_name", f"Cluster {c}")
                    for c in batch_preds
                ]
                scored_df["marketing_strategy"] = [
                    (personas.get(c) or personas.get(str(c)) or {}).get("marketing_strategy", "Standard Engagement")
                    for c in batch_preds
                ]

                st.success(f"Successfully processed {len(scored_df)} customer records!")
                st.dataframe(scored_df, use_container_width=True, height=350)

                csv_buf = io.StringIO()
                scored_df.to_csv(csv_buf, index=False)
                st.download_button(
                    label="Download Scored Cohort Dataset (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="customeriq_batch_scored.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Batch Scoring Error: {ex}")
