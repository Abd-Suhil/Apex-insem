"""
K-Means Clustering Studio Page
Interactive Machine Learning workspace for hyperparameter tuning, Elbow/Silhouette analysis,
model training, and full artifact persistence (models + metadata).
Engineered for multi-resolution screen adaptability.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.ui.theme import get_plotly_theme
from src.preprocessor import CustomerPreprocessor
from src.segmentation_model import CustomerKMeansModel
from src.evaluation import evaluate_clusters
from src.profiling import generate_cluster_profiles, assign_persona_metadata


def render_clustering_studio_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="K-Means Clustering Studio",
        subtitle="Configure feature projections, evaluate optimal centroids via WCSS & Silhouette analysis, and train production models.",
        badge="MACHINE LEARNING ENGINE"
    )

    df = st.session_state.get("df")
    if df is None:
        st.warning("No dataset loaded.")
        return

    # Configuration controls in 2 columns
    cfg_col1, cfg_col2 = st.columns([1.2, 1])

    with cfg_col1:
        st.markdown("<div class='kpi-title'>Feature Engineering & Scaling</div>", unsafe_allow_html=True)
        available_features = [c for c in ["annual_income", "spending_score", "age"] if c in df.columns]
        selected_features = st.multiselect(
            "Select Features for Clustering:",
            options=available_features,
            default=st.session_state.get("selected_features", ["annual_income", "spending_score"]),
            key="studio_feature_select"
        )
        if len(selected_features) < 2:
            st.warning("K-Means requires at least 2 numerical features. Defaulting to Income & Spending Score.")
            selected_features = ["annual_income", "spending_score"]

        scaler_choice = st.selectbox(
            "Feature Normalization Technique:",
            options=["StandardScaler (Zero-Mean, Unit-Variance)", "MinMaxScaler (0 to 1 Normalization)"],
            index=0,
            key="studio_scaler_select"
        )
        scaler_type = "standard" if "Standard" in scaler_choice else "minmax"

    with cfg_col2:
        st.markdown("<div class='kpi-title'>Hyperparameter Calibration</div>", unsafe_allow_html=True)
        current_k = st.session_state.get("k_clusters", 5)
        k_slider = st.slider("Target Number of Clusters (K):", min_value=2, max_value=10, value=current_k, key="studio_k_slider")
        rand_seed = st.number_input("Random State Seed:", min_value=0, max_value=999, value=42, key="studio_seed_input")
        n_init_val = st.slider("Centroid Initialization Restarts (n_init):", min_value=5, max_value=20, value=10, key="studio_ninit_slider")

    # Optimal K Detection Section
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Convergence & Optimization Analysis (Elbow vs. Silhouette)</div>", unsafe_allow_html=True)

    # Preprocess current features to run analysis
    prep_temp = CustomerPreprocessor(features=selected_features, scaler_type=scaler_type, include_pca=True)
    X_scaled, X_pca = prep_temp.fit_transform(df)

    with st.spinner("Calculating Within-Cluster Sum of Squares (WCSS) & Silhouette scores..."):
        opt_analysis = CustomerKMeansModel.compute_elbow_and_silhouette(
            X_scaled, k_min=2, k_max=10, random_state=rand_seed
        )

    best_k_sil = opt_analysis["best_k_silhouette"]
    best_k_elb = opt_analysis["best_k_elbow"]

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        render_info_banner(f"<b>Elbow Method Recommendation:</b> Diminishing returns inflection occurs at <b>K = {best_k_elb}</b>.")
    with opt_col2:
        render_info_banner(f"<b>Max Silhouette Score Recommendation:</b> Peak cluster cohesion and separation at <b>K = {best_k_sil}</b>.")

    # Side-by-side Plotly Curves with responsive heights
    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=opt_analysis["full_k"],
            y=opt_analysis["inertias"],
            mode="lines+markers",
            line=dict(color="#6366F1", width=2.5),
            marker=dict(size=7, color="#818CF8"),
            name="Inertia (WCSS)"
        ))
        fig_elbow.add_vline(
            x=best_k_elb, line_dash="dash", line_color="#EF4444",
            annotation_text=f"Elbow K={best_k_elb}", annotation_position="top right"
        )
        fig_elbow.update_layout(
            title="Elbow Curve (WCSS vs. K)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster Sum of Squares",
            template="plotly_dark" if theme == "dark" else "plotly_white",
            height=340,
            **pl_theme
        )
        st.plotly_chart(fig_elbow, use_container_width=True, config={"responsive": True})

    with curve_col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=opt_analysis["silhouette_k_range"],
            y=opt_analysis["silhouette_scores"],
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=7, color="#34D399"),
            name="Silhouette Score"
        ))
        fig_sil.add_vline(
            x=best_k_sil, line_dash="dash", line_color="#F59E0B",
            annotation_text=f"Peak K={best_k_sil}", annotation_position="top right"
        )
        fig_sil.update_layout(
            title="Silhouette Score Curve vs. K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Score ([-1, 1])",
            template="plotly_dark" if theme == "dark" else "plotly_white",
            height=340,
            **pl_theme
        )
        st.plotly_chart(fig_sil, use_container_width=True, config={"responsive": True})

    # Model Training Action Card
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    train_box_col1, train_box_col2 = st.columns([2, 1])

    with train_box_col1:
        st.markdown(f"""
        <b>Active Model Configuration:</b><br/>
        • <b>Selected Features:</b> <code>{selected_features}</code><br/>
        • <b>Normalization:</b> <code>{scaler_choice}</code><br/>
        • <b>Target Clusters:</b> <code>K = {k_slider}</code> | <b>Seeding Algorithm:</b> <code>k-means++</code>
        """, unsafe_allow_html=True)

    with train_box_col2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("Execute Pipeline & Save Model", use_container_width=True, key="studio_exec_btn"):
            with st.spinner("Fitting Scaler & Training K-Means Model..."):
                # 1. Fit Preprocessor
                prep = CustomerPreprocessor(features=selected_features, scaler_type=scaler_type, include_pca=True)
                X_s, X_p = prep.fit_transform(df)

                # 2. Fit Model
                trained_model = CustomerKMeansModel(
                    n_clusters=k_slider,
                    random_state=rand_seed,
                    n_init=n_init_val
                )
                labels = trained_model.fit_predict(X_s)

                # 3. Evaluation
                eval_metrics = evaluate_clusters(X_s, labels)

                # 4. Profiling & Personas
                df_updated = df.copy()
                df_updated["cluster"] = labels
                profile_df = generate_cluster_profiles(df_updated, cluster_col="cluster")
                personas_data = assign_persona_metadata(profile_df, df_updated)

                # 5. Full Persistence to Disk (models + complete metadata JSON)
                os.makedirs("models", exist_ok=True)
                trained_model.save("models/kmeans_model.joblib")
                prep.save("models/preprocessor.joblib")

                centers_scaled = trained_model.cluster_centers_
                centers_orig = prep.inverse_transform(centers_scaled)

                metadata = {
                    "n_clusters": k_slider,
                    "features": selected_features,
                    "scaler_type": scaler_type,
                    "random_state": rand_seed,
                    "metrics": {
                        "silhouette_score": eval_metrics["silhouette_score"],
                        "davies_bouldin_index": eval_metrics["davies_bouldin_index"],
                        "calinski_harabasz_score": eval_metrics["calinski_harabasz_score"]
                    },
                    "optimal_k_analysis": {
                        "k_range": opt_analysis["k_range"],
                        "inertias": opt_analysis["inertias"],
                        "silhouette_scores": opt_analysis["silhouette_scores"],
                        "best_k_silhouette": best_k_sil,
                        "best_k_elbow": best_k_elb
                    },
                    "cluster_centers_scaled": centers_scaled.tolist(),
                    "cluster_centers_original": centers_orig.tolist(),
                    "personas": personas_data
                }

                with open("models/cluster_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # Update session state across workspace
                st.session_state["model"] = trained_model
                st.session_state["preprocessor"] = prep
                st.session_state["df"] = df_updated
                st.session_state["k_clusters"] = k_slider
                st.session_state["selected_features"] = selected_features
                st.session_state["metrics"] = eval_metrics
                st.session_state["personas"] = personas_data
                st.session_state["model_trained"] = True

                st.success(f"Model successfully trained & persisted with K={k_slider}! Silhouette Score: {eval_metrics['silhouette_score']:.4f}")
                st.rerun()
