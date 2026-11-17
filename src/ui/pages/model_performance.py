"""
Model Performance Page
Mathematical diagnostics, unsupervised evaluation metrics, cluster separation, and algorithmic bounds.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.ui.theme import get_plotly_theme
from src.segmentation_model import CustomerKMeansModel


def render_model_performance_page():
    theme = st.session_state.get("theme", "dark")
    pl_theme = get_plotly_theme(theme)

    render_header(
        title="Model Performance & Diagnostics",
        subtitle="Unsupervised mathematical validation, cluster separation, compactness, and algorithmic limits.",
        badge="ALGORITHM AUDIT"
    )

    df = st.session_state.get("df")
    model = st.session_state.get("model")
    prep = st.session_state.get("preprocessor")
    metrics = st.session_state.get("metrics", {})
    features = st.session_state.get("selected_features", ["annual_income", "spending_score"])

    if model is None or df is None or not metrics:
        st.warning("No trained model diagnostics found. Please train a model in Clustering Studio.")
        return

    # 1. Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_kpi_card("Silhouette Score", f"{metrics.get('silhouette_score', 0):.4f}", "Range: [-1.0, +1.0] (Higher is better)")
    with m2:
        render_kpi_card("Davies-Bouldin Index", f"{metrics.get('davies_bouldin_index', 0):.4f}", "Range: [0, +inf) (Lower is better)")
    with m3:
        render_kpi_card("Calinski-Harabasz", f"{metrics.get('calinski_harabasz_score', 0):.1f}", "Variance Ratio Criterion")
    with m4:
        inertia_val = getattr(model, "inertia_", 0.0)
        render_kpi_card("Total Inertia (WCSS)", f"{inertia_val:.1f}", "Sum of squared distances")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Per-Cluster Silhouette Scores & Size Distribution
    st.markdown("<div class='kpi-title'>Per-Cluster Silhouette Cohesion Analysis</div>", unsafe_allow_html=True)
    c_sil_data = metrics.get("per_cluster_silhouette", {})

    if c_sil_data:
        sil_df = pd.DataFrame({
            "Cluster": [f"Cluster {k}" for k in c_sil_data.keys()],
            "Avg Silhouette Score": list(c_sil_data.values()),
            "Size (Count)": [metrics.get("cluster_sizes", {}).get(k, metrics.get("cluster_sizes", {}).get(str(k), 0)) for k in c_sil_data.keys()]
        })

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            fig_sil_bar = px.bar(
                sil_df,
                x="Cluster",
                y="Avg Silhouette Score",
                color="Avg Silhouette Score",
                color_continuous_scale="Purples",
                title="Cluster Cohesion (Average Silhouette per Cluster)",
                template="plotly_dark" if theme == "dark" else "plotly_white",
                height=320
            )
            fig_sil_bar.update_layout(**pl_theme)
            fig_sil_bar.add_hline(y=metrics.get("silhouette_score", 0), line_dash="dash", line_color="#10B981")
            st.plotly_chart(fig_sil_bar, use_container_width=True, config={"responsive": True})

        with s_col2:
            fig_pie_size = px.pie(
                sil_df,
                names="Cluster",
                values="Size (Count)",
                title="Cluster Size Equitability",
                template="plotly_dark" if theme == "dark" else "plotly_white",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                height=320
            )
            fig_pie_size.update_layout(**pl_theme)
            st.plotly_chart(fig_pie_size, use_container_width=True, config={"responsive": True})

    # 3. Centroid Coordinates Table (Scaled vs. Original)
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Mathematical Centroid Coordinates</div>", unsafe_allow_html=True)

    centers_scaled = model.cluster_centers_
    centers_orig = prep.inverse_transform(centers_scaled)

    centroid_records = []
    for i in range(len(centers_scaled)):
        rec = {"Cluster": f"Cluster {i}"}
        for f_idx, feat in enumerate(features):
            rec[f"{feat.replace('_', ' ').title()} (Raw)"] = round(centers_orig[i, f_idx], 2)
            rec[f"{feat.replace('_', ' ').title()} (Scaled Z-Score)"] = round(centers_scaled[i, f_idx], 3)
        centroid_records.append(rec)

    st.dataframe(pd.DataFrame(centroid_records), use_container_width=True)

    # 4. Model Assumptions & Algorithmic Boundary Conditions
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>Algorithmic Boundaries & Model Limitations</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="iq-card">
        <b>1. Spherical Cluster Geometry Assumption:</b><br/>
        K-Means minimizes variance around point centroids, implicitly assuming isotropic (spherical) clusters with similar spatial variance. If customer clusters follow curved, elongated, or non-linear manifolds, density-based algorithms (such as DBSCAN or HDBSCAN) may provide superior boundary estimation.
        <br/><br/>
        <b>2. Sensitivity to Outlier Vectors:</b><br/>
        Centroid recalculation takes the arithmetic mean of all cluster members. Extreme outlier transactions can pull centroids away from true density centers. Robust scaling or Winsorization should be considered if heavy outlier distributions arise.
        <br/><br/>
        <b>3. Absence of Supervised Ground-Truth Labels:</b><br/>
        Because customer segmentation is fundamentally an unsupervised task, traditional supervised metrics (such as Accuracy, Precision, Recall, or ROC-AUC) are mathematically inapplicable. Validation relies on geometric separation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and empirical commercial utility.
    </div>
    """, unsafe_allow_html=True)
