"""
CustomerIQ – Intelligent Customer Segmentation & Business Analytics
Main Application Entry Point orchestrating modular navigation and session lifecycle.
Engineered for multi-resolution screen adaptability.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import json
import streamlit as st
import pandas as pd

from src.ui.theme import get_theme_css
from src.data_loader import load_customer_data
from src.preprocessor import CustomerPreprocessor
from src.segmentation_model import CustomerKMeansModel
from src.evaluation import evaluate_clusters
from src.profiling import generate_cluster_profiles, assign_persona_metadata

# Import Modular Page Renderers
from src.ui.pages.overview import render_overview_page
from src.ui.pages.data_explorer import render_data_explorer_page
from src.ui.pages.analytics import render_analytics_page
from src.ui.pages.clustering_studio import render_clustering_studio_page
from src.ui.pages.segments import render_segments_page
from src.ui.pages.prediction import render_prediction_page
from src.ui.pages.bi_insights import render_bi_insights_page
from src.ui.pages.model_performance import render_model_performance_page
from src.ui.pages.reports import render_reports_page
from src.ui.pages.settings_page import render_settings_page


# Global Page Configuration
st.set_page_config(
    page_title="CustomerIQ – Customer Segmentation & Business Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Ensure all required dataset and model artifacts are initialized in session state."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Overview"

    # Default Feature Space
    if "selected_features" not in st.session_state:
        st.session_state["selected_features"] = ["annual_income", "spending_score"]

    if "k_clusters" not in st.session_state:
        st.session_state["k_clusters"] = 5

    # Load Dataset
    if "df" not in st.session_state:
        default_csv = "data/mall_customers.csv"
        if os.path.exists(default_csv):
            df_loaded = load_customer_data(default_csv)
            st.session_state["df"] = df_loaded
            st.session_state["dataset_name"] = "Mall Customers Benchmark"
        else:
            st.session_state["df"] = None

    # Load Model and Preprocessor if on disk
    model_path = "models/kmeans_model.joblib"
    prep_path = "models/preprocessor.joblib"
    meta_path = "models/cluster_metadata.json"

    if "model" not in st.session_state:
        if os.path.exists(model_path) and os.path.exists(prep_path):
            try:
                model = CustomerKMeansModel.load(model_path)
                prep = CustomerPreprocessor.load(prep_path)
                st.session_state["model"] = model
                st.session_state["preprocessor"] = prep
                st.session_state["k_clusters"] = model.n_clusters
                st.session_state["selected_features"] = prep.features

                # Ensure df has clusters
                df = st.session_state.get("df")
                if df is not None:
                    X_s, _ = prep.transform(df)
                    df["cluster"] = model.predict(X_s)
                    st.session_state["df"] = df

                # Load metadata
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    st.session_state["metrics"] = meta.get("metrics", {})
                    # Ensure personas keys are both ints and strings for seamless lookup
                    personas_raw = meta.get("personas", {})
                    p_dict = {}
                    for k, v in personas_raw.items():
                        p_dict[int(k)] = v
                        p_dict[str(k)] = v
                    st.session_state["personas"] = p_dict
                else:
                    if df is not None:
                        st.session_state["metrics"] = evaluate_clusters(X_s, df["cluster"].values)
                        profile_df = generate_cluster_profiles(df)
                        p_gen = assign_persona_metadata(profile_df, df)
                        p_dict = {}
                        for k, v in p_gen.items():
                            p_dict[int(k)] = v
                            p_dict[str(k)] = v
                        st.session_state["personas"] = p_dict

            except Exception as e:
                st.sidebar.error(f"Error loading models: {e}")


def main():
    initialize_session_state()

    # Apply SaaS Dynamic Theme CSS
    current_theme = st.session_state.get("theme", "dark")
    st.markdown(get_theme_css(current_theme), unsafe_allow_html=True)

    # Sidebar Navigation Branding
    with st.sidebar:
        st.markdown("""
        <div style="padding: 6px 0 16px 0;">
            <div style="font-size: 1.35rem; font-weight: 800; color: #6366F1; letter-spacing: -0.02em;">CustomerIQ</div>
            <div style="font-size: 0.78rem; color: #9CA3AF; margin-top: 1px;">Turn Customer Data into Business Intelligence.</div>
        </div>
        """, unsafe_allow_html=True)

        nav_options = [
            "Overview",
            "Data Explorer",
            "Exploratory Analytics",
            "Clustering Studio",
            "Customer Segments",
            "Customer Prediction",
            "Business Intelligence",
            "Model Performance",
            "Reports & Exports",
            "Settings"
        ]

        # Use synchronized session state for radio
        current_nav = st.session_state.get("nav_selection", "Overview")
        default_index = nav_options.index(current_nav) if current_nav in nav_options else 0

        selected_nav = st.radio(
            "Navigation Menu",
            options=nav_options,
            index=default_index,
            key="nav_selection",
            label_visibility="collapsed"
        )

        # System Status Footer
        df = st.session_state.get("df")
        k_val = st.session_state.get("k_clusters", 5)
        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.72rem; color: #9CA3AF; line-height: 1.6;">
            <b>Workspace Status:</b><br/>
            • Active Base: <code>{len(df) if df is not None else 0:,} profiles</code><br/>
            • Active Clusters: <code>K = {k_val}</code><br/>
            • Theme: <code>{current_theme.capitalize()}</code>
        </div>
        """, unsafe_allow_html=True)

    # Route to Appropriate Page Renderer
    pages_router = {
        "Overview": render_overview_page,
        "Data Explorer": render_data_explorer_page,
        "Exploratory Analytics": render_analytics_page,
        "Clustering Studio": render_clustering_studio_page,
        "Customer Segments": render_segments_page,
        "Customer Prediction": render_prediction_page,
        "Business Intelligence": render_bi_insights_page,
        "Model Performance": render_model_performance_page,
        "Reports & Exports": render_reports_page,
        "Settings": render_settings_page,
    }

    renderer = pages_router.get(selected_nav, render_overview_page)
    renderer()


if __name__ == "__main__":
    main()
