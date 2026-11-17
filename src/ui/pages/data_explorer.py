"""
Data Explorer Page
Dataset ingestion, interactive filtering, dimensional validation, and data export.
Engineered for multi-resolution screen adaptability.
"""

import streamlit as st
import pandas as pd
from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.data_loader import load_customer_data, get_dataset_summary


def render_data_explorer_page():
    render_header(
        title="Data Explorer & Management",
        subtitle="Inspect, validate, filter, and audit customer records across all demographic dimensions.",
        badge="DATA OPS"
    )

    df_current = st.session_state.get("df")
    if df_current is None:
        st.warning("No dataset currently loaded.")
        return

    # Upload custom CSV card
    st.markdown("<div class='kpi-title'>Dataset Ingestion</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload Custom Customer CSV Dataset",
        type=["csv"],
        help="Expected columns: Annual Income, Spending Score, Age (optional), Gender (optional)"
    )

    if uploaded_file is not None:
        try:
            new_df = load_customer_data(uploaded_file)
            st.success(f"Dataset successfully validated! Ingested {len(new_df)} records.")
            if st.button("Apply Ingested Dataset to Workspace", key="apply_uploaded_btn", use_container_width=True):
                # If we have a model and preprocessor, automatically segment the new dataset
                model = st.session_state.get("model")
                prep = st.session_state.get("preprocessor")
                if model is not None and prep is not None:
                    try:
                        X_s, _ = prep.transform(new_df)
                        new_df["cluster"] = model.predict(X_s)
                    except Exception:
                        if "cluster" in new_df.columns:
                            new_df.drop(columns=["cluster"], inplace=True)
                
                st.session_state["df"] = new_df
                st.session_state["dataset_name"] = uploaded_file.name
                st.rerun()
        except Exception as ex:
            st.error(f"Validation Failure: {ex}")

    # Dataset Profile Metrics
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        render_kpi_card("Total Rows", f"{len(df_current):,}", "Active dataset records")
    with m_col2:
        render_kpi_card("Total Columns", f"{len(df_current.columns)}", "Detected feature schema")
    with m_col3:
        missing_count = int(df_current.isnull().sum().sum())
        render_kpi_card("Missing Values", f"{missing_count}", "Clean (0 missing)" if missing_count == 0 else "Action required")
    with m_col4:
        dup_count = int(df_current.duplicated().sum())
        render_kpi_card("Duplicate Rows", f"{dup_count}", "Unique customer records")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Interactive Query & Filtering Controls
    with st.expander("Filter & Search Controls", expanded=True):
        f_row1_col1, f_row1_col2 = st.columns([1.5, 1])
        with f_row1_col1:
            search_query = st.text_input("Search Customer ID:", placeholder="e.g. 42", key="cust_search_input")
        with f_row1_col2:
            if "gender" in df_current.columns:
                genders_available = sorted(list(df_current["gender"].dropna().unique()))
                selected_genders = st.multiselect("Gender:", options=genders_available, default=genders_available, key="gender_filter_input")
            else:
                selected_genders = []

        f_row2_col1, f_row2_col2, f_row2_col3 = st.columns(3)
        with f_row2_col1:
            min_inc, max_inc = int(df_current["annual_income"].min()), int(df_current["annual_income"].max())
            inc_range = st.slider("Annual Income ($k):", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc), key="income_range_slider")
        with f_row2_col2:
            min_spd, max_spd = int(df_current["spending_score"].min()), int(df_current["spending_score"].max())
            spd_range = st.slider("Spending Score (1–100):", min_value=min_spd, max_value=max_spd, value=(min_spd, max_spd), key="spending_range_slider")
        with f_row2_col3:
            if "age" in df_current.columns:
                min_age, max_age = int(df_current["age"].min()), int(df_current["age"].max())
                age_range = st.slider("Age Range:", min_value=min_age, max_value=max_age, value=(min_age, max_age), key="age_range_slider")
            else:
                age_range = None

    # Apply Filter Logic (Copy to ensure original data is never modified)
    df_filtered = df_current.copy()

    if search_query.strip():
        if "customer_id" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["customer_id"].astype(str).str.contains(search_query.strip())]

    if selected_genders and "gender" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["gender"].isin(selected_genders)]

    df_filtered = df_filtered[
        (df_filtered["annual_income"] >= inc_range[0]) & 
        (df_filtered["annual_income"] <= inc_range[1]) &
        (df_filtered["spending_score"] >= spd_range[0]) & 
        (df_filtered["spending_score"] <= spd_range[1])
    ]

    if age_range and "age" in df_filtered.columns:
        df_filtered = df_filtered[
            (df_filtered["age"] >= age_range[0]) & 
            (df_filtered["age"] <= age_range[1])
        ]

    st.markdown(f"<div class='kpi-title' style='margin-top: 15px;'>Dataset Records ({len(df_filtered)} matches of {len(df_current)})</div>", unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, height=360)

    # Column Schema Inspection & Export
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    sch_col, dl_col = st.columns([2, 1])

    with sch_col:
        schema_df = pd.DataFrame({
            "Column Name": df_current.columns,
            "Data Type": [str(t) for t in df_current.dtypes],
            "Non-Null Count": [f"{df_current[c].count()} / {len(df_current)}" for c in df_current.columns],
            "Sample Value": [str(df_current[c].iloc[0]) for c in df_current.columns]
        })
        with st.expander("Feature Schema & Types"):
            st.dataframe(schema_df, use_container_width=True)

    with dl_col:
        csv_data = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv_data,
            file_name="customeriq_filtered_data.csv",
            mime="text/csv",
            use_container_width=True
        )
