"""
Reports & Exports Page
Comprehensive artifact exporting: CSV datasets, cluster summaries, high-res PNG plots,
and automated executive PDF generation using ReportLab.
Engineered for multi-resolution screen adaptability.
"""

import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.ui.components import render_header, render_kpi_card, render_info_banner
from src.profiling import generate_cluster_profiles
from src.reporting.pdf_report import generate_executive_pdf


def render_reports_page():
    render_header(
        title="Reports & Executive Exports",
        subtitle="Generate executive PDF dossiers, export structured CSV datasets, and download publication-ready figures.",
        badge="EXPORT ENGINE"
    )

    df = st.session_state.get("df")
    model = st.session_state.get("model")
    metrics = st.session_state.get("metrics", {})
    personas = st.session_state.get("personas", {})
    features = st.session_state.get("selected_features", ["annual_income", "spending_score"])
    k_clusters = st.session_state.get("k_clusters", 5)

    if df is None:
        st.warning("No dataset loaded.")
        return

    # 1. Executive PDF Dossier Generation Card
    st.markdown("<div class='kpi-title'>Executive PDF Dossier Generator</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        Generate an executive-ready PDF report including dataset metrics, mathematical methodology,
        unsupervised performance scores, cluster persona summaries, and commercial action playbooks.
        """)

        if st.button("Generate Executive PDF Report (ReportLab)", use_container_width=True, key="gen_pdf_btn"):
            with st.spinner("Compiling ReportLab Document Object Model and generating PDF..."):
                try:
                    pdf_bytes = generate_executive_pdf(
                        df=df,
                        metrics=metrics,
                        personas=personas,
                        features=features,
                        k_clusters=k_clusters
                    )
                    st.success("Executive PDF Report successfully generated!")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"CustomerIQ_Executive_Report_K{k_clusters}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as ex:
                    st.error(f"PDF Generation error: {ex}")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 2. Structured Data CSV Exports
    st.markdown("<div class='kpi-title'>Structured Data Exports (CSV)</div>", unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        st.markdown("<b>Enriched Clustered Dataset</b><br/><small>Full records with cluster IDs & persona tags</small>", unsafe_allow_html=True)
        if "cluster" in df.columns:
            df_export = df.copy()
            df_export["persona_name"] = df_export["cluster"].apply(
                lambda c: (personas.get(c) or personas.get(str(c)) or {}).get("persona_name", f"Cluster {c}")
            )
            csv_enriched = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Clustered CSV",
                data=csv_enriched,
                file_name="customeriq_clustered_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Clusters not yet computed.")

    with exp_col2:
        st.markdown("<b>Raw Cleaned Dataset</b><br/><small>Validated input records prior to clustering</small>", unsafe_allow_html=True)
        raw_cols = [c for c in df.columns if c not in ["cluster", "persona_name"]]
        csv_cleaned = df[raw_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_cleaned,
            file_name="customeriq_cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp_col3:
        st.markdown("<b>Cluster Summary Profiles</b><br/><small>Centroid means, counts, and market share</small>", unsafe_allow_html=True)
        if "cluster" in df.columns:
            profile_df = generate_cluster_profiles(df, cluster_col="cluster")
            csv_summary = profile_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Summary CSV",
                data=csv_summary,
                file_name="customeriq_cluster_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Clusters not yet computed.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 3. High-Resolution Publication Figure Export (PNG)
    st.markdown("<div class='kpi-title'>Publication Figure Export (High-Resolution PNG)</div>", unsafe_allow_html=True)

    if "cluster" in df.columns and len(features) >= 2:
        f1, f2 = features[0], features[1]

        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        sns.scatterplot(
            data=df,
            x=f1,
            y=f2,
            hue="cluster",
            palette="tab10",
            s=80,
            alpha=0.9,
            ax=ax
        )
        ax.set_title(f"CustomerIQ: {f1.replace('_', ' ').title()} vs. {f2.replace('_', ' ').title()} Segmentation Map", color="#FFFFFF", fontsize=11, pad=10)
        ax.set_xlabel(f1.replace('_', ' ').title(), color="#94A3B8", fontsize=10)
        ax.set_ylabel(f2.replace('_', ' ').title(), color="#94A3B8", fontsize=10)
        ax.tick_params(colors="#94A3B8")
        ax.grid(color="#1E293B", linestyle="--", linewidth=0.5)

        legend = ax.legend(title="Cluster", facecolor="#1E293B", edgecolor="#334155")
        if legend:
            plt.setp(legend.get_texts(), color="#FFFFFF")
            plt.setp(legend.get_title(), color="#FFFFFF")

        img_buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        st.download_button(
            label="Download High-Res Segmentation Map (300 DPI PNG)",
            data=img_buf.getvalue(),
            file_name="customeriq_segmentation_map.png",
            mime="image/png",
            use_container_width=True
        )
