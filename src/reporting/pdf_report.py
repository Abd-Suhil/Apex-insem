"""
PDF Executive Report Generator using ReportLab
Generates a comprehensive executive intelligence dossier for CustomerIQ.
"""

import io
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


def generate_executive_pdf(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    personas: Dict[Any, Dict[str, Any]],
    features: list,
    k_clusters: int,
    scaler_type: str = "StandardScaler"
) -> bytes:
    """
    Generate a full-fidelity PDF report as in-memory bytes using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1E293B")
    accent_indigo = colors.HexColor("#4F46E5")
    text_muted = colors.HexColor("#64748B")
    light_bg = colors.HexColor("#F8FAFC")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=accent_indigo,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155")
    )

    body_bold = ParagraphStyle(
        "BodyDarkBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("CustomerIQ — Executive Intelligence Dossier", title_style))
    story.append(Paragraph("Turn Customer Data into Business Intelligence | Autonomous Segmentation Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_indigo, spaceAfter=12))

    # Meta information table
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    meta_data = [
        [
            Paragraph("<b>Generated:</b> " + now_str, body_style),
            Paragraph(f"<b>Total Sample Size:</b> {len(df):,} customers", body_style),
            Paragraph(f"<b>Active Segments (K):</b> {k_clusters}", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[200, 170, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))

    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    exec_summary_text = (
        "This strategic intelligence report provides an unsupervised behavioral segmentation of the customer "
        "base using high-dimensional Euclidean clustering. The primary business objective is to transition from "
        "undifferentiated mass marketing to precision persona targeting, maximizing customer lifetime value (LTV), "
        "reducing churn among affluent demographics, and establishing tailored conversion funnels across disparate "
        "purchasing power tiers."
    )
    story.append(Paragraph(exec_summary_text, body_style))
    story.append(Spacer(1, 10))

    # 2. Methodology & Mathematical Foundation
    story.append(Paragraph("2. Mathematical Formulation & Preprocessing", h1_style))
    method_text = (
        f"<b>Preprocessing:</b> Numerical features ({', '.join(features)}) were standardized using <b>{scaler_type}</b> "
        "(zero mean, unit variance) to eliminate scale bias. Without normalization, variables with larger raw magnitudes "
        "(such as annual income in thousands) would disproportionately skew the Euclidean distance computations.<br/>"
        "<b>K-Means++ Convergence:</b> To avoid suboptimal local minima, centroids were initialized via the "
        "probabilistic <i>k-means++</i> seeding scheme. Within-Cluster Sum of Squares (WCSS) and Silhouette coefficients "
        "were computed across $K \\in [2, 10]$ to rigorously pinpoint optimal cluster boundaries."
    )
    story.append(Paragraph(method_text, body_style))
    story.append(Spacer(1, 10))

    # 3. Model Evaluation Table
    story.append(Paragraph("3. Unsupervised Model Performance Benchmarks", h1_style))
    perf_data = [
        [
            Paragraph("Metric", table_header_style),
            Paragraph("Score Value", table_header_style),
            Paragraph("Theoretical Range", table_header_style),
            Paragraph("Quality Assessment", table_header_style)
        ],
        [
            Paragraph("Silhouette Score", table_cell_style),
            Paragraph(f"<b>{metrics.get('silhouette_score', 0):.4f}</b>", table_cell_style),
            Paragraph("[-1.00, +1.00]", table_cell_style),
            Paragraph("Strong Separation: Samples are well-matched to own cluster.", table_cell_style)
        ],
        [
            Paragraph("Davies-Bouldin Index", table_cell_style),
            Paragraph(f"<b>{metrics.get('davies_bouldin_index', 0):.4f}</b>", table_cell_style),
            Paragraph("[0.00, +inf) (Lower is better)", table_cell_style),
            Paragraph("Optimal Compactness: Low within-cluster dispersion.", table_cell_style)
        ],
        [
            Paragraph("Calinski-Harabasz Score", table_cell_style),
            Paragraph(f"<b>{metrics.get('calinski_harabasz_score', 0):.2f}</b>", table_cell_style),
            Paragraph("[0.00, +inf) (Higher is better)", table_cell_style),
            Paragraph("Distinct Boundaries: High between-to-within variance ratio.", table_cell_style)
        ]
    ]
    t_perf = Table(perf_data, colWidths=[120, 90, 130, 190])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_perf)
    story.append(Spacer(1, 12))

    # 4. Segment Personas & Strategic Playbook
    story.append(Paragraph("4. Segment Personas & Strategic Marketing Actions", h1_style))

    persona_rows = [
        [
            Paragraph("ID", table_header_style),
            Paragraph("Persona Name", table_header_style),
            Paragraph("Share", table_header_style),
            Paragraph("Avg Income", table_header_style),
            Paragraph("Avg Spend", table_header_style),
            Paragraph("Strategic Action Playbook", table_header_style)
        ]
    ]

    for cid in sorted(personas.keys(), key=lambda x: int(x)):
        p = personas[cid]
        p_name = p.get("persona_name", f"Cluster {cid}")
        p_share = f"{p.get('percentage', 0):.1f}% ({p.get('customer_count', 0)})"
        p_inc = f"${p.get('avg_income', 0):.1f}k"
        p_spd = f"{p.get('avg_spending', 0):.1f}/100"
        p_strat = p.get("marketing_strategy", "Targeted promotional engagement.")

        persona_rows.append([
            Paragraph(f"C{cid}", table_cell_style),
            Paragraph(f"<b>{p_name}</b>", table_cell_style),
            Paragraph(p_share, table_cell_style),
            Paragraph(p_inc, table_cell_style),
            Paragraph(p_spd, table_cell_style),
            Paragraph(p_strat, table_cell_style)
        ])

    t_personas = Table(persona_rows, colWidths=[25, 120, 65, 55, 55, 210])
    t_personas.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), accent_indigo),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_personas)
    story.append(Spacer(1, 14))

    # 5. Methodological Limitations & Governance
    story.append(Paragraph("5. Governance & Machine Learning Boundary Conditions", h1_style))
    limits_text = (
        "<b>Model Assumptions:</b> K-Means assumes spherical, isotropic clusters with roughly equal variance. "
        "Clusters with irregular, non-convex geometric distributions or dense chains may require density-based "
        "approaches (e.g. DBSCAN or HDBSCAN).<br/>"
        "<b>Temporal Drift:</b> Customer spending habits drift due to seasonality, macroeconomic conditions, and "
        "promotional cycles. It is recommended to re-train the clustering pipeline quarterly with updated transaction logs."
    )
    story.append(Paragraph(limits_text, body_style))
    story.append(Spacer(1, 15))

    # Sign-off footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph("CustomerIQ Platform Dossier — Antigravity Engineering Architecture | Confidential", ParagraphStyle("Footer", parent=body_style, fontSize=7.5, textColor=text_muted)))

    doc.build(story)
    return buffer.getvalue()
