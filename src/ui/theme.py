"""
CustomerIQ Theme System: Dark/Light SaaS Styling and Plotly Templates
Engineered for ultra-responsive multi-resolution scaling (1366x768 to 4K displays).
"""

from typing import Dict, Any


def get_theme_css(theme: str = "dark") -> str:
    """Return responsive, resolution-adaptive CSS for modern SaaS layout."""
    is_dark = theme.lower() == "dark"

    bg_color = "#0B0F19" if is_dark else "#F8FAFC"
    card_bg = "#111827" if is_dark else "#FFFFFF"
    card_border = "#1F2937" if is_dark else "#E5E7EB"
    text_primary = "#F9FAFB" if is_dark else "#111827"
    text_secondary = "#9CA3AF" if is_dark else "#6B7280"
    accent = "#6366F1"  # Indigo
    accent_hover = "#4F46E5"
    sidebar_bg = "#030712" if is_dark else "#F3F4F6"
    badge_bg = "rgba(99, 102, 241, 0.15)" if is_dark else "rgba(99, 102, 241, 0.1)"

    css = f"""
    <style>
        /* Global Reset & Typography */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: {text_primary};
        }}

        .stApp {{
            background-color: {bg_color};
        }}

        /* Fluid Responsive Canvas Container - Fits any screen resolution */
        .block-container {{
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
            padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
            max-width: 100% !important;
            width: 100% !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            border-right: 1px solid {card_border};
            width: clamp(240px, 18vw, 300px) !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown h1, 
        section[data-testid="stSidebar"] .stMarkdown h2, 
        section[data-testid="stSidebar"] .stMarkdown h3 {{
            color: {text_primary} !important;
            font-weight: 700;
        }}

        /* Card Architecture */
        .iq-card {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: clamp(14px, 1.5vw, 22px);
            box-shadow: 0 1px 3px rgba(0, 0, 0, {'0.3' if is_dark else '0.05'});
            margin-bottom: 16px;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .iq-card:hover {{
            border-color: {accent};
            box-shadow: 0 4px 12px rgba(99, 102, 241, {'0.25' if is_dark else '0.1'});
        }}

        /* KPI Metric Container - Adaptive font sizing */
        .kpi-container {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 12px;
            padding: clamp(12px, 1.2vw, 18px);
            box-shadow: 0 1px 3px rgba(0,0,0,{'0.2' if is_dark else '0.04'});
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        .kpi-title {{
            font-size: clamp(0.68rem, 0.75vw, 0.82rem);
            font-weight: 600;
            color: {text_secondary};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .kpi-value {{
            font-size: clamp(1.3rem, 1.8vw, 1.9rem);
            font-weight: 700;
            color: {text_primary};
            line-height: 1.2;
            word-break: break-word;
        }}
        .kpi-subtitle {{
            font-size: clamp(0.68rem, 0.72vw, 0.78rem);
            color: {text_secondary};
            margin-top: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Section Headings */
        .iq-header {{
            margin-bottom: clamp(16px, 1.8vw, 24px);
        }}
        .iq-title {{
            font-size: clamp(1.35rem, 1.8vw, 1.85rem);
            font-weight: 700;
            color: {text_primary};
            letter-spacing: -0.02em;
            margin-bottom: 4px;
            line-height: 1.25;
        }}
        .iq-subtitle {{
            font-size: clamp(0.82rem, 0.95vw, 0.98rem);
            color: {text_secondary};
            line-height: 1.4;
        }}
        .iq-badge {{
            display: inline-block;
            background-color: {badge_bg};
            color: {accent};
            font-size: clamp(0.65rem, 0.75vw, 0.75rem);
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 20px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-bottom: 6px;
        }}

        /* Button Enhancements */
        .stButton>button {{
            background: linear-gradient(135deg, {accent} 0%, {accent_hover} 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: clamp(0.82rem, 0.88vw, 0.92rem) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 18px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2) !important;
            white-space: nowrap !important;
        }}
        .stButton>button:hover {{
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
            transform: translateY(-1px) !important;
        }}

        /* Secondary/Download Buttons */
        .stDownloadButton>button {{
            background-color: transparent !important;
            border: 1px solid {accent} !important;
            color: {text_primary} !important;
            font-weight: 600 !important;
            font-size: clamp(0.8rem, 0.85vw, 0.9rem) !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            white-space: nowrap !important;
        }}
        .stDownloadButton>button:hover {{
            background-color: {badge_bg} !important;
        }}

        /* Plotly and Chart Container Resiliency */
        .js-plotly-plot, .plot-container {{
            width: 100% !important;
            max-width: 100% !important;
        }}

        /* Table & DataFrame Overrides */
        div[data-testid="stDataFrame"] {{
            border: 1px solid {card_border};
            border-radius: 8px;
            overflow: hidden;
            width: 100% !important;
        }}

        /* Clean Radio and Navigation Spacing */
        div[data-testid="stRadio"] > div {{
            gap: 8px;
        }}

        /* Info Callout Banners */
        .info-callout {{
            background-color: {'rgba(99, 102, 241, 0.08)' if is_dark else 'rgba(99, 102, 241, 0.05)'};
            border-left: 4px solid {accent};
            border-radius: 6px;
            padding: 10px 14px;
            color: {text_primary};
            font-size: 0.88rem;
            margin-bottom: 14px;
        }}

        /* Media Queries for Smaller Resolutions (Laptops / 1366x768) */
        @media (max-width: 1200px) {{
            .kpi-container {{
                padding: 10px 12px;
            }}
            .kpi-value {{
                font-size: 1.35rem;
            }}
            .iq-title {{
                font-size: 1.45rem;
            }}
        }}
    </style>
    """
    return css


def get_plotly_theme(theme: str = "dark") -> Dict[str, Any]:
    """Get Plotly layout template dictionary for dark/light SaaS style."""
    is_dark = theme.lower() == "dark"

    bg_color = "rgba(0,0,0,0)"
    grid_color = "#1F2937" if is_dark else "#F1F5F9"
    text_color = "#F3F4F6" if is_dark else "#1F2937"
    subtext_color = "#9CA3AF" if is_dark else "#64748B"

    # Distinctive SaaS color palette
    palette = [
        "#6366F1",  # Indigo
        "#EC4899",  # Pink
        "#10B981",  # Emerald
        "#F59E0B",  # Amber
        "#3B82F6",  # Blue
        "#8B5CF6",  # Purple
        "#06B6D4",  # Cyan
        "#EF4444"   # Red
    ]

    layout = {
        "autosize": True,
        "paper_bgcolor": bg_color,
        "plot_bgcolor": bg_color,
        "font": {
            "family": "'Plus Jakarta Sans', -apple-system, sans-serif",
            "color": text_color,
            "size": 11
        },
        "colorway": palette,
        "xaxis": {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"color": subtext_color, "size": 10},
            "title": {"font": {"color": text_color, "size": 11}}
        },
        "yaxis": {
            "gridcolor": grid_color,
            "linecolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"color": subtext_color, "size": 10},
            "title": {"font": {"color": text_color, "size": 11}}
        },
        "legend": {
            "font": {"color": text_color, "size": 10},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)"
        },
        "hoverlabel": {
            "bgcolor": "#1F2937" if is_dark else "#FFFFFF",
            "bordercolor": "#374151" if is_dark else "#E5E7EB",
            "font": {"color": "#F9FAFB" if is_dark else "#111827", "size": 11}
        },
        "margin": {"l": 40, "r": 20, "t": 45, "b": 35}
    }
    return layout
