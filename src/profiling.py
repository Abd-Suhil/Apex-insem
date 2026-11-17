"""
Customer Profiling and Persona Generation Module
Assigns human-understandable persona names, descriptions, and marketing recommendations
to customer clusters based on their statistical centroids.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


# Pre-defined marketing strategy blueprints for classic segmentation patterns
STRATEGY_TEMPLATES = {
    "VIP / High Rollers": {
        "description": "High income earners with high spending scores. Highly lucrative, brand-conscious, and value exclusivity.",
        "marketing_strategy": "Direct high-tier loyalty programs, early access to luxury product launches, VIP events, personalized concierge, and premium bundled services.",
        "retention_priority": "Critical - Maximum lifetime value (LTV)."
    },
    "Careful Savers / High Potential": {
        "description": "High income earners with conservative spending habits. Prudent, analytical, and seek tangible quality or ROI.",
        "marketing_strategy": "Emphasize product quality, durability, premium warranties, and financial rewards/cashback. Targeted premium marketing highlighting long-term value.",
        "retention_priority": "High - Huge untapped wallet share."
    },
    "Youth Trendsetters / Enthusiasts": {
        "description": "Modest income earners with disproportionately high spending scores. Trend-driven, impulsive, and socially motivated.",
        "marketing_strategy": "Engage via social media campaigns, influencer partnerships, flash discounts, limited editions, and flexible payment plans (BNPL).",
        "retention_priority": "Medium - High brand advocacy but price-sensitive to macro changes."
    },
    "Budget Conscious / Thrifty": {
        "description": "Modest income earners who spend conservatively. Highly price-sensitive, practical, and focused on essential utilities.",
        "marketing_strategy": "Value-oriented promotions, clearance events, bulk buy discounts, reward points on essentials, and transparent pricing without hidden fees.",
        "retention_priority": "Low-to-Medium - Focus on automated cost-effective retention."
    },
    "Mainstream / Everyday Balancers": {
        "description": "Average income earners with moderate spending behavior. Stable, predictable, and representative of the mass middle market.",
        "marketing_strategy": "Seasonal discounts, family packages, standard loyalty tier progression, cross-category recommendations, and customer satisfaction surveys.",
        "retention_priority": "Medium-High - Volume anchor of overall business."
    }
}


def generate_cluster_profiles(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Compute statistical profile (mean, min, max, count) for each customer cluster.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in [cluster_col, "customer_id"]]
    
    grouped = df.groupby(cluster_col)
    
    # Calculate means
    profile = grouped[numeric_cols].mean().round(2)
    profile.columns = [f"mean_{c}" for c in profile.columns]
    
    # Add count and percentage
    counts = grouped.size()
    total = len(df)
    profile["customer_count"] = counts
    profile["percentage"] = (counts / total * 100).round(2)
    
    return profile.reset_index()


def assign_persona_metadata(
    profile_df: pd.DataFrame,
    overall_df: pd.DataFrame
) -> Dict[int, Dict[str, Any]]:
    """
    Dynamically assign descriptive business personas and actionable strategies to clusters.
    Uses relative comparisons against global averages for Income and Spending Score.
    """
    global_income_mean = overall_df["annual_income"].mean() if "annual_income" in overall_df else 50
    global_spend_mean = overall_df["spending_score"].mean() if "spending_score" in overall_df else 50
    global_age_mean = overall_df["age"].mean() if "age" in overall_df else 38

    personas = {}

    for _, row in profile_df.iterrows():
        cluster_id = int(row["cluster"])
        income = row.get("mean_annual_income", global_income_mean)
        spending = row.get("mean_spending_score", global_spend_mean)
        age = row.get("mean_age", global_age_mean)

        income_diff = income - global_income_mean
        spend_diff = spending - global_spend_mean

        # Classification heuristics based on income and spending relative to benchmark
        if income_diff >= 5 and spend_diff >= 5:
            persona_name = "VIP / High Rollers"
        elif income_diff >= 5 and spend_diff < -5:
            persona_name = "Careful Savers / High Potential"
        elif income_diff < -5 and spend_diff >= 5:
            persona_name = "Youth Trendsetters / Enthusiasts"
        elif income_diff < -5 and spend_diff < -5:
            persona_name = "Budget Conscious / Thrifty"
        else:
            persona_name = "Mainstream / Everyday Balancers"

        template = STRATEGY_TEMPLATES.get(persona_name, STRATEGY_TEMPLATES["Mainstream / Everyday Balancers"])

        personas[cluster_id] = {
            "cluster_id": cluster_id,
            "persona_name": persona_name,
            "customer_count": int(row.get("customer_count", 0)),
            "percentage": float(row.get("percentage", 0.0)),
            "avg_income": float(income),
            "avg_spending": float(spending),
            "avg_age": float(age) if "mean_age" in row else None,
            "description": template["description"],
            "marketing_strategy": template["marketing_strategy"],
            "retention_priority": template["retention_priority"]
        }

    return personas
