"""
Evaluation Module for Customer Clustering
Computes Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and cluster distributions.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)


def evaluate_clusters(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive clustering quality metrics.

    Args:
        X: Feature matrix used for clustering (preferably scaled)
        labels: Cluster assignment labels

    Returns:
        Dictionary of metrics including silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
    """
    n_unique_labels = len(np.unique(labels))
    if n_unique_labels <= 1 or n_unique_labels >= len(X):
        return {
            "error": "Metrics require between 2 and n-1 clusters.",
            "n_clusters": n_unique_labels
        }

    sil_score = float(silhouette_score(X, labels))
    db_score = float(davies_bouldin_score(X, labels))
    ch_score = float(calinski_harabasz_score(X, labels))
    sample_sil = silhouette_samples(X, labels)

    # Per-cluster statistics
    cluster_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    total_samples = len(labels)
    cluster_percentages = {k: round((v / total_samples) * 100, 2) for k, v in cluster_counts.items()}

    per_cluster_sil = {}
    for cl in sorted(list(set(labels))):
        per_cluster_sil[int(cl)] = float(np.mean(sample_sil[labels == cl]))

    return {
        "n_clusters": n_unique_labels,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_index": round(db_score, 4),
        "calinski_harabasz_score": round(ch_score, 2),
        "cluster_sizes": cluster_counts,
        "cluster_percentages": cluster_percentages,
        "per_cluster_silhouette": per_cluster_sil,
        "sample_silhouette_values": sample_sil
    }


def format_evaluation_report(metrics: Dict[str, Any]) -> str:
    """Format evaluation metrics into a human-readable text report."""
    lines = [
        "========================================",
        "      CLUSTERING EVALUATION REPORT      ",
        "========================================",
        f"Total Clusters:           {metrics.get('n_clusters')}",
        f"Silhouette Score:         {metrics.get('silhouette_score')} (range [-1, 1], higher is better)",
        f"Davies-Bouldin Index:     {metrics.get('davies_bouldin_index')} (lower is better)",
        f"Calinski-Harabasz Score:  {metrics.get('calinski_harabasz_score')} (higher is better)",
        "",
        "--- Cluster Size Distribution ---"
    ]
    sizes = metrics.get("cluster_sizes", {})
    pcts = metrics.get("cluster_percentages", {})
    sil_per = metrics.get("per_cluster_silhouette", {})
    for cl, count in sizes.items():
        pct = pcts.get(cl, 0)
        sil = sil_per.get(cl, 0)
        lines.append(f"  Cluster {cl}: {count} customers ({pct}%) | Avg Silhouette: {sil:.4f}")
    lines.append("========================================")
    return "\n".join(lines)
