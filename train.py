"""
End-to-End Training Pipeline for Customer Segmentation using K-Means
Loads data, preprocesses, determines optimal clusters, trains model, evaluates, and saves artifacts.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import json
import argparse
from typing import Optional, List
import pandas as pd
import numpy as np

from src.data_loader import load_customer_data, get_dataset_summary
from src.preprocessor import CustomerPreprocessor
from src.segmentation_model import CustomerKMeansModel
from src.evaluation import evaluate_clusters, format_evaluation_report
from src.profiling import generate_cluster_profiles, assign_persona_metadata


def train_pipeline(
    data_path: str = "data/mall_customers.csv",
    output_dir: str = "models",
    k: Optional[int] = None,
    features: Optional[List[str]] = None,
    scaler_type: str = "standard",
    random_state: int = 42
):
    """
    Execute full training pipeline.
    """
    print(f"\n[1/6] Loading customer data from: {data_path}")
    df = load_customer_data(data_path)
    summary = get_dataset_summary(df)
    print(f"Loaded {len(df)} records. Features detected: {list(df.columns)}")

    # Default features: Annual Income and Spending Score
    selected_features = features or ["annual_income", "spending_score"]
    print(f"[2/6] Selected features for clustering: {selected_features}")

    # Preprocessing
    preprocessor = CustomerPreprocessor(
        features=selected_features,
        scaler_type=scaler_type,
        include_pca=True,
        pca_components=2
    )
    X_scaled, X_pca = preprocessor.fit_transform(df)

    # Determine optimal K if not specified
    print("[3/6] Evaluating optimal cluster count (Elbow & Silhouette analysis)...")
    opt_analysis = CustomerKMeansModel.compute_elbow_and_silhouette(
        X_scaled,
        k_min=2,
        k_max=10,
        random_state=random_state
    )
    best_k_sil = opt_analysis["best_k_silhouette"]
    best_k_elbow = opt_analysis["best_k_elbow"]
    print(f"  > Optimal K via Silhouette Score: {best_k_sil}")
    print(f"  > Optimal K via Elbow Method:     {best_k_elbow}")

    final_k = k if k is not None else best_k_sil
    print(f"[4/6] Training K-Means model with K={final_k} (random_state={random_state})...")
    model = CustomerKMeansModel(n_clusters=final_k, random_state=random_state)
    labels = model.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    df["cluster"] = labels

    # Evaluation
    print("[5/6] Evaluating model quality metrics...")
    metrics = evaluate_clusters(X_scaled, labels)
    print(format_evaluation_report(metrics))

    # Profiling & Personas
    print("[6/6] Generating cluster personas & marketing recommendations...")
    profile_df = generate_cluster_profiles(df, cluster_col="cluster")
    personas = assign_persona_metadata(profile_df, df)

    # Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "kmeans_model.joblib")
    prep_path = os.path.join(output_dir, "preprocessor.joblib")
    meta_path = os.path.join(output_dir, "cluster_metadata.json")
    summary_path = os.path.join(output_dir, "training_summary.json")

    model.save(model_path)
    preprocessor.save(prep_path)

    # Prepare metadata
    cluster_centers_scaled = model.cluster_centers_
    cluster_centers_orig = preprocessor.inverse_transform(cluster_centers_scaled)

    metadata = {
        "n_clusters": final_k,
        "features": selected_features,
        "scaler_type": scaler_type,
        "random_state": random_state,
        "metrics": {
            "silhouette_score": metrics["silhouette_score"],
            "davies_bouldin_index": metrics["davies_bouldin_index"],
            "calinski_harabasz_score": metrics["calinski_harabasz_score"]
        },
        "optimal_k_analysis": {
            "k_range": opt_analysis["k_range"],
            "inertias": opt_analysis["inertias"],
            "silhouette_scores": opt_analysis["silhouette_scores"],
            "best_k_silhouette": best_k_sil,
            "best_k_elbow": best_k_elbow
        },
        "cluster_centers_scaled": cluster_centers_scaled.tolist(),
        "cluster_centers_original": cluster_centers_orig.tolist(),
        "personas": personas
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_summary": summary,
            "metrics": metadata["metrics"],
            "cluster_sizes": metrics["cluster_sizes"]
        }, f, indent=2)

    print(f"\nArtifacts successfully saved to '{output_dir}/':")
    print(f"  - Model:         {model_path}")
    print(f"  - Preprocessor:  {prep_path}")
    print(f"  - Metadata:      {meta_path}")
    print(f"  - Summary:       {summary_path}")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "df": df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Customer Segmentation K-Means Model")
    parser.add_argument("--data", type=str, default="data/mall_customers.csv", help="Path to CSV dataset")
    parser.add_argument("--output", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("--features", nargs="+", default=["annual_income", "spending_score"], help="Features to use")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "minmax"], help="Scaler type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_pipeline(
        data_path=args.data,
        output_dir=args.output,
        k=args.k,
        features=args.features,
        scaler_type=args.scaler,
        random_state=args.seed
    )
