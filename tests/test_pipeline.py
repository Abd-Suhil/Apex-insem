"""
Unit and Integration Tests for Customer Segmentation System using Python unittest
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import unittest
import numpy as np
import pandas as pd

from src.data_loader import load_customer_data, normalize_column_name, get_dataset_summary
from src.preprocessor import CustomerPreprocessor
from src.segmentation_model import CustomerKMeansModel
from src.evaluation import evaluate_clusters, format_evaluation_report
from src.profiling import generate_cluster_profiles, assign_persona_metadata
from predict import CustomerSegmentPredictor


class TestCustomerSegmentation(unittest.TestCase):
    """Comprehensive test cases for the Customer Segmentation pipeline."""

    def setUp(self):
        """Prepare synthetic customer dataset for testing."""
        self.sample_df = pd.DataFrame({
            "CustomerID": list(range(1, 21)),
            "Gender": ["Male", "Female"] * 10,
            "Age": [19, 21, 20, 23, 31, 22, 35, 23, 64, 30, 45, 52, 28, 33, 40, 22, 58, 29, 36, 48],
            "Annual Income (k$)": [15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 70, 72, 75, 78, 80, 82, 85, 87, 88, 90],
            "Spending Score (1-100)": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 75, 80, 15, 20, 85, 78, 12, 18, 88, 16]
        })

    def test_normalize_column_name(self):
        self.assertEqual(normalize_column_name("Annual Income (k$)"), "annual_income")
        self.assertEqual(normalize_column_name("Spending Score (1-100)"), "spending_score")
        self.assertEqual(normalize_column_name("CustomerID"), "customer_id")
        self.assertEqual(normalize_column_name("Gender"), "gender")
        self.assertEqual(normalize_column_name("Age"), "age")

    def test_load_customer_data(self):
        df = load_customer_data(self.sample_df)
        self.assertEqual(len(df), 20)
        self.assertIn("annual_income", df.columns)
        self.assertIn("spending_score", df.columns)
        self.assertIn("customer_id", df.columns)

        summary = get_dataset_summary(df)
        self.assertEqual(summary["total_records"], 20)
        self.assertIsNotNone(summary["income_mean"])

    def test_preprocessor(self):
        df = load_customer_data(self.sample_df)
        prep = CustomerPreprocessor(features=["annual_income", "spending_score"])
        X_scaled, X_pca = prep.fit_transform(df)

        self.assertEqual(X_scaled.shape, (20, 2))
        self.assertEqual(X_pca.shape, (20, 2))
        self.assertTrue(np.allclose(X_scaled.mean(axis=0), [0.0, 0.0], atol=1e-5))

        X_orig = prep.inverse_transform(X_scaled)
        self.assertTrue(np.allclose(X_orig, df[["annual_income", "spending_score"]].values))

    def test_kmeans_and_evaluation(self):
        df = load_customer_data(self.sample_df)
        prep = CustomerPreprocessor()
        X_scaled, _ = prep.fit_transform(df)

        model = CustomerKMeansModel(n_clusters=3, random_state=42)
        labels = model.fit_predict(X_scaled)
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(np.unique(labels)), 3)

        metrics = evaluate_clusters(X_scaled, labels)
        self.assertIn("silhouette_score", metrics)
        self.assertIn("davies_bouldin_index", metrics)
        self.assertIn("calinski_harabasz_score", metrics)
        self.assertEqual(metrics["n_clusters"], 3)

        report = format_evaluation_report(metrics)
        self.assertIn("CLUSTERING EVALUATION REPORT", report)

    def test_elbow_and_silhouette(self):
        df = load_customer_data(self.sample_df)
        prep = CustomerPreprocessor()
        X_scaled, _ = prep.fit_transform(df)

        analysis = CustomerKMeansModel.compute_elbow_and_silhouette(X_scaled, k_min=2, k_max=4)
        self.assertIn("inertias", analysis)
        self.assertIn("silhouette_scores", analysis)
        self.assertEqual(len(analysis["silhouette_scores"]), 3)

    def test_profiling_and_personas(self):
        df = load_customer_data(self.sample_df)
        prep = CustomerPreprocessor()
        X_scaled, _ = prep.fit_transform(df)
        model = CustomerKMeansModel(n_clusters=3, random_state=42)
        df["cluster"] = model.fit_predict(X_scaled)

        profile = generate_cluster_profiles(df)
        self.assertIn("mean_annual_income", profile.columns)
        self.assertIn("customer_count", profile.columns)

        personas = assign_persona_metadata(profile, df)
        self.assertIn(0, personas)
        self.assertIn("persona_name", personas[0])
        self.assertIn("marketing_strategy", personas[0])

    def test_predictor_e2e(self):
        if os.path.exists("models/kmeans_model.joblib"):
            predictor = CustomerSegmentPredictor(models_dir="models")
            res = predictor.predict_single(annual_income=85, spending_score=85)
            self.assertIn("cluster", res)
            self.assertIn("persona_name", res)
            self.assertIn("marketing_strategy", res)


if __name__ == "__main__":
    unittest.main()
