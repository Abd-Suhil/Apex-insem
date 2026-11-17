"""
Inference Module for Customer Segmentation
Assigns individual customer records or batches of customers to clusters with persona recommendations.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import json
import argparse
from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np

from src.data_loader import load_customer_data
from src.preprocessor import CustomerPreprocessor
from src.segmentation_model import CustomerKMeansModel


class CustomerSegmentPredictor:
    """Inference engine to classify new customers into existing clusters."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "kmeans_model.joblib")
        self.prep_path = os.path.join(models_dir, "preprocessor.joblib")
        self.meta_path = os.path.join(models_dir, "cluster_metadata.json")

        if not os.path.exists(self.model_path) or not os.path.exists(self.prep_path):
            raise FileNotFoundError(
                f"Model or Preprocessor not found in '{models_dir}'. Please run train.py first."
            )

        self.model = CustomerKMeansModel.load(self.model_path)
        self.preprocessor = CustomerPreprocessor.load(self.prep_path)

        self.metadata = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.personas = self.metadata.get("personas", {})

    def predict_single(
        self,
        annual_income: float,
        spending_score: float,
        age: Optional[float] = None,
        gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assign a single customer to a cluster and retrieve persona recommendations.
        """
        sample_dict = {
            "annual_income": [float(annual_income)],
            "spending_score": [float(spending_score)]
        }
        if age is not None:
            sample_dict["age"] = [float(age)]
        if gender is not None:
            sample_dict["gender"] = [str(gender)]

        df_single = pd.DataFrame(sample_dict)
        return self.predict_batch(df_single).iloc[0].to_dict()

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict clusters and attach personas for a DataFrame of customers.
        """
        df_clean = load_customer_data(df)
        X_scaled, _ = self.preprocessor.transform(df_clean)
        cluster_preds = self.model.predict(X_scaled)

        df_result = df_clean.copy()
        df_result["cluster"] = cluster_preds

        # Map persona metadata
        persona_names = []
        descriptions = []
        strategies = []

        for c in cluster_preds:
            p_data = self.personas.get(str(c)) or self.personas.get(c, {})
            persona_names.append(p_data.get("persona_name", f"Cluster {c}"))
            descriptions.append(p_data.get("description", "Standard customer group"))
            strategies.append(p_data.get("marketing_strategy", "Standard promotions"))

        df_result["persona_name"] = persona_names
        df_result["description"] = descriptions
        df_result["marketing_strategy"] = strategies

        return df_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Customer Segment")
    parser.add_argument("--income", type=float, help="Annual income (k$)")
    parser.add_argument("--spending", type=float, help="Spending score (1-100)")
    parser.add_argument("--age", type=float, default=30.0, help="Customer age")
    parser.add_argument("--gender", type=str, default="Female", help="Customer gender")
    parser.add_argument("--batch-input", type=str, help="Path to input CSV for batch prediction")
    parser.add_argument("--batch-output", type=str, default="data/predictions.csv", help="Path to output CSV")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with saved models")

    args = parser.parse_args()
    predictor = CustomerSegmentPredictor(models_dir=args.models_dir)

    if args.batch_input:
        df_in = pd.read_csv(args.batch_input)
        df_out = predictor.predict_batch(df_in)
        df_out.to_csv(args.batch_output, index=False)
        print(f"Batch predictions saved to: {args.batch_output}")
        print(df_out[["customer_id", "cluster", "persona_name"]].head(10))
    elif args.income is not None and args.spending is not None:
        res = predictor.predict_single(
            annual_income=args.income,
            spending_score=args.spending,
            age=args.age,
            gender=args.gender
        )
        print("\n================ CUSTOMER SEGMENTATION RESULT ================")
        print(f"Assigned Cluster:      Cluster {res['cluster']}")
        print(f"Customer Persona:      {res['persona_name']}")
        print(f"Profile Description:   {res['description']}")
        print(f"Target Strategy:       {res['marketing_strategy']}")
        print("==============================================================")
    else:
        print("Please provide --income and --spending, or --batch-input.")
