"""
Data Loader Module for Customer Segmentation
Handles dataset loading, column normalization, schema validation, and summary stats.
"""

from typing import Union, Tuple, Dict, Any
import io
import pandas as pd


STANDARD_COLUMNS = {
    "customer_id": ["customerid", "customer_id", "id", "cust_id"],
    "gender": ["gender", "sex"],
    "age": ["age", "customer_age"],
    "annual_income": ["annual income (k$)", "annual_income", "annual_income_k", "income", "annualincome"],
    "spending_score": ["spending score (1-100)", "spending_score", "spending_score_1_100", "score", "spendingscore"]
}


def normalize_column_name(col_name: str) -> str:
    """Normalize column string for flexible matching."""
    cleaned = col_name.strip().lower().replace("_", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("$", "")
    for target, variations in STANDARD_COLUMNS.items():
        for var in variations:
            clean_var = var.replace("_", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace("$", "")
            if cleaned == clean_var:
                return target
    return col_name.strip()


def load_customer_data(source: Union[str, io.BytesIO, io.StringIO, pd.DataFrame]) -> pd.DataFrame:
    """
    Load customer dataset from filepath, buffer, or existing DataFrame.
    Normalizes column names to standard identifiers:
    'customer_id', 'gender', 'age', 'annual_income', 'spending_score'.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, (str, io.BytesIO, io.StringIO)):
        df = pd.read_csv(source)
    else:
        raise ValueError("Unsupported data source type. Must be filepath, buffer, or DataFrame.")

    # Normalize columns
    rename_map = {}
    for col in df.columns:
        norm = normalize_column_name(col)
        rename_map[col] = norm
    df = df.rename(columns=rename_map)

    # Basic validations: Annual Income and Spending Score are fundamental
    required_features = ["annual_income", "spending_score"]
    missing_req = [f for f in required_features if f not in df.columns]
    if missing_req:
        raise ValueError(
            f"Dataset is missing required features: {missing_req}. "
            f"Found columns: {list(df.columns)}"
        )

    # If customer_id not present, generate consecutive IDs
    if "customer_id" not in df.columns:
        df["customer_id"] = range(1, len(df) + 1)

    # Clean types
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["annual_income"] = pd.to_numeric(df["annual_income"], errors="coerce")
    df["spending_score"] = pd.to_numeric(df["spending_score"], errors="coerce")

    # Drop rows where required features are null
    df = df.dropna(subset=required_features).reset_index(drop=True)

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate quick overview statistics of the loaded dataset."""
    summary = {
        "total_records": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "age_mean": float(df["age"].mean()) if "age" in df else None,
        "income_mean": float(df["annual_income"].mean()) if "annual_income" in df else None,
        "spending_mean": float(df["spending_score"].mean()) if "spending_score" in df else None,
        "gender_breakdown": df["gender"].value_counts().to_dict() if "gender" in df else {}
    }
    return summary
