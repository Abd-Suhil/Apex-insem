"""
Preprocessor Module for Customer Segmentation
Handles feature selection, missing value imputation, scaling, and dimensionality reduction.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import joblib


class CustomerPreprocessor:
    """Preprocessor for transforming customer data for clustering algorithms."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        scaler_type: str = "standard",
        include_pca: bool = True,
        pca_components: int = 2
    ):
        """
        Args:
            features: List of feature column names to use. Default: ['annual_income', 'spending_score']
            scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
            include_pca: Whether to fit a PCA transformer for dimensionality reduction
            pca_components: Number of PCA components for visualization (typically 2)
        """
        self.features = features or ["annual_income", "spending_score"]
        self.scaler_type = scaler_type.lower()
        self.include_pca = include_pca
        self.pca_components = pca_components

        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.pca = PCA(n_components=pca_components) if include_pca else None
        self.is_fitted = False
        self.feature_names_in_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "CustomerPreprocessor":
        """Fit scaler and optional PCA on the specified features of the dataframe."""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"Features {missing} not found in dataframe. Available: {list(df.columns)}")

        X = df[self.features].values
        self.scaler.fit(X)
        self.feature_names_in_ = list(self.features)

        if self.include_pca and self.pca is not None:
            # PCA requires n_components <= n_features
            n_comp = min(self.pca_components, len(self.features))
            self.pca.n_components = n_comp
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform dataframe features.
        Returns:
            Tuple of (scaled_features_array, pca_features_array_or_None)
        """
        if not self.is_fitted:
            raise RuntimeError("CustomerPreprocessor is not fitted. Call fit() first.")

        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise KeyError(f"Features {missing} not found in dataframe.")

        X = df[self.features].values
        X_scaled = self.scaler.transform(X)

        X_pca = None
        if self.include_pca and self.pca is not None:
            X_pca = self.pca.transform(X_scaled)

        return X_scaled, X_pca

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit scaler/PCA and transform in one step."""
        return self.fit(df).transform(df)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Invert scaled features back to original domain."""
        if not self.is_fitted:
            raise RuntimeError("CustomerPreprocessor is not fitted.")
        return self.scaler.inverse_transform(X_scaled)

    def save(self, filepath: str) -> None:
        """Save fitted preprocessor to disk using joblib."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(filepath)
