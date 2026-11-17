"""
Segmentation Model Module for K-Means Clustering
Includes KMeans training, Elbow Method, Silhouette Optimization, and inference.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib


class CustomerKMeansModel:
    """K-Means Clustering model wrapper for customer segmentation."""

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> "CustomerKMeansModel":
        """Fit KMeans model on feature matrix X (typically scaled)."""
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index for samples in X."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster labels."""
        self.fit(X)
        return self.model.labels_

    @property
    def cluster_centers_(self) -> np.ndarray:
        """Return coordinates of cluster centers in feature space."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.model.cluster_centers_

    @property
    def inertia_(self) -> float:
        """Sum of squared distances of samples to their closest cluster center (WCSS)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        return float(self.model.inertia_)

    @staticmethod
    def compute_elbow_and_silhouette(
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Compute WCSS (Inertia) and Silhouette Scores for a range of k values.
        Also computes optimal k via Knee locator and max silhouette score.
        """
        k_values = list(range(k_min, k_max + 1))
        inertias = []
        silhouettes = []

        # Also get k=1 inertia for elbow plot if k_min > 1
        full_k = [1] + k_values if k_min > 1 else k_values
        full_inertias = []
        for k in full_k:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
            km.fit(X)
            full_inertias.append(float(km.inertia_))
            if k >= 2:
                inertias.append(float(km.inertia_))
                score = silhouette_score(X, km.labels_)
                silhouettes.append(float(score))

        # 1. Optimal K by Max Silhouette Score
        best_k_silhouette = k_values[int(np.argmax(silhouettes))]

        # 2. Optimal K by Elbow method (maximum distance to chord connecting first and last point)
        k_arr = np.array(full_k)
        i_arr = np.array(full_inertias)
        # Vector from first point to last point
        p1 = np.array([k_arr[0], i_arr[0]])
        p2 = np.array([k_arr[-1], i_arr[-1]])
        line_vec = p2 - p1
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        distances = []
        for p in zip(k_arr, i_arr):
            point = np.array(p)
            vec_to_point = point - p1
            # Perpendicular distance to line
            proj = np.dot(vec_to_point, line_vec_norm) * line_vec_norm
            perp = vec_to_point - proj
            distances.append(np.linalg.norm(perp))

        best_k_elbow = int(k_arr[np.argmax(distances)])

        return {
            "k_range": k_values,
            "full_k": full_k,
            "inertias": full_inertias,
            "silhouette_k_range": k_values,
            "silhouette_scores": silhouettes,
            "best_k_silhouette": best_k_silhouette,
            "best_k_elbow": best_k_elbow
        }

    def save(self, filepath: str) -> None:
        """Save model to disk using joblib."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CustomerKMeansModel":
        """Load model from disk."""
        return joblib.load(filepath)
