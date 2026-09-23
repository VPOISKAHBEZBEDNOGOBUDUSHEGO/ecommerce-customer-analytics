"""Customer clustering based on standardized RFM metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_customers(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign K-means clusters and return data plus an interpretable profile."""
    features = ["Recency", "Frequency", "Monetary"]
    missing = set(features).difference(rfm.columns)
    if missing:
        raise ValueError(
            f"Не хватает колонок для кластеризации: {', '.join(sorted(missing))}"
        )
    if not 2 <= n_clusters < len(rfm):
        raise ValueError("Число кластеров должно быть от 2 до числа клиентов минус 1")

    scaled = StandardScaler().fit_transform(rfm[features])
    labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(scaled)
    clustered = rfm.copy()
    clustered["Cluster"] = labels
    profile = clustered.groupby("Cluster").agg(
        customers=("Recency", "size"),
        mean_recency=("Recency", "mean"),
        mean_frequency=("Frequency", "mean"),
        mean_monetary=("Monetary", "mean"),
    )
    return clustered, profile
