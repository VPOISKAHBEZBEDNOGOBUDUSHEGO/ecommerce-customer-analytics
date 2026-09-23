"""Current-snapshot churn-risk classification for the case study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ChurnRiskResult:
    model: Pipeline
    report: dict
    test_size: int
    threshold_days: int


def train_churn_risk_classifier(
    rfm: pd.DataFrame,
    threshold_days: int = 180,
    test_size: float = 0.30,
    random_state: int = 51,
) -> ChurnRiskResult:
    """Classify a recency-based risk flag from Frequency and Monetary.

    This is a cross-sectional classification exercise, not a forward-in-time
    churn forecast. A temporal validation set would be required for that claim.
    """
    required = {"Recency", "Frequency", "Monetary"}
    missing = required.difference(rfm.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {', '.join(sorted(missing))}")
    if threshold_days <= 0:
        raise ValueError("Порог неактивности должен быть положительным")

    target = (rfm["Recency"] > threshold_days).astype(int)
    if target.nunique() < 2 or target.value_counts().min() < 2:
        raise ValueError("Для обучения нужны оба класса минимум с двумя наблюдениями")

    features = rfm[["Frequency", "Monetary"]]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight="balanced", random_state=random_state),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    return ChurnRiskResult(
        model=model,
        report=report,
        test_size=len(x_test),
        threshold_days=threshold_days,
    )
