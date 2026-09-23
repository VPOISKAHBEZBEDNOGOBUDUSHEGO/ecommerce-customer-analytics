"""Segment-level LTV scenarios with explicit assumptions."""

from __future__ import annotations

import pandas as pd


def calculate_segment_ltv(
    rfm: pd.DataFrame,
    annual_churn_rate: float = 0.20,
    discount_rate: float = 0.15,
    horizon_years: int = 5,
    cac: float = 2000,
) -> pd.DataFrame:
    """Calculate historical, simple and discounted revenue LTV by segment."""
    required = {"Segment", "Monetary"}
    missing = required.difference(rfm.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {', '.join(sorted(missing))}")
    if not 0 < annual_churn_rate <= 1:
        raise ValueError("annual_churn_rate должен находиться в диапазоне (0, 1]")
    if discount_rate <= -1:
        raise ValueError("discount_rate должен быть больше -1")
    if horizon_years < 1 or cac < 0:
        raise ValueError("Горизонт должен быть положительным, CAC - неотрицательным")

    historical = (
        rfm.groupby("Segment")["Monetary"].mean().rename("historical_revenue_ltv")
    )
    simple = (historical / annual_churn_rate).rename("simple_revenue_ltv")
    discount_factor = sum(
        1 / (1 + discount_rate) ** year for year in range(horizon_years)
    )
    discounted = (historical * discount_factor).rename("discounted_revenue_ltv")
    result = pd.concat([historical, simple, discounted], axis=1)
    result["ltv_to_cac"] = (
        result["discounted_revenue_ltv"] / cac if cac else float("inf")
    )
    return result.sort_values("discounted_revenue_ltv", ascending=False)
