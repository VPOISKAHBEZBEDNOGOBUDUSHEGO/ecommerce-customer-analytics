"""RFM scoring, segmentation and segment-level summaries."""

from __future__ import annotations

import pandas as pd

RFM_COLUMNS = {"CustomerID", "InvoiceDate", "InvoiceNo", "Revenue"}


def identify_segment(row: pd.Series) -> str:
    """Map RFM scores to mutually exclusive business segments."""
    recency = int(row["R_score"])
    frequency = int(row["F_score"])
    monetary = int(row["M_score"])

    if recency == 5 and frequency == 5 and monetary == 5:
        return "Champion"
    if recency == 1 and frequency >= 4 and monetary >= 4:
        return "Can't Lose"
    if recency == 2 and frequency >= 4 and monetary >= 4:
        return "At Risk"
    if recency <= 2 and frequency <= 2 and monetary <= 2:
        return "Lost"
    if recency >= 4 and frequency <= 2:
        return "New"
    if recency >= 4 and frequency in {2, 3} and monetary in {2, 3}:
        return "Potential"
    if frequency >= 4 and monetary >= 2 and recency >= 2:
        return "Loyal"
    return "Other"


def build_rfm(
    transactions: pd.DataFrame,
    analysis_date: pd.Timestamp | None = None,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Aggregate transactions to customer-level RFM metrics and scores."""
    missing = RFM_COLUMNS.difference(transactions.columns)
    if missing:
        raise ValueError(f"Не хватает колонок для RFM: {', '.join(sorted(missing))}")
    if quantiles < 2:
        raise ValueError("Число квантилей должно быть не меньше двух")

    dates = pd.to_datetime(transactions["InvoiceDate"], errors="raise")
    analysis_date = (
        dates.max() + pd.Timedelta(days=1)
        if analysis_date is None
        else pd.Timestamp(analysis_date)
    )
    if analysis_date <= dates.max():
        raise ValueError("Дата анализа должна быть позже последней транзакции")

    work = transactions.copy()
    work["InvoiceDate"] = dates
    rfm = work.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda values: (analysis_date - values.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    if len(rfm) < quantiles:
        raise ValueError("Для квантильной оценки недостаточно клиентов")

    ascending_labels = list(range(1, quantiles + 1))
    descending_labels = list(reversed(ascending_labels))
    rfm["R_score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), quantiles, labels=descending_labels
    ).astype(int)
    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), quantiles, labels=ascending_labels
    ).astype(int)
    rfm["M_score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), quantiles, labels=ascending_labels
    ).astype(int)
    rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)
    rfm["RFM_code"] = (
        rfm["R_score"].astype(str)
        + rfm["F_score"].astype(str)
        + rfm["M_score"].astype(str)
    )
    rfm["Segment"] = rfm.apply(identify_segment, axis=1)
    return rfm.sort_index()


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Summarize customer count, revenue and average value by segment."""
    required = {"Segment", "Recency", "Monetary"}
    missing = required.difference(rfm.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {', '.join(sorted(missing))}")

    summary = rfm.groupby("Segment").agg(
        customers=("Recency", "size"),
        revenue=("Monetary", "sum"),
        average_customer_value=("Monetary", "mean"),
    )
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    summary["revenue_share"] = summary["revenue"] / summary["revenue"].sum()
    return summary.sort_values("revenue", ascending=False)
