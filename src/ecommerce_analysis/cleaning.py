"""Transaction cleaning with an auditable row-count report."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
}


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    duplicates_removed: int
    missing_customer_removed: int
    missing_description_removed: int
    invalid_date_removed: int
    cancellations_removed: int
    nonpositive_quantity_removed: int
    nonpositive_price_removed: int
    output_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _drop_by_mask(frame: pd.DataFrame, mask: pd.Series) -> tuple[pd.DataFrame, int]:
    removed = int(mask.sum())
    return frame.loc[~mask].copy(), removed


def clean_transactions(data: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean raw Online Retail rows and return both data and removal counts."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(
            f"Не хватает обязательных колонок: {', '.join(sorted(missing))}"
        )

    frame = data.copy()
    input_rows = len(frame)
    duplicates_removed = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    frame, missing_customer_removed = _drop_by_mask(frame, frame["CustomerID"].isna())
    frame, missing_description_removed = _drop_by_mask(
        frame, frame["Description"].isna()
    )

    parsed_dates = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame, invalid_date_removed = _drop_by_mask(frame, parsed_dates.isna())
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"])

    frame, cancellations_removed = _drop_by_mask(
        frame, frame["InvoiceNo"].astype(str).str.startswith("C")
    )
    frame, nonpositive_quantity_removed = _drop_by_mask(frame, frame["Quantity"] <= 0)
    frame, nonpositive_price_removed = _drop_by_mask(frame, frame["UnitPrice"] <= 0)

    frame["Revenue"] = frame["Quantity"] * frame["UnitPrice"]
    frame = frame.reset_index(drop=True)

    report = CleaningReport(
        input_rows=input_rows,
        duplicates_removed=duplicates_removed,
        missing_customer_removed=missing_customer_removed,
        missing_description_removed=missing_description_removed,
        invalid_date_removed=invalid_date_removed,
        cancellations_removed=cancellations_removed,
        nonpositive_quantity_removed=nonpositive_quantity_removed,
        nonpositive_price_removed=nonpositive_price_removed,
        output_rows=len(frame),
    )
    return frame, report
