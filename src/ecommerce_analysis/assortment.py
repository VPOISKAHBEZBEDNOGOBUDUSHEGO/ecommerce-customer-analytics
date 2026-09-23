"""ABC/XYZ assortment analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_CODES = {"POST", "DOT", "M", "BANK CHARGES"}


def build_abc_xyz(transactions: pd.DataFrame) -> pd.DataFrame:
    """Classify products by revenue contribution and monthly demand stability."""
    required = {"StockCode", "InvoiceDate", "Quantity", "Revenue"}
    missing = required.difference(transactions.columns)
    if missing:
        raise ValueError(
            f"Не хватает колонок для ABC/XYZ: {', '.join(sorted(missing))}"
        )

    data = transactions.loc[~transactions["StockCode"].isin(SERVICE_CODES)].copy()
    if data.empty:
        raise ValueError("После исключения служебных кодов не осталось товаров")

    abc = (
        data.groupby("StockCode")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .to_frame()
    )
    abc = abc.rename(columns={"Revenue": "revenue"})
    abc["revenue_share"] = abc["revenue"] / abc["revenue"].sum()
    abc["cumulative_revenue_share"] = abc["revenue_share"].cumsum()
    abc["ABC"] = np.select(
        [
            abc["cumulative_revenue_share"] <= 0.80,
            abc["cumulative_revenue_share"] <= 0.95,
        ],
        ["A", "B"],
        default="C",
    )

    data["month"] = pd.to_datetime(data["InvoiceDate"]).dt.to_period("M")
    monthly = data.pivot_table(
        index="StockCode",
        columns="month",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )
    xyz = pd.DataFrame(
        {
            "mean_monthly_sales": monthly.mean(axis=1),
            "std_monthly_sales": monthly.std(axis=1, ddof=0),
        }
    )
    xyz["coefficient_of_variation"] = xyz["std_monthly_sales"] / xyz[
        "mean_monthly_sales"
    ].replace(0, np.nan)
    xyz["XYZ"] = np.select(
        [xyz["coefficient_of_variation"] < 0.5, xyz["coefficient_of_variation"] < 1.0],
        ["X", "Y"],
        default="Z",
    )

    result = abc.join(xyz, how="left")
    result["Grade"] = result["ABC"] + result["XYZ"]
    return result
