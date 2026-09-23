import pandas as pd

from ecommerce_analysis.cleaning import clean_transactions


def test_clean_transactions_filters_invalid_rows_and_reports_counts():
    valid = {
        "InvoiceNo": "100",
        "StockCode": "A",
        "Description": "Item",
        "Quantity": 2,
        "InvoiceDate": "2024-01-01",
        "UnitPrice": 10.0,
        "CustomerID": 1,
    }
    rows = [
        valid,
        valid,
        {**valid, "InvoiceNo": "C101"},
        {**valid, "InvoiceNo": "102", "Quantity": 0},
        {**valid, "InvoiceNo": "103", "UnitPrice": -1},
        {**valid, "InvoiceNo": "104", "CustomerID": None},
    ]

    clean, report = clean_transactions(pd.DataFrame(rows))

    assert len(clean) == 1
    assert clean.iloc[0]["Revenue"] == 20
    assert report.duplicates_removed == 1
    assert report.cancellations_removed == 1
    assert report.nonpositive_quantity_removed == 1
    assert report.nonpositive_price_removed == 1
    assert report.missing_customer_removed == 1
