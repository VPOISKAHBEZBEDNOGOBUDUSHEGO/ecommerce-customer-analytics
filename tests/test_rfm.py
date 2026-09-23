import pandas as pd

from ecommerce_analysis.rfm import build_rfm, identify_segment


def test_cant_lose_and_at_risk_are_both_reachable():
    cant_lose = pd.Series({"R_score": 1, "F_score": 5, "M_score": 5})
    at_risk = pd.Series({"R_score": 2, "F_score": 5, "M_score": 5})

    assert identify_segment(cant_lose) == "Can't Lose"
    assert identify_segment(at_risk) == "At Risk"


def test_build_rfm_returns_one_row_per_customer():
    rows = []
    for customer in range(1, 11):
        rows.append(
            {
                "CustomerID": customer,
                "InvoiceNo": f"I{customer}",
                "InvoiceDate": pd.Timestamp("2024-01-01") + pd.Timedelta(days=customer),
                "Revenue": customer * 10,
            }
        )
    result = build_rfm(pd.DataFrame(rows), analysis_date=pd.Timestamp("2024-02-01"))

    assert len(result) == 10
    assert {"R_score", "F_score", "M_score", "Segment"}.issubset(result.columns)
