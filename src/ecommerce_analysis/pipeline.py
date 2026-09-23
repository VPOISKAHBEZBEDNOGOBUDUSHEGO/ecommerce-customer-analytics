"""Command-line pipeline that rebuilds the main analytical artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .assortment import build_abc_xyz
from .churn import train_churn_risk_classifier
from .cleaning import clean_transactions
from .clustering import cluster_customers
from .experiments import required_sample_size_per_group, simulate_retention_experiment
from .ltv import calculate_segment_ltv
from .rfm import build_rfm, segment_summary


def run_pipeline(input_path: Path, output_dir: Path) -> dict:
    """Run deterministic analyses and save machine-readable outputs."""
    if not input_path.exists():
        raise FileNotFoundError(f"Файл данных не найден: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(input_path)
    clean, cleaning_report = clean_transactions(raw)
    rfm = build_rfm(clean)
    segments = segment_summary(rfm)
    assortment = build_abc_xyz(clean)
    clustered, cluster_profile = cluster_customers(rfm)
    ltv = calculate_segment_ltv(rfm)
    experiment = simulate_retention_experiment()
    churn_risk = train_churn_risk_classifier(rfm)

    clean.to_pickle(output_dir / "transactions_clean.pkl")
    rfm.to_csv(output_dir / "rfm_customers.csv")
    segments.to_csv(output_dir / "segment_summary.csv")
    assortment.to_csv(output_dir / "abc_xyz.csv")
    clustered.to_csv(output_dir / "customer_clusters.csv")
    cluster_profile.to_csv(output_dir / "cluster_profile.csv")
    ltv.to_csv(output_dir / "segment_ltv.csv")

    summary = {
        "cleaning": cleaning_report.to_dict(),
        "customers": len(rfm),
        "products": len(assortment),
        "experiment": {
            "control_rate": experiment.control_rate,
            "treatment_rate": experiment.treatment_rate,
            "absolute_lift": experiment.absolute_lift,
            "relative_lift": experiment.relative_lift,
            "ci_95": [experiment.ci_low, experiment.ci_high],
            "p_value": experiment.p_value,
            "required_sample_size_per_group": required_sample_size_per_group(
                0.10, 0.15
            ),
            "synthetic": True,
        },
        "churn_risk": {
            "definition": f"Recency > {churn_risk.threshold_days} days",
            "test_size": churn_risk.test_size,
            "precision": churn_risk.report["1"]["precision"],
            "recall": churn_risk.report["1"]["recall"],
            "f1_score": churn_risk.report["1"]["f1-score"],
            "cross_sectional_not_forecast": True,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the Online Retail analysis artifacts"
    )
    parser.add_argument("--input", type=Path, default=Path("Online Retail.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    summary = run_pipeline(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
