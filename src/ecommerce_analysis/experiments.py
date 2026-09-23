"""Reproducible binary A/B-test simulation and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm


@dataclass(frozen=True)
class ABTestResult:
    contingency: pd.DataFrame
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    ci_low: float
    ci_high: float
    chi2: float
    p_value: float


def _as_binary(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} должен быть непустым одномерным массивом")
    if not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} должен содержать только 0 и 1")
    return array.astype(int)


def analyze_binary_ab(control, treatment, alpha: float = 0.05) -> ABTestResult:
    """Compare two binary samples; the contingency table is built from observations."""
    if not 0 < alpha < 1:
        raise ValueError("alpha должен находиться между 0 и 1")
    control = _as_binary(control, "control")
    treatment = _as_binary(treatment, "treatment")

    table = pd.DataFrame(
        [
            [int(control.sum()), int(control.size - control.sum())],
            [int(treatment.sum()), int(treatment.size - treatment.sum())],
        ],
        index=["control", "treatment"],
        columns=["returned", "not_returned"],
    )
    chi2, p_value, _, _ = chi2_contingency(table.to_numpy())
    control_rate = float(control.mean())
    treatment_rate = float(treatment.mean())
    absolute_lift = treatment_rate - control_rate
    relative_lift = absolute_lift / control_rate if control_rate else float("inf")
    standard_error = sqrt(
        control_rate * (1 - control_rate) / control.size
        + treatment_rate * (1 - treatment_rate) / treatment.size
    )
    critical = norm.ppf(1 - alpha / 2)

    return ABTestResult(
        contingency=table,
        control_rate=control_rate,
        treatment_rate=treatment_rate,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        ci_low=absolute_lift - critical * standard_error,
        ci_high=absolute_lift + critical * standard_error,
        chi2=float(chi2),
        p_value=float(p_value),
    )


def simulate_retention_experiment(
    n_per_group: int = 84,
    control_rate: float = 0.10,
    treatment_rate: float = 0.15,
    seed: int = 17,
) -> ABTestResult:
    """Generate a documented synthetic retention experiment and analyze it."""
    if n_per_group < 2:
        raise ValueError("В каждой группе должно быть не меньше двух наблюдений")
    if not 0 <= control_rate <= 1 or not 0 <= treatment_rate <= 1:
        raise ValueError("Вероятности должны находиться в диапазоне от 0 до 1")

    random = np.random.RandomState(seed)
    control = random.binomial(1, control_rate, size=n_per_group)
    treatment = random.binomial(1, treatment_rate, size=n_per_group)
    return analyze_binary_ab(control, treatment)


def required_sample_size_per_group(
    baseline_rate: float,
    target_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate equal group size for a two-sided comparison of proportions."""
    if not 0 < baseline_rate < 1 or not 0 < target_rate < 1:
        raise ValueError("Доли должны находиться строго между 0 и 1")
    if baseline_rate == target_rate:
        raise ValueError("Ожидаемый эффект должен быть ненулевым")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha и power должны находиться между 0 и 1")

    pooled = (baseline_rate + target_rate) / 2
    numerator = (
        norm.ppf(1 - alpha / 2) * sqrt(2 * pooled * (1 - pooled))
        + norm.ppf(power)
        * sqrt(baseline_rate * (1 - baseline_rate) + target_rate * (1 - target_rate))
    ) ** 2
    return ceil(numerator / (target_rate - baseline_rate) ** 2)
