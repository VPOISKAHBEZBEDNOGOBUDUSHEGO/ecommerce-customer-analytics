import numpy as np
import pytest

from ecommerce_analysis.experiments import (
    analyze_binary_ab,
    required_sample_size_per_group,
    simulate_retention_experiment,
)


def test_contingency_table_is_derived_from_generated_groups():
    control = np.array([1, 0, 0, 1, 0])
    treatment = np.array([1, 1, 1, 0, 0])
    result = analyze_binary_ab(control, treatment)

    assert result.contingency.loc["control"].tolist() == [2, 3]
    assert result.contingency.loc["treatment"].tolist() == [3, 2]
    assert result.absolute_lift == pytest.approx(0.2)


def test_simulation_is_reproducible_and_matches_documented_example():
    first = simulate_retention_experiment()
    second = simulate_retention_experiment()

    assert first.contingency.equals(second.contingency)
    assert first.contingency.loc["control", "returned"] == 5
    assert first.contingency.loc["treatment", "returned"] == 16


def test_power_calculation_returns_practical_group_size():
    assert required_sample_size_per_group(0.10, 0.15) > 84
