import numpy as np

from experiments.bottleneck_eda import (
    population_stability_index,
    ratio_bin_diagnostics,
    stable_binary_auc,
)


def test_population_stability_is_zero_for_identical_samples() -> None:
    values = np.linspace(-2.0, 2.0, 1000)

    assert population_stability_index(values, values) == 0.0


def test_population_stability_detects_a_shift() -> None:
    reference = np.linspace(-2.0, 2.0, 1000)
    shifted = reference + 1.5

    assert population_stability_index(reference, shifted) > 0.5


def test_stable_binary_auc_keeps_first_period_direction() -> None:
    target = np.asarray([0, 0, 1, 1])
    increasing = np.asarray([0.0, 0.2, 0.8, 1.0])

    result = stable_binary_auc(increasing, target, increasing, target)

    assert result["direction"] == 1
    assert result["stable_auc"] == 1.0


def test_ratio_bins_expose_high_power_underprediction() -> None:
    capacity = 100.0
    truth = np.asarray([85.0, 90.0, 95.0])
    prediction = np.asarray([70.0, 70.0, 70.0])

    rows = ratio_bin_diagnostics(truth, prediction, capacity)
    high_rows = [row for row in rows if row["bin"].startswith("0.8-") or row["bin"].startswith("0.9-")]

    assert high_rows
    assert all(row["bias_kwh"] < 0.0 for row in high_rows)
    assert all(row["miss_8pct_fraction"] == 1.0 for row in high_rows)
