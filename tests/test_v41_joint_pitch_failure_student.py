import numpy as np

from src.v41_joint_pitch_failure_student import (
    N_CLASSES,
    _load_year_cache,
    _save_year_cache,
    conditional_rates,
    joint_label,
    raw_predictions,
    temperature,
)


def test_joint_label_maps_type_major_and_preserves_missing():
    output = joint_label(np.array([0, 1, 2, -1]), np.array([3, 0, 2, 1]))
    np.testing.assert_array_equal(output, [3, 4, 10, -1])


def test_temperature_is_normalized_and_order_preserving():
    probability = np.array([[0.1, 0.3, 0.6]])
    transformed = temperature(probability, 1.25)
    np.testing.assert_allclose(transformed.sum(axis=1), 1.0)
    assert np.array_equal(np.argsort(transformed[0]), np.argsort(probability[0]))


def test_conditional_rates_and_raw_predictions_are_finite():
    label = np.arange(N_CLASSES, dtype=np.int16)
    target = (label % 4 >= 2).astype(float)
    mode_rate, joint_rate = conditional_rates(
        target, label, np.ones(N_CLASSES)
    )
    probability = np.full((2, N_CLASSES), 1.0 / N_CLASSES)
    mode_raw, joint_raw = raw_predictions(probability, mode_rate, joint_rate)
    assert np.isfinite(mode_rate).all()
    assert np.isfinite(joint_rate).all()
    assert np.isfinite(mode_raw).all()
    assert np.isfinite(joint_raw).all()


def test_year_cache_round_trip(tmp_path):
    target = np.array([0.0, 1.0])
    bank = {"a": np.array([0.2, 0.8]), "b": np.array([0.3, 0.7])}
    metrics = [{"audit_year": 2022.0, "joint_accuracy": 0.25}]
    path = tmp_path / "raw_o2022.npz"
    _save_year_cache(path, target, bank, metrics)
    loaded, loaded_metrics = _load_year_cache(path, target)
    assert set(loaded) == set(bank)
    for name in bank:
        np.testing.assert_allclose(loaded[name], bank[name])
    assert loaded_metrics == metrics
