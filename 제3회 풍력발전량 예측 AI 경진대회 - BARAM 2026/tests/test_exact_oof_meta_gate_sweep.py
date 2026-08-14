from experiments.exact_oof_meta_gate_sweep import (
    Policy,
    default_alphas,
    default_thresholds,
    select_policy,
)


def _record(
    threshold: float,
    alpha: float,
    score: float,
    minimum: float,
    movement: float,
) -> dict[str, object]:
    return {
        "policy": Policy(threshold, alpha).to_dict(),
        "metrics": {
            "delta": {"score": score, "one_minus_nmae": 0.001, "ficr": 0.001}
        },
        "min_seed_score_delta": minimum,
        "months_improved": 2,
        "mean_absolute_movement_kwh": movement,
    }


def test_fine_grid_contains_reference_policy() -> None:
    assert 0.55 in default_thresholds()
    assert 0.25 in default_alphas()
    assert len(default_thresholds()) == 31
    assert len(default_alphas()) == 17


def test_policy_selection_prioritizes_worst_seed_stability() -> None:
    high_mean = _record(0.54, 0.30, score=0.003, minimum=0.0002, movement=15.0)
    stable = _record(0.56, 0.20, score=0.002, minimum=0.0004, movement=10.0)
    assert select_policy([high_mean, stable])["policy"] == stable["policy"]


def test_policy_selection_rejects_negative_component() -> None:
    rejected = _record(0.54, 0.30, score=0.003, minimum=0.001, movement=15.0)
    rejected["metrics"]["delta"]["ficr"] = -0.001
    accepted = _record(0.56, 0.20, score=0.002, minimum=0.0004, movement=10.0)
    assert select_policy([rejected, accepted])["policy"] == accepted["policy"]
