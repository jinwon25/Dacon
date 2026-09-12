import numpy as np

from src.archive.v260_mechanism_complement_experts import compose_experts


def _routes(n: int) -> dict[str, np.ndarray]:
    return {
        "deployed": np.ones(n, dtype=bool),
        "pressure_boundary_agreement": np.zeros(n, dtype=bool),
        "nonpressure_same_hand": np.zeros(n, dtype=bool),
        "nonpressure_opposite_hand_high52": np.zeros(n, dtype=bool),
    }


def test_command_and_complement_experts_are_disjoint() -> None:
    parent = np.array([0.4, 0.4])
    base = np.array([0.6, 0.6])
    # Row 0 command moves toward center; row 1 moves farther away.
    command = np.array([0.5, 0.7])
    complement = np.array([0.2, 0.2])
    _baseline, candidate, selected, command_scope = compose_experts(
        parent, base, command, [complement], _routes(2)
    )
    assert command_scope.tolist() == [True, False]
    assert selected.tolist() == [True, True]
    # Deployed route weight is 0.30. Fallbacks are 0.54 and 0.50.
    np.testing.assert_allclose(candidate, [0.442, 0.43])


def test_command_only_leaves_complement_rows_at_v244_baseline() -> None:
    parent = np.array([0.4, 0.4])
    base = np.array([0.6, 0.6])
    command = np.array([0.5, 0.7])
    baseline, candidate, selected, _scope = compose_experts(
        parent, base, command, [], _routes(2)
    )
    assert selected.tolist() == [True, False]
    assert candidate[1] == baseline[1]
