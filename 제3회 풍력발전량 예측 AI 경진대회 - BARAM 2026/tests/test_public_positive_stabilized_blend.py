import numpy as np
import pytest

from experiments.public_positive_stabilized_blend import compose_candidate


TARGETS = ("kpx_group_1", "kpx_group_2", "kpx_group_3")


def _surface(value: float) -> dict[str, np.ndarray]:
    return {target: np.asarray([value, value + 1.0]) for target in TARGETS}


def test_compose_candidate_adds_weighted_stabilizer_and_freezes_group3() -> None:
    active = _surface(10.0)
    exact = _surface(14.0)
    trajectory = _surface(6.0)
    group1 = np.asarray([20.0, 21.0])
    group2 = np.asarray([30.0, 31.0])

    result = compose_candidate(
        active,
        group1,
        group2,
        exact,
        trajectory,
        stabilizer_alpha=0.5,
        exact_share=0.75,
    )

    # Stabilizer displacement is .75 * 4 + .25 * -4 = 2; alpha adds 1.
    np.testing.assert_allclose(result["kpx_group_1"], [21.0, 22.0])
    np.testing.assert_allclose(result["kpx_group_2"], [31.0, 32.0])
    np.testing.assert_array_equal(result["kpx_group_3"], active["kpx_group_3"])


@pytest.mark.parametrize("exact_share", [-0.01, 1.01])
def test_compose_candidate_rejects_invalid_exact_share(exact_share: float) -> None:
    active = _surface(10.0)
    with pytest.raises(ValueError, match="exact share"):
        compose_candidate(
            active,
            np.asarray([10.0, 11.0]),
            np.asarray([10.0, 11.0]),
            active,
            active,
            stabilizer_alpha=0.01,
            exact_share=exact_share,
        )
