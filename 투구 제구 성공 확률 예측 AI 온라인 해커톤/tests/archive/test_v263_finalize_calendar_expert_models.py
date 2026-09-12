import numpy as np

from src.archive.v263_finalize_calendar_expert_models import deployment_fit_mask


def test_deployment_fit_mask_matches_strict_futures_policy() -> None:
    season = np.array([2021, 2022, 2023, 2024, 2025])
    is_futures = np.array([False, True, True, False, False])
    assert deployment_fit_mask(season, is_futures).tolist() == [
        True,
        False,
        True,
        True,
        False,
    ]
