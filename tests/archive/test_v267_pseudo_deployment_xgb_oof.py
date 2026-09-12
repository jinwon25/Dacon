import pytest

from src.archive.v267_pseudo_deployment_xgb_oof import pseudo_fit_years


def test_pseudo_fit_years_are_strictly_prior_and_expand_forward() -> None:
    assert pseudo_fit_years(2022) == (2021,)
    assert pseudo_fit_years(2024) == (2021, 2022, 2023)


def test_pseudo_fit_years_reject_start_year() -> None:
    with pytest.raises(ValueError, match="follow"):
        pseudo_fit_years(2021)
