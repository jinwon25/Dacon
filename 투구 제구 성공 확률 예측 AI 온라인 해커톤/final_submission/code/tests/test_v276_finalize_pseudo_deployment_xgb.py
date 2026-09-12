from src.archive.v276_finalize_pseudo_deployment_xgb import FIT_YEARS


def test_v276_final_fit_uses_all_available_pseudo_years_once() -> None:
    assert FIT_YEARS == (2021, 2022, 2023, 2024)
