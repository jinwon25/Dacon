from src.archive.v269_pseudo_deployment_lightgbm_oof import PROTOCOL


def test_v269_protocol_is_distinct_pseudo_deployment_family() -> None:
    assert "PSEUDO_DEPLOYMENT_LIGHTGBM" in PROTOCOL
