from src.archive.v274_pseudo_deployment_trackman_expert_oof import EXPERTS


def test_v274_experts_have_distinct_reproducible_stems_and_seeds() -> None:
    assert EXPERTS["command"]["stem"] != EXPERTS["batter"]["stem"]
    assert EXPERTS["command"]["random_state"] != EXPERTS["batter"]["random_state"]
