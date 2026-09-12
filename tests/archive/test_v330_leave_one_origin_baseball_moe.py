from src.archive.v330_leave_one_origin_baseball_moe import ORIGINS, PROTOCOL


def test_v330_protocol_has_three_origins() -> None:
    assert PROTOCOL == "V330_LEAVE_ONE_ORIGIN_BASEBALL_MOE_V1"
    assert ORIGINS == ("full_2022", "late_2023", "full_2024")
