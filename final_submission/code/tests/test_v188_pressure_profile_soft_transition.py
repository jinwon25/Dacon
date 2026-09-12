import pandas as pd

from src.archive.v188_pressure_profile_soft_transition import soft_source_candidates


def test_soft_source_gate_keeps_only_small_bounded_candidate() -> None:
    frame = pd.DataFrame(
        {
            "minimum_gain": [-0.3, -0.6, -0.2, -0.2],
            "mean_gain": [0.1, 0.2, -0.1, 0.1],
            "worst_month_gain": [-0.8, -0.5, -0.5, -1.2],
            "eta": [0.05, 0.05, 0.05, 0.05],
        }
    )
    selected = soft_source_candidates(frame)
    assert selected.index.tolist() == [0]
