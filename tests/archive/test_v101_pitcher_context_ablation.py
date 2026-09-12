import pandas as pd

from src.archive.v101_pitcher_context_ablation import _contexts


def test_baseball_context_bands_are_deterministic():
    frame = pd.DataFrame(
        {
            "num_runners_on": [0, 1, 2],
            "inning": [3, 4, 8],
            "score_diff_pitcher_team": [1, -3, 5],
        }
    )
    out = _contexts(frame)
    assert out["stretch"].tolist() == [0, 1, 1]
    assert out["inning_band"].tolist() == [0, 1, 2]
    assert out["score_band"].tolist() == [0, 1, 2]
