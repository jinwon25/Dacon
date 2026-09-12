import numpy as np
import pandas as pd

from src.archive.v29_anchor_route_screen import _candidate


def test_candidate_preserves_non_anchor_rows_and_matches_parent_at_point_one() -> None:
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.5, 0.6],
            "domain3": ["R_ANCHOR", "R_CORE", "R_ANCHOR"],
        }
    )
    direct = np.array([0.8, 0.9, 0.2])
    route = np.array([True, False, False])
    parent, candidate = _candidate(frame, direct, route, 0.1, 0.1)
    assert np.allclose(candidate, parent)
    assert candidate[1] == 0.5
