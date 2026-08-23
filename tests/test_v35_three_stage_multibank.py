import numpy as np
import pandas as pd

from src.core.banks import signal_family
from src.v35_three_stage_multibank import _candidate, family_prefilter


def test_signal_family_uses_stable_prefix():
    assert signal_family("exact::l2_leaves7") == "exact"
    assert signal_family("mode::conditional_mode") == "mode"


def test_family_prefilter_enforces_equal_identity_quota():
    rows = []
    for family in ("exact", "mode"):
        for index in range(4):
            rows.append(
                {
                    "signal": f"{family}::{index}",
                    "family": family,
                    "selection_score": 10.0 - index,
                    "gain": 20.0 - index,
                }
            )
    selected = family_prefilter(pd.DataFrame(rows), quota=2)
    assert selected == ["exact::0", "exact::1", "mode::0", "mode::1"]


def test_disjoint_members_add_their_declared_directions():
    frame = pd.DataFrame(
        {
            "v22": [0.4, 0.4, 0.4],
            "v25": [0.4, 0.4, 0.4],
            "v21": [0.3, 0.3, 0.3],
            "domain3": ["R_CORE", "R_ANCHOR", "F"],
        }
    )
    bank = {
        "exact::x": np.array([0.6, 0.6, 0.6]),
        "mode::y": np.array([0.5, 0.5, 0.5]),
    }
    members = [
        {
            "signal": "exact::x",
            "direction": "toward_parent",
            "domain": "R_CORE",
            "weight": 0.1,
        },
        {
            "signal": "mode::y",
            "direction": "delta_v21",
            "domain": "F",
            "weight": 0.2,
        },
    ]
    candidate, active = _candidate(frame, bank, members)
    assert np.allclose(candidate, [0.42, 0.4, 0.44])
    assert active.tolist() == [True, False, True]
